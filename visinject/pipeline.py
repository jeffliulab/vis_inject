"""
VisInject End-to-End Pipeline
==============================
Combines UniversalAttack (multi-model pixel optimization) with AnyAttack_LAION400M
(decoder-based noise fusion) into a single pipeline.

Input:  target_prompt (string) + clean_image(s) (paths)
Output: adversarial image(s) that look natural but hijack VLM responses

Pipeline:
  1. UniversalAttack: optimize a universal adversarial image for the target prompt
  2. AnyAttack fusion: encode universal image via CLIP, decode noise, apply to clean images
  3. (Optional) Evaluate: test adversarial images against target VLMs

Usage:
    # Full pipeline
    python pipeline.py --target-phrase "Sure, here it is" \
                       --clean-images ../demos/demo_images/ORIGIN_dog.png

    # Skip UniversalAttack (reuse existing universal image)
    python pipeline.py --universal-image outputs/universal/universal_final.png \
                       --clean-images ../demos/demo_images/ORIGIN_dog.png

    # With evaluation
    python pipeline.py --target-phrase "Sure, here it is" \
                       --clean-images ../demos/demo_images/ORIGIN_dog.png \
                       --evaluate
"""

import argparse
import hashlib
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    UNIVERSAL_ATTACK_CONFIG, ATTACK_TARGETS, ANYATTACK_CONFIG, OUTPUT_CONFIG,
)

# Import UniversalAttack components from demos using isolated loading
# to avoid module name collisions (config, models, evaluate, dataset).
from utils import _import_from_file

_demo_s3_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demos", "demo_S3_UniversalAttack"))

# attack.py has complex imports (config, models/, etc.) so we must let it
# resolve its own dependencies via sys.path — but sandboxed.
_saved_modules = {k: sys.modules.pop(k) for k in list(sys.modules) if k in ("config", "models", "dataset")}
_saved_path = sys.path.copy()
sys.path.insert(0, _demo_s3_dir)
_attack_mod = _import_from_file("_s3_attack", os.path.join(_demo_s3_dir, "attack.py"))
_dataset_mod = _import_from_file("_s3_dataset", os.path.join(_demo_s3_dir, "dataset.py"))
get_wrapper_for_model = _attack_mod.get_wrapper_for_model
apply_gaussian_blur = _attack_mod.apply_gaussian_blur
compute_quantization_sigma = _attack_mod.compute_quantization_sigma
AttackDataset = _dataset_mod.AttackDataset
# Restore saved modules but keep _demo_s3_dir in sys.path — it's needed
# at runtime when get_wrapper_for_model() calls importlib.import_module("models.xxx")
sys.modules.update(_saved_modules)


def _cache_key(target_phrase: str, target_models: list[str]) -> str:
    """Generate a short hash for caching universal attack results."""
    content = f"{target_phrase}|{'_'.join(sorted(target_models))}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def run_universal_attack(
    target_phrase: str,
    target_models: list[str],
    config: dict,
    output_dir: str,
    checkpoint_dir: str,
    device: torch.device,
    log_every: int = 50,
    save_every: int = 200,
) -> str:
    """
    Step 1: Optimize a universal adversarial image via multi-model pixel optimization.

    Returns path to the saved universal image.
    """
    # Check cache
    cache_id = _cache_key(target_phrase, target_models)
    cached_ckpt = os.path.join(checkpoint_dir, f"universal_{cache_id}.pt")
    cached_img = os.path.join(output_dir, f"universal_{cache_id}.png")

    if os.path.exists(cached_img) and os.path.exists(cached_ckpt):
        print(f"[Cache hit] Reusing universal image: {cached_img}")
        return cached_img

    is_multi = len(target_models) > 1
    gamma = config.get("gamma_multi") if is_multi else config.get("gamma_single")
    H, W = config["image_size"]
    num_steps = config["num_steps"]

    # Initialize adversarial image: z = z0 + gamma * tanh(z1)
    z0 = torch.full((1, 3, H, W), 0.5, device=device)
    z1 = torch.randn(1, 3, H, W, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([z1], lr=config["lr"])
    dataset = AttackDataset()
    sigma = 1.0 / 255

    # Load target models
    print(f"\nLoading {len(target_models)} target model(s)...")
    wrappers = {}
    for model_key in target_models:
        print(f"  Loading {model_key}...")
        wrappers[model_key] = get_wrapper_for_model(model_key, device)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\nStarting UniversalAttack optimization:")
    print(f"  Steps: {num_steps} | Gamma: {gamma} | Mode: {'multi' if is_multi else 'single'}")
    print(f"  Target: \"{target_phrase}\"")
    print(f"  Models: {target_models}")
    print(f"  Dataset: {dataset.num_safe} safe + {dataset.num_adversarial} adversarial questions\n")

    best_loss = float("inf")
    t0 = time.time()

    for step in range(num_steps):
        question = dataset.sample()

        if config.get("multi_answer"):
            target = random.choice(config["answer_pool"])
        else:
            target = target_phrase

        perturbation = gamma * torch.tanh(z1)

        if config.get("gaussian_blur"):
            perturbation = apply_gaussian_blur(
                perturbation,
                kernel_size=config["blur_kernel_size"],
                sigma=config["blur_sigma"],
            )

        z_clean = z0 + perturbation

        if config.get("quant_robustness"):
            noise = torch.randn_like(z1) * sigma
            z = torch.clamp(z_clean + noise, 0, 1)
        else:
            z = torch.clamp(z_clean, 0, 1)

        if config.get("localize"):
            scale = random.uniform(config["localize_scale_min"], config["localize_scale_max"])
            crop_h, crop_w = int(H * scale), int(W * scale)
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
            z_crop = z[:, :, top:top + crop_h, left:left + crop_w]
            z_input = F.interpolate(z_crop, size=(H, W), mode="bilinear", align_corners=False)
        else:
            z_input = z

        total_loss = torch.tensor(0.0, device=device)
        for model_key, wrapper in wrappers.items():
            loss = wrapper.compute_masked_ce_loss(z_input, question, target)
            total_loss = total_loss + loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if config.get("quant_robustness") and step % 10 == 0:
            with torch.no_grad():
                z_for_sigma = torch.clamp(z0 + gamma * torch.tanh(z1), 0, 1)
                sigma = compute_quantization_sigma(z_for_sigma)

        loss_val = total_loss.item()
        if loss_val < best_loss:
            best_loss = loss_val

        if step % log_every == 0:
            elapsed = time.time() - t0
            print(f"  Step {step:5d}/{num_steps} | Loss: {loss_val:.4f} | "
                  f"Best: {best_loss:.4f} | Time: {elapsed:.0f}s")

        if step % save_every == 0 and step > 0:
            with torch.no_grad():
                z_save = torch.clamp(z0 + gamma * torch.tanh(z1), 0, 1)
                path = os.path.join(output_dir, f"universal_step{step:05d}.png")
                torchvision.utils.save_image(z_save[0], path)

    # Save final image and checkpoint
    with torch.no_grad():
        z_final = torch.clamp(z0 + gamma * torch.tanh(z1), 0, 1)
        torchvision.utils.save_image(z_final[0], cached_img)

        torch.save({
            "z0": z0.cpu(), "z1": z1.cpu(), "gamma": gamma,
            "step": num_steps, "best_loss": best_loss,
            "target_models": target_models, "target_phrase": target_phrase,
        }, cached_ckpt)

    # Unload all models
    for wrapper in wrappers.values():
        wrapper.unload()
    del wrappers
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\nUniversalAttack done! {num_steps} steps in {elapsed:.0f}s | Best loss: {best_loss:.4f}")
    print(f"  Image: {cached_img}")
    print(f"  Checkpoint: {cached_ckpt}")

    return cached_img


def run_anyattack_fusion(
    universal_image_path: str,
    clean_image_paths: list[str],
    config: dict,
    output_dir: str,
    device: torch.device,
) -> list[str]:
    """
    Step 2: Fuse universal adversarial semantics into clean images via AnyAttack Decoder.

    Returns list of adversarial image paths.
    """
    from generate import generate
    return generate(
        universal_image_path=universal_image_path,
        clean_image_paths=clean_image_paths,
        decoder_path=config["decoder_path"],
        clip_model=config["clip_model"],
        embed_dim=config["embed_dim"],
        eps=config["eps"],
        image_size=config["image_size"],
        output_dir=output_dir,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="VisInject end-to-end pipeline")
    ua_cfg = UNIVERSAL_ATTACK_CONFIG
    aa_cfg = ANYATTACK_CONFIG
    out_cfg = OUTPUT_CONFIG

    # UniversalAttack args
    parser.add_argument("--target-phrase", type=str, default=ua_cfg["target_phrase"])
    parser.add_argument("--target-models", nargs="+", default=ATTACK_TARGETS)
    parser.add_argument("--num-steps", type=int, default=ua_cfg["num_steps"])
    parser.add_argument("--image-size", type=int, nargs=2, default=list(ua_cfg["image_size"]))

    # AnyAttack fusion args
    parser.add_argument("--clean-images", type=str, nargs="+", required=True)
    parser.add_argument("--decoder-path", type=str, default=aa_cfg["decoder_path"])
    parser.add_argument("--eps", type=float, default=aa_cfg["eps"])

    # Skip UniversalAttack if universal image already exists
    parser.add_argument("--universal-image", type=str, default=None,
                        help="Skip UniversalAttack, use this universal image directly")

    # Evaluation
    parser.add_argument("--evaluate", action="store_true", help="Run legacy ASR evaluation")
    parser.add_argument("--generate-pairs", action="store_true",
                        help="Generate response pairs (clean vs adv) for LLM-as-judge evaluation")
    parser.add_argument("--eval-vlms", nargs="+", default=None,
                        help="VLMs to evaluate against (default: from config)")

    # General
    parser.add_argument("--output-dir", type=str, default=out_cfg["base_dir"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"=" * 60)

    # ── Step 1: UniversalAttack ──
    if args.universal_image and os.path.exists(args.universal_image):
        universal_path = args.universal_image
        print(f"\n[Step 1] Skipping UniversalAttack, using: {universal_path}")
    else:
        print(f"\n[Step 1] Running UniversalAttack...")
        ua_config = dict(ua_cfg)
        ua_config["image_size"] = tuple(args.image_size)
        ua_config["num_steps"] = args.num_steps
        universal_path = run_universal_attack(
            target_phrase=args.target_phrase,
            target_models=args.target_models,
            config=ua_config,
            output_dir=os.path.join(args.output_dir, "universal"),
            checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
            device=device,
            log_every=out_cfg["log_every"],
            save_every=out_cfg["save_every"],
        )

    # ── Step 2: AnyAttack Fusion ──
    print(f"\n[Step 2] Running AnyAttack fusion...")
    aa_config = dict(aa_cfg)
    aa_config["decoder_path"] = args.decoder_path
    aa_config["eps"] = args.eps

    adv_paths = run_anyattack_fusion(
        universal_image_path=universal_path,
        clean_image_paths=args.clean_images,
        config=aa_config,
        output_dir=os.path.join(args.output_dir, "adversarial"),
        device=device,
    )

    # ── Step 3: Generate response pairs for LLM-as-judge (optional) ──
    if args.generate_pairs:
        print(f"\n[Step 3] Generating response pairs for LLM-as-judge...")
        from evaluate import generate_response_pairs
        eval_vlms = args.eval_vlms or args.target_models
        results_dir = os.path.join(args.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        for adv_path, clean_path in zip(adv_paths, args.clean_images):
            name = os.path.splitext(os.path.basename(clean_path))[0]
            pairs_path = os.path.join(results_dir, f"response_pairs_{name}.json")
            generate_response_pairs(
                adv_image_path=adv_path,
                clean_image_path=clean_path,
                target_phrase=args.target_phrase,
                target_vlms=eval_vlms,
                device=device,
                output_path=pairs_path,
            )

    # ── Step 3b: Legacy ASR evaluation (optional) ──
    if args.evaluate:
        print(f"\n[Step 3b] Running legacy ASR evaluation...")
        from evaluate import run_evaluation
        run_evaluation(
            adv_paths=adv_paths,
            clean_paths=args.clean_images,
            universal_path=universal_path,
            target_phrase=args.target_phrase,
            output_dir=os.path.join(args.output_dir, "results"),
            device=device,
            eval_vlms=args.eval_vlms,
        )

    print(f"\n{'=' * 60}")
    print(f"VisInject pipeline complete!")
    print(f"  Universal image: {universal_path}")
    print(f"  Adversarial images: {adv_paths}")
    if args.generate_pairs:
        print(f"  Response pairs: {os.path.join(args.output_dir, 'results', 'response_pairs_*.json')}")
    if args.evaluate:
        print(f"  Legacy results: {os.path.join(args.output_dir, 'results')}")


if __name__ == "__main__":
    main()
