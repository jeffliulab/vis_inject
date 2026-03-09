"""
Universal Adversarial Attack: core pixel optimization loop.

Optimizes a single synthetic image z = z0 + gamma*tanh(z1) to force
one or more MLLMs to respond with a target phrase for any question.

Supports:
  - Single-model and multi-model attacks (via config.target_models)
  - Quantization robustness (random noise calibrated to int8 error)
  - Multi-answer attack (randomly sampled target phrases)
  - Gaussian blur on perturbation
  - Localization (random crop)

Usage:
    python attack.py
    python attack.py --num-steps 3000 --target-models qwen2_5_vl_3b phi_3_5_vision
"""

import argparse
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.dirname(__file__))
from config import ATTACK_CONFIG, OUTPUT_CONFIG
from dataset import AttackDataset


FAMILY_WRAPPER_MAP = {
    "qwen":     ("models.qwen_wrapper",    "QwenWrapper"),
    "blip2":    ("models.blip2_wrapper",   "Blip2Wrapper"),
    "deepseek": ("models.deepseek_wrapper", "DeepSeekWrapper"),
    "llava":    ("models.llava_wrapper",   "LlavaWrapper"),
    "phi":      ("models.phi_wrapper",     "PhiWrapper"),
    "llama":    ("models.llama_wrapper",   "LlamaVisionWrapper"),
}


def get_wrapper_for_model(model_key: str, device: torch.device):
    """Instantiate the appropriate wrapper based on model family."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from model_registry import get_model_info
    info = get_model_info(model_key)
    family = info["family"]

    if family not in FAMILY_WRAPPER_MAP:
        available = list(FAMILY_WRAPPER_MAP.keys())
        raise NotImplementedError(
            f"No wrapper for family '{family}'. "
            f"Available families: {available}. "
            f"Implement a wrapper in models/ and add it to FAMILY_WRAPPER_MAP."
        )

    module_path, class_name = FAMILY_WRAPPER_MAP[family]
    import importlib
    mod = importlib.import_module(module_path)
    WrapperCls = getattr(mod, class_name)

    wrapper = WrapperCls(model_key, device)
    wrapper.load()
    return wrapper


def apply_gaussian_blur(perturbation: torch.Tensor,
                        kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Apply Gaussian blur to the perturbation tensor."""
    channels = perturbation.shape[1]
    x = torch.arange(kernel_size, dtype=torch.float32, device=perturbation.device)
    x = x - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    return F.conv2d(perturbation, kernel, padding=pad, groups=channels)


def compute_quantization_sigma(z_float: torch.Tensor) -> float:
    """Calibrate noise sigma to match int8 quantization error."""
    with torch.no_grad():
        z_quant = (z_float * 255).round() / 255
        return (z_float - z_quant).std().item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine gamma
    is_multi = len(args.target_models) > 1
    if args.gamma is not None:
        gamma = args.gamma
    else:
        gamma = ATTACK_CONFIG["gamma_multi"] if is_multi else ATTACK_CONFIG["gamma_single"]
    print(f"Mode: {'multi-model' if is_multi else 'single-model'}, gamma={gamma}")

    # Image size
    H, W = args.image_size

    # Initialize adversarial image: z = z0 + gamma * tanh(z1)
    z0 = torch.full((1, 3, H, W), 0.5, device=device)
    z1 = torch.randn(1, 3, H, W, device=device, requires_grad=True)

    optimizer = torch.optim.AdamW([z1], lr=args.lr)
    dataset = AttackDataset()
    sigma = 1.0 / 255  # initial quantization noise

    # Load target models
    print(f"\nLoading {len(args.target_models)} target model(s)...")
    wrappers = {}
    for model_key in args.target_models:
        wrappers[model_key] = get_wrapper_for_model(model_key, device)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\nStarting optimization: {args.num_steps} steps")
    print(f"Target phrase: \"{args.target_phrase}\"")
    print(f"Dataset: {dataset.num_safe} safe + {dataset.num_adversarial} adversarial questions")
    print()

    best_loss = float("inf")
    t0 = time.time()

    for step in range(args.num_steps):
        question = dataset.sample()

        # Select target answer
        if args.multi_answer:
            target = random.choice(ATTACK_CONFIG["answer_pool"])
        else:
            target = args.target_phrase

        # Construct adversarial image
        perturbation = gamma * torch.tanh(z1)

        if args.gaussian_blur:
            perturbation = apply_gaussian_blur(
                perturbation,
                kernel_size=ATTACK_CONFIG["blur_kernel_size"],
                sigma=ATTACK_CONFIG["blur_sigma"],
            )

        z_clean = z0 + perturbation

        # Quantization robustness: add calibrated noise
        if args.quant_robustness:
            noise = torch.randn_like(z1) * sigma
            z = torch.clamp(z_clean + noise, 0, 1)
        else:
            z = torch.clamp(z_clean, 0, 1)

        # Localization: random crop
        if args.localize:
            scale = random.uniform(
                ATTACK_CONFIG["localize_scale_min"],
                ATTACK_CONFIG["localize_scale_max"],
            )
            crop_h, crop_w = int(H * scale), int(W * scale)
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
            z_crop = z[:, :, top:top + crop_h, left:left + crop_w]
            z_input = F.interpolate(z_crop, size=(H, W), mode="bilinear",
                                    align_corners=False)
        else:
            z_input = z

        # Compute loss across all target models
        total_loss = torch.tensor(0.0, device=device)
        for model_key, wrapper in wrappers.items():
            loss = wrapper.compute_masked_ce_loss(z_input, question, target)
            total_loss = total_loss + loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update quantization sigma
        if args.quant_robustness and step % 10 == 0:
            with torch.no_grad():
                z_for_sigma = torch.clamp(z0 + gamma * torch.tanh(z1), 0, 1)
                sigma = compute_quantization_sigma(z_for_sigma)

        loss_val = total_loss.item()
        if loss_val < best_loss:
            best_loss = loss_val

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"Step {step:5d}/{args.num_steps} | "
                  f"Loss: {loss_val:.4f} | Best: {best_loss:.4f} | "
                  f"Sigma: {sigma:.6f} | Time: {elapsed:.0f}s | "
                  f"Q: {question[:50]}...")

        # Save intermediate images
        if step % args.save_every == 0 and step > 0:
            with torch.no_grad():
                z_save = torch.clamp(z0 + gamma * torch.tanh(z1), 0, 1)
                path = os.path.join(args.output_dir, f"universal_step{step:05d}.png")
                torchvision.utils.save_image(z_save[0], path)

    # Final save
    with torch.no_grad():
        z_final = torch.clamp(z0 + gamma * torch.tanh(z1), 0, 1)
        final_path = os.path.join(args.output_dir, "universal_final.png")
        torchvision.utils.save_image(z_final[0], final_path)

        # Also save quantized version
        z_quant = (z_final * 255).round().byte()
        quant_path = os.path.join(args.output_dir, "universal_final_quant.png")
        torchvision.utils.save_image(z_quant.float() / 255, quant_path)

        # Save checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, "universal_attack.pt")
        torch.save({
            "z0": z0.cpu(),
            "z1": z1.cpu(),
            "gamma": gamma,
            "step": args.num_steps,
            "best_loss": best_loss,
            "target_models": args.target_models,
            "target_phrase": args.target_phrase,
        }, ckpt_path)

    elapsed = time.time() - t0
    print(f"\nDone! {args.num_steps} steps in {elapsed:.0f}s")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final image: {final_path}")
    print(f"Checkpoint:  {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Adversarial Attack")
    cfg = ATTACK_CONFIG
    out = OUTPUT_CONFIG
    parser.add_argument("--target-models", nargs="+", default=cfg["target_models"])
    parser.add_argument("--gamma", type=float, default=cfg["gamma"])
    parser.add_argument("--lr", type=float, default=cfg["lr"])
    parser.add_argument("--num-steps", type=int, default=cfg["num_steps"])
    parser.add_argument("--target-phrase", type=str, default=cfg["target_phrase"])
    parser.add_argument("--image-size", type=int, nargs=2, default=list(cfg["image_size"]))
    parser.add_argument("--quant-robustness", action="store_true",
                        default=cfg["quant_robustness"])
    parser.add_argument("--no-quant-robustness", action="store_false",
                        dest="quant_robustness")
    parser.add_argument("--gaussian-blur", action="store_true",
                        default=cfg["gaussian_blur"])
    parser.add_argument("--multi-answer", action="store_true",
                        default=cfg["multi_answer"])
    parser.add_argument("--localize", action="store_true", default=cfg["localize"])
    parser.add_argument("--output-dir", type=str, default=out["output_dir"])
    parser.add_argument("--checkpoint-dir", type=str, default=out["checkpoint_dir"])
    parser.add_argument("--log-every", type=int, default=out["log_every"])
    parser.add_argument("--save-every", type=int, default=out["save_every"])
    main(parser.parse_args())
