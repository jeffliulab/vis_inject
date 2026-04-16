"""
AnyAttack_LAION400M Fusion: inject adversarial semantics into clean images.

Takes a universal adversarial image (from UniversalAttack) and fuses its
CLIP semantics into one or more clean images via the AnyAttack Decoder.

The Decoder generates noise from the *semantic embedding* of the universal image,
not its visual appearance. All clean images share the same noise pattern.

Usage:
    python generate.py --universal-image outputs/experiments/exp_url_2m/universal/universal_<hash>.png \
                       --clean-images images/ORIGIN_dog.png

    python generate.py --universal-image outputs/experiments/exp_url_2m/universal/universal_<hash>.png \
                       --clean-images images/ORIGIN_dog.png images/ORIGIN_cat.png \
                       --decoder-path checkpoints/coco_bi.pt
"""

import argparse
import os
import sys

import torch
import torchvision

from src.config import ANYATTACK_CONFIG, OUTPUT_CONFIG
from src.utils import load_image, load_decoder, compute_psnr, CLIPEncoder


def generate(
    universal_image_path: str,
    clean_image_paths: list[str],
    decoder_path: str,
    clip_model: str,
    embed_dim: int,
    eps: float,
    image_size: int,
    output_dir: str,
    device: torch.device,
) -> list[str]:
    """
    Fuse universal adversarial semantics into clean images.

    Returns list of output file paths.
    """
    print(f"Loading CLIP {clip_model}...")
    clip_encoder = CLIPEncoder(clip_model).to(device)

    print(f"Loading Decoder: {os.path.basename(decoder_path)}...")
    decoder = load_decoder(decoder_path, embed_dim, device)

    # Load and encode universal image -> target embedding
    universal = load_image(universal_image_path, size=image_size).to(device)
    with torch.no_grad():
        emb_target = clip_encoder.encode_img(universal)
        noise = decoder(emb_target)
        noise = torch.clamp(noise, -eps, eps)

    print(f"Noise L-inf: {noise.abs().max().item():.4f} (budget: {eps:.4f})")
    print(f"Generating {len(clean_image_paths)} adversarial image(s)...\n")

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    for clean_path in clean_image_paths:
        clean = load_image(clean_path, size=image_size).to(device)

        with torch.no_grad():
            adv = torch.clamp(clean + noise, 0, 1)
            psnr = compute_psnr(clean, adv)

        basename = os.path.splitext(os.path.basename(clean_path))[0]
        out_path = os.path.join(output_dir, f"adv_{basename}.png")
        torchvision.utils.save_image(adv[0], out_path)
        output_paths.append(out_path)

        print(f"  {os.path.basename(clean_path)} -> {os.path.basename(out_path)} | PSNR: {psnr:.1f} dB")

    # Cleanup
    del clip_encoder, decoder
    torch.cuda.empty_cache()

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="AnyAttack fusion: inject adversarial semantics into clean images")
    cfg = ANYATTACK_CONFIG
    parser.add_argument("--universal-image", type=str, required=True,
                        help="Path to universal adversarial image from UniversalAttack")
    parser.add_argument("--clean-images", type=str, nargs="+", required=True,
                        help="Path(s) to clean images to inject into")
    parser.add_argument("--decoder-path", type=str, default=cfg["decoder_path"])
    parser.add_argument("--eps", type=float, default=cfg["eps"])
    parser.add_argument("--output-dir", type=str, default=OUTPUT_CONFIG["adversarial_dir"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if not os.path.exists(args.decoder_path):
        print(f"Decoder checkpoint not found: {args.decoder_path}")
        print("Download it with: python data_preparation/models/download_decoder_weights.py")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    output_paths = generate(
        universal_image_path=args.universal_image,
        clean_image_paths=args.clean_images,
        decoder_path=args.decoder_path,
        clip_model=cfg["clip_model"],
        embed_dim=cfg["embed_dim"],
        eps=args.eps,
        image_size=cfg["image_size"],
        output_dir=args.output_dir,
        device=device,
    )

    print(f"\nDone! {len(output_paths)} adversarial image(s) saved to {args.output_dir}")


if __name__ == "__main__":
    main()
