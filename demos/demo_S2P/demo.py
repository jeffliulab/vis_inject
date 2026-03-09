"""
Generate adversarial images using official AnyAttack weights.

Usage:
    python demo.py --clean-image ../demo_images/ORIGIN_dog.png \
                   --target-image ../demo_images/ORIGIN_cat.png

    python demo.py --clean-image ../demo_images/ORIGIN_dog.png \
                   --target-image ../demo_images/ORIGIN_cat.png \
                   --decoder-path checkpoints/coco_cos.pt
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.dirname(__file__))
from config import ATTACK_CONFIG, WEIGHTS_CONFIG, OUTPUT_CONFIG

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_S2_AnyAttack"))
from models import CLIPEncoder, Decoder
from dataset import load_image


def main():
    parser = argparse.ArgumentParser(description="AnyAttack demo (official weights)")
    parser.add_argument("--clean-image", type=str, required=True)
    parser.add_argument("--target-image", type=str, required=True)
    parser.add_argument("--decoder-path", type=str, default=WEIGHTS_CONFIG["local_path"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--eps", type=float, default=ATTACK_CONFIG["eps"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if not os.path.exists(args.decoder_path):
        print(f"Checkpoint not found: {args.decoder_path}")
        print("Run 'python download_weights.py' first.")
        sys.exit(1)

    if args.output is None:
        os.makedirs(OUTPUT_CONFIG["output_dir"], exist_ok=True)
        args.output = os.path.join(OUTPUT_CONFIG["output_dir"], "adversarial.png")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading CLIP {ATTACK_CONFIG['clip_model']}...")
    clip_encoder = CLIPEncoder(ATTACK_CONFIG["clip_model"]).to(device)

    print(f"Loading Decoder: {os.path.basename(args.decoder_path)}...")
    decoder = Decoder(embed_dim=ATTACK_CONFIG["embed_dim"]).to(device).eval()
    ckpt = torch.load(args.decoder_path, map_location="cpu", weights_only=False)
    state = ckpt.get("decoder_state_dict", ckpt)
    remapped = {}
    for k, v in state.items():
        k = k.removeprefix("module.")
        k = k.replace("upsample_blocks.", "blocks.")
        k = k.replace("final_conv.", "head.")
        remapped[k] = v
    decoder.load_state_dict(remapped)

    clean = load_image(args.clean_image).to(device)
    target = load_image(args.target_image).to(device)

    with torch.no_grad():
        emb_target = clip_encoder.encode_img(target)
        emb_clean = clip_encoder.encode_img(clean)

        noise = decoder(emb_target)
        noise = torch.clamp(noise, -args.eps, args.eps)
        adv = torch.clamp(clean + noise, 0, 1)

        emb_adv = clip_encoder.encode_img(adv)

        emb_clean_n = F.normalize(emb_clean, p=2, dim=1)
        emb_target_n = F.normalize(emb_target, p=2, dim=1)
        emb_adv_n = F.normalize(emb_adv, p=2, dim=1)

        sim_clean_target = (emb_clean_n * emb_target_n).sum().item()
        sim_adv_target = (emb_adv_n * emb_target_n).sum().item()
        sim_clean_adv = (emb_clean_n * emb_adv_n).sum().item()

    torchvision.utils.save_image(adv[0], args.output)

    psnr = -10 * torch.log10(torch.mean((clean - adv) ** 2)).item()

    print(f"\n{'='*50}")
    print(f"Clean image:  {args.clean_image}")
    print(f"Target image: {args.target_image}")
    print(f"Output:       {args.output}")
    print(f"{'='*50}")
    print(f"Noise L-inf:  {noise.abs().max().item():.4f} (budget: {args.eps:.4f})")
    print(f"PSNR:         {psnr:.1f} dB")
    print(f"{'='*50}")
    print(f"CLIP Similarity (clean  <-> target): {sim_clean_target:.4f}")
    print(f"CLIP Similarity (adv    <-> target): {sim_adv_target:.4f}")
    print(f"CLIP Similarity (clean  <-> adv):    {sim_clean_adv:.4f}")
    print(f"Similarity shift:                    {sim_adv_target - sim_clean_target:+.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
