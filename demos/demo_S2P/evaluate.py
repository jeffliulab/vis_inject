"""
Evaluate adversarial images against target VLMs.

Tests whether the VLM "sees" the target image's content when looking at
the adversarial image. Generates captions for clean, adversarial, and
target images side by side.

Usage:
    python evaluate.py --adv-image outputs/adversarial.png \
                       --clean-image ../demo_images/ORIGIN_dog.png \
                       --target-image ../demo_images/ORIGIN_cat.png

    python evaluate.py --adv-image outputs/adversarial.png \
                       --clean-image ../demo_images/ORIGIN_dog.png \
                       --target-image ../demo_images/ORIGIN_cat.png \
                       --target-vlms blip2_opt_2_7b
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from config import EVAL_CONFIG, OUTPUT_CONFIG

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model_registry import get_model_info


def load_vlm(model_key: str, device):
    """Load a VLM from model_registry."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    info = get_model_info(model_key)
    hf_id = info["hf_id"]
    dtype = torch.bfloat16 if info["dtype"] == "bf16" else torch.float16

    print(f"  Loading {info['short_name']} ({hf_id})...")
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        hf_id, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )
    model.eval()
    return model, processor, info


def caption_image(model, processor, image_path: str, device) -> str:
    """Generate a caption for a single image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100)
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate AnyAttack adversarial images")
    parser.add_argument("--adv-image", type=str, required=True)
    parser.add_argument("--clean-image", type=str, default=None)
    parser.add_argument("--target-image", type=str, default=None)
    parser.add_argument("--target-vlms", nargs="+", default=EVAL_CONFIG["target_vlms"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        os.makedirs(OUTPUT_CONFIG["results_dir"], exist_ok=True)
        args.output = os.path.join(OUTPUT_CONFIG["results_dir"], "eval_results.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"VLMs to evaluate: {args.target_vlms}\n")

    results = {}

    for vlm_key in args.target_vlms:
        print(f"[{vlm_key}]")
        try:
            model, processor, info = load_vlm(vlm_key, device)
        except Exception as e:
            print(f"  Failed to load: {e}\n")
            continue

        vlm_result = {"model": info["short_name"]}

        print(f"  Captioning adversarial image...")
        vlm_result["adv_caption"] = caption_image(model, processor, args.adv_image, device)
        print(f"    Adv caption:    {vlm_result['adv_caption']}")

        if args.clean_image:
            print(f"  Captioning clean image...")
            vlm_result["clean_caption"] = caption_image(model, processor, args.clean_image, device)
            print(f"    Clean caption:  {vlm_result['clean_caption']}")

        if args.target_image:
            print(f"  Captioning target image...")
            vlm_result["target_caption"] = caption_image(model, processor, args.target_image, device)
            print(f"    Target caption: {vlm_result['target_caption']}")

        results[vlm_key] = vlm_result
        print()

        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {args.output}")


if __name__ == "__main__":
    main()
