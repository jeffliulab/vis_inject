"""
Download official AnyAttack decoder weights from HuggingFace.

Usage:
    python download_decoder_weights.py                          # download coco_bi.pt
    python download_decoder_weights.py --checkpoint coco_cos.pt # other checkpoint
"""

import argparse
import os
import sys


HF_REPO = "jiamingzz/anyattack"
DEFAULT_CHECKPOINT = "coco_bi.pt"


def download(checkpoint: str = DEFAULT_CHECKPOINT, output_dir: str = None):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    if output_dir is None:
        # Default: project_root/checkpoints/
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "checkpoints"
        )
    os.makedirs(output_dir, exist_ok=True)

    remote_path = f"checkpoints/{checkpoint}"

    print(f"Downloading {remote_path} from {HF_REPO} ...")
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=remote_path,
        local_dir=os.path.dirname(output_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Saved to: {path}")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Size: {size_mb:.1f} MB")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AnyAttack decoder weights")
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="Which checkpoint to download (default: coco_bi.pt). "
             "Options: coco_bi.pt, coco_cos.pt, pre-trained.pt, "
             "flickr30k_bi.pt, flickr30k_cos.pt, snli_ve_cos.pt",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <project_root>/checkpoints/)",
    )
    args = parser.parse_args()
    download(args.checkpoint, args.output_dir)
