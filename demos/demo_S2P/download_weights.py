"""
Download official AnyAttack weights from HuggingFace.

Usage:
    python download_weights.py
    python download_weights.py --checkpoint coco_cos.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import WEIGHTS_CONFIG


def download(checkpoint: str = "checkpoints/coco_bi.pt"):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    repo_id = WEIGHTS_CONFIG["hf_repo"]
    local_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(local_dir, exist_ok=True)

    filename = os.path.basename(checkpoint)
    remote_path = f"checkpoints/{filename}"

    print(f"Downloading {remote_path} from {repo_id} ...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=remote_path,
        local_dir=os.path.dirname(local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Saved to: {path}")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Size: {size_mb:.1f} MB")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AnyAttack weights")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/coco_bi.pt",
        help="Which checkpoint to download (default: checkpoints/coco_bi.pt). "
             "Options: coco_bi.pt, coco_cos.pt, pre-trained.pt, "
             "flickr30k_bi.pt, flickr30k_cos.pt, snli_ve_cos.pt",
    )
    args = parser.parse_args()
    download(args.checkpoint)
