"""
Verify LAION-Art WebDataset download completeness.

Usage:
    python verify_dataset.py
    python verify_dataset.py --data-dir /path/to/webdataset
    python verify_dataset.py --data-dir /path/to/webdataset --show-samples 5
"""

import argparse
import glob
import os
import tarfile
import sys
from pathlib import Path


DEFAULT_DATA_DIR = "/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset"
EXPECTED_TOTAL_IMAGES = 8_000_000
EXPECTED_IMAGES_PER_SHARD = 10_000


def count_shard_images(tar_path: str) -> dict:
    """Count images and check integrity of a single tar shard."""
    result = {"path": tar_path, "images": 0, "texts": 0, "size_mb": 0, "error": None}
    try:
        result["size_mb"] = os.path.getsize(tar_path) / (1024 * 1024)
        with tarfile.open(tar_path, "r") as t:
            members = t.getmembers()
            result["images"] = sum(1 for m in members if m.name.endswith((".jpg", ".png", ".webp")))
            result["texts"] = sum(1 for m in members if m.name.endswith(".txt"))
    except Exception as e:
        result["error"] = str(e)
    return result


def show_samples(tar_path: str, n: int = 3):
    """Display a few sample entries from a tar shard."""
    try:
        with tarfile.open(tar_path, "r") as t:
            txt_members = [m for m in t.getmembers() if m.name.endswith(".txt")][:n]
            for m in txt_members:
                f = t.extractfile(m)
                if f:
                    caption = f.read().decode("utf-8", errors="replace").strip()
                    img_name = m.name.replace(".txt", ".jpg")
                    print(f"    {img_name}: {caption[:100]}...")
    except Exception as e:
        print(f"    Error reading samples: {e}")


def main():
    parser = argparse.ArgumentParser(description="Verify LAION-Art WebDataset download")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Path to WebDataset directory containing .tar files")
    parser.add_argument("--show-samples", type=int, default=0,
                        help="Show N sample captions from the first shard")
    parser.add_argument("--detailed", action="store_true",
                        help="Show per-shard statistics")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        sys.exit(1)

    tar_files = sorted(glob.glob(str(data_dir / "*.tar")))
    if not tar_files:
        print(f"[ERROR] No .tar files found in: {data_dir}")
        sys.exit(1)

    print(f"LAION-Art Dataset Verification")
    print(f"{'=' * 50}")
    print(f"Directory : {data_dir}")
    print(f"Tar shards: {len(tar_files)}")
    print()

    total_images = 0
    total_texts = 0
    total_size_mb = 0
    errors = []

    for i, tf in enumerate(tar_files):
        result = count_shard_images(tf)
        total_images += result["images"]
        total_texts += result["texts"]
        total_size_mb += result["size_mb"]

        if result["error"]:
            errors.append(result)

        if args.detailed:
            status = "ERR" if result["error"] else "OK"
            print(f"  [{status}] {os.path.basename(tf)}: "
                  f"{result['images']:,} images, "
                  f"{result['size_mb']:.1f} MB"
                  f"{' -- ' + result['error'] if result['error'] else ''}")

        if (i + 1) % 50 == 0:
            print(f"  ... checked {i + 1}/{len(tar_files)} shards "
                  f"({total_images:,} images so far)")

    print()
    print(f"Results")
    print(f"{'-' * 50}")
    print(f"Total shards     : {len(tar_files)}")
    print(f"Total images     : {total_images:,}")
    print(f"Total captions   : {total_texts:,}")
    print(f"Total size       : {total_size_mb / 1024:.1f} GB")
    print(f"Avg images/shard : {total_images // max(len(tar_files), 1):,}")
    print(f"Corrupted shards : {len(errors)}")

    coverage = total_images / EXPECTED_TOTAL_IMAGES * 100
    print()
    print(f"Coverage")
    print(f"{'-' * 50}")
    print(f"Expected (ideal) : {EXPECTED_TOTAL_IMAGES:,}")
    print(f"Actual           : {total_images:,} ({coverage:.1f}%)")

    if coverage >= 60:
        print(f"[OK] Dataset has sufficient coverage for pre-training.")
    elif coverage >= 30:
        print(f"[WARN] Dataset coverage is low. Pre-training may be weaker.")
        print(f"       Consider resubmitting the download job.")
    else:
        print(f"[ERROR] Dataset coverage is too low. Resubmit download_laion_art.sh.")

    if errors:
        print()
        print(f"Corrupted Shards ({len(errors)})")
        print(f"{'-' * 50}")
        for e in errors:
            print(f"  {os.path.basename(e['path'])}: {e['error']}")
        print(f"  Consider deleting these and rerunning the download script.")

    if args.show_samples > 0:
        print()
        print(f"Sample Entries (from first shard)")
        print(f"{'-' * 50}")
        show_samples(tar_files[0], args.show_samples)

    print()


if __name__ == "__main__":
    main()
