"""
Download all HuggingFace models required by the VisInject pipeline.

Usage:
    python download_all_models.py                # download quick stage (3 VLMs + CLIP)
    python download_all_models.py --stage full   # download all 5 VLMs + CLIP
    python download_all_models.py --stage quick  # same as default
    python download_all_models.py --list         # list models and status
"""

import argparse
import os
import sys

MODELS = {
    "quick": [
        ("Qwen/Qwen2.5-VL-3B-Instruct", "~6 GB"),
        ("Salesforce/blip2-opt-2.7b", "~5 GB"),
        ("deepseek-ai/deepseek-vl-1.3b-chat", "~4 GB"),
        ("openai/clip-vit-base-patch32", "~600 MB"),
    ],
    "extra": [
        ("liuhaotian/llava-v1.5-7b", "~14 GB"),
        ("microsoft/Phi-3.5-vision-instruct", "~8 GB"),
    ],
}


def list_models(cache_dir: str):
    """List all models and their cache status."""
    hub_dir = os.path.join(cache_dir, "hub")
    print(f"Model cache: {cache_dir}")
    print(f"Hub dir:     {hub_dir}")
    print()

    for stage, models in MODELS.items():
        print(f"--- Stage: {stage} ---")
        for repo_id, size in models:
            dir_name = f"models--{repo_id.replace('/', '--')}"
            model_path = os.path.join(hub_dir, dir_name)
            exists = os.path.isdir(model_path)
            status = "CACHED" if exists else "MISSING"
            print(f"  [{status:7s}] {repo_id:45s} ({size})")
        print()


def download_models(stage: str, cache_dir: str):
    """Download models for the specified stage."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("  pip install huggingface_hub")
        sys.exit(1)

    os.environ.setdefault("HF_HOME", cache_dir)

    models = list(MODELS["quick"])
    if stage == "full":
        models.extend(MODELS["extra"])

    print(f"Stage: {stage}")
    print(f"Cache: {cache_dir}")
    print(f"Models to download: {len(models)}")
    print()

    for i, (repo_id, size) in enumerate(models, 1):
        dir_name = f"models--{repo_id.replace('/', '--')}"
        model_path = os.path.join(cache_dir, "hub", dir_name)

        if os.path.isdir(model_path):
            print(f"[{i}/{len(models)}] SKIP (cached) {repo_id}")
            continue

        print(f"[{i}/{len(models)}] Downloading {repo_id} ({size})...")
        try:
            snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
            )
            print(f"  Done.")
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  You may need to accept terms at https://huggingface.co/{repo_id}")

    print()
    print("All done.")


if __name__ == "__main__":
    default_cache = os.environ.get(
        "HF_HOME",
        "/cluster/tufts/c26sp1ee0141/pliu07/model_cache"
    )

    parser = argparse.ArgumentParser(description="Download VisInject pipeline models")
    parser.add_argument("--stage", choices=["quick", "full"], default="quick",
        help="quick: 3 VLMs + CLIP (~16GB). full: 5 VLMs + CLIP (~38GB)")
    parser.add_argument("--cache-dir", type=str, default=default_cache,
        help=f"HuggingFace cache directory (default: {default_cache})")
    parser.add_argument("--list", action="store_true",
        help="List models and their cache status")
    args = parser.parse_args()

    if args.list:
        list_models(args.cache_dir)
    else:
        download_models(args.stage, args.cache_dir)
