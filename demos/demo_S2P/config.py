"""
demo_S2P Configuration
=======================
Uses official AnyAttack pre-trained + fine-tuned weights from HuggingFace
for inference-only adversarial image generation and evaluation.

No training required. Runs on a local GPU (RTX 4090 or similar).

Paper: https://arxiv.org/abs/2410.05346
Weights: https://huggingface.co/jiamingzz/anyattack
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model_registry import init_model_env

init_model_env()

# ── HuggingFace weights ──────────────────────────────────────────
WEIGHTS_CONFIG = {
    "hf_repo": "jiamingzz/anyattack",
    "checkpoint": "checkpoints/coco_bi.pt",
    "local_path": os.path.join(os.path.dirname(__file__), "checkpoints", "coco_bi.pt"),
}

# ── Attack parameters (must match the Decoder training config) ───
ATTACK_CONFIG = {
    "eps": 16 / 255,
    "clip_model": "ViT-B/32",
    "embed_dim": 512,
    "image_size": 224,
}

# ── Evaluation VLMs ──────────────────────────────────────────────
EVAL_CONFIG = {
    "target_vlms": [
        "blip2_opt_2_7b",
        "deepseek_vl_1_3b",
        "qwen2_5_vl_3b",
    ],
}

# ── Paths ────────────────────────────────────────────────────────
OUTPUT_CONFIG = {
    "output_dir": os.path.join(os.path.dirname(__file__), "outputs"),
    "results_dir": os.path.join(os.path.dirname(__file__), "outputs", "results"),
}
