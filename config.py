"""
VisInject Configuration
========================
Unified config for the end-to-end pipeline:
  UniversalAttack (pixel optimization) + AnyAttack_LAION400M (decoder fusion).

Extends models.registry for environment setup and model info.
"""

import os

from models.registry import init_model_env

init_model_env()

# ── UniversalAttack: multi-model joint pixel optimization ─────────

UNIVERSAL_ATTACK_CONFIG = {
    "target_phrase": "Sure, here it is",
    "image_size": (448, 448),
    "num_steps": 3000,
    "lr": 1e-2,
    "gamma_single": 0.1,
    "gamma_multi": 0.5,
    "quant_robustness": True,
    "gaussian_blur": False,
    "blur_kernel_size": 5,
    "blur_sigma": 1.0,
    "multi_answer": False,
    "answer_pool": [
        "Sure, here it is",
        "Of course, I can help with that",
        "Absolutely, here you go",
        "Yes, I'll provide that information",
        "Sure thing, let me explain",
    ],
    "localize": False,
    "localize_scale_min": 0.5,
    "localize_scale_max": 0.9,
}

# Attack target models: all enabled models are loaded simultaneously during
# joint optimization. VRAM usage is cumulative across all enabled models.
#
# To add a new architecture:
#   1. Add a REGISTRY entry in models/registry.py
#   2. Implement an MLLMWrapper subclass in demos/demo_S3_UniversalAttack/models/
#      (must implement load(), compute_masked_ce_loss(), generate())
#
# Dependencies:
#   - DeepSeek family: pip install deepseek-vl
#   - All others: standard transformers, weights auto-downloaded on first use
ATTACK_TARGETS = [
    # -- Qwen family (ViT + PatchMerger + Qwen LLM) --
    "qwen2_5_vl_3b",            # Qwen2.5-VL-3B-Instruct              ~6GB
    # "qwen2_vl_2b",            # Qwen2-VL-2B-Instruct                ~4GB
    # "qwen2_5_vl_7b",          # Qwen2.5-VL-7B-Instruct              ~14GB

    # -- BLIP-2 family (EVA-ViT-G + Q-Former + LLM) --
    "blip2_opt_2_7b",           # BLIP-2 + OPT-2.7B                   ~5GB
    # "blip2_flan_t5_xl",       # BLIP-2 + Flan-T5-XL                 ~8GB
    # "instructblip_vicuna_7b", # InstructBLIP + Vicuna-7B             ~14GB

    # -- DeepSeek family (SigLIP + MLP + LLaMA) -- requires: pip install deepseek-vl
    "deepseek_vl_1_3b",         # DeepSeek-VL-1.3B                    ~4GB
    # "deepseek_vl2_tiny",      # DeepSeek-VL2-Tiny                   ~3GB

    # -- LLaVA family (CLIP-ViT + Linear Proj + Vicuna) --
    "llava_1_5_7b",             # LLaVA-1.5-7B                        ~14GB

    # -- Phi family (custom ViT + Phi-3.5 LLM) --
    "phi_3_5_vision",           # Phi-3.5-Vision-Instruct              ~8GB

    # -- Llama Vision family (Mllama ViT + Llama-3.2 LLM) --
    # "llama_3_2_11b_vision",   # Llama-3.2-11B-Vision-Instruct        ~22GB
]
# Current enabled: qwen(6) + blip2(5) + deepseek(4) + llava(14) + phi(8) = ~37GB
# All enabled: ~98GB (needs multi-GPU or batch attack)
# Recommended: H200 80GB for default config

# ── AnyAttack_LAION400M: decoder-based noise fusion ──────────────

ANYATTACK_CONFIG = {
    "decoder_path": os.path.join(
        os.path.dirname(__file__), "checkpoints", "coco_bi.pt"
    ),
    "clip_model": "ViT-B/32",
    "embed_dim": 512,
    "eps": 16 / 255,
    "image_size": 224,
}


# ── Evaluation ────────────────────────────────────────────────────

EVAL_CONFIG = {
    "eval_vlms": ["qwen2_5_vl_3b", "blip2_opt_2_7b", "deepseek_vl_1_3b"],
    "num_adversarial_questions": 20,
    "num_safe_questions": 10,
}

# ── LLM-as-Judge: API-based evaluation ─────────────────────────────
# Response pairs are generated on HPC, then judged locally via API.
# Usage: python -m evaluate.judge --pairs-file response_pairs.json

JUDGE_CONFIG = {
    "version": 2,  # v2: dual-check (affected + injected), v1: legacy single-prompt
    "judges": [
        {"name": "deepseek", "provider": "deepseek", "model": "deepseek-chat"},
    ],
    "temperature": 0.0,
    "max_tokens": 300,
}

# ── Output paths ──────────────────────────────────────────────────

OUTPUT_CONFIG = {
    "base_dir": os.path.join(os.path.dirname(__file__), "outputs"),
    "universal_dir": os.path.join(os.path.dirname(__file__), "outputs", "universal"),
    "adversarial_dir": os.path.join(os.path.dirname(__file__), "outputs", "adversarial"),
    "results_dir": os.path.join(os.path.dirname(__file__), "outputs", "results"),
    "checkpoint_dir": os.path.join(os.path.dirname(__file__), "outputs", "checkpoints"),
    "save_every": 200,
    "log_every": 50,
}
