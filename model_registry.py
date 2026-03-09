"""
VisInject 统一模型注册表
========================
所有 demo 通过此文件获取模型元数据，无需各自硬编码模型 ID 和参数。

新增模型：在 REGISTRY 字典中加一个条目，其他代码无需改动。
新增 demo：在 demo 的 config.py 顶部调用 init_model_env()，然后
          通过 get_model_info() 引用所需模型。

缓存位置：
  默认写入本文件同级的 model_cache/ 目录（已 gitignored）。
  若用户已在系统环境中设置了 HF_HOME（如 D:\\hugging_face），
  init_model_env() 使用 setdefault，不会覆盖已有设置。

Quick Reference (注册键速查表)
──────────────────────────────────────────────────────────────────
  Registry Key              │ Model Name              │ VRAM  │ S3 Wrapper       │ Family
  ─────────────────────────┼─────────────────────────┼───────┼──────────────────┼────────
  qwen2_5_vl_3b            │ Qwen2.5-VL-3B           │  6 GB │ QwenWrapper      │ qwen
  qwen2_vl_2b              │ Qwen2-VL-2B             │  4 GB │ QwenWrapper      │ qwen
  qwen2_5_vl_7b            │ Qwen2.5-VL-7B           │ 14 GB │ QwenWrapper      │ qwen
  blip2_opt_2_7b           │ BLIP-2-OPT-2.7B         │  5 GB │ Blip2Wrapper     │ blip2
  blip2_flan_t5_xl         │ BLIP-2-FlanT5-XL        │  8 GB │ Blip2Wrapper     │ blip2
  instructblip_vicuna_7b   │ InstructBLIP-Vicuna-7B   │ 14 GB │ Blip2Wrapper     │ blip2
  deepseek_vl_1_3b         │ DeepSeek-VL-1.3B         │  4 GB │ DeepSeekWrapper  │ deepseek
  deepseek_vl2_tiny        │ DeepSeek-VL2-Tiny        │  3 GB │ DeepSeekWrapper  │ deepseek
  llava_1_5_7b             │ LLaVA-1.5-7B             │ 14 GB │ LlavaWrapper     │ llava
  phi_3_5_vision           │ Phi-3.5-Vision           │  8 GB │ PhiWrapper       │ phi
  llama_3_2_11b_vision     │ Llama-3.2-11B-Vision     │ 22 GB │ LlamaVisionWrapper│ llama
  clip_vit_l14             │ CLIP-ViT-L/14            │  1 GB │ N/A (encoder)    │ clip
  clip_vit_b32             │ CLIP-ViT-B/32            │ <1 GB │ N/A (encoder)    │ clip
  minigpt4_vicuna_7b       │ MiniGPT-4-Vicuna-7B      │ 14 GB │ N/A (未实现)      │ minigpt4
  ─────────────────────────┴─────────────────────────┴───────┴──────────────────┴────────

  demo_S3 用法示例:
    python demo.py --target-models qwen2_5_vl_3b              # 单模型 (~6 GB)
    python attack.py --target-models qwen2_vl_2b blip2_opt_2_7b  # 多模型 (~9 GB, 显存累加)

quick view model summary:
python -c "from model_registry import print_registry_summary; print_registry_summary()"

"""

import os
from pathlib import Path
from typing import Optional

# 统一缓存目录：所有 HuggingFace 模型下载到此处（gitignored）
MODEL_CACHE_DIR = Path(__file__).parent / "model_cache"


def init_model_env() -> str:
    """
    统一设置 HuggingFace 缓存位置。
    必须在任何 transformers / huggingface_hub import 之前调用。
    使用 setdefault：若系统已设置 HF_HOME，则保持不变。
    返回实际使用的缓存路径字符串。
    """
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
    return cache_path


# ============================================================
# 模型注册表
# 每个条目包含从 transformers 加载模型所需的全部元数据。
# 新增模型：在此添加一个字典条目，其他代码自动感知。
# ============================================================

REGISTRY: dict = {

    # ── Qwen 系列 ──────────────────────────────────────────────────
    "qwen2_5_vl_3b": {
        "hf_id":      "Qwen/Qwen2.5-VL-3B-Instruct",
        "short_name": "Qwen2.5-VL-3B",
        "family":     "qwen",
        "dtype":      "bf16",
        "img_size":   392,
        "norm_mean":  [0.48145466, 0.4578275,  0.40821073],
        "norm_std":   [0.26862954, 0.26130258, 0.27577711],
        "vram_bf16_gb":         6.0,
        "encoder_vram_bf16_gb": 0.6,
        "used_by":    ["demo3", "demo_S1", "demo_S2_eval", "demo_S3"],
    },

    "qwen2_vl_2b": {
        "hf_id":      "Qwen/Qwen2-VL-2B-Instruct",
        "short_name": "Qwen2-VL-2B",
        "family":     "qwen",
        "dtype":      "bf16",
        "img_size":   392,
        "norm_mean":  [0.48145466, 0.4578275,  0.40821073],
        "norm_std":   [0.26862954, 0.26130258, 0.27577711],
        "vram_bf16_gb":         4.0,
        "encoder_vram_bf16_gb": 0.6,
        "used_by":    ["demo_S3"],  # Universal Attack 原论文目标模型
    },

    "qwen2_5_vl_7b": {
        "hf_id":      "Qwen/Qwen2.5-VL-7B-Instruct",
        "short_name": "Qwen2.5-VL-7B",
        "family":     "qwen",
        "dtype":      "bf16",
        "img_size":   392,
        "norm_mean":  [0.48145466, 0.4578275,  0.40821073],
        "norm_std":   [0.26862954, 0.26130258, 0.27577711],
        "vram_bf16_gb":         14.0,
        "encoder_vram_bf16_gb": 0.6,
        "used_by":    ["demo_S1"],
    },

    # ── BLIP-2 系列 ────────────────────────────────────────────────
    "blip2_opt_2_7b": {
        "hf_id":      "Salesforce/blip2-opt-2.7b",
        "short_name": "BLIP-2-OPT-2.7B",
        "family":     "blip2",
        "dtype":      "fp16",
        "img_size":   224,
        "norm_mean":  [0.48145466, 0.4578275,  0.40821073],
        "norm_std":   [0.26862954, 0.26130258, 0.27577711],
        "vram_bf16_gb":         5.4,
        "encoder_vram_bf16_gb": 2.0,
        "used_by":    ["demo1", "demo_S1", "demo_S2_eval"],
    },

    "blip2_flan_t5_xl": {
        "hf_id":      "Salesforce/blip2-flan-t5-xl",
        "short_name": "BLIP-2-FlanT5-XL",
        "family":     "blip2",
        "dtype":      "fp16",
        "img_size":   224,
        "norm_mean":  [0.48145466, 0.4578275,  0.40821073],
        "norm_std":   [0.26862954, 0.26130258, 0.27577711],
        "vram_bf16_gb":         8.0,
        "encoder_vram_bf16_gb": 2.0,
        "used_by":    ["demo_S2_eval"],
    },

    "instructblip_vicuna_7b": {
        "hf_id":      "Salesforce/instructblip-vicuna-7b",
        "short_name": "InstructBLIP-Vicuna-7B",
        "family":     "blip2",
        "dtype":      "fp16",
        "img_size":   224,
        "norm_mean":  [0.48145466, 0.4578275,  0.40821073],
        "norm_std":   [0.26862954, 0.26130258, 0.27577711],
        "vram_bf16_gb":         14.0,
        "encoder_vram_bf16_gb": 2.0,
        "used_by":    ["demo_S2_eval"],  # AnyAttack 原论文评估目标
    },

    # ── DeepSeek 系列 ──────────────────────────────────────────────
    "deepseek_vl_1_3b": {
        "hf_id":      "deepseek-ai/deepseek-vl-1.3b-chat",
        "short_name": "DeepSeek-VL-1.3B",
        "family":     "deepseek",
        "dtype":      "bf16",
        "img_size":   384,
        "norm_mean":  [0.5, 0.5, 0.5],
        "norm_std":   [0.5, 0.5, 0.5],
        "vram_bf16_gb":         3.8,
        "encoder_vram_bf16_gb": 1.0,
        "used_by":    ["demo2", "demo_S1"],
    },

    "deepseek_vl2_tiny": {
        "hf_id":      "deepseek-ai/deepseek-vl2-tiny",
        "short_name": "DeepSeek-VL2-Tiny",
        "family":     "deepseek",
        "dtype":      "bf16",
        "img_size":   384,
        "norm_mean":  [0.5, 0.5, 0.5],
        "norm_std":   [0.5, 0.5, 0.5],
        "vram_bf16_gb":         3.0,
        "encoder_vram_bf16_gb": 0.8,
        "used_by":    ["demo_S1"],
    },

    # ── CLIP 系列（纯视觉编码器）──────────────────────────────────
    "clip_vit_l14": {
        "hf_id":      "openai/clip-vit-large-patch14",
        "short_name": "CLIP-ViT-L/14",
        "family":     "clip",
        "dtype":      "fp16",
        "img_size":   224,
        "norm_mean":  [0.48145466, 0.4578275,  0.40821073],
        "norm_std":   [0.26862954, 0.26130258, 0.27577711],
        "vram_bf16_gb":         0.9,
        "encoder_vram_bf16_gb": 0.9,
        "used_by":    ["demo_S1", "demo_C1"],
    },

    "clip_vit_b32": {
        "hf_id":      "openai/clip-vit-base-patch32",
        "short_name": "CLIP-ViT-B/32",
        "family":     "clip",
        "dtype":      "fp16",
        "img_size":   224,
        "norm_mean":  [0.48145466, 0.4578275,  0.40821073],
        "norm_std":   [0.26862954, 0.26130258, 0.27577711],
        "vram_bf16_gb":         0.3,
        "encoder_vram_bf16_gb": 0.3,
        "used_by":    ["demo_S2"],  # AnyAttack surrogate encoder
    },

    # ── LLaVA 系列 ────────────────────────────────────────────────
    "llava_1_5_7b": {
        "hf_id":      "llava-hf/llava-1.5-7b-hf",
        "short_name": "LLaVA-1.5-7B",
        "family":     "llava",
        "dtype":      "fp16",
        "img_size":   336,
        "norm_mean":  [0.48145466, 0.4578275,  0.40821073],
        "norm_std":   [0.26862954, 0.26130258, 0.27577711],
        "vram_bf16_gb":         14.0,
        "encoder_vram_bf16_gb": 0.9,
        "used_by":    ["demo_S3"],  # Universal Attack 原论文目标模型
    },

    # ── Llama Vision 系列 ─────────────────────────────────────────
    "llama_3_2_11b_vision": {
        "hf_id":      "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "short_name": "Llama-3.2-11B-Vision",
        "family":     "llama",
        "dtype":      "bf16",
        "img_size":   560,
        "norm_mean":  [0.48145466, 0.4578275,  0.40821073],
        "norm_std":   [0.26862954, 0.26130258, 0.27577711],
        "vram_bf16_gb":         22.0,
        "encoder_vram_bf16_gb": 1.2,
        "used_by":    ["demo_S3"],  # Universal Attack 原论文目标模型
    },

    # ── Phi Vision 系列 ───────────────────────────────────────────
    "phi_3_5_vision": {
        "hf_id":      "microsoft/Phi-3.5-vision-instruct",
        "short_name": "Phi-3.5-Vision",
        "family":     "phi",
        "dtype":      "bf16",
        "img_size":   336,
        "norm_mean":  [0.48145466, 0.4578275,  0.40821073],
        "norm_std":   [0.26862954, 0.26130258, 0.27577711],
        "vram_bf16_gb":         8.0,
        "encoder_vram_bf16_gb": 0.9,
        "used_by":    ["demo_S3"],  # Universal Attack 原论文目标模型
    },

    # ── MiniGPT-4 系列 ────────────────────────────────────────────
    "minigpt4_vicuna_7b": {
        "hf_id":      "Vision-CAIR/MiniGPT-4",
        "short_name": "MiniGPT-4-Vicuna-7B",
        "family":     "minigpt4",
        "dtype":      "fp16",
        "img_size":   224,
        "norm_mean":  [0.48145466, 0.4578275,  0.40821073],
        "norm_std":   [0.26862954, 0.26130258, 0.27577711],
        "vram_bf16_gb":         14.0,
        "encoder_vram_bf16_gb": 2.0,
        "used_by":    ["demo_S2_eval"],  # AnyAttack 原论文评估目标
    },

    # ── InternVL 系列（扩展预留）──────────────────────────────────
    # "internvl2_2b": {
    #     "hf_id":      "OpenGVLab/InternVL2-2B",
    #     "short_name": "InternVL2-2B",
    #     "family":     "internvl",
    #     "dtype":      "bf16",
    #     "img_size":   448,
    #     "norm_mean":  [0.485, 0.456, 0.406],
    #     "norm_std":   [0.229, 0.224, 0.225],
    #     "vram_bf16_gb":         4.0,
    #     "encoder_vram_bf16_gb": 1.2,
    # },
}


# ============================================================
# 查询接口
# ============================================================

def get_model_info(key: str) -> dict:
    """
    按注册键名获取模型元数据字典（返回副本，防止意外修改）。
    键不存在时抛出包含所有可用键名的 KeyError。
    """
    if key not in REGISTRY:
        available = list(REGISTRY.keys())
        raise KeyError(
            f"模型 '{key}' 未在注册表中。\n"
            f"可用模型: {available}"
        )
    return dict(REGISTRY[key])


def list_models(family: Optional[str] = None) -> list:
    """
    列出所有已注册模型的键名。
    可通过 family 参数过滤（如 'qwen', 'blip2', 'deepseek', 'clip'）。
    """
    if family:
        return [k for k, v in REGISTRY.items() if v.get("family") == family]
    return list(REGISTRY.keys())


def get_hf_id(key: str) -> str:
    """快捷方法：直接获取 HuggingFace 模型 ID"""
    return get_model_info(key)["hf_id"]


def print_registry_summary():
    """打印注册表概览，便于调试"""
    print(f"\n{'='*60}")
    print(f"VisInject 模型注册表（共 {len(REGISTRY)} 个）")
    print(f"缓存目录: {os.environ.get('HF_HOME', str(MODEL_CACHE_DIR))}")
    print(f"{'='*60}")
    families = {}
    for key, info in REGISTRY.items():
        fam = info.get("family", "other")
        families.setdefault(fam, []).append(key)
    for fam, keys in sorted(families.items()):
        print(f"\n  [{fam.upper()}]")
        for k in keys:
            info = REGISTRY[k]
            print(f"    {k:<25} {info['hf_id']:<45} "
                  f"img={info['img_size']}  vram={info['vram_bf16_gb']}GB")
    print()


if __name__ == "__main__":
    init_model_env()
    print_registry_summary()
