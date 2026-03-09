# demo_S2P: AnyAttack with Official Pre-trained Weights

[中文版](README_zh.md)

Inference-only demo using official AnyAttack weights (pre-trained on LAION-400M + fine-tuned on COCO). No training required -- runs on a local GPU (RTX 4090 or similar).

- Paper: https://arxiv.org/abs/2410.05346
- Weights: https://huggingface.co/jiamingzz/anyattack
- Training code: [`demo_S2_AnyAttack/`](../demo_S2_AnyAttack/)

## Quick Start

```bash
cd demos/demo_S2P

# Step 1: Download official weights (~100 MB)
python download_weights.py

# Step 2: Generate adversarial image
python demo.py --clean-image ../demo_images/ORIGIN_dog.png \
               --target-image ../demo_images/ORIGIN_cat.png

# Step 3: Evaluate against VLMs
python evaluate.py --adv-image outputs/adversarial.png \
                   --clean-image ../demo_images/ORIGIN_dog.png \
                   --target-image ../demo_images/ORIGIN_cat.png
```

## How It Works

```
Target Image ──> CLIP ViT-B/32 ──> 512-dim Embedding ──> Decoder (coco_bi.pt) ──> Noise
                                                                                    │
                                                            Clean Image + Noise ──> Adversarial Image
                                                                                    │
                                                                                    v
                                                               VLM "sees" target image content
```

The Decoder was trained by the AnyAttack authors on LAION-400M (400M images) with self-supervised contrastive learning, then fine-tuned on COCO with multi-encoder losses (CLIP + EVA02 + ViT-B/16). The resulting model can generate adversarial noise that transfers across VLMs.

## Available Checkpoints

| Checkpoint | Training Data | Loss Function | Best For |
|------------|--------------|---------------|----------|
| `coco_bi.pt` (default) | COCO | BiContrastive | General use, paper's main experiments |
| `coco_cos.pt` | COCO | Cosine | Alternative loss variant |
| `pre-trained.pt` | LAION-400M only | InfoNCE | Base model, no fine-tuning |
| `flickr30k_bi.pt` | Flickr30k | BiContrastive | Flickr-specific tasks |
| `flickr30k_cos.pt` | Flickr30k | Cosine | Flickr-specific tasks |
| `snli_ve_cos.pt` | SNLI-VE | Cosine | Visual entailment tasks |

To use a different checkpoint:

```bash
python download_weights.py --checkpoint coco_cos.pt
python demo.py --decoder-path checkpoints/coco_cos.pt \
               --clean-image ../demo_images/ORIGIN_dog.png \
               --target-image ../demo_images/ORIGIN_cat.png
```

## Evaluation VLMs

The demo evaluates against three VLMs with different vision encoders:

| VLM | Vision Encoder | In Original Paper? | VRAM |
|-----|---------------|-------------------|------|
| BLIP-2 (OPT-2.7B) | EVA-ViT-G | Yes | ~6 GB |
| DeepSeek-VL-1.3B | SigLIP-L | No | ~4 GB |
| Qwen2.5-VL-3B | ViT-L | No | ~6 GB |

To evaluate against a single VLM:

```bash
python evaluate.py --adv-image outputs/adversarial.png \
                   --target-vlms blip2_opt_2_7b
```

## File Structure

```
demo_S2P/
├── config.py               # Configuration (HF repo, attack params, eval VLMs)
├── download_weights.py     # Download official weights from HuggingFace
├── demo.py                 # Generate adversarial image (single pair)
├── evaluate.py             # Evaluate against VLMs (caption comparison)
├── README.md               # This file
├── README_zh.md            # Chinese documentation
├── checkpoints/            # Downloaded weights (gitignored)
└── outputs/                # Generated images and results (gitignored)
```

## Relationship to demo_S2_AnyAttack

| | demo_S2_AnyAttack | demo_S2P |
|---|---|---|
| Purpose | Full reproduction (train from scratch) | Inference with official weights |
| Training | Pre-train on LAION-Art + fine-tune on COCO | None |
| Hardware | HPC (H200) | Local GPU (RTX 4090) |
| Weights | Self-trained | Downloaded from HuggingFace |
| Models/code | CLIPEncoder, Decoder, losses, dataset | Reuses S2's models and dataset |

## References

```bibtex
@inproceedings{zhang2025anyattack,
  title={Anyattack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models},
  author={Zhang, Jiaming and Ye, Junhong and Ma, Xingjun and Li, Yige and Yang, Yunfan and Chen, Yunhao and Sang, Jitao and Yeung, Dit-Yan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2025}
}
```
