# demo_S2P：使用 AnyAttack 官方预训练权重

[English](README.md)

纯推理 demo，使用 AnyAttack 官方权重（在 LAION-400M 上预训练 + 在 COCO 上微调）。无需训练，本地 GPU（RTX 4090 等）即可运行。

- 论文：https://arxiv.org/abs/2410.05346
- 权重：https://huggingface.co/jiamingzz/anyattack
- 训练代码：[`demo_S2_AnyAttack/`](../demo_S2_AnyAttack/)

## 快速开始

```bash
cd demos/demo_S2P

# 第 1 步：下载官方权重（约 100 MB）
python download_weights.py

# 第 2 步：生成对抗图片
python demo.py --clean-image ../demo_images/ORIGIN_dog.png \
               --target-image ../demo_images/ORIGIN_cat.png

# 第 3 步：评估（让 VLM 看对抗图片，对比描述是否偏向目标图片）
python evaluate.py --adv-image outputs/adversarial.png \
                   --clean-image ../demo_images/ORIGIN_dog.png \
                   --target-image ../demo_images/ORIGIN_cat.png
```

## 工作原理

```
目标图片 ──> CLIP ViT-B/32 ──> 512维 Embedding ──> Decoder (coco_bi.pt) ──> 噪声
                                                                              │
                                                      干净图片 + 噪声 ──> 对抗图片
                                                                              │
                                                                              v
                                                         VLM "看到"的是目标图片的内容
```

Decoder 由 AnyAttack 作者在 LAION-400M（4 亿张图片）上通过自监督对比学习预训练，然后在 COCO 上用多编码器损失（CLIP + EVA02 + ViT-B/16）微调。生成的噪声可以跨模型迁移。

## 可用的 Checkpoint

| 文件 | 训练数据 | 损失函数 | 适用场景 |
|------|---------|---------|---------|
| `coco_bi.pt`（默认） | COCO | BiContrastive | 通用，论文主实验用 |
| `coco_cos.pt` | COCO | Cosine | 另一种损失函数版本 |
| `pre-trained.pt` | 仅 LAION-400M | InfoNCE | 基础模型，未微调 |
| `flickr30k_bi.pt` | Flickr30k | BiContrastive | Flickr 相关任务 |
| `flickr30k_cos.pt` | Flickr30k | Cosine | Flickr 相关任务 |
| `snli_ve_cos.pt` | SNLI-VE | Cosine | 视觉蕴含任务 |

使用其他 checkpoint：

```bash
python download_weights.py --checkpoint coco_cos.pt
python demo.py --decoder-path checkpoints/coco_cos.pt \
               --clean-image ../demo_images/ORIGIN_dog.png \
               --target-image ../demo_images/ORIGIN_cat.png
```

## 评估 VLM

评估三个使用不同视觉编码器的 VLM：

| VLM | 视觉编码器 | 原论文是否测试 | 显存 |
|-----|-----------|--------------|------|
| BLIP-2 (OPT-2.7B) | EVA-ViT-G | 是 | ~6 GB |
| DeepSeek-VL-1.3B | SigLIP-L | 否 | ~4 GB |
| Qwen2.5-VL-3B | ViT-L | 否 | ~6 GB |

只评估单个 VLM：

```bash
python evaluate.py --adv-image outputs/adversarial.png \
                   --target-vlms blip2_opt_2_7b
```

## 文件结构

```
demo_S2P/
├── config.py               # 配置（HF 仓库、攻击参数、评估 VLM）
├── download_weights.py     # 从 HuggingFace 下载官方权重
├── demo.py                 # 生成对抗图片（单对图片）
├── evaluate.py             # 评估（对比 VLM 对三张图片的描述）
├── README.md               # 英文文档
├── README_zh.md            # 本文件
├── checkpoints/            # 下载的权重（gitignored）
└── outputs/                # 生成的图片和结果（gitignored）
```

## 与 demo_S2_AnyAttack 的关系

| | demo_S2_AnyAttack | demo_S2P |
|---|---|---|
| 定位 | 完整复现（从零训练） | 使用官方权重推理 |
| 训练 | LAION-Art 预训练 + COCO 微调 | 无需训练 |
| 硬件 | HPC（H200） | 本地 GPU（RTX 4090） |
| 权重 | 自行训练 | 从 HuggingFace 下载 |
| 模型代码 | CLIPEncoder、Decoder、losses、dataset | 复用 S2 的模型和数据加载代码 |

## 参考文献

```bibtex
@inproceedings{zhang2025anyattack,
  title={Anyattack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models},
  author={Zhang, Jiaming and Ye, Junhong and Ma, Xingjun and Li, Yige and Yang, Yunfan and Chen, Yunhao and Sang, Jitao and Yeung, Dit-Yan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2025}
}
```
