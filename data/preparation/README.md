# `data/preparation/` — 数据/模型下载工具

VisInject 运行时所需的两类外部资源（VLM 权重 + AnyAttack decoder 权重）的下载脚本。**所有下载都建议在 HPC 登录节点跑**（计算节点访问不了 HuggingFace CDN）。

## 目录结构

```
data/preparation/
├── README.md                       # 本文件
└── models/
    ├── download_all_models.py      # 5 个 VLM + CLIP（HuggingFace）
    └── download_decoder_weights.py # AnyAttack coco_bi.pt
```

## 1. 下载 VLM 权重

```bash
# 在登录节点跑（有 internet）
export HF_HOME=/cluster/tufts/c26sp1ee0141/pliu07/model_cache

# 三个核心 VLM（约 13 GB）
python data/preparation/models/download_all_models.py --stage quick
# = qwen2_5_vl_3b + blip2_opt_2_7b + clip-vit-base-patch32

# 全部 5 个 VLM（约 37 GB）
python data/preparation/models/download_all_models.py --stage full
# = quick + deepseek_vl_1_3b + llava_1_5_7b + phi_3_5_vision
```

## 2. 下载 AnyAttack Decoder 权重

```bash
python data/preparation/models/download_decoder_weights.py
# 输出: data/checkpoints/coco_bi.pt （约 320 MB）
```

来自 AnyAttack (CVPR 2025) 官方 HuggingFace 仓库 `jiamingzz/anyattack`。VisInject 直接复用这份预训练权重，没有自训练 decoder（评估过 LAION-Art 18K 图片相比原论文 LAION-400M 差距过大，性价比不足）。

## HPC 路径

| 资源 | 路径 |
|---|---|
| Conda env | `/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject` |
| HF 模型缓存 | `/cluster/tufts/c26sp1ee0141/pliu07/model_cache` |
| Decoder 权重（仓库内） | `data/checkpoints/coco_bi.pt` |
| Work dir | `/cluster/tufts/c26sp1ee0141/pliu07/vis_inject` |

完整 HPC 工作流见 [`../docs/HPC_GUIDE.md`](../docs/HPC_GUIDE.md)。
