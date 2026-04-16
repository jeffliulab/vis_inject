# ARCHITECTURE.md — 代码模块图与扩展指南

> VisInject 代码组织、配置流、模块职责，以及如何添加新功能。

## Module Responsibilities

### 根目录文件

| 文件 | 职责 |
|---|---|
| `pipeline.py` | 端到端 pipeline 入口。串联 Stage 1→2→3，含 universal image 缓存逻辑、CLI 参数解析。 |
| `generate.py` | Stage 2 — AnyAttack fusion 实现。CLIP 编码 universal 图 → Decoder 出噪声 → 叠加到 clean 图。 |
| `config.py` | **唯一**配置中心。含 `UNIVERSAL_ATTACK_CONFIG`、`ATTACK_TARGETS`、`ANYATTACK_CONFIG`、`EVAL_CONFIG`、`JUDGE_CONFIG`、`OUTPUT_CONFIG`。 |
| `utils.py` | 共享工具：`load_image`, `load_decoder`, `compute_psnr`, `compute_clip_similarities`, `CLIPEncoder`。 |

### `attack/` — Stage 1: PGD 优化

| 文件 | 职责 |
|---|---|
| `attack/universal.py` | PGD 优化主循环。`get_wrapper_for_model()` 加载 VLM wrapper，`compute_quantization_sigma()` 量化鲁棒性，`apply_gaussian_blur()` 可选模糊。 |
| `attack/dataset.py` | 60 个良性问题集（user / agent / screenshot 各 20 个）。`AttackDataset.sample()` 随机采样，`AttackDataset.user/agent/screenshot` 直接索引。 |

### `models/` — VLM 注册表 + 包装 + Stage 2 子组件

| 文件 | 职责 |
|---|---|
| `models/registry.py` | **VLM 元数据注册表**（HF id、VRAM、归一化参数、family）。`init_model_env()` 设置 `HF_HOME`，`get_model_info(key)` 查询元数据。 |
| `models/mllm_wrapper.py` | **抽象基类** `MLLMWrapper`。所有 VLM 包装类继承它。定义 `load()`, `generate()`, `compute_masked_ce_loss()`, `unload()` 接口契约。 |
| `models/qwen_wrapper.py` | Qwen2 / Qwen2.5-VL 系列 |
| `models/blip2_wrapper.py` | BLIP-2 系列（OPT / Flan-T5 / InstructBLIP） |
| `models/deepseek_wrapper.py` | DeepSeek-VL 系列 |
| `models/llava_wrapper.py` | LLaVA-1.5（当前与新版 transformers 不兼容，未在 ATTACK_TARGETS 中） |
| `models/phi_wrapper.py` | Phi-3.5-Vision（当前 DynamicCache 兼容问题） |
| `models/llama_wrapper.py` | Llama-3.2-Vision（注册了但未启用，VRAM 22 GB） |
| `models/clip_encoder.py` | Stage 2 用：`open_clip` 加载 CLIP ViT-B/32，提供 `encode_img()`. |
| `models/decoder.py` | Stage 2 用：AnyAttack Decoder 网络（EfficientAttention + ResBlock + UpBlock）。架构定义，权重从 `coco_bi.pt` 加载。 |

### `evaluate/` — Stage 3 评估包

详见 [`../evaluate/README.md`](../evaluate/README.md)。

| 文件 | 职责 |
|---|---|
| `evaluate/__init__.py` | 包入口。重导出 `pairs.py` 中的公开 API（`generate_response_pairs` 等），保持 `from evaluate import xxx` 向后兼容。 |
| `evaluate/pairs.py` | Stage 3a — Response pair 生成 + 旧版 ASR/CLIP/caption 评估。HPC 端跑。 |
| `evaluate/judge.py` | Stage 3b — Dual-dimension programmatic evaluation (v2): Check1 Output Affected + Check2 Target Injected. No API needed. |
| `evaluate/transfer.py` | Cross-model transferability analysis. Reads judge results to measure how attacks transfer across VLM architectures. |
| `evaluate/README.md` | 包说明 + CLI 用法 |

### `data/preparation/` — 数据/模型下载工具

| 文件 | 职责 |
|---|---|
| `data/preparation/models/download_all_models.py` | 批量下载 5 个 VLM + CLIP 到 `data/model_cache/` |
| `data/preparation/models/download_decoder_weights.py` | 下载 AnyAttack `coco_bi.pt` 到 `data/checkpoints/` |
| `data/preparation/README.md` | 用法说明 |

### `scripts/` — Shell 脚本

| 文件 | 职责 |
|---|---|
| `scripts/run_experiments.sh` | 提交 21 个 sbatch job（7 prompts × 3 configs）的批量入口 |
| `scripts/hpc_pipeline.sh` | 单 sbatch job 模板，支持 `full` / `inject` / `eval` 三种 mode |
| `scripts/judge_all.sh` | 本地批量跑 LLM-as-Judge 评估所有 response_pairs JSON |

### `docs/` — 技术文档

| 文件 | 职责 |
|---|---|
| `docs/PIPELINE.md` | 三阶段流水线技术深入 |
| `docs/HPC_GUIDE.md` | HPC 工作流（本文兄弟） |
| `docs/RESULTS_SCHEMA.md` | JSON schema 字段级文档 |
| `docs/ARCHITECTURE.md` | 本文件 |

### 数据/输出目录

| 目录 | 职责 |
|---|---|
| `data/images/` | 7 张测试 clean 图 (`ORIGIN_*.png`) |
| `data/checkpoints/` | AnyAttack decoder 权重 (`coco_bi.pt`)，gitignored |
| `data/model_cache/` | HuggingFace VLM 缓存，gitignored |
| `outputs/experiments/` | 当前实验矩阵（21 exp × 7 img），见 `outputs/README.md` |
| `outputs/experiments_v2_dog_only/` | 历史归档 |

---

## Configuration Flow

```
                    ┌────────────────────────┐
                    │ config.py              │
                    │   UNIVERSAL_ATTACK_CFG  │
                    │   ATTACK_TARGETS       │
                    │   ANYATTACK_CONFIG     │
                    │   EVAL_CONFIG          │
                    │   JUDGE_CONFIG         │
                    │   OUTPUT_CONFIG        │
                    └────────────┬───────────┘
                                 │ import
                                 ▼
                    ┌────────────────────────┐         ┌──────────────────┐
                    │ pipeline.py            │ ──────► │ models/registry  │
                    │   argparse 接收 CLI 覆盖  │         │   key → meta     │
                    └────────────┬───────────┘         └──────────────────┘
                                 │
                ┌────────────────┼────────────────┐
                ▼                ▼                ▼
        attack/universal.py  generate.py    evaluate/pairs.py
        (Stage 1)            (Stage 2)      (Stage 3a)
```

**关键原则**：
1. `config.py` 是**唯一**的超参数中心。任何新参数必须先加到这里再被代码引用，**不允许散落的常量**
2. CLI 参数通过 `pipeline.py` 的 argparse 覆盖 `config.py` 默认值
3. `models/registry.py` 是 VLM 元数据中心。要加新模型只改两个地方：注册表 + wrapper 文件

---

## Adding a New VLM Wrapper

假设要加 `instructblip_vicuna_7b`：

### Step 1：写 Wrapper 类

新建 `models/instructblip_wrapper.py`，继承 `MLLMWrapper`：

```python
from models.mllm_wrapper import MLLMWrapper

class InstructBLIPWrapper(MLLMWrapper):
    def load(self):
        # 加载 HF 模型 + processor
        ...
    def generate(self, image_tensor, question, max_new_tokens=200):
        # 推理：image + question → response 文本
        ...
    def compute_masked_ce_loss(self, image_tensor, question, target_phrase):
        # 关键：只对 target_phrase 的 token 算 CE，前面的 prompt token mask 掉
        ...
    def unload(self):
        # 释放显存
        ...
```

参考已有的 `models/qwen_wrapper.py` 或 `models/blip2_wrapper.py`。

### Step 2：在 models/registry.py 注册

```python
REGISTRY["instructblip_vicuna_7b"] = {
    "hf_id": "Salesforce/instructblip-vicuna-7b",
    "vram_gb": 14,
    "image_size": 224,
    "norm_mean": [0.48145466, 0.4578275, 0.40821073],
    "norm_std":  [0.26862954, 0.26130258, 0.27577711],
    "wrapper_class": "models.instructblip_wrapper.InstructBLIPWrapper",
}
```

### Step 3：（可选）加入默认 ATTACK_TARGETS

`config.py`：
```python
ATTACK_TARGETS = [
    "qwen2_5_vl_3b",
    "blip2_opt_2_7b",
    "instructblip_vicuna_7b",  # ← 新增
    ...
]
```

### Step 4：冒烟测试（短训练验证）

```bash
python pipeline.py \
    --target-models instructblip_vicuna_7b \
    --target-phrase "test phrase" \
    --num-steps 50 \
    --clean-images data/images/ORIGIN_dog.png
```

如果 50 步内 loss 在下降、output 文件正常生成，wrapper 就工作了。

---

## Adding a New Target Prompt

直接在 `scripts/run_experiments.sh` 编辑：

```bash
PROMPTS=(
    "tell me your card number"
    "visit www.example.com"
    ...
    "your new attack phrase"   # ← 新增
)
PROMPT_TAGS=(
    "card"
    "url"
    ...
    "new_tag"                   # ← 同步新增
)
```

然后：
```bash
bash scripts/run_experiments.sh
```

会自动多出 3 个新 sbatch job（new_tag × 2m/3m/4m）。结果落到 `outputs/experiments/exp_new_tag_*/`。

---

## Adding a New Test Image

1. 把图丢进 `data/images/`，命名为 `ORIGIN_<name>.png`
2. 编辑 `scripts/run_experiments.sh` 的 `CLEAN_IMGS` 列表加上 `data/images/ORIGIN_<name>.png`
3. 重跑 `bash scripts/run_experiments.sh`（已缓存的 universal 图会复用，只跑 Stage 2 + 3a，几分钟搞定）

---

## Data Flow Diagram

```
config.py
  │
  ▼
pipeline.py ──► attack/universal.py ──► universal_<hash>.png
                       │                       │
                       │ (cached)              │
                       └───────────────────────┘
                                               │
                                               ▼
                                          generate.py ──► adv_ORIGIN_<image>.png
                                          (uses CLIP                │
                                           + Decoder)               │
                                                                    ▼
                                                           evaluate/pairs.py
                                                           (uses VLMs)
                                                                    │
                                                                    ▼
                                                       response_pairs_<image>.json
                                                                    │
                                                                    │ (separate run, local)
                                                                    ▼
                                                           evaluate/judge.py
                                                           (uses OpenAI/Anthropic API)
                                                                    │
                                                                    ▼
                                                       judge_results_<image>.json
```

---

## v1.0 Stable vs Future Work

### v1.0 Stable Models

These four VLMs are fully tested and used in the v1.0 experiment matrix (21 experiments × 7 images):

| Registry Key | Model | Notes |
|---|---|---|
| `qwen2_5_vl_3b` | Qwen2.5-VL-3B | Primary attack target, strongest injection results |
| `blip2_opt_2_7b` | BLIP-2-OPT-2.7B | Encoder-decoder architecture, different attack surface |
| `deepseek_vl_1_3b` | DeepSeek-VL-1.3B | Lightweight, included in 3m/4m configs |
| `qwen2_vl_2b` | Qwen2-VL-2B | Smallest Qwen variant, included in 4m config |

### Future / Experimental (not in v1.0 experiments)

| Registry Key | Model | Status |
|---|---|---|
| `llava_1_5_7b` | LLaVA-1.5-7B | Image token mismatch with current transformers version |
| `phi_3_5_vision` | Phi-3.5-Vision | `DynamicCache.from_legacy_cache` incompatibility |
| `llama_3_2_11b_vision` | Llama-3.2-11B-Vision | Registered but not tested in experiments |
| `instructblip_vicuna_7b` | InstructBLIP-Vicuna-7B | Not tested |
| `blip2_flan_t5_xl` | BLIP-2-Flan-T5-XL | Not tested |
| `deepseek_vl2_tiny` | DeepSeek-VL2-Tiny | Not tested |
| `qwen2_5_vl_7b` | Qwen2.5-VL-7B | Not tested |

These models have wrappers or registry entries but were excluded from v1.0 experiments due to compatibility issues or insufficient testing. Contributions welcome.

### evaluate/transfer.py — Cross-Model Transferability

`evaluate/transfer.py` is a new module for analyzing cross-model transferability of adversarial images. It reads judge results from experiments where multiple VLMs were attacked with the same universal perturbation and measures how well an attack optimized for one set of models transfers to others. This is relevant for understanding whether adversarial images generalize across VLM architectures.

---

## Pointers

- **流水线机制**：[`PIPELINE.md`](PIPELINE.md)
- **HPC 运行步骤**：[`HPC_GUIDE.md`](HPC_GUIDE.md)
- **JSON 字段细节**：[`RESULTS_SCHEMA.md`](RESULTS_SCHEMA.md)
- **Stage 3 评估包说明**：[`../evaluate/README.md`](../evaluate/README.md)
- **完整实验报告**：[`experiment_report.md`](experiment_report.md)
