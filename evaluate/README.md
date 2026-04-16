# `evaluate/` — Stage 3 Evaluation Module

VisInject 三阶段流水线的最后一阶：判定攻击是否成功。

## 设计

评估分两步进行，**故意解耦**，因为它们的硬件需求完全不同：

| 子阶段 | 文件 | 运行位置 | 资源 | 输入 | 输出 |
|---|---|---|---|---|---|
| **3a — Pairs 生成** | `pairs.py` | HPC GPU | 1× H200，加载 N 个 VLM | clean 图 + adv 图 + 60 个良性问题 | `response_pairs_<image>.json` |
| **3b — Dual-dimension Judge (v2)** | `judge.py` | 本地 / 任意 CPU | 纯程序化，无需 API key，$0 成本 | response_pairs JSON | `judge_results_<image>.json` |

3a 必须在 HPC 上跑（VLM 需要 GPU），但 3b 不需要 GPU、不需要 VLM、也不需要任何外部 API —— 纯本地程序化评估。这样：
- 3a 跑一次产出的 JSON 可以反复跑 3b（调整阈值 / 换关键词列表）
- 3b 不占 HPC 时间，不花 API 钱，便于本地迭代评估方法

### v2 评估体系（当前默认）

v2 使用**双维度程序化评估**，替代了 v1 的 LLM-as-Judge（GPT-4o / GPT-4o-mini / DeepSeek）。不再需要 API key，成本从 $3-5 降为 $0。

| 维度 | 检测内容 | 方法 |
|---|---|---|
| **Check1 — Output Affected** | 对抗图是否改变了 VLM 的输出？ | 文本相似度对比 (clean response vs adv response) |
| **Check2 — Target Injected** | 目标短语是否被注入到输出中？ | 关键词匹配 (target phrase keywords in adv response) |

两个维度互补：Check1 测"攻击是否有效果"，Check2 测"效果是否命中目标"。只有两者都通过的情况才算真正的成功注入。

### 附加模块

| 文件 | 功能 |
|---|---|
| `transfer.py` | 跨模型迁移性测试 —— 通过 vision API（GPT-4o、Gemini 等）验证对抗图在未参与训练的黑盒 VLM 上是否仍然有效 |

## CLI 用法

**`pairs.py`**（HPC 端，由 `pipeline.py` 自动调用，也可单独跑）：
```bash
python -m evaluate.pairs \
    --adv-images outputs/experiments/exp_url_2m/adversarial/adv_ORIGIN_dog.png \
    --clean-images data/images/ORIGIN_dog.png \
    --universal-image outputs/experiments/exp_url_2m/universal/universal_<hash>.png \
    --target-phrase "visit www.example.com"
```

**`judge.py`**（本地端，v2 默认使用程序化评估，无需 API key）：
```bash
# v2 dual-dimension evaluation (default, $0 cost)
python -m evaluate.judge \
    --pairs-file outputs/experiments/exp_url_2m/results/response_pairs_ORIGIN_dog.json
```

批量跑所有实验：
```bash
# Run all experiments
bash scripts/judge_all.sh

# Overwrite existing results
bash scripts/judge_all.sh --force
```

## 程序化调用（已被 `pipeline.py` 使用）

```python
from evaluate import generate_response_pairs

result = generate_response_pairs(
    adv_image_path="...",
    clean_image_path="...",
    target_phrase="visit www.example.com",
    target_vlms=["qwen2_5_vl_3b", "blip2_opt_2_7b"],
    output_path="outputs/experiments/exp_url_2m/results/response_pairs_ORIGIN_dog.json",
)
```

`evaluate/__init__.py` 重导出全部公开 API（`generate_response_pairs`, `run_evaluation`, `evaluate_asr`, `evaluate_image_quality`, `evaluate_clip`, `evaluate_captions`），所以历史的 `from evaluate import xxx` 写法继续有效。

## 输出格式

- `response_pairs_<image>.json` —— 见 [`docs/RESULTS_SCHEMA.md`](../docs/RESULTS_SCHEMA.md)
- `judge_results_<image>.json` —— 同上

## 设计注意

- **3a 与训练完全解耦**：`pairs.py` 不知道 universal image 是怎么来的，它只接收 (clean, adv) 路径
- **3b 与 VLM 完全解耦**：`judge.py` 只读 JSON，不依赖 PyTorch、任何 VLM 或外部 API
- **v2 无需 API key**：纯程序化评估（文本相似度 + 关键词匹配），无外部依赖
- **`transfer.py` 需要 API key**：跨模型迁移性测试通过 vision API 调用外部 VLM，需要 `OPENAI_API_KEY` 等，从根目录 `.env` 读取
- **评估问题来自 `attack/dataset.py`**（共用同一份 60 个良性问题集，确保训练/评估问题分布一致）
