# `outputs/` — 实验输出索引

> 所有实验产物的根目录。**完全 gitignored**，结构由 `pipeline.py` 和 `scripts/run_experiments.sh` 自动生成。

## 当前结构

```
outputs/
├── README.md                        # 本文件
│
├── experiments/                     # ★ 当前实验矩阵 (21 exp × 7 image = 147 response_pairs)
│   ├── README.md                    # 矩阵描述 + 每实验布局
│   └── exp_<prompt>_<config>/       # 例：exp_url_2m, exp_email_3m
│       ├── universal/
│       │   └── universal_<hash>.png       # Stage 1 通用对抗图
│       ├── adversarial/
│       │   └── adv_ORIGIN_<image>.png     # Stage 2 对抗照片（每张 clean 图一份）
│       └── results/
│           ├── response_pairs_ORIGIN_<image>.json   # Stage 3a 输出
│           └── judge_results_ORIGIN_<image>.json    # Stage 3b 输出（跑 judge 后才有）
│
└── experiments_v2_dog_only/         # 历史归档（v2，仅 dog 图，3 prompts × 3 configs）
    ├── README.md
    └── exp_<prompt>_<config>/
        ├── response_pairs_ORIGIN_dog.json
        └── judge_results.json
```

## 实验矩阵

详见 [`experiments/README.md`](experiments/README.md)。

| 维度 | 数量 | 说明 |
|---|---|---|
| 实验组 | **21** | 7 prompts × 3 model configs |
| 测试图 | **7** | dog / cat / bill / kpop / webpage / code / chat |
| 评估问题/图 | **15** | 3 类（user/agent/screenshot）× 5 个 |
| 总 response pairs | **6,615** | 21 × 7 × 3 VLM × 15 |

## 单次运行的临时目录

`hpc_pipeline.sh` 在 `full` 模式下若不指定 `--output-dir`，会向 `outputs/universal/`、`outputs/adversarial/`、`outputs/results/` 写入。**这些目录在本仓库不入库**（已加入 `.gitignore`），只是 HPC 单次实验的临时落地。批量实验（`run_experiments.sh`）总是用 `--output-dir outputs/exp_<tag>` 显式指定，不会污染这些临时目录。

## 想看格式细节？

JSON 字段级 schema 见 [`../docs/RESULTS_SCHEMA.md`](../docs/RESULTS_SCHEMA.md)。
