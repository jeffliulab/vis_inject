# CLAUDE.md — VisInject Agent Guide

> 给 Claude / 其他 agent 的项目入门指南。目标：60 秒内理解定位，知道边界，知道去哪里找深入信息。

## Project Essence

VisInject 是一个**对视觉语言模型（VLM）做对抗性 prompt injection 的研究代码库**。

威胁模型：攻击者只能改图，不能改用户的提问。攻击目标是让 VLM 在用户上传一张"看起来正常"的照片并问"describe this image"时，回答中夹带攻击者指定的内容（信用卡诱导、URL、品牌植入、虚假信息等）。

技术路径（三阶段）：
1. **Stage 1 — UniversalAttack**：在灰图上做 PGD 优化，得到一张"通用对抗图"，使多个目标 VLM 都倾向输出 target phrase
2. **Stage 2 — AnyAttack Fusion**：用 CLIP 编码 universal 图，再用预训练 AnyAttack Decoder 把语义解码成有界噪声（eps=16/255），叠加到任意 clean 照片上得到对抗照片（PSNR ≈ 25 dB，人眼无法区分）
3. **Stage 3 — LLM-as-Judge**：对 (clean, adv) 图分别让 VLM 回答 60 个良性问题，再用 GPT-4o-mini 对比两份回答打分（0-10）

**这是研究代码，不是生产部署**。不会上 HuggingFace Space（多 VLM 联合加载，免费 tier 不够），不会做完整测试套件，重点是清晰、可复现、可写报告。

## Topology（每件东西在哪）

```
VisInject/
├── CLAUDE.md                    ← 你正在读
├── README.md / README_CN.md     ← 双语门面
├── 实验报告.md                   ← 实验叙事主档（中文，正式报告）
│
├── pipeline.py                  ← 端到端入口（CLI + cache）
├── generate.py                  ← Stage 2 实现
├── config.py                    ← 唯一配置中心
├── utils.py                     ← 共享工具
│
├── attack/                      ← Stage 1：PGD 优化
│   ├── universal.py
│   └── dataset.py               ← 60 良性问题集
│
├── models/                      ← VLM wrapper + Stage 2 子组件
│   ├── registry.py              ← VLM 元数据注册表（原 model_registry.py）
│   ├── mllm_wrapper.py          ← 抽象基类（接口契约）
│   ├── qwen_wrapper.py / blip2_wrapper.py / deepseek_wrapper.py / ...
│   ├── clip_encoder.py          ← Stage 2 用
│   └── decoder.py               ← Stage 2 用（AnyAttack 网络结构）
│
├── evaluate/                    ← Stage 3 评估包
│   ├── __init__.py              ← 重导出公开 API（向后兼容 from evaluate import）
│   ├── pairs.py                 ← Stage 3a: response pair 生成（HPC GPU）
│   ├── judge.py                 ← Stage 3b: LLM-as-Judge（本地 API）
│   └── README.md
│
├── scripts/                     ← Shell 脚本
│   ├── run_experiments.sh       ← HPC 批量提交（21 个 sbatch job）
│   ├── hpc_pipeline.sh          ← 单 sbatch job 模板
│   └── judge_all.sh             ← 本地批量跑 judge
│
├── demo/                        ← Gradio 演示（两个版本）
│   ├── README.md                ← 对比与导航
│   ├── space_demo/              ← 精简版，CPU-only，可部署到 HF Space
│   │   ├── app.py               ← 仅 Stage 2 fusion（复用预生成 universal 图）
│   │   ├── requirements.txt
│   │   └── README.md
│   └── full_demo/               ← 完整双语演示，本地 GPU
│       ├── web_demo.py          ← Stage 1 + Stage 2 + 评估
│       └── README.md
│
├── docs/                        ← 技术文档
│   ├── PIPELINE.md              ← 三阶段攻击机制
│   ├── HPC_GUIDE.md             ← Tufts HPC 工作流
│   ├── RESULTS_SCHEMA.md        ← JSON schema 字段级
│   └── ARCHITECTURE.md          ← 代码模块图 + 扩展指南
│
├── data_preparation/            ← 数据/模型下载工具
│   ├── README.md
│   └── models/
│       ├── download_all_models.py
│       └── download_decoder_weights.py
│
├── images/                      ← 7 张测试 clean 图
├── checkpoints/                 ← coco_bi.pt（gitignored）
├── model_cache/                 ← HF 缓存（gitignored）
│
└── outputs/
    ├── README.md
    ├── experiments/             ← 21 实验 × 7 图 = 147 response_pairs
    │   ├── README.md
    │   └── exp_<prompt>_<config>/
    │       ├── universal/       ← Stage 1 输出
    │       ├── adversarial/     ← Stage 2 输出
    │       └── results/         ← Stage 3a + 3b 输出
    └── experiments_v2_dog_only/ ← 历史归档（只读）
```

## Code Conventions

- **扁平根布局**——`pipeline.py`、`generate.py`、`config.py` 等都在根。**不要引入 `src/` 重构**
- **`config.py` 是所有超参数的单一数据源**。新参数先加这里，不允许散落常量
- **所有 VLM 通过 `models/registry.py` 注册**。加新 VLM 看 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **不破坏 `models/mllm_wrapper.py` 的接口契约**（修改前要 grep 所有 wrapper）
- **outputs 命名约定**：`exp_<prompt>_<config>/{universal,adversarial,results}/`，详见 [`docs/RESULTS_SCHEMA.md`](docs/RESULTS_SCHEMA.md)
- **`evaluate/` 是包**，`from evaluate import generate_response_pairs` 通过 `__init__.py` 重导出。CLI 用 `python -m evaluate.pairs` / `python -m evaluate.judge`，**不要**直接 `python evaluate/pairs.py`（sys.path 会出问题）

## HPC Specifics (Tufts)

| 项 | 值 |
|---|---|
| Cluster work dir | `/cluster/tufts/c26sp1ee0141/pliu07/vis_inject` |
| Conda env | `/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject` |
| HF cache | `/cluster/tufts/c26sp1ee0141/pliu07/model_cache` |
| GPU | H200 80GB 单卡（partition `gpu`） |
| 提交单 job | `sbatch scripts/hpc_pipeline.sh full images/ORIGIN_dog.png` |
| 提交 21 job 矩阵 | `bash scripts/run_experiments.sh` |
| 监控 | `squeue -u pliu07` |

完整 SLURM workflow + troubleshooting 在 [`docs/HPC_GUIDE.md`](docs/HPC_GUIDE.md)。

## Rules for Agents

1. **研究代码，不要加测试套件 / CI**，除非用户明确要求
2. **不要动 `outputs/experiments/` 内容**——这是冻结的实验数据
3. **不要动 `outputs/experiments_v2_dog_only/`**——历史归档
4. **不要删改 `实验报告.md` 的科学叙事内容**——只能在顶部加引用、追加新章节、或更新过期的命令字符串
5. **优先编辑现有文件而不是新建**
6. **`scripts/judge_all.sh` 会花 API 钱**（GPT-4o-mini 全跑约 $3-5），运行前**必须**明确用户确认
7. **HPC 脚本的绝对路径不要"规范化"或参数化**，它们就是 Tufts 集群专用的，参数化反而引入风险
8. **`.env` 含 API key，gitignored**。绝不提交、绝不完整 cat 出来、绝不写进 commit message
9. **修改 `config.py` 默认值前 grep 所有调用方**，确保了解影响范围
10. **如果发现死代码或陈旧引用**，优先清理而不是绕过——除非用户明确要保留作为历史

## Deep Dives

| 想了解 | 看这里 |
|---|---|
| 三阶段流水线机制（技术原理） | [`docs/PIPELINE.md`](docs/PIPELINE.md) |
| HPC 工作流（怎么跑实验） | [`docs/HPC_GUIDE.md`](docs/HPC_GUIDE.md) |
| JSON 输出字段含义 | [`docs/RESULTS_SCHEMA.md`](docs/RESULTS_SCHEMA.md) |
| 代码模块图 + 怎么加新 VLM/prompt | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Stage 3 评估包的使用 | [`evaluate/README.md`](evaluate/README.md) |
| 实验设计 + 结果叙事 | [`实验报告.md`](实验报告.md) |
| 数据/模型下载 | [`data_preparation/README.md`](data_preparation/README.md) |
| 实验输出目录结构 | [`outputs/README.md`](outputs/README.md) |

---

## Session Log

> 短的运行性日志：每次大改动留下"做了什么 / 为什么 / 留意什么"。新条目追加到末尾，不要删旧条目。

### 2026-04-07 — 项目大整理 + 第三轮实验完成 + HF 部署

**起点**：第二轮实验只测了 1 张图（dog），代码散乱，没有 HF 资源。

**完成的事**（按时间顺序）：

1. **多图片实验扩展**
   - `scripts/run_experiments.sh` 加 4 个新 prompt（email / news / ad / obey），加 7 张测试图（原 dog + cat / bill / kpop / webpage / code / chat）
   - HPC 上提交 21 个 sbatch job（7 prompts × 3 model configs），全部跑完
   - 下载本地：147 个 `response_pairs_*.json` + 147 张 adv 图 + 21 张 universal 图（约 32 MB）
   - 旧的 9 个 dog-only 实验归档到 `outputs/experiments_v2_dog_only/`

2. **放弃自训练 art decoder**
   - 一度尝试自训练 AnyAttack Decoder（写了 `train_decoder.py`、下载 18K LAION-Art 图片）
   - 评估后放弃：相比原论文 LAION-400M 差 ~22000×，性价比太低
   - 全部清理（HPC 端 + 本地），决定继续复用 `jiamingzz/anyattack` 的 `coco_bi.pt`

3. **项目结构大整理**（参考 `D:/Projects/Agent规范` 但不照搬）
   - 新建 `docs/`：4 篇深入文档（PIPELINE / HPC_GUIDE / RESULTS_SCHEMA / ARCHITECTURE）
   - 新建 `scripts/`：把 3 个 shell 脚本从根目录移进来
   - 新建 `evaluate/` 包：`evaluate.py` → `evaluate/pairs.py`，`judge.py` → `evaluate/judge.py`，新建 `__init__.py` 重导出公开 API（向后兼容 `from evaluate import xxx`），新建 `evaluate/README.md`
   - 新建 `demo/{space_demo,full_demo}/`：原来 `web_demo.py` 移到 `demo/full_demo/`，新写一份 `demo/space_demo/app.py`（仅 Stage 2，CPU 可跑）
   - `model_registry.py` → `models/registry.py`（属于 models 包的一部分）
   - 删除：`view_results.py`（陈旧）、`data_preparation/laion_art/`（放弃的 art decoder 训练）、`data_preparation/demo_images/`（遗留）
   - 新建 `CLAUDE.md`（本文件）作为 agent 入门指南
   - 全面同步 `README.md` / `README_CN.md` / `实验报告.md` / `.gitignore`（清掉幽灵路径）

4. **根目录瘦身**
   - 之前根目录 6 个 .py：`pipeline.py / generate.py / config.py / utils.py / model_registry.py / web_demo.py`
   - 整理后 4 个：`pipeline.py / generate.py / config.py / utils.py`（每个都是入口或共享工具，留根合理）

5. **HF 部署**
   - 创建 [`jeffliulab/visinject` (Dataset)](https://huggingface.co/datasets/jeffliulab/visinject)：21 实验完整数据 + Dataset Card + v2 归档
   - 创建 [`jeffliulab/visinject` (Space)](https://huggingface.co/spaces/jeffliulab/visinject)：Stage 2 fusion 演示，CPU 可跑，运行时从 HF Hub 拉资源
   - **不创建** Model repo（项目不训练任何 nn.Module 权重，唯一的 `coco_bi.pt` 是借的）

**部署路上踩的坑**：
- HF Space 默认 Python 3.13，3.13 删了 `audioop`，pydub→Gradio 链断 → 在 README YAML 加 `python_version: "3.11"`
- Gradio 4.44 import `HfFolder`，但 huggingface_hub 1.0+ 删了它 → 升 Gradio 到 5.x

**未做的事**（明确推迟）：
- ❌ Judge 结果增量上传到 HF Dataset

**留给下一次 agent 的注意事项**：
- HF Dataset 增量上传用 `hf upload jeffliulab/visinject <local> <remote> --repo-type dataset --commit-message "Add judge results"`

### 2026-04-16 — v2 评估系统 + 迁移性测试 + v1.0 封版

**起点**：v1 LLM-as-Judge 混淆"输出干扰"与"目标注入"（报告 50.5% "注入率"实为干扰率），缺少迁移性研究。

**完成的事**：

1. **v2 评估系统**
   - 重写 `evaluate/judge.py`：双维度评估（Check1 Output Affected + Check2 Target Injected）
   - 不再需要 LLM API，改为程序化评估（文本相似度 + 关键词匹配）
   - 7 个 Claude agent 并行判断全部 6,615 pairs → 写出 147 个 `judge_results_*.json`
   - 更新 `config.py`（JUDGE_CONFIG 改为单 DeepSeek judge + version 2）
   - 更新 `scripts/judge_all.sh`（支持 `--force` 和 `--version` 参数）

2. **核心发现修正**
   - v1: "注入率 50.5%" → v2: "干扰率 66.3%，注入率 0.23%"
   - 10 个与目标相关的注入案例：2 确认 + 3 部分 + 5 弱
   - 关键洞察：攻击是"破坏性"的（干扰 VLM 输出），不是"建设性"的（植入目标内容）

3. **注入案例整理**
   - 创建 `outputs/succeed_injection_examples/`：12 张图片（4 clean + 8 adv）+ manifest.json
   - Demo 新增 "Injection Cases" Tab（Gradio，clean vs adv 图片/回答对比）

4. **迁移性测试**
   - 新建 `evaluate/transfer.py`：通过 vision API 测试跨模型迁移
   - 手动测试 GPT-4o：最强注入案例（URL + 代码截图）在 GPT-4o 上完全失败
   - GPT-4o 主动识别对抗噪声为"distortion, artifacts"，正确恢复代码内容

5. **v1.0 封版**
   - README.md / README_CN.md 按 agent-rules 规范重写（双语 badge、Highlights、Key Results）
   - 全部 docs/ 文档更新
   - 实验报告全面更新（Section 2.4, 6.3-6.9, 7, 8, 9, 10）
   - 清理临时文件，更新 .gitignore
   - 创建 v1.0 tag + v1.0-release branch
