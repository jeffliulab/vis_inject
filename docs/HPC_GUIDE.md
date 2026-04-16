# HPC_GUIDE.md — Tufts HPC 工作流

> **Note**: This guide is specific to the Tufts University HPC cluster (pax). For other SLURM-based clusters, adapt the module paths, partition names, and conda environment paths accordingly. The Tufts-specific paths and commands below are intentionally hardcoded and should not be parameterized.

> VisInject 在 Tufts HPC（pax 集群）上完整跑一轮实验的可复现指南。
> 假设你已经有 Tufts cluster 账号、SSH 配置好、能访问 c26sp1ee0141 group。

## Cluster Paths

| 角色 | 路径 |
|---|---|
| **Work dir**（git 仓库 checkout 处） | `/cluster/tufts/c26sp1ee0141/pliu07/vis_inject` |
| **Conda env** | `/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject` |
| **HF 模型缓存** | `/cluster/tufts/c26sp1ee0141/pliu07/model_cache` |
| **SLURM logs** | `/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/logs` |
| **GPU 类型** | H200 80GB 单卡（partition `gpu`） |

所有这些都在 `scripts/hpc_pipeline.sh` 头部固定，作为 SLURM job 的环境变量加载。

## One-Time Setup

### 1. SSH + 克隆仓库

```bash
ssh tufts-login                    # ssh config 别名
cd /cluster/tufts/c26sp1ee0141/pliu07
git clone <visinject-repo-url> vis_inject
cd vis_inject
```

### 2. 创建/激活 conda 环境

如果还没有 `visinject` env：
```bash
module load anaconda
conda create -y -p /cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject python=3.10
conda activate /cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate open_clip_torch pillow tqdm
pip install deepseek-vl einops sentencepiece attrdict   # DeepSeek-VL 额外依赖
```

### 3. 下载 VLM 权重（在登录节点跑，不要 sbatch — 计算节点访问不了 HF CDN）

```bash
cd /cluster/tufts/c26sp1ee0141/pliu07/vis_inject
export HF_HOME=/cluster/tufts/c26sp1ee0141/pliu07/model_cache

# 三个核心 VLM（约 13 GB）
/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject/bin/python \
    data/preparation/models/download_all_models.py --stage quick

# 全部 5 个 VLM（约 37 GB）
/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject/bin/python \
    data/preparation/models/download_all_models.py --stage full
```

### 4. 下载 AnyAttack Decoder 权重（约 320 MB）

```bash
/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject/bin/python \
    data/preparation/models/download_decoder_weights.py
# 输出落到 data/checkpoints/coco_bi.pt
```

---

## Running a Single Experiment

`scripts/hpc_pipeline.sh` 是单 sbatch job 的模板，支持三种 mode：

```bash
# 完整 3 阶段（默认）
sbatch scripts/hpc_pipeline.sh full data/images/ORIGIN_dog.png

# 只跑 Stage 2（已有 universal 图）
sbatch scripts/hpc_pipeline.sh inject data/images/ORIGIN_dog.png

# 只跑 Stage 3a（已有 adv 图）
sbatch scripts/hpc_pipeline.sh eval data/images/ORIGIN_dog.png
```

监控：
```bash
squeue -u $USER
tail -f logs/slurm_<JOBID>.out
```

---

## Running the Full Experiment Matrix

`scripts/run_experiments.sh` 提交 21 个独立 sbatch job（7 prompts × 3 model configs），每个 job 在 7 张测试图上跑完整 3 阶段：

```bash
cd /cluster/tufts/c26sp1ee0141/pliu07/vis_inject
bash scripts/run_experiments.sh
```

结果落在：
```
outputs/exp_<prompt>_<config>/
├── universal/universal_<hash>.png      # 通用对抗图（缓存复用）
├── adversarial/adv_ORIGIN_<image>.png  # 7 张对抗照片
└── results/response_pairs_<image>.json # 7 个 response_pairs JSON
```

监控：
```bash
squeue -u $USER                                   # 看排队
ls outputs/                                       # 看哪些跑完
tail -f logs/exp_card_2m_<JOBID>.out              # 看具体一个 job
```

**资源占用**（参考第三轮实验）：
- 2m 配置：~32 GB RAM、6 小时（含 7 图推理）
- 3m 配置：~48 GB RAM、8 小时
- 4m 配置：~64 GB RAM、10 小时

总实验时间约 30-40 小时（21 jobs，看 GPU 排队情况）。

---

## Downloading Results to Local

从本地工作站执行：

```bash
cd d:/Projects/VisInject  # 本地仓库根

# 完整下载 outputs/experiments/ (含中间快照，约 240 MB)
scp -r tufts-login:/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/outputs/exp_* outputs/experiments/

# 清理冗余的中间 step 快照和 .pt checkpoint，节省空间到 ~32 MB
cd outputs/experiments
find . -name "universal_step*.png" -delete
find . -name "*.pt" -delete
```

或用 `rsync`（断点续传更稳）：
```bash
rsync -avz --exclude="universal_step*.png" --exclude="*.pt" \
    tufts-login:/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/outputs/exp_*/ \
    outputs/experiments/
```

验证完整性（每个实验应该有 7 个 response_pairs JSON）：
```bash
cd outputs/experiments
for d in exp_*; do
    n=$(ls $d/results/response_pairs_*.json 2>/dev/null | wc -l)
    echo "$d: $n response_pairs"
done
```

---

## Running the Judge Locally (no GPU)

LLM-as-Judge 完全在本地跑（纯 API 调用），不需要 HPC，不需要 GPU。

### 准备 API key

在项目根创建 `.env`（已经被 `.gitignore` 排除）：
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...   # 可选
```

### 单文件评估

```bash
python -m evaluate.judge \
    --pairs-file outputs/experiments/exp_url_2m/results/response_pairs_ORIGIN_dog.json \
    --judges gpt-4o-mini
# 输出: outputs/experiments/exp_url_2m/results/judge_results_ORIGIN_dog.json
```

### 批量所有实验

```bash
bash scripts/judge_all.sh --judges gpt-4o-mini
```

会扫描 `outputs/experiments/exp_*/results/response_pairs_*.json`，对每个未评估过的文件跑 judge，输出同目录的 `judge_results_*.json`。已评估的会自动跳过。

**成本估算**（GPT-4o-mini）：
- 21 实验 × 7 图 × 3 VLM × 15 Q ≈ **6,615 API 调用**
- 每次 ~500 token in + ~150 token out ≈ $0.0001
- 总计 **~$3-5 美元**

---

## Troubleshooting

| 症状 | 原因 | 解决 |
|---|---|---|
| `Bus error (core dumped)` 在加载 LLaVA | CPU RAM 不足 | sbatch `--mem` 提到 80G/96G |
| `NFS Stale file handle` | 集群文件系统抽风 | 重新提交任务 |
| `ModuleNotFoundError: deepseek_vl` | pip 装的是空包 | 从 GitHub 源码装：`pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git` |
| `DynamicCache.from_legacy_cache` AttributeError | Phi-3.5 与新版 transformers 不兼容 | 用 Qwen2-VL-2B 替代 |
| `LLaVA image token mismatch` | LLaVA wrapper 与新版 transformers 不兼容 | 暂时从 ATTACK_TARGETS 移除 LLaVA |
| `AutoModelForVision2Seq` 缺失 | transformers 版本太老 | `evaluate/pairs.py` 已用 try/except 跳过 caption 评估 |
| `decoder weights not found: data/checkpoints/coco_bi.pt` | 权重没下载 | 跑 `python data/preparation/models/download_decoder_weights.py` |
| Universal 训练超 SLURM 时限 | num_steps 设太高或模型太大 | 降到 `--num-steps 1500` 或换 2m 配置 |
| HF cache 重复下载 | 没设 `HF_HOME` 环境变量 | sbatch 脚本会 `export HF_HOME=...`，本地手动跑也要设 |
