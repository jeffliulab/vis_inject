# demo_S2_AnyAttack

Reproduction of **"AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models"** (CVPR 2025).

- Paper: [https://arxiv.org/abs/2410.05346](https://arxiv.org/abs/2410.05346)
- Original code: [https://github.com/jiamingzhang94/AnyAttack](https://github.com/jiamingzhang94/AnyAttack)

## Overview

AnyAttack trains a **Decoder** network that generates adversarial perturbations for **any** input image, making it appear as a specified target image to Vision-Language Models. The attack transfers across multiple VLMs without requiring access to any specific target model during training.

## Architecture

```
Target Image ──> CLIP ViT-B/32 (frozen) ──> Embedding (512-dim)
                                                │
                                                v
                                           Decoder (~28M params)
                                                │
                                                v
                                        Noise (224×224×3)
                                                │
                                         clamp [-ε, ε]
                                                │
Clean Image ──────────────────────────> (+) ──> Adversarial Image
```

### Decoder Architecture

- **Input**: 512-dim CLIP embedding
- **FC**: 512 → 256 × 14 × 14
- **4× (ResBlock + UpBlock)**: 256 → 128 → 64 → 32 → 16
  - Each ResBlock includes EfficientAttention (linear-complexity spatial self-attention)
- **Head**: Conv2d(16 → 3), raw output (clamped externally)
- **Parameters**: ~28M

## Mathematical Principles

### Phase 1: Self-supervised Pre-training (LAION-Art)

The Decoder F learns to generate adversarial noise such that the CLIP embedding of the perturbed image matches the original image's embedding.

**InfoNCE contrastive loss with K-augmentation:**

Given a batch of images {x_i}, extract embeddings e_i = E(x_i), generate noise δ_i = F(e_i), then create K adversarial copies by adding noise to shuffled images:

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\text{sim}(e_i, \bar{e}'*i) / \tau)}{\sum*{j=1}^{B} \exp(\text{sim}(e_i, \bar{e}'_j) / \tau)}$$

where $\bar{e}'*i = \frac{1}{K}\sum*{k=1}^{K} E(x_{\sigma_k(i)} + \delta_i)$ averages K shuffled adversarial embeddings, and τ is an annealing temperature (1.0 → 0.07).

### Phase 2: Fine-tuning (COCO)

Adapt the Decoder with additional auxiliary encoder losses for cross-model transferability:

$$\mathcal{L}*{total} = \mathcal{L}*{CLIP} + \mathcal{L}*{EVA} + \mathcal{L}*{ImageNet}$$

Each loss term uses either BiContrastiveLoss (bidirectional image-text contrastive) or DirectMatchingLoss (cosine similarity maximization).

---

## Training Pipeline

### Step 0: Download LAION-Art Dataset

Download scripts are located in `demos/LAION_ART_DATA/`. All commands should be run on the HPC cluster.

#### 0.1 Download Parquet Metadata (run on login node, requires network)

```bash
cd /path/to/vis_inject/demos/LAION_ART_DATA
bash download_parquet_metadata.sh
```

This should download 128 parquet shards into the metadata directory. Verify:

```bash
ls /cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/metadata/*.parquet | wc -l
# Should output 128
```

#### 0.2 Submit Image Download Job

```bash
cd /path/to/vis_inject/demos/LAION_ART_DATA

# Test download (100 images, verify environment)
sbatch download_laion_art.sh test

# Full download (~8 million images)
sbatch download_laion_art.sh full

# Resume download (continue after timeout or interruption)
sbatch download_laion_art.sh resume
```

> **Note**: HPC jobs have time limits (e.g., 24 hours). When a job times out, submit a new job with `resume` mode to continue from where it left off.

#### 0.3 Check Download Progress

```bash
# Check if the job is running
squeue -u $USER

# Number of downloaded tar shards (each shard contains thousands of images)
ls /cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset/*.tar 2>/dev/null | wc -l

# Total size of downloaded data
du -sh /cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset/

# View resume state file (contains completed parquet shard indices)
cat /cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset/.download_state.json

# View the latest download job log (DOWNLOAD REPORT at the end)
# Use this command to check percentage of dataset downloading
ls -t /cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/logs/download_*.out | head -1 | xargs tail -50
```

#### 0.4 Verify Data Integrity

```bash
cd /path/to/vis_inject/demos/LAION_ART_DATA

# Quick check: verify tar file structure for all shards
python verify_dataset.py --data-dir /cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset

# Deep check: decode every image to verify readability (slower)
python verify_dataset.py --data-dir /cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset --check-images
```

#### 0.5 Check Disk Space

```bash
df -h /cluster/tufts/c26sp1ee0141/pliu07/
du -sh /cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/
du -sh /cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/*/
```

---

### Step 1: Pre-train on LAION-Art

#### Submit via SLURM (recommended)

```bash
cd /path/to/vis_inject/demos/demo_S2_AnyAttack

sbatch hpc_train.sh pretrain

# Monitor training logs
tail -f logs/slurm_<JOBID>.out
```

Default parameters for pretrain mode in `hpc_train.sh`:


| Parameter      | Value                | Description                                                       |
| -------------- | -------------------- | ----------------------------------------------------------------- |
| `--tar-dir`    | LAION_ART/webdataset | Directory containing tar shards                                   |
| `--batch-size` | 600                  | Images per batch                                                  |
| `--epochs`     | 5                    | Training epochs                                                   |
| `--max-shards` | 120                  | Only use the first 120 shards (for partially downloaded datasets) |


#### Specifying Shard Range

If the dataset is still downloading, use `--max-shards` to limit training to the first N complete shards, avoiding reads from incomplete tar files:

```bash
# Option 1: Edit the --max-shards value in hpc_train.sh
# Option 2: Run manually (without sbatch)
python pretrain.py \
    --tar-dir /path/to/LAION_ART/webdataset \
    --batch-size 600 \
    --epochs 5 \
    --max-shards 100   # Only use the first 100 shards
```

Once the full dataset is downloaded, remove the `--max-shards` line from `hpc_train.sh` to train on all shards.

The log will confirm the shard count at startup:

```
Using 100 / 523 shards
```

> **Fault tolerance**: Training will not crash even if some tar shards are corrupted or incomplete. WebDataset is configured with a `warn_and_continue` handler that prints a warning and skips damaged shards automatically.

#### Resuming Training

Checkpoints are saved automatically every 500 batches and at the end of each epoch to `checkpoints/pre-trained.pt`. Each checkpoint contains model weights, optimizer state, learning rate scheduler state, epoch, and global_step.

If training is interrupted (HPC job timeout, node failure, etc.), resume from the last checkpoint with `--checkpoint`:

```bash
python pretrain.py \
    --tar-dir /path/to/LAION_ART/webdataset \
    --checkpoint checkpoints/pre-trained.pt \
    --max-shards 100
```

The log will show:

```
Loading checkpoint: checkpoints/pre-trained.pt
Resumed from epoch 2, step 15000
```

#### Expected Training Time

- 1× NVIDIA H200, batch_size=600: ~3-5 hours (full dataset, 5 epochs)
- Proportionally shorter with fewer shards

---

### Step 2: Fine-tune on COCO

```bash
# Via SLURM
sbatch hpc_train.sh finetune

# Or manually
python finetune.py \
    --pretrain-checkpoint checkpoints/pre-trained.pt \
    --use-auxiliary \
    --epochs 20
```

Fine-tuning uses BiContrastiveLoss across three encoders (CLIP + EVA02-Large + ViT-B/16) to improve cross-model transferability of the generated noise.

Expected time: ~1-2 hours (1× H200).

---

### Step 3: Generate Adversarial Images

```bash
# Via SLURM
sbatch hpc_train.sh generate

# Single-pair demo
python demo.py \
    --decoder-path checkpoints/finetuned.pt \
    --clean-image dog.jpg \
    --target-image cat.jpg

# Batch generation
python generate_adv.py \
    --decoder-path checkpoints/finetuned.pt \
    --clean-dir data/clean \
    --target-dir data/target
```

---

### Step 4: Evaluate

```bash
# Via SLURM
sbatch hpc_train.sh evaluate

# Or manually
python evaluate.py \
    --adv-image adversarial.png \
    --target-image cat.jpg \
    --clean-image dog.jpg \
    --target-vlms qwen2_5_vl_3b blip2_opt_2_7b
```

---

## SLURM Quick Reference

`hpc_train.sh` supports four modes, selected by the first argument:

```bash
sbatch hpc_train.sh pretrain    # Pre-train on LAION-Art
sbatch hpc_train.sh finetune    # Fine-tune on COCO
sbatch hpc_train.sh generate    # Generate adversarial images
sbatch hpc_train.sh evaluate    # Evaluate against VLMs
```

Job management:

```bash
squeue -u $USER                    # List jobs (PD=pending, R=running)
scontrol show job <JOBID>          # Job details
scancel <JOBID>                    # Cancel a job
tail -f logs/slurm_<JOBID>.out     # Stream training logs
tail -f logs/slurm_<JOBID>.err     # Stream error output
```

SLURM resource allocation (configured in `hpc_train.sh`):


| Resource   | Value   | Flag                 |
| ---------- | ------- | -------------------- |
| GPU        | 1× H200 | `--gres=gpu:h200:1`  |
| CPU        | 8 cores | `-c 8`               |
| Memory     | 64 GB   | `--mem=64G`          |
| Time limit | 2 days  | `--time=02-00:00:00` |


---

## Key Parameters


| Parameter               | Default           | Description                             |
| ----------------------- | ----------------- | --------------------------------------- |
| `eps`                   | 16/255            | L∞ perturbation budget                  |
| `clip_model`            | ViT-B/32          | CLIP surrogate encoder                  |
| `batch_size` (pretrain) | 600               | Pre-training batch size                 |
| `batch_size` (finetune) | 100               | Fine-tuning batch size                  |
| `chunk`                 | 5                 | K-augmentation copies                   |
| `lr`                    | 1e-4              | Learning rate (both phases)             |
| `max_shards`            | None (all)        | Limit number of tar shards for training |
| `checkpoint_every`      | 500               | Save checkpoint every N batches         |
| `criterion`             | BiContrastiveLoss | Fine-tuning loss function               |


## File Structure

```
demo_S2_AnyAttack/
├── config.py            # All configuration parameters
├── requirements.txt     # Python dependencies
├── hpc_train.sh         # SLURM job script (pretrain/finetune/generate/evaluate)
├── models/
│   ├── __init__.py
│   ├── clip_encoder.py  # CLIP ViT-B/32 wrapper (via open_clip)
│   └── decoder.py       # Decoder with ResBlock + EfficientAttention
├── losses.py            # DynamicInfoNCELoss, BiContrastiveLoss, DirectMatchingLoss
├── dataset.py           # LAION-Art WebDataset, COCO, ImageFolder loaders
├── pretrain.py          # Self-supervised pre-training on LAION-Art
├── finetune.py          # Fine-tuning on COCO with auxiliary encoders
├── generate_adv.py      # Batch adversarial image generation
├── evaluate.py          # Evaluation against target VLMs
├── demo.py              # Quick single-pair demo
├── checkpoints/         # Saved model weights (gitignored)
└── logs/                # SLURM output logs (gitignored)
```

## Differences from Original Paper


| Aspect            | Original                          | This Reproduction                   |
| ----------------- | --------------------------------- | ----------------------------------- |
| Pre-training data | LAION-400M (400M images)          | LAION-Art (8M images)               |
| Pre-training GPUs | 3-4× A100                         | 1× H200                             |
| Fine-tuning GPUs  | 2× GPU (DDP)                      | 1× H200                             |
| CLIP loading      | Bundled OpenAI CLIP               | open_clip library                   |
| Evaluation VLMs   | BLIP/BLIP2/InstructBLIP/MiniGPT-4 | Qwen2.5-VL-3B, BLIP-2, + extensible |


## Troubleshooting

### `tarfile.ReadError: unexpected end of data` during training

The dataset is not fully downloaded and some tar files are incomplete. Solutions:

1. Use `--max-shards N` to limit training to the first N complete shards
2. The code is already configured with `warn_and_continue` fault tolerance, so isolated corrupted shards will be skipped automatically

### How to resume after training interruption

```bash
python pretrain.py --checkpoint checkpoints/pre-trained.pt
```

The checkpoint contains epoch, global_step, model weights, optimizer state, and scheduler state. Training resumes from the exact point of interruption.

### How to check how many shards are being used

Look for this line in the training log at startup:

```
Using 100 / 523 shards
```

The first number is the actual shards in use; the second is the total number of shards in the directory.

### SLURM job timeout

The default time limit is 2 days. To extend it, modify the `--time` parameter in `hpc_train.sh`. Alternatively, leverage the checkpoint mechanism: after a timeout, resubmit the job with `--checkpoint` to resume training.

## References

```bibtex
@inproceedings{zhang2025anyattack,
  title={Anyattack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models},
  author={Zhang, Jiaming and Ye, Junhong and Ma, Xingjun and Li, Yige and Yang, Yunfan and Chen, Yunhao and Sang, Jitao and Yeung, Dit-Yan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2025}
}
```

