# LAION-Art Dataset for AnyAttack Pre-training

## Overview

LAION-Art is a curated subset of LAION-5B containing **~8 million** high-quality images
(aesthetic score > 8, watermark probability < 0.8, unsafe probability < 0.5).

This dataset is used for self-supervised pre-training of the AnyAttack Decoder network
(see `demo_S2_AnyAttack`).

**Paper**: [AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models](https://arxiv.org/abs/2410.05346) (CVPR 2025)

## Storage Requirements

| Item | Size |
|------|------|
| Parquet metadata | ~500 MB |
| Images (224x224 WebDataset) | ~80-150 GB |
| **Total** | **~80-150 GB** |

## Download Instructions

### Step 1: Submit the download job

```bash
# On the HPC login node:
sbatch download_laion_art.sh
```

### Step 2: Monitor progress

```bash
# Check job status
squeue -u $USER

# Watch the log file
tail -f /cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/logs/download_<JOBID>.out
```

### Step 3: If the job times out, resubmit

`img2dataset` is **resumable by design** -- it tracks completed shards and skips them
on rerun. Simply resubmit the same script:

```bash
sbatch download_laion_art.sh
```

Repeat until the download completes. You can verify completeness with:

```bash
python verify_dataset.py
```

### Step 4: Verify the dataset

```bash
python verify_dataset.py --data-dir /cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset
```

## Output Structure

```
/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/
  metadata/
    laion-art.parquet        # URL + caption metadata
  webdataset/
    00000.tar                # Each tar contains ~10K image-text pairs
    00001.tar
    ...
    00xxx.tar
  logs/
    download_<jobid>.out     # SLURM output logs
    download_<jobid>.err
```

Each `.tar` shard contains files like:
```
000000000.jpg    # Image (224x224, JPEG quality 95)
000000000.txt    # Caption text
000000000.json   # Metadata (similarity, aesthetic score, etc.)
```

## Usage in AnyAttack Pre-training

The WebDataset tar files are loaded directly by `demo_S2_AnyAttack/pretrain.py`
using the `webdataset` library for efficient streaming:

```python
import webdataset as wds

dataset = (
    wds.WebDataset(tar_files, resampled=True, shardshuffle=True)
    .shuffle(5000)
    .decode("pil")
    .to_tuple("jpg", "txt")
    .map(transform_fn)
    .batched(batch_size)
)
```

## Troubleshooting

**Q: Many URLs return 404 errors?**
A: This is expected. LAION datasets are crawled from the web and URLs expire over time.
Typically 20-40% of URLs are dead. img2dataset handles this gracefully and logs failures.
You should still get 5-6 million usable images, which is sufficient for pre-training.

**Q: The download is very slow?**
A: Ensure the HPC compute nodes have internet access. If only login nodes can reach
the internet, you may need to run the download interactively on a login node (use `screen`
or `tmux` to prevent session termination). Adjust `--processes_count` and `--thread_count`
based on your network bandwidth.

**Q: "Permission denied" or "Disk quota exceeded"?**
A: Check your storage quota with `quota -s` or equivalent. LAION-Art needs 80-150 GB.
Consider using `$SCRATCH` if your home directory quota is limited.
