# VisInject Demos

This directory contains all experimental demos for the VisInject project, progressing from basic single-model PGD attacks to advanced generative adversarial approaches.

## Quick Navigation

| Demo | Target Model | Method | Status |
|------|-------------|--------|--------|
| [Demo_0](#demo_0-clip-cross-modal-embedding-attack) | CLIP ViT-L/14 | Embedding alignment via PGD | Completed |
| [Demo1](#demo1-blip-2-end-to-end-pgd) | BLIP-2 (OPT-2.7B) | End-to-end CE + PGD | Completed |
| [Demo2](#demo2-deepseek-vl-pgd) | DeepSeek-VL 1.3B | End-to-end CE + PGD | Completed |
| [Demo3](#demo3-qwen25-vl-pgd) | Qwen2.5-VL-3B | End-to-end CE + PGD | Completed |
| [Demo_S1](#demo_s1-stegoencoder) | Multi-VLM | U-Net generative encoder | Abandoned |
| [Demo_S2](#demo_s2-anyattack) | Cross-model (CLIP proxy) | Self-supervised Decoder | In Progress |
| [Demo_S2P](#demo_s2p-anyattack-official-weights) | Cross-model (CLIP proxy) | Official pre-trained Decoder | Ready |
| [Demo_S3](#demo_s3-universalattack) | Qwen / multi-model | Direct pixel optimization | Code Complete |

## Research Progression

```
Stage 1: Single-Model PGD            Stage 2: Generative           Stage 3: Final
┌──────────────────────┐     ┌──────────────────────────┐     ┌──────────────┐
│ Demo_0  CLIP ViT     │     │ Demo_S1  StegoEncoder    │     │              │
│ Demo1   BLIP-2       │ ──> │ Demo_S2  AnyAttack       │ ──> │  VisInject   │
│ Demo2   DeepSeek-VL  │     │ Demo_S3  UniversalAttack │     │  S3 + S2     │
│ Demo3   Qwen2.5-VL   │     └──────────────────────────┘     └──────────────┘
└──────────────────────┘
```

---

## Demo_0: CLIP Cross-Modal Embedding Attack

**Directory**: [`demo_0_CLIP_ViT/`](demo_0_CLIP_ViT/)

Uses PGD to align a clean image's CLIP visual embedding with a target text's CLIP text embedding. Operates entirely in the CLIP ViT-L/14 shared embedding space.

**Key idea**: Minimize `1 - cos(E_v(x + delta), E_t(target_text))` under L-infinity constraint.

| Detail | Value |
|--------|-------|
| Vision encoder | CLIP ViT-L/14 |
| Input size | 224 x 224 |
| Perturbation | L-inf, eps = 16/255 |
| VRAM | ~1.6 GB |
| Steps | 500-1000 |

**Results**: CLIP similarity increased from 0.126 to 0.68. PSNR 22-27 dB, SSIM 0.50-0.73.

**Key finding**: Cross-modal alignment works within CLIP, but does **not** transfer to other VLMs (BLIP-2, DeepSeek, Qwen) since they use different vision encoders (EVA-CLIP, SigLIP, etc.).

### Usage

```bash
cd demo_0_CLIP_ViT
python run_attack.py
```

---

## Demo1: BLIP-2 End-to-End PGD

**Directory**: [`demo1_BLIP2/`](demo1_BLIP2/)

First successful end-to-end VLM attack. Gradients flow from the cross-entropy loss through the entire BLIP-2 pipeline (OPT-2.7B -> Q-Former -> EVA-ViT-G) back to input pixels.

```
Image --> EVA-ViT-G --> Q-Former (32 queries) --> Linear Proj --> OPT-2.7B --> CE Loss --> PGD
```

| Detail | Value |
|--------|-------|
| Vision encoder | EVA-ViT-G |
| Bridge | Q-Former (32 learnable query tokens) |
| LLM | OPT-2.7B |
| Input size | 224 x 224 |
| Perturbation | L-inf, eps = 32/255 |
| VRAM | ~6 GB |

**Results**: CE loss 10.7 -> ~0.0. 100% ASR on both direct tensor and PNG round-trip (with QAA).

### Usage

```bash
cd demo1_BLIP2
python simple_demo.py                     # Quick attack
python pgd_attack.py                      # Full attack with metrics
python test_inference.py                  # Verify adversarial image
```

---

## Demo2: DeepSeek-VL PGD

**Directory**: [`demo2_DeepSeekVL_1/`](demo2_DeepSeekVL_1/)

Extends the PGD attack methodology to DeepSeek-VL 1.3B, proving it generalizes across different VLM architectures (SigLIP vs EVA-CLIP, LLaMA vs OPT).

```
Image --> SigLIP-L (576 patches) --> MLP Aligner (1024->2048) --> LLaMA-1.3B --> CE Loss --> PGD
```

| Detail | Value |
|--------|-------|
| Vision encoder | SigLIP-L |
| Bridge | MLP Aligner (2-layer linear) |
| LLM | LLaMA-1.3B (24 layers) |
| Input size | 384 x 384 |
| Perturbation | L-inf, eps = 32/255 |
| VRAM | ~5-8 GB |

**Key technique**: Manual embedding concatenation to ensure full gradient flow through the model.

### Usage

```bash
cd demo2_DeepSeekVL_1
python simple_demo.py
python pgd_attack.py
```

---

## Demo3: Qwen2.5-VL PGD

**Directory**: [`demo3_Qwen_2_5_VL_3B/`](demo3_Qwen_2_5_VL_3B/)

Cleanest and fastest PGD attack implementation. Uses Qwen2.5-VL's native `model.forward(labels=...)` with differentiable `pixel_values`, avoiding manual embedding assembly.

```
Image --> ViT-L (32 layers, 784 patches) --> PatchMerger (2x2, 196 tokens) --> Qwen2.5-3B --> CE Loss --> PGD
```

| Detail | Value |
|--------|-------|
| Vision encoder | ViT-L (32 layers, d=1280) |
| Bridge | PatchMerger (2x2 merge, 784 -> 196 tokens) |
| LLM | Qwen2.5-3B (36 layers, mRoPE) |
| Input size | 392 x 392 |
| Perturbation | L-inf, eps = 32/255 |
| VRAM | ~12 GB |

**Results**: CE loss 10.74 -> 0.00 in ~50 steps. 100% ASR with QAA. Fastest convergence among all demos.

### Usage

```bash
cd demo3_Qwen_2_5_VL_3B
python simple_demo.py
python pgd_attack.py
```

---

## Demo_S1: StegoEncoder

**Directory**: [`demo_S1_Small_Model/`](demo_S1_Small_Model/) | **Status**: Abandoned

First attempt at a generative (amortized) attack model. Trains a U-Net (~55M params) to produce adversarial perturbations conditioned on a target prompt, with DCT mid-frequency constraints to preserve image quality.

```
Clean Image + Prompt --> U-Net --> Raw Noise --> DCT Filter (bands 3-15) --> L-inf Norm --> Gaussian Blur --> Adversarial Image
```

| Detail | Value |
|--------|-------|
| Network | U-Net, 4 scales (64/128/256/512 ch), ~55M params |
| Conditioning | Optional FiLM (text embedding modulates ResBlocks) |
| Frequency constraint | 8x8 block DCT, mid-frequency bands 3-15 |
| Target VLMs | BLIP-2, DeepSeek-VL, Qwen2.5-VL |

**Results after 1500 epochs**: CE loss only dropped from 10.70 to ~10.12. ASR 0%. Estimated ~1400+ epochs per image to converge.

**Why abandoned**: Training is orders of magnitude slower than direct PGD, and the DCT constraint severely limits attack capacity. Replaced by Demo_S2 (AnyAttack) and Demo_S3 (UniversalAttack), which use proven architectures from published papers.

### File Structure

```
demo_S1_Small_Model/
├── config.py                   # Global configuration
├── models/stego_encoder.py     # U-Net with DCT constraints
├── encoders/                   # VLM encoder wrappers (BLIP-2, DeepSeek, Qwen)
├── prompts/                    # Prompt strategies (fixed keyword, style injection, etc.)
├── vlms/                       # VLM inference wrappers
├── training/                   # Trainers (proxy, e2e, RL)
├── losses.py                   # Multi-component loss functions
├── run_demo.py                 # Training entry point
└── test/                       # Unit and integration tests
```

---

## Demo_S2: AnyAttack

**Directory**: [`demo_S2_AnyAttack/`](demo_S2_AnyAttack/) | **Paper**: [CVPR 2025](https://arxiv.org/abs/2410.05346)

Trains a Decoder network (~10M params) that takes a CLIP embedding and generates adversarial noise. The attack transfers across VLMs because it operates through the CLIP surrogate model rather than any specific target VLM.

**Two-phase training**:
1. **Self-supervised pre-training** on LAION-Art (~8M images) with InfoNCE contrastive loss and K-augmentation
2. **Fine-tuning** on COCO with multi-encoder losses (CLIP + EVA + ViT-B/16) for cross-model transferability

```
Target Image --> CLIP ViT-B/32 (frozen) --> 512-dim Embedding --> Decoder --> Noise --> clamp [-eps, eps]
Clean Image + Noise --> Adversarial Image
```

| Detail | Value |
|--------|-------|
| Surrogate encoder | CLIP ViT-B/32 (frozen) |
| Decoder | FC + 4x (ResBlock + EfficientAttention + Upsample), ~10M params |
| Pre-training data | LAION-Art (830K images downloaded so far) |
| Fine-tuning data | MS-COCO |
| Perturbation | L-inf, eps = 16/255 |
| Pre-training time | ~3-5h on 1x H200 |
| Fine-tuning time | ~1-2h on 1x H200 |

See [`demo_S2_AnyAttack/README.md`](demo_S2_AnyAttack/README.md) for full details.

### Quick Start

```bash
cd demo_S2_AnyAttack

# Pre-train on LAION-Art
sbatch hpc_train.sh pretrain

# Fine-tune on COCO
sbatch hpc_train.sh finetune

# Generate adversarial image
python demo.py --decoder-path checkpoints/finetuned.pt \
               --clean-image dog.jpg --target-image cat.jpg
```

---

## Demo_S2P: AnyAttack Official Weights

**Directory**: [`demo_S2P/`](demo_S2P/) | **Status**: Ready

Inference-only demo using AnyAttack's **official pre-trained weights** from HuggingFace (pre-trained on LAION-400M + fine-tuned on COCO). No training required -- runs on a local GPU (RTX 4090).

Uses the same Decoder architecture as Demo_S2, but with the authors' fully trained `coco_bi.pt` checkpoint. Evaluates adversarial transferability against BLIP-2, DeepSeek-VL, and Qwen2.5-VL.

### Quick Start

```bash
cd demo_S2P
python download_weights.py
python demo.py --clean-image ../demo_images/ORIGIN_dog.png \
               --target-image ../demo_images/ORIGIN_cat.png
python evaluate.py --adv-image outputs/adversarial.png \
                   --clean-image ../demo_images/ORIGIN_dog.png \
                   --target-image ../demo_images/ORIGIN_cat.png
```

---

## Demo_S3: UniversalAttack

**Directory**: [`demo_S3_UniversalAttack/`](demo_S3_UniversalAttack/) | **Paper**: [arXiv 2502.07987](https://arxiv.org/abs/2502.07987)

Directly optimizes a single universal image's pixel values so that any MLLM will respond with a target phrase (e.g., "Sure, here it is") regardless of the user's question. No neural network is trained -- only the image pixels are optimized via AdamW.

**Parameterization**: `z = clip(z0 + gamma * tanh(z1) + noise, 0, 1)`, where `z1` is the trainable parameter and `gamma` controls perturbation scale.

```
z0 (gray 0.5) + gamma * tanh(z1) + quantization noise --> VLM (frozen, grad flows) --> Masked CE Loss --> backprop to z1
```

| Detail | Value |
|--------|-------|
| Trainable parameters | Image pixels only (z1 tensor) |
| Optimizer | AdamW, lr = 0.01 |
| Steps | 2000 (single-model) / 3000 (multi-model) |
| Training time | ~30 min (single) / ~1-2h (multi) on 1x H200 |
| gamma | 0.1 (single-model) / 0.5 (multi-model) |
| Quantization robustness | Calibrated Gaussian noise |

Supports both **single-model** and **multi-model** attacks, controlled by `--target-models`. Also supports multi-answer, Gaussian blur, and localization attack variants.

See [`demo_S3_UniversalAttack/README.md`](demo_S3_UniversalAttack/README.md) for full details.

### Quick Start

```bash
cd demo_S3_UniversalAttack

# Single-model attack
python attack.py --target-models qwen2_5_vl_3b --num-steps 2000

# Multi-model attack
python attack.py --target-models qwen2_5_vl_3b phi_3_5_vision --num-steps 3000

# Evaluate
python evaluate.py --image outputs/universal_final.png --target-models qwen2_5_vl_3b
```

---

## Supporting Directories

### `LAION_ART_DATA/`

Scripts for downloading the LAION-Art dataset (~8M images) used for Demo_S2 pre-training.

- `download_laion_art.sh` -- SLURM job script with test/full/resume modes
- `download_images.py` -- Robust multi-threaded image downloader with retry, resume, and comprehensive logging
- `download_parquet_metadata.sh` -- Downloads 128 parquet metadata shards from HuggingFace
- `verify_dataset.py` -- Integrity checker for downloaded WebDataset shards

### `demo_images/`

Source images used across demos (cat, dog, kpop, bill) and generated adversarial examples.

### `demo_screenshots/`

Screenshots and presentation materials demonstrating attack results.

---

## Shared Infrastructure

### Model Registry

All demos use the centralized [`model_registry.py`](../model_registry.py) at the project root. It stores HuggingFace model IDs, image sizes, normalization parameters, and VRAM estimates for 11+ VLMs. To add a new model, add an entry to `REGISTRY` -- no demo code changes needed.

### Common Attack Flow

All PGD-based demos (Demo_0 through Demo3) follow the same pattern:

1. Load the target VLM with gradients enabled
2. Tokenize the target phrase into token IDs
3. Initialize perturbation `delta = 0`
4. For each PGD step:
   - Forward pass: `image + delta` through the VLM
   - Compute masked cross-entropy loss on target tokens
   - Backpropagate to get `grad_delta`
   - Update: `delta = delta - alpha * sign(grad_delta)`
   - Project: `delta = clamp(delta, -eps, eps)`
5. Apply Quantization-Aware Attack (QAA): `image_quant = round(image * 255) / 255`
6. Save adversarial image as PNG

---

## Hardware Requirements

| Demo | Min VRAM | Recommended GPU |
|------|----------|----------------|
| Demo_0 | 2 GB | Any GPU |
| Demo1 | 6 GB | RTX 3060+ |
| Demo2 | 6 GB | RTX 3060+ |
| Demo3 | 12 GB | RTX 3090 / A100 |
| Demo_S1 | 14 GB | RTX 4090 / A100 |
| Demo_S2 (train) | 20 GB | A100 / H100 / H200 |
| Demo_S3 (train) | 12 GB | RTX 3090+ |

All demos have been tested on NVIDIA H200 (80 GB HBM3) via SLURM on the Tufts HPC cluster.

---

## Final VisInject Pipeline

The final system combines S3 and S2:

1. **S3** optimizes a universal adversarial image that encodes a target prompt into visual features
2. The abstract image from S3 is fed through **CLIP ViT-B/32** to get a 512-dim embedding
3. **S2's Decoder** generates bounded noise from that embedding
4. The noise is added to any natural carrier image, producing a natural-looking adversarial image that carries the injected prompt

```
Target Prompt --> [S3 Pixel Optimization] --> Abstract Image --> [CLIP Encode] --> [S2 Decoder] --> Noise
                                                                                                     |
                                                                       Carrier Image + Noise --> VisInject Image
```

See the [project README](../README.md) for the full architecture and timeline.
