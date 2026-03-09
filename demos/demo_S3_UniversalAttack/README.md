# demo_S3_UniversalAttack

Reproduction of **"Universal Adversarial Attack on Aligned Multimodal LLMs"** (arXiv 2502.07987, 2025).

- Paper: https://arxiv.org/abs/2502.07987

## Overview

This method crafts a **single universal adversarial image** that, when shown alongside any text prompt, forces an aligned multimodal LLM to respond with a target phrase (e.g., "Sure, here it is"), effectively overriding safety alignment. The attack works by directly optimizing pixel values through gradient backpropagation across the entire MLLM.

No neural network is trained -- the method directly optimizes the pixels of one synthetic image.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Optimization Loop               │
                    │                                         │
  z0 (grayscale) ──┤──> z = z0 + γ·tanh(z1) + ε             │
  z1 (trainable) ──┤                │                         │
  ε ~ N(0,σ²) ────┤          clip [0,1]                      │
                    │                │                         │
                    │         ┌──────v──────┐                  │
                    │         │Vision Encoder│ ◄── grad flows  │
                    │         │  (frozen)    │     through     │
                    │         └──────┬──────┘                  │
                    │                │                         │
                    │         ┌──────v──────┐                  │
  Random question ──┤────────>│   LLM Head  │                  │
  from dataset      │         │  (frozen)   │                  │
                    │         └──────┬──────┘                  │
                    │                │                         │
                    │     Masked CE Loss                       │
                    │     (target tokens only)                 │
                    │           │                              │
                    │     backprop to z1                       │
                    └───────────┘─────────────────────────────┘
```

## Mathematical Principles

### Image Parameterization

The adversarial image z is parameterized as:

$$z = \text{clamp}(z_0 + \gamma \cdot \tanh(z_1) + \varepsilon, \; 0, \; 1)$$


$$z = \text{clip}(z_0 + g(z_1) + \epsilon, \ 0, \ 1)$$

where:
- $z_0$: grayscale base image (all pixels = 0.5)
- $z_1$: trainable tensor (same shape as image)
- $g(z_1) = \gamma \cdot \tanh(z_1)$: bounded perturbation
- $\gamma = 0.1$ (single-model) or $0.5$ (multi-model)
- $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$: quantization robustness noise

### Loss Function

Masked cross-entropy loss on target answer tokens only:

$$\min_{z_1} \; \mathcal{L} = -\sum_{t \in \text{target}} \log P(y_t \mid y_{<t}, \; q, \; z)$$


$$\mathcal{L}_{LLM}(y | x, z) = -\sum_{t \in \text{target}} \log P(y_t | y_{<t}, x, z)$$

where $x$ is a random question and $y$ is the target answer ("Sure, here it is").

### Multi-Model Universality

For joint optimization across M models:

$$\mathcal{L} = \sum_{i=1}^{M} \mathcal{L}_{LLM_i}(y | x, z)$$

### Quantization Robustness

At each step, $\sigma$ is calibrated to match int8 quantization error:

$$\sigma = \text{std}(z_{\text{float}} - \lfloor z_{\text{float}} \cdot 255 \rceil / 255)$$

### Enhancement Variants

- **Multi-Answer (MA)**: Target randomly sampled from a pool of affirmative phrases
- **Gaussian Blur**: Smooths perturbation to reduce high-frequency artifacts
- **Localization**: Random crop + resize to inject perturbation into sub-regions

## Training Pipeline

### Step 1: Single-model attack (~30 min on 1× H200)

```bash
python attack.py --target-models qwen2_5_vl_3b --num-steps 2000

# Or via SLURM:
sbatch hpc_train.sh attack
```

### Step 2: Multi-model attack (optional, ~1-2 hours)

```bash
python attack.py \
    --target-models qwen2_5_vl_3b qwen2_vl_2b \
    --num-steps 3000 --multi-answer --quant-robustness

# Or via SLURM:
sbatch hpc_train.sh attack-multi
```

### Step 3: Evaluate

```bash
python evaluate.py \
    --image outputs/universal_final.png \
    --target-models qwen2_5_vl_3b
```

### Quick demo (500 steps)

```bash
python demo.py --num-steps 500
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.1 / 0.5 | Perturbation scale (auto: single/multi) |
| `lr` | 1e-2 | AdamW learning rate (pixel optimization) |
| `num_steps` | 2000 | Optimization steps |
| `target_phrase` | "Sure, here it is" | Target response prefix |
| `image_size` | 448×448 | Adversarial image resolution |
| `quant_robustness` | True | Add quantization noise |
| `gaussian_blur` | False | Smooth perturbation |
| `multi_answer` | False | Random target from pool |
| `localize` | False | Random crop attack |

## Switching Single-model vs Multi-model

All controlled by the `target_models` config or CLI argument:

```bash
# Single-model (gamma auto = 0.1)
python attack.py --target-models qwen2_5_vl_3b

# Multi-model (gamma auto = 0.5)
python attack.py --target-models qwen2_5_vl_3b phi_3_5_vision llava_1_5_7b
```

## File Structure

```
demo_S3_UniversalAttack/
├── config.py            # All configuration parameters
├── requirements.txt     # Python dependencies
├── models/
│   ├── __init__.py
│   ├── mllm_wrapper.py  # Abstract MLLM wrapper interface
│   └── qwen_wrapper.py  # Qwen2-VL / Qwen2.5-VL implementation
├── dataset.py           # 100 safe + 50 adversarial training questions
├── attack.py            # Core pixel optimization loop
├── evaluate.py          # Attack success rate evaluation
├── demo.py              # Quick demo script
└── hpc_train.sh         # SLURM job script (attack/attack-multi/evaluate/demo)
```

## Adding New Target Models

1. Add model to `model_registry.py` (root level)
2. Create a wrapper in `models/` inheriting from `MLLMWrapper`
3. Register the wrapper in `attack.py:get_wrapper_for_model()`
4. Add model key to `target_models` in config or CLI

Example for Phi-3.5-Vision:
```python
# models/phi_wrapper.py
class PhiWrapper(MLLMWrapper):
    def load(self):
        # Load Phi-3.5-vision-instruct
        ...
    def compute_masked_ce_loss(self, image, question, target_answer):
        # Build inputs and compute loss
        ...
```

## Differences from Original Paper

| Aspect | Original | This Reproduction |
|--------|----------|-------------------|
| Target models | Phi/Llama/Qwen2-VL-2B/Llava | Qwen2.5-VL-3B (extensible) |
| Code | Not released | Implemented from paper Section 3 |
| Judge model | Gemma-3-4B-it | Simple prefix matching (extensible) |
| Multi-model training | All models loaded simultaneously | Sequential loading to save VRAM |

## Expected Results

Based on the paper (Table 2, SafeBench ASR %):

| Model | Single-Answer | Multi-Answer |
|-------|--------------|--------------|
| Phi-3.5 | 15.0% | 81.3% |
| Llama-3.2-11B | 15.0% | 70.4% |
| Qwen2-VL-2B | 21.4% | 79.3% |
| Llava-1.5-7B | 44.0% | 46.0% |

Multi-answer variant significantly improves attack success rates.

## References

```bibtex
@article{rahmatullaev2025universal,
  title={Universal Adversarial Attack on Aligned Multimodal LLMs},
  author={Rahmatullaev, Temurbek and Druzhinina, Polina and Kurdiukov, Nikita
          and Mikhalchuk, Matvey and Kuznetsov, Andrey and Razzhigaev, Anton},
  journal={arXiv preprint arXiv:2502.07987},
  year={2025}
}
```
