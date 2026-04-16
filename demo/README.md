# `demo/` — Gradio demos

Two separate Gradio apps are provided, differing in what part of the
VisInject pipeline they execute and what hardware they require.

```
demo/
├── README.md          # (this file)
├── space_demo/        # Stripped, CPU-only, HF Space compatible
│   ├── app.py         # Gradio app — Stage 2 fusion only
│   ├── requirements.txt
│   └── README.md
└── full_demo/         # Original full-pipeline demo (local, GPU required)
    ├── web_demo.py    # Gradio app — Stage 1 + Stage 2 + evaluation
    └── README.md
```

## Which one do I want?

| Feature                        | `space_demo`             | `full_demo`           |
| ------------------------------ | ------------------------ | --------------------- |
| Stage 1 (PGD training)         | No (uses precomputed)    | Yes                   |
| Stage 2 (AnyAttack fusion)     | Yes                      | Yes                   |
| VLM-based ASR evaluation       | No                       | No (always offline)   |
| Hardware                       | CPU, 2 vCPU / 16 GB      | GPU, 11+ GB VRAM      |
| HF Space compatible            | Yes                      | No                    |
| Typical latency per generation | ~2–5 s                   | minutes (full mode)   |

- Pick [`space_demo/`](space_demo/README.md) if you just want to try
  the VisInject attack on an image you upload, or if you plan to host
  the demo on Hugging Face Spaces. It reuses the 7 precomputed
  universal adversarial images from the 21-experiment matrix
  (`outputs/experiments/exp_<tag>_2m/universal/`).
- Pick [`full_demo/`](full_demo/README.md) if you want to re-train
  universal images from scratch against arbitrary target phrases and
  arbitrary VLMs. Requires a local GPU.

## `space_demo` Tabs

The Space demo (`space_demo/app.py`) has two tabs:

**Tab 1 — Try VisInject**: The interactive fusion tool. Upload any
image, pick a target phrase and model config, and the app generates an
adversarial image in real time (Stage 2 only, CPU).

**Tab 2 — Injection Cases**: A gallery of 10 confirmed injection cases
from the experiment matrix. For each case:
- Clean vs adversarial image shown side-by-side
- VLM response comparison (clean response vs adversarial response)
- Cases are categorized by injection strength:
  - **Confirmed** (2 cases) — target phrase clearly appears in adversarial response
  - **Partial** (3 cases) — target phrase partially or indirectly surfaces
  - **Weak** (5 cases) — subtle behavioral shift, target not literally present

Case data is loaded from `outputs/succeed_injection_examples/manifest.json`.

## Relation to the batch experiments

The production experiments are still driven by
[`scripts/run_experiments.sh`](../scripts/run_experiments.sh) on the
HPC. Neither demo writes to `outputs/experiments/`; both are meant for
exploratory / qualitative use only.
