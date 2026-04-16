# VisInject Full Demo (local, GPU required)

A bilingual (EN/CN) Gradio web UI that drives the **full** VisInject
pipeline end-to-end: Stage 1 (multi-model PGD universal optimization) +
Stage 2 (AnyAttack Decoder fusion), plus an ASR evaluation tab. This is
the same app that used to live at `demo/web_demo.py`, now moved one
level deeper to make room for the Space-compatible stripped version
under `../space_demo/`.

## Hardware requirements

- **GPU with at least ~11 GB VRAM.** Stage 1 loads multiple VLMs
  simultaneously for joint optimization. With the default
  `ATTACK_TARGETS` (`qwen2_5_vl_3b` + `blip2_opt_2_7b`) VRAM usage is
  around 11 GB; enabling more targets scales this up linearly.
- **CUDA** (the app auto-falls back to CPU, but Stage 1 is impractical
  on CPU — expect tens of minutes to hours per run).
- `data/checkpoints/coco_bi.pt` for Stage 2 (download with
  `python data/preparation/models/download_decoder_weights.py`).

## Usage

Run from the project root:

```bash
python demo/full_demo/web_demo.py
python demo/full_demo/web_demo.py --port 7861 --share
python demo/full_demo/web_demo.py --lang cn
```

Options:

- `--port <int>` — Gradio server port (default `7860`)
- `--share` — publish a temporary public URL via Gradio sharing
- `--lang {en,cn}` — choose UI language

## Modes

- **Quick mode** — reuse the most recent cached universal image in
  `outputs/universal/` (skips Stage 1, returns in seconds)
- **Full mode** — re-run Stage 1 PGD from scratch with the chosen target
  phrase / target models / number of steps

## Tabs

- **Generate** — pick a clean image, a target phrase, a set of target
  VLMs, then run Stage 1 + Stage 2 and inspect PSNR / noise L-inf
- **Evaluate** — upload an adversarial image and run per-VLM ASR
  measurement (requires the chosen VLM to be loadable on the local GPU)

## Deployability

This demo is **not** deployable to a Hugging Face Space free tier:
Stage 1 requires GPU memory and model downloads that exceed the free
tier budget. For a Space-compatible version that reuses precomputed
universal images and runs Stage 2 only, see
[`../space_demo/README.md`](../space_demo/README.md).
