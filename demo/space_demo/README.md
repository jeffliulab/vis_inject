# VisInject Space Demo (Stage 2 only)

> **🚀 Live demo deployed at**: https://huggingface.co/spaces/jeffliulab/visinject
>
> The deployed Space uses a slightly different `app.py` that loads the
> universal images from [`jeffliulab/visinject` (HF Dataset)](https://huggingface.co/datasets/jeffliulab/visinject)
> and the decoder from [`jiamingzz/anyattack`](https://huggingface.co/jiamingzz/anyattack)
> at runtime, instead of expecting them on local disk. The Space repo
> source lives at https://huggingface.co/spaces/jeffliulab/visinject/tree/main.

A stripped-down, CPU-only Gradio app that demonstrates the VisInject
**Stage 2 fusion** step (AnyAttack Decoder) using precomputed universal
adversarial images from the 21-experiment matrix. It is small enough to
run on a Hugging Face Space free tier: Stage 1 (multi-model PGD
optimization) is **not** executed here — the app simply picks one of the
7 precomputed `universal_*.png` files and fuses it into the image you
upload.

The `app.py` in **this** directory is the **local-mode** variant: it
expects `data/checkpoints/coco_bi.pt` and the universal images on local disk
(useful when developing the Space against your own changes before
pushing). The deployed version is in the Space repo, not this directory.

## Limitations

- **CPU only.** Device is hard-coded to `cpu`. The Stage 2 pipeline
  (CLIP ViT-B/32 encoder + AnyAttack Decoder) runs comfortably on a
  free-tier Space, but each generation still takes ~2–5 s.
- **No real-time PGD.** Stage 1 requires loading multiple VLMs in GPU
  memory and takes minutes per run. See `../full_demo/` for that version.
- **No VLM verification.** The app does not load any VLM, so ASR cannot
  be measured here. Download the adv image and paste it into ChatGPT (or
  any other VLM) to verify the injection manually.

## Local usage

From the project root:

```bash
python demo/space_demo/app.py
```

Then open http://localhost:7860.

Prerequisites:

1. `data/checkpoints/coco_bi.pt` must exist. Download it with:
   ```bash
   python data/preparation/models/download_decoder_weights.py
   ```
2. `outputs/experiments/exp_<tag>_2m/universal/universal_*.png` must
   exist for each of the 7 attack prompts. These are produced by the
   Stage 1 pipeline (`scripts/run_experiments.sh`).

## HF Space deployment notes

The code in this directory is ready to deploy, but you will need to
upload a few binary assets alongside it (they are gitignored in the main
repo and therefore not pushed automatically):

- `data/checkpoints/coco_bi.pt` (~320 MB) — the AnyAttack Decoder weights
- `outputs/experiments/exp_<tag>_2m/universal/universal_*.png` for
  `<tag>` in `{card, url, apple, email, news, ad, obey}` (7 files, each
  ~600 KB)

Suggested Space configuration:

- **SDK:** `gradio`
- **Hardware:** `CPU Basic` (2 vCPU / 16 GB — free tier)
- **App file:** `app.py` (mirror this directory at the Space repo root,
  plus the two asset paths above)

You can either commit the 7 universal images + decoder weight directly
to the Space repo (Git LFS recommended for `coco_bi.pt`), or load them
from an HF Dataset reference at runtime.

## Relationship to the full pipeline

| Stage | Runs here? |
| --- | --- |
| Stage 1 — universal PGD optimization (needs VLMs + GPU) | No (uses precomputed) |
| Stage 2 — AnyAttack Decoder fusion (CPU-friendly) | Yes |
| Stage 3 — VLM-based ASR evaluation | No |

For the full interactive pipeline (Stage 1 + Stage 2) see
[`../full_demo/README.md`](../full_demo/README.md).
