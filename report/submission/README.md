# VisInject — EE141 Final Submission

Three files, one folder.

| File | What it is |
|---|---|
| `VisInject_Final_Report.pdf` | Written report, 18 pages, native LaTeX. |
| `VisInject_Slides.pptx`      | Presentation deck, 25 slides, 16:9 widescreen. |
| `VisInject_Code.zip`         | Core source code at submission time. Keeps `src/`, `attack/`, `models/`, `evaluate/`, `scripts/`, `demo/`, `data/`, `outputs/` (curated case studies only — model weights and the HuggingFace cache are excluded by `.gitignore`), plus `README.md`, `LICENSE`, `.gitignore`, and `.env.example`. The agent guide (`CLAUDE.md`), the technical-docs folder (`docs/`), the Chinese README, and the report subtree itself are NOT included — they are workflow artefacts, not part of the code deliverable. |

## Project links

- Code: <https://github.com/jeffliulab/vis-inject>
- Dataset: <https://huggingface.co/datasets/jeffliulab/visinject>
- Demo: <https://huggingface.co/spaces/jeffliulab/visinject>

## How to rebuild these artefacts from the code

From the project root:

```bash
# Rebuild the slide deck
python report/scripts/build_slides.py
# → writes report/slides/VisInject_final.pptx

# Rebuild the PDF report
cd report/pdf && make
# → writes report/pdf/main.pdf
```

Build prerequisites:

- TeX Live or MacTeX (`pdflatex`, `latexmk`, `bibtex`)
- Python 3.10+ with `python-pptx`, `matplotlib`, `Pillow` (see `report/scripts/requirements.txt`)

## Author

Pang Liu — `pang.liu@tufts.edu` (or `jeff.pang.liu@gmail.com` after graduation)
