"""
VisInject Stage 3: Evaluation module.

Two-step pipeline:
  Stage 3a (HPC, GPU)  — pairs.py: query target VLMs on (clean, adv) image pairs
                          and save responses as JSON.
  Stage 3b (local, API) — judge.py: GPT-4o / DeepSeek scores each pair for
                          adversarial injection success.

Public API (re-exported here for backward compatibility with existing
imports in pipeline.py and demo/web_demo.py):

Note: pairs.py imports are lazy to allow `python -m evaluate.judge` to run
without GPU/PyTorch dependencies.
"""


def __getattr__(name):
    _pairs_exports = {
        "generate_response_pairs",
        "run_evaluation",
        "evaluate_asr",
        "evaluate_image_quality",
        "evaluate_clip",
        "evaluate_captions",
    }
    if name in _pairs_exports:
        from . import pairs
        return getattr(pairs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "generate_response_pairs",
    "run_evaluation",
    "evaluate_asr",
    "evaluate_image_quality",
    "evaluate_clip",
    "evaluate_captions",
]
