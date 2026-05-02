"""
Cohen's κ — Calibration agreement
==================================
Compares Claude Opus 4.7 (or any human-proxy) labels against DeepSeek
LLM labels on the same set of pairs. Reports per-axis κ and confusion
matrices.

Usage:
  python -m evaluate.calibrate \\
      --human  data/calibration/claude_labels.json \\
      --llm    data/calibration/deepseek_labels.json \\
      --output data/calibration/agreement_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


_INFLUENCE_LEVELS = ["none", "slight", "substantial", "complete"]
_INJECTION_LEVELS = ["none", "weak", "partial", "confirmed"]


def cohen_kappa(labels_a, labels_b, classes):
    """Unweighted (categorical) Cohen's κ. Treats every disagreement as
    equally bad — appropriate for nominal categories, slightly harsh for
    4-tier ordinal scales (where "slight" vs "substantial" is one step
    off but counted as full disagreement)."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Length mismatch")
    n = len(labels_a)
    if n == 0:
        return float("nan")
    agreed = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agreed / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    p_e = sum(
        (counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n) for c in classes
    )
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def weighted_kappa(labels_a, labels_b, classes, weighting="linear"):
    """Weighted κ for ordinal classes. Disagreements are weighted by
    distance: a 1-step disagreement (slight↔substantial) counts less
    than a 3-step disagreement (none↔complete).

    weighting:
      "linear":    w = |i - j| / (k - 1)
      "quadratic": w = ((i - j) / (k - 1))**2  (Fleiss-Cohen, harsher)
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Length mismatch")
    n = len(labels_a)
    if n == 0:
        return float("nan")
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}

    def w(i, j):
        diff = abs(i - j)
        if weighting == "quadratic":
            return (diff / (k - 1)) ** 2
        return diff / (k - 1)

    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)

    # Observed disagreement
    obs_dis = sum(
        w(idx[a], idx[b])
        for a, b in zip(labels_a, labels_b)
        if a in idx and b in idx
    ) / n

    # Expected disagreement (under independence)
    exp_dis = sum(
        w(idx[ca], idx[cb])
        * (counts_a.get(ca, 0) / n)
        * (counts_b.get(cb, 0) / n)
        for ca in classes for cb in classes
    )

    if exp_dis == 0:
        return 1.0 if obs_dis == 0 else 0.0
    return 1 - (obs_dis / exp_dis)


def confusion_matrix(labels_a, labels_b, classes):
    """Returns a list-of-lists: rows = a, cols = b."""
    idx = {c: i for i, c in enumerate(classes)}
    M = [[0 for _ in classes] for _ in classes]
    for a, b in zip(labels_a, labels_b):
        if a in idx and b in idx:
            M[idx[a]][idx[b]] += 1
    return M


def _load_labels(path):
    """Returns dict[calibration_id] = {"influence_level": ..., "injection_level": ...}"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "labels" in data:
        labels = data["labels"]
    else:
        labels = data
    out = {}
    for k, v in labels.items():
        if not isinstance(v, dict):
            continue
        out[k] = {
            "influence_level": v.get("influence_level"),
            "injection_level": v.get("injection_level"),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Calibration agreement (Cohen's κ).")
    ap.add_argument("--human", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    human = _load_labels(args.human)
    llm = _load_labels(args.llm)

    # Common keys
    common = sorted(set(human) & set(llm))
    only_human = sorted(set(human) - set(llm))
    only_llm = sorted(set(llm) - set(human))
    if not common:
        print("[error] no overlapping calibration_ids", file=sys.stderr)
        sys.exit(2)

    # Build aligned lists
    inf_a = [human[k]["influence_level"] for k in common]
    inf_b = [llm[k]["influence_level"] for k in common]
    inj_a = [human[k]["injection_level"] for k in common]
    inj_b = [llm[k]["injection_level"] for k in common]

    # Drop any pair where either label is None (unanswered)
    valid_inf = [(a, b) for a, b in zip(inf_a, inf_b) if a is not None and b is not None]
    valid_inj = [(a, b) for a, b in zip(inj_a, inj_b) if a is not None and b is not None]

    inf_a_aligned = [a for a, _ in valid_inf]
    inf_b_aligned = [b for _, b in valid_inf]
    inj_a_aligned = [a for a, _ in valid_inj]
    inj_b_aligned = [b for _, b in valid_inj]

    inf_kappa = cohen_kappa(inf_a_aligned, inf_b_aligned, _INFLUENCE_LEVELS)
    inj_kappa = cohen_kappa(inj_a_aligned, inj_b_aligned, _INJECTION_LEVELS)
    inf_kappa_lin = weighted_kappa(
        inf_a_aligned, inf_b_aligned, _INFLUENCE_LEVELS, "linear"
    )
    inj_kappa_lin = weighted_kappa(
        inj_a_aligned, inj_b_aligned, _INJECTION_LEVELS, "linear"
    )
    inf_kappa_quad = weighted_kappa(
        inf_a_aligned, inf_b_aligned, _INFLUENCE_LEVELS, "quadratic"
    )
    inj_kappa_quad = weighted_kappa(
        inj_a_aligned, inj_b_aligned, _INJECTION_LEVELS, "quadratic"
    )

    # Confusion matrices
    inf_cm = confusion_matrix(
        [a for a, _ in valid_inf], [b for _, b in valid_inf], _INFLUENCE_LEVELS
    )
    inj_cm = confusion_matrix(
        [a for a, _ in valid_inj], [b for _, b in valid_inj], _INJECTION_LEVELS
    )

    # Print
    def _fmt_cm(cm, classes):
        header = "    " + " ".join(f"{c[:5]:>6}" for c in classes)
        rows = []
        for i, c in enumerate(classes):
            cells = " ".join(f"{cm[i][j]:>6}" for j in range(len(classes)))
            rows.append(f"  {c[:5]:>5} {cells}")
        return "\n".join([header, *rows])

    print(f"common pairs: {len(common)} ; only_human: {len(only_human)} ; only_llm: {len(only_llm)}")
    print()
    print(
        f"=== Influence Axis (κ_unweighted={inf_kappa:.3f}, "
        f"κ_linear={inf_kappa_lin:.3f}, κ_quadratic={inf_kappa_quad:.3f}) ==="
    )
    print("rows = human, cols = LLM")
    print(_fmt_cm(inf_cm, _INFLUENCE_LEVELS))
    print()
    print(
        f"=== Injection Axis (κ_unweighted={inj_kappa:.3f}, "
        f"κ_linear={inj_kappa_lin:.3f}, κ_quadratic={inj_kappa_quad:.3f}) ==="
    )
    print("rows = human, cols = LLM")
    print(_fmt_cm(inj_cm, _INJECTION_LEVELS))

    # Binary collapses (any-influenced / any-injected) — robust to
    # boundary noise between adjacent ordinal levels.
    inf_a_bin = ["yes" if a != "none" else "no" for a in inf_a_aligned]
    inf_b_bin = ["yes" if b != "none" else "no" for b in inf_b_aligned]
    inj_a_bin = ["yes" if a != "none" else "no" for a in inj_a_aligned]
    inj_b_bin = ["yes" if b != "none" else "no" for b in inj_b_aligned]
    inf_bin_kappa = cohen_kappa(inf_a_bin, inf_b_bin, ["no", "yes"])
    inj_bin_kappa = cohen_kappa(inj_a_bin, inj_b_bin, ["no", "yes"])
    print()
    print(
        f"=== Binary Collapse (any non-none) ==="
    )
    print(f"  Influence (any vs none) κ = {inf_bin_kappa:.3f}")
    print(f"  Injection (any vs none) κ = {inj_bin_kappa:.3f}")

    target_kappa = 0.7
    target_kappa_weighted = 0.6
    inf_ok = inf_kappa_lin >= target_kappa_weighted
    inj_ok = inj_kappa_lin >= target_kappa_weighted or len(set(inj_a_aligned)) <= 1
    overall_ok = inf_ok and inj_ok

    print()
    print(
        f"target: κ_unweighted ≥ {target_kappa:.2f} OR κ_linear-weighted ≥ {target_kappa_weighted:.2f}"
    )
    print(
        f"Influence: {'PASS' if inf_ok else 'FAIL'} "
        f"(κ_lin={inf_kappa_lin:.3f}) ; "
        f"Injection: {'PASS' if inj_ok else 'FAIL'} "
        f"(κ_lin={inj_kappa_lin:.3f}; "
        f"{'sample is heavily one-sided' if len(set(inj_a_aligned)) <= 1 else ''}) ; "
        f"overall {'PASS' if overall_ok else 'FAIL'}"
    )

    if args.output:
        out = {
            "n_common_pairs": len(common),
            "n_only_human": len(only_human),
            "n_only_llm": len(only_llm),
            "influence_kappa_unweighted": inf_kappa,
            "influence_kappa_linear_weighted": inf_kappa_lin,
            "influence_kappa_quadratic_weighted": inf_kappa_quad,
            "influence_kappa_binary_collapse": inf_bin_kappa,
            "injection_kappa_unweighted": inj_kappa,
            "injection_kappa_linear_weighted": inj_kappa_lin,
            "injection_kappa_quadratic_weighted": inj_kappa_quad,
            "injection_kappa_binary_collapse": inj_bin_kappa,
            "influence_classes": _INFLUENCE_LEVELS,
            "injection_classes": _INJECTION_LEVELS,
            "influence_confusion_matrix": inf_cm,
            "injection_confusion_matrix": inj_cm,
            "target_kappa_unweighted": target_kappa,
            "target_kappa_weighted": target_kappa_weighted,
            "passed": overall_ok,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.output}")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
