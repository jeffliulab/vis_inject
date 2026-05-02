"""
Stratified sampling of a calibration set
=========================================
Selects 100 (clean, adv, target, question, vlm) tuples from the full
6,615-pair sweep for human (Claude Opus 4.7) labelling. Stratification:

  - Stratify on prompt_tag (7 strata) — equal coverage
  - Within each prompt, ensure mix of VLMs (3 architectures present in
    response_pairs)
  - Within each (prompt, vlm), ensure mix of question categories
    (user / agent / screenshot)
  - Random with a fixed seed for reproducibility

Output: ``calibration_set.json`` with 100 entries containing the inputs
the labeler will see (target_phrase, question, response_clean,
response_adv) and provenance (which file, which VLM, which question).

The labeller never sees the LLM's output until after labelling.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_all_pairs(pairs_dir):
    """Load every (clean, adv, target, question) tuple as a flat list."""
    pairs_dir = Path(pairs_dir).resolve()
    out = []
    for pf in sorted(pairs_dir.glob("exp_*/results/response_pairs_*.json")):
        data = json.loads(pf.read_text(encoding="utf-8"))
        target = data["metadata"]["target_phrase"]
        # exp_<prompt>_<config>
        exp_dir = pf.parent.parent.name
        parts = exp_dir.split("_")  # ['exp', 'card', '2m']
        prompt_tag = parts[1] if len(parts) >= 3 else "unknown"
        config = parts[2] if len(parts) >= 3 else "unknown"
        image = pf.stem.replace("response_pairs_", "")  # ORIGIN_dog
        for vlm_key, qa_list in data["pairs"].items():
            for qa in qa_list:
                out.append({
                    "experiment": exp_dir,
                    "prompt_tag": prompt_tag,
                    "model_config": config,
                    "image": image,
                    "vlm": vlm_key,
                    "category": qa.get("category", "unknown"),
                    "question": qa["question"],
                    "response_clean": qa["response_clean"],
                    "response_adv": qa["response_adv"],
                    "target_phrase": target,
                    "source_file": str(pf.relative_to(pairs_dir.parent)),
                })
    return out


def sample_stratified(all_pairs, n=100, seed=42):
    """Stratified sampling.

    Strategy:
      - Group by (prompt_tag, vlm, category)
      - Round-robin pick from groups until n filled
      - Within a group, random.sample
    """
    rnd = random.Random(seed)
    groups = defaultdict(list)
    for p in all_pairs:
        groups[(p["prompt_tag"], p["vlm"], p["category"])].append(p)

    # Shuffle within each group
    for g in groups.values():
        rnd.shuffle(g)

    # Round-robin
    keys = sorted(groups.keys())
    rnd.shuffle(keys)
    picks = []
    while len(picks) < n:
        any_left = False
        for k in keys:
            if groups[k]:
                picks.append(groups[k].pop())
                any_left = True
                if len(picks) >= n:
                    break
        if not any_left:
            break

    return picks


def main():
    ap = argparse.ArgumentParser(description="Sample calibration set.")
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_pairs = load_all_pairs(args.pairs_dir)
    print(f"loaded {len(all_pairs)} total pairs")

    sample = sample_stratified(all_pairs, n=args.n, seed=args.seed)
    print(f"sampled {len(sample)} for calibration (seed={args.seed})")

    # Distribution sanity print
    from collections import Counter
    print("by prompt:", Counter(p["prompt_tag"] for p in sample))
    print("by vlm   :", Counter(p["vlm"] for p in sample))
    print("by cat   :", Counter(p["category"] for p in sample))

    # Add stable IDs
    for i, p in enumerate(sample):
        p["calibration_id"] = f"cal_{i:03d}"

    out = {
        "schema_version": "v3",
        "n": len(sample),
        "seed": args.seed,
        "pairs": sample,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
