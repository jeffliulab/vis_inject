#!/bin/bash
# Batch-judge all response_pairs JSON files across experiments.
#
# Usage (from project root):
#   bash scripts/judge_all.sh                          # v2 judge all experiments (default)
#   bash scripts/judge_all.sh --version v1             # legacy v1 mode
#   bash scripts/judge_all.sh --judges gpt-4o-mini     # specific judge(s) (v1 only)
#   bash scripts/judge_all.sh --exp exp_card_2m        # specific experiment only
#   bash scripts/judge_all.sh --force                  # overwrite existing judge results
#
# Expects experiment results in outputs/experiments/exp_*/results/response_pairs_*.json

set -euo pipefail

JUDGE_ARGS=""
EXP_FILTER=""
FORCE=false
VERSION="v2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --judges)
            shift
            JUDGE_NAMES=""
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                JUDGE_NAMES="$JUDGE_NAMES $1"
                shift
            done
            JUDGE_ARGS="--judges $JUDGE_NAMES"
            ;;
        --exp)      EXP_FILTER="$2"; shift 2 ;;
        --version)  VERSION="$2"; shift 2 ;;
        --force)    FORCE=true; shift ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

EXPERIMENTS_DIR="outputs/experiments"
TOTAL=0
DONE=0
SKIPPED=0

echo "===== VisInject Batch Judge (${VERSION}) ====="
echo ""

for exp_dir in "${EXPERIMENTS_DIR}"/exp_*; do
    [ -d "$exp_dir" ] || continue
    exp_name=$(basename "$exp_dir")

    # Filter if specified
    if [ -n "$EXP_FILTER" ] && [ "$exp_name" != "$EXP_FILTER" ]; then
        continue
    fi

    # Find response_pairs files (check both layouts)
    pairs_files=()
    for f in "$exp_dir"/response_pairs_*.json "$exp_dir"/results/response_pairs_*.json; do
        [ -f "$f" ] && pairs_files+=("$f")
    done

    if [ ${#pairs_files[@]} -eq 0 ]; then
        echo "[SKIP] $exp_name: no response_pairs files found"
        continue
    fi

    echo "[EXP]  $exp_name (${#pairs_files[@]} image(s))"

    for pairs_file in "${pairs_files[@]}"; do
        TOTAL=$((TOTAL + 1))
        basename_file=$(basename "$pairs_file")
        image_name="${basename_file#response_pairs_}"
        image_name="${image_name%.json}"

        # Output goes next to the pairs file
        out_dir=$(dirname "$pairs_file")
        out_file="${out_dir}/judge_results_${image_name}.json"

        # Skip if already judged (unless --force)
        if [ -f "$out_file" ] && [ "$FORCE" = false ]; then
            echo "       [skip] $image_name (already judged, use --force to overwrite)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo "       [judge] $image_name ..."
        python -m evaluate.judge --pairs-file "$pairs_file" --output "$out_file" --version "$VERSION" ${JUDGE_ARGS} || {
            echo "       [FAIL] $image_name"
            continue
        }
        DONE=$((DONE + 1))
    done
done

echo ""
echo "===== Done ====="
echo "Total: $TOTAL | Judged: $DONE | Skipped: $SKIPPED"
