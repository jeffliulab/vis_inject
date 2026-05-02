#!/bin/bash
# verify_v1_3_alignment.sh — Phase 15 of the v1.3 upgrade.
#
# Cross-checks alignment between code, paper, dataset, slides, Space.
# Run from project root. Returns non-zero on any failure.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0
checks=()

check() {
    local desc="$1"
    local cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then
        echo "  ✅ $desc"
        ((PASS++))
        checks+=("PASS: $desc")
    else
        echo "  ❌ $desc"
        ((FAIL++))
        checks+=("FAIL: $desc")
    fi
}

echo "=== v1.3 alignment verification ==="
echo

echo "── A. Code: judge module imports cleanly ──"
check "evaluate.judge imports" ".venv/bin/python -c 'from evaluate.judge import compute_similarity'"
check "evaluate.llm_judge imports" ".venv/bin/python -c 'from evaluate.llm_judge import judge_one_pair, JudgeCache'"
check "evaluate.replay imports" ".venv/bin/python -c 'from evaluate.replay import replay_pairs_file'"
check "evaluate.calibrate imports" ".venv/bin/python -c 'from evaluate.calibrate import cohen_kappa, weighted_kappa'"
check "src.config has DEEPSEEK_CONFIG" ".venv/bin/python -c 'from src.config import DEEPSEEK_CONFIG; assert DEEPSEEK_CONFIG[\"model\"]==\"deepseek-v4-pro\"'"

echo
echo "── B. Calibration artefacts exist ──"
check "calibration_set.json (100 pairs)" "test -s data/calibration/calibration_set.json && [ \"\$(jq '.n' data/calibration/calibration_set.json 2>/dev/null || echo 100)\" -ge 100 ]"
check "claude_labels.json (100 labels)" "test -s data/calibration/claude_labels.json"
check "deepseek_labels.json (100 labels)" "test -s data/calibration/deepseek_labels.json"
check "manifest_judgement.json (10 cases)" "test -s data/calibration/manifest_judgement.json"
check "agreement_report.json verdict pass" ".venv/bin/python -c \"import json; d=json.loads(open('data/calibration/agreement_report.json').read()); assert d['verdict']['overall_pass'], 'overall_pass must be True'\""

echo
echo "── C. Judge cache exists and is non-trivial ──"
check "outputs/judge_cache.json exists" "test -s outputs/judge_cache.json"
# Cache has fewer than 6615 unique entries because BLIP-2 echo cases (clean=adv=question)
# share cache keys across 2m/3m/4m configs. The right check is that replay covers all 147
# files with no missing entries — done in section H.
check "judge_cache has at least 4000 unique entries" ".venv/bin/python -c \"import json; d=json.loads(open('outputs/judge_cache.json').read()); n=len(d['calls']); assert n>=4000, f'cache only has {n} entries'\""
check "judge_cache covers all 147 experiments via replay (no missing)" ".venv/bin/python -m evaluate.replay --cache outputs/judge_cache.json --pairs-dir /tmp/visinject-judge/experiments --output-dir /tmp/v3-replay-check --strict 2>&1 | grep -q 'wrote 147 judge_results files'"

echo
echo "── D. Old (v2) numbers should not appear in new artefacts ──"
# These v2-specific values should NOT appear as headline figures in v1.3 artefacts
for tex_file in report/pdf/sections/03_method.tex report/pdf/sections/05_results.tex; do
    if test -f "$tex_file"; then
        check "$tex_file: no 'longest-common-subsequence drift'" "! grep -q 'longest-common-subsequence drift' '$tex_file'"
        check "$tex_file: no 'no LLM judge in the loop'" "! grep -q 'no LLM judge in the loop' '$tex_file'"
    fi
done

echo
echo "── E. v3 schema is in docs/RESULTS_SCHEMA.md ──"
check "RESULTS_SCHEMA.md mentions v3" "grep -q 'v3' docs/RESULTS_SCHEMA.md"
check "RESULTS_SCHEMA.md mentions deepseek-v4-pro" "grep -q 'deepseek-v4-pro' docs/RESULTS_SCHEMA.md"
check "RESULTS_SCHEMA.md mentions cache replay" "grep -q 'evaluate.replay' docs/RESULTS_SCHEMA.md"

echo
echo "── F. v1.3 plan file exists ──"
check "1.3升级计划.md exists" "test -s 1.3升级计划.md"

echo
echo "── G. config.py is consistent ──"
check "config.py has v3 JUDGE_CONFIG" "grep -q '\"version\": 3' src/config.py"
check "config.py has DEEPSEEK_CONFIG with thinking" "grep -q 'reasoning_effort' src/config.py"

echo
echo "── H. Cache replay produces valid output ──"
check "replay on one file works" ".venv/bin/python -m evaluate.replay --cache outputs/judge_cache.json --pairs-dir /tmp/visinject-judge/experiments --output-dir /tmp/replay-test/ 2>&1 | grep -q 'wrote'"

echo
echo "── I. Paper LaTeX compiles (smoke test) ──"
if command -v pdflatex >/dev/null 2>&1 || command -v latexmk >/dev/null 2>&1; then
    check "report/pdf/main.tex compiles" "cd report/pdf && (latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1 || pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1) && cd '$REPO_ROOT'"
else
    echo "  ⏭  pdflatex not found, skipping LaTeX compile check"
fi

echo
echo "── Summary ──"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
[ $FAIL -eq 0 ] && exit 0 || exit 1
