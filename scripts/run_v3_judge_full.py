"""Run v3 dual-axis judge over the full 6,615-pair sweep. Background-friendly."""

import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluate.llm_judge import judge_one_pair, _make_client, JudgeCache
from src.config import DEEPSEEK_CONFIG


def main():
    cache_path = REPO_ROOT / "outputs" / "judge_cache.json"
    cache = JudgeCache.load_or_init(cache_path, DEEPSEEK_CONFIG["model"])
    client = _make_client(DEEPSEEK_CONFIG)

    pairs_dir = Path("/tmp/visinject-judge/experiments")
    files = sorted(pairs_dir.glob("exp_*/results/response_pairs_*.json"))
    print(f"[{time.strftime('%H:%M:%S')}] judging {len(files)} files using {DEEPSEEK_CONFIG['model']}")
    print(f"[{time.strftime('%H:%M:%S')}] cache: {len(cache.calls)} pre-warmed entries")
    sys.stdout.flush()

    work = []
    for f in files:
        data = json.loads(f.read_text())
        target = data["metadata"]["target_phrase"]
        for vlm, qa_list in data["pairs"].items():
            for i, qa in enumerate(qa_list):
                work.append({
                    "file": str(f), "vlm": vlm, "qa_idx": i,
                    "target": target, "question": qa["question"],
                    "rc": qa["response_clean"], "ra": qa["response_adv"],
                })

    print(f"[{time.strftime('%H:%M:%S')}] total work items: {len(work)}")
    sys.stdout.flush()

    errs = []
    t0 = time.time()
    done = made = hits = 0

    def _judge(item, idx):
        try:
            out, hit = judge_one_pair(
                client, DEEPSEEK_CONFIG, item["target"], item["question"],
                item["rc"], item["ra"], cache=cache
            )
            return idx, out, hit, None
        except Exception as e:
            return idx, None, False, str(e)

    with ThreadPoolExecutor(max_workers=DEEPSEEK_CONFIG.get("max_concurrent", 5)) as ex:
        futures = [ex.submit(_judge, w, i) for i, w in enumerate(work)]
        for f in as_completed(futures):
            idx, out, hit, err = f.result()
            done += 1
            if err:
                errs.append((idx, err))
            elif hit:
                hits += 1
            else:
                made += 1
            if done % 50 == 0:
                cache.save()
                elapsed = time.time() - t0
                rate = done / elapsed
                eta_min = (len(work) - done) / rate / 60 if rate > 0 else 0
                print(
                    f"[{time.strftime('%H:%M:%S')}] {done}/{len(work)}  "
                    f"hits={hits}  made={made}  errs={len(errs)}  "
                    f"rate={rate:.1f}/s  ETA={eta_min:.1f}min"
                )
                sys.stdout.flush()

    cache.save()
    print(
        f"\n[{time.strftime('%H:%M:%S')}] DONE: {done}/{len(work)}  "
        f"hits={hits}  made={made}  errs={len(errs)}  "
        f"total_time={(time.time()-t0)/60:.1f}min"
    )

    if errs:
        Path("outputs/judge_errors.json").write_text(
            json.dumps([
                {"idx": i, "err": e, "vlm": work[i]["vlm"], "q": work[i]["question"][:120]}
                for i, e in errs
            ], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  saved {len(errs)} errors to outputs/judge_errors.json")
        # Retry once with bigger max_tokens for errors
        print(f"\n[{time.strftime('%H:%M:%S')}] retrying {len(errs)} errors with max_tokens=8192…")
        sys.stdout.flush()
        cfg = dict(DEEPSEEK_CONFIG)
        cfg["max_tokens"] = 8192
        retry_made = retry_failed = 0
        for idx, _ in errs:
            w = work[idx]
            try:
                out, hit = judge_one_pair(
                    client, cfg, w["target"], w["question"], w["rc"], w["ra"], cache=cache
                )
                retry_made += 1
            except Exception as e:
                retry_failed += 1
                print(f"  RETRY-FAIL  {w['vlm']}/{w['question'][:50]}: {str(e)[:80]}")
        cache.save()
        print(f"  retry_made={retry_made}  retry_failed={retry_failed}")


if __name__ == "__main__":
    main()
