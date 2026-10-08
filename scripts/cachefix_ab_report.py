"""Compare the two arms of scripts/cl2_cachefix_ab.sh (linear_cache_rewind False vs True).

Prints, per (language, task, draft, gamma):
  - overall acceptance, speed-up and timings for each arm
  - the paired per-sentence acceptance difference (fix - old) with a 95% CI;
    both arms are seeded per sentence, so sentence i is the same prompt in both
  - acceptance at each octile position, to see whether the stale state compounds
  - chrF/BLEU for translation (both arms translate the same 400 test sentences)

Usage: uv run python scripts/cachefix_ab_report.py [--tag cachefix-ab]
"""

import argparse
import math
from collections import defaultdict
from typing import Any

import wandb

from src.config.config import WANDB_ENTITY, WANDB_PROJECT

SUMMARY_KEYS = [
    ("sentence_avg_acceptance_rate", "alpha (sentence avg)"),
    ("token_weighted_acceptance_rate", "alpha (token weighted)"),
    ("speedup_factor", "speed-up f"),
    ("average_draft_time", "draft time (s)"),
    ("average_verifier_time", "verifier time (s)"),
    ("chrf2", "chrF++ (translation)"),
    ("bleu", "BLEU (translation)"),
]


def per_sentence_alpha(run) -> dict[int, float]:
    rows = run.scan_history(keys=["sentence_idx", "sentence/acceptance_rate"])
    return {int(r["sentence_idx"]): r["sentence/acceptance_rate"] for r in rows}


def fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="cachefix-ab")
    args = parser.parse_args()

    runs = wandb.Api().runs(
        path=f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"tags": {"$in": [args.tag]}, "state": "finished"},
        lazy=False,
    )
    # Latest finished run per arm, in case a cell was resubmitted
    cells: dict[tuple, dict[bool, Any]] = defaultdict(dict)
    for run in sorted(runs, key=lambda r: r.created_at):
        c = run.config
        key = (c["language_code"], c["task"], c.get("draft_model") or "ngram", c["gamma"])
        cells[key][bool(c.get("linear_cache_rewind", True))] = run

    if not cells:
        print(f"No finished runs tagged {args.tag!r}")
        return

    for key, arms in sorted(cells.items()):
        print(f"\n=== lang={key[0]} task={key[1]} draft={key[2]} gamma={key[3]} ===")
        if set(arms) != {False, True}:
            print(f"  only arm(s) {sorted(arms)} finished; skipping")
            continue
        old, new = arms[False], arms[True]
        if old.config.get("seed") is None or old.config.get("seed") != new.config.get("seed"):
            print("  WARNING: arms are not seeded identically; paired differences include sampling noise")

        print(f"  {'':26s} {'old (no fix)':>13s} {'fix':>10s} {'diff':>10s}")
        for k, label in SUMMARY_KEYS:
            a, b = old.summary.get(k), new.summary.get(k)
            d = b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            print(f"  {label:26s} {fmt(a):>13s} {fmt(b):>10s} {fmt(d):>10s}")

        a_old, a_new = per_sentence_alpha(old), per_sentence_alpha(new)
        shared = sorted(set(a_old) & set(a_new))
        diffs = [a_new[i] - a_old[i] for i in shared]
        if len(diffs) >= 2:
            mean = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
            half = 1.96 * sd / math.sqrt(len(diffs))
            identical = sum(d == 0 for d in diffs)
            print(
                f"  paired alpha diff (fix - old): {mean:+.4f} "
                f"[{mean - half:+.4f}, {mean + half:+.4f}] over {len(diffs)} sentences "
                f"({identical} identical)"
            )

        octiles = sorted(
            (int(k.rsplit("_", 1)[1]), k) for k in new.summary.keys() if k.startswith("acceptance_rate_pos_")
        )
        if octiles:
            print("  acceptance by output position (old -> fix):")
            for pos, k in octiles:
                a, b = old.summary.get(k), new.summary.get(k)
                print(f"    token {pos:4d}: {fmt(a)} -> {fmt(b)}")


if __name__ == "__main__":
    main()
