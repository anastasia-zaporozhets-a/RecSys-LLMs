#!/usr/bin/env python3
"""C) Final submissions for the two baselines, through the same pipeline.

Same protocol as src/step3_final.py and src/step3_tune.py:
  * refit on weeks 11-20 + 21-30 (the 11-30 training window);
  * candidate pool = labelled programmes airing in the 31-42 schedule;
  * exactly 5 distinct ids per user, popularity fill, popularity baseline for
    cold-start users, repeats allowed;
  * the same automatic checks are run on both files.

Writes:
  exam/submission_popularity.csv   (B1: popularity over weeks 11-30)
  exam/submission_history.csv      (B2: the user's own training programmes)

Usage:  .venv/bin/python src/step3_baselines_final.py
"""
from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import final_build as FB  # noqa: E402

REPORT: list[str] = []


def log(msg="", flush=True):
    print(msg, flush=flush)
    REPORT.append(str(msg))


def hr(t):
    log("")
    log("=" * 104)
    log(t)
    log("=" * 104)


def main():
    t0 = time.perf_counter()
    hr("C) FINAL SUBMISSIONS FOR THE BASELINES (refit on weeks 11-30, predict 31-42)")
    log(f"Environment : Python {sys.version.split()[0]} {platform.platform()}")
    log("Both files use the identical final pipeline as src/step3_final.py / src/step3_tune.py.")

    for method, fname, title in (
        ("popularity", "submission_popularity.csv", "B1 POPULARITY"),
        ("history", "submission_history.csv", "B2 PERSONAL HISTORY"),
    ):
        hr(f"{title} -> exam/{fname}")
        recs, info = FB.build_final(method, alpha=0.3, beta=0.0, log=log)
        checks = FB.check_and_write(recs, H.EXAM / fname, info["pool"], info["popularity"], log=log)
        log(f"  checks: {checks}")

    log(f"\nTotal runtime: {time.perf_counter() - t0:.1f}s")
    (H.OUT / "step3_baselines.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"\nWrote {H.OUT / 'step3_baselines.md'}")


if __name__ == "__main__":
    main()
