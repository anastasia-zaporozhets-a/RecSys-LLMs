#!/usr/bin/env python3
"""Equal-weight RRF of three existing submission files; no model training.

Run: .venv/bin/python src/ensemble_final.py
Writes submission_v8_ens.csv and out/ensemble_final.md. Selection was made
after seeing Kaggle scores. Ranks are one-based; ties use v6_3 order, then
ascending programme ID for items absent from v6_3. Exact fractions prevent
floating-point addition order from breaking mathematical ties.
"""
from collections import defaultdict
from fractions import Fraction
import hashlib

import pandas as pd

import final_build as FB
import harness as H

SOURCES = ("submission_v6_3.csv", "submission_v7_1.csv", "submission_v7_2.csv")
REPORT = []


def log(message=""):
    print(message, flush=True)
    REPORT.append(str(message))


def combine(rankings):
    scores = defaultdict(Fraction)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] += Fraction(1, 60 + rank)
    base_order = {item: rank for rank, item in enumerate(rankings[0], 1)}
    return sorted(scores, key=lambda item: (-scores[item], base_order.get(item, 6), item))[:5]


def read_source(name, users, pool):
    path = H.EXAM / name
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert frame.columns.tolist() == ["user_id", "tv_show_id"], name
    assert len(frame) == len(users) == H.EXPECTED_SUB_USERS, name
    assert frame.user_id.is_unique and frame.user_id.tolist() == users, name
    rows = [list(map(int, value.split())) for value in frame.tv_show_id]
    assert all(len(row) == len(set(row)) == 5 and 0 not in row and set(row) <= pool for row in rows), name
    log(f"- `{name}`: PASS — {len(rows):,} ordered unique users; five distinct nonzero in-pool IDs per row.")
    log(f"  SHA-256: `{hashlib.sha256(path.read_bytes()).hexdigest()}`")
    return dict(zip(users, rows))


def main():
    log("# Final rank-average ensemble")
    log()
    log("**Selection note: this ensemble was chosen after seeing Kaggle scores.**")
    log()
    log("- Components: submission_v6_3, submission_v7_1, submission_v7_2; equal weights 1/1/1.")
    log("- Score: sum of `1 / (60 + rank)` across the three files, with one-based ranks.")
    log("- Ties: v6_3 order first; items absent from v6_3 follow those present, with ascending ID as the remaining tie-break.")
    log("- Exact rational scores; five distinct IDs; no model training.")
    log()
    log("## Source validation")
    users = pd.read_csv(H.DATA / "submission.csv", dtype=str).user_id.tolist()
    pool = H.load_pool(H.EVAL_BLOCK_FINAL)
    sources = [read_source(name, users, pool) for name in SOURCES]
    recs = {user: combine([source[user] for source in sources]) for user in users}
    assert all(len(row) == len(set(row)) == 5 and set(row) <= pool for row in recs.values())
    log()
    log("## Output checks")
    log("```text")
    path = H.EXAM / "submission_v8_ens.csv"
    # Every source contributes five valid IDs, so the union always supplies
    # five ensemble recommendations; the standard writer never needs a fill.
    FB.check_and_write(recs, path, pool, pd.Series(dtype=float), log=log)
    written = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert written.user_id.tolist() == users and written.user_id.is_unique
    assert [list(map(int, value.split())) for value in written.tv_show_id] == [recs[user] for user in users]
    log("```")
    log(f"SHA-256: `{hashlib.sha256(path.read_bytes()).hexdigest()}`")
    log()
    log("## Diversity versus inputs")
    for name, source in zip(SOURCES, sources):
        changed = sum(recs[user] != source[user] for user in users)
        log(f"- `{name}`: {changed:,}/{len(users):,} user rankings differ from the ensemble.")
    H.OUT.mkdir(exist_ok=True)
    report_path = H.OUT / "ensemble_final.md"
    report_path.write_text("\n".join(REPORT).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
