#!/usr/bin/env python3
"""Step 3: final refit on 11-20 + 21-30, predictions for weeks 31-42.

Protocol (same as the validation harness):
  * training interactions = block 11-20 + block 21-30 (no viewing data from
    weeks 31-42 exists in the dataset, and none is used);
  * candidate pool = labelled programmes that air in the 31-42 schedule
    (the test schedule is used only to know which programmes air where/when);
  * every one of the 1,260 submission users gets exactly 5 distinct ids:
    method ranking first, popularity fill from the training-blocks popularity
    baseline (restricted to the pool), popularity baseline for cold-start users;
  * repeats (already-watched programmes) are allowed.

Outputs:
  out/test_<method>.csv        user_id + 5 ids
  out/submission_final.csv     Kaggle format: user_id, "id id id id id"
  out/step3_final.md           log of the run, incl. the two example users
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import m2_content as M2  # noqa: E402

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
REPORT: list[str] = []


def log(msg="", flush=True):
    print(msg, flush=flush)
    REPORT.append(str(msg))


def hr(t):
    log("")
    log("=" * 100)
    log(t)
    log("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.3, help="PPR alpha (selected on validation)")
    ap.add_argument("--method", default="ppr", choices=["ppr", "popularity", "history", "hybrid"])
    ap.add_argument("--w-ppr", type=float, default=1.0)
    ap.add_argument("--w-hist", type=float, default=0.0)
    ap.add_argument("--w-pop", type=float, default=0.0)
    args = ap.parse_args()

    t_script = time.perf_counter()
    hr("STEP 3 - FINAL FIT (11-20 + 21-30) AND PREDICTION FOR WEEKS 31-42")
    log(f"Environment : Python {sys.version.split()[0]} {platform.platform()}; "
        f"numpy {np.__version__}; pandas {pd.__version__}")
    log(f"Seed        : {RANDOM_SEED} (no stochastic component)")
    log(f"Method      : {args.method}" + (f" (alpha={args.alpha})" if args.method == "ppr" else ""))

    inter = H.load_interactions()
    progs = H.load_programmes()
    train_sum = H.get_user_train(inter, H.TRAIN_BLOCKS_FINAL)
    user_train_set, user_train_list = H.build_user_histories(train_sum)
    pool = H.load_pool(H.EVAL_BLOCK_FINAL)
    airtime = H.load_airtime(H.EVAL_BLOCK_FINAL)
    sub = pd.read_csv(H.DATA / "submission.csv", dtype={"user_id": str})
    sub_users = sub["user_id"].tolist()
    log(f"  training interactions (11-20 + 21-30): {len(train_sum):,} user-programme pairs, "
        f"{len(user_train_set):,} users")
    log(f"  candidate pool (labelled programmes airing in 31-42): {len(pool):,}")
    log(f"  submission users: {len(sub_users):,}")
    cold = [u for u in sub_users if not user_train_set.get(u)]
    log(f"  submission users with no training interaction (cold start): {len(cold)} -> popularity baseline")

    pop = H.build_popularity(train_sum, pool)
    pop_ids = [int(it) for it in pop.index if int(it) in pool]
    log(f"  popularity baseline (distinct training users per programme, pool-restricted): "
        f"top-5 = {pop_ids[:5]}")

    # ---- method scores -------------------------------------------------
    ppr_scores = None
    if args.method in ("ppr", "hybrid"):
        user_list = sorted(user_train_set.keys())
        item_list = sorted(set(train_sum["tv_show_id"].astype(int).tolist()))
        user2idx = {u: i for i, u in enumerate(user_list)}
        item2idx = {it: i for i, it in enumerate(item_list)}
        ppr_scores = H.ppr_batched(sub_users, user2idx, item2idx, user_list, item_list,
                                   user_train_list, user_train_set, pool, args.alpha)
    if args.method == "ppr":
        method_scores = {u: ppr_scores.get(u, []) for u in sub_users}
    elif args.method == "hybrid":
        CAND_CUT = 200
        pop_rank = [int(it) for it in pop.index if int(it) in pool][:CAND_CUT]
        method_scores = {}
        for u in sub_users:
            ppr_rank = [it for it, _ in ppr_scores.get(u, [])][:CAND_CUT]
            hist_rank = [it for it, _ in user_train_list.get(u, []) if it in pool][:CAND_CUT]
            method_scores[u] = H.rank_fusion([ppr_rank, hist_rank, pop_rank],
                                            [args.w_ppr, args.w_hist, args.w_pop], k=5)
    elif args.method == "history":
        method_scores = {u: [(it, 0.0) for it, _ in user_train_list.get(u, [])] for u in sub_users}
    else:
        method_scores = {u: [] for u in sub_users}
    recs, fallback = H.make_recs(method_scores, pool, pop, user_train_set, set(cold), 5)
    log(f"  fallback users served by the popularity baseline: {fallback}")

    # ---- save ----------------------------------------------------------
    out_rows = []
    for u in sub_users:
        lst = list(dict.fromkeys(int(x) for x in recs.get(u, [])[:5]))
        assert len(lst) == 5 and len(set(lst)) == 5, f"bad list for {u}: {lst}"
        out_rows.append((u, lst))
    test_path = H.OUT / f"test_{args.method}.csv"
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id"] + [f"tv_show_id_{i}" for i in range(1, 6)])
        for u, lst in out_rows:
            w.writerow([u] + lst)
    log(f"  wrote {test_path} ({len(out_rows):,} rows)")

    sub_path = H.OUT / "submission_final.csv"
    with open(sub_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "tv_show_id"])
        for u, lst in out_rows:
            w.writerow([u, " ".join(str(x) for x in lst)])
    log(f"  wrote {sub_path} (Kaggle format: user_id + 5 space-separated ids)")

    # ---- example users --------------------------------------------------
    hr("EXAMPLE USERS (same two across every method)")
    # Heavy and light viewer among the 1,260 submission users, chosen by
    # training seconds, restricted to users that also have 21-30 activity so
    # that a real target exists for the comparison.
    ev_all = inter[inter["block"] == "21-30"]
    active_21_30 = set(ev_all["user_id"].unique())
    sub_set = set(sub_users)
    tot = train_sum.groupby("user_id")["overlap_seconds"].sum()
    cand = tot[[u in sub_set and u in active_21_30 for u in tot.index]]
    heavy = cand.idxmax()
    light = cand.idxmin()
    log(f"  heavy viewer : {heavy} (training seconds = {int(tot[heavy]):,}; "
        f"{int((train_sum['user_id'] == heavy).sum())} programmes)")
    log(f"  light viewer : {light} (training seconds = {int(tot[light]):,}; "
        f"{int((train_sum['user_id'] == light).sum())} programmes)")
    title_map = dict(zip(progs["tv_show_id"].astype(int), progs["title"].astype(str)))
    cat_map = dict(zip(progs["tv_show_id"].astype(int), progs["category"].astype(str)))

    def show(uid, lst, tag):
        log(f"  {tag} | user {uid}:")
        for rank, pid in enumerate(lst, 1):
            log(f"    {rank}. id={pid:<12} cat={cat_map.get(pid, '?'):<16} "
                f"title={title_map.get(pid, '?')[:60]}")

    for uid, name in ((heavy, "heavy viewer"), (light, "light viewer")):
        log("")
        show(uid, [int(x) for x in recs.get(uid, [])], f"{args.method} final (31-42 pool) - {name}")
    if cold:
        log("")
        show(cold[0], [int(x) for x in recs.get(cold[0], [])], "popularity fallback - cold-start user")

    log("")
    log(f"Total runtime: {time.perf_counter() - t_script:.1f}s")
    (H.OUT / "step3_final.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"\nWrote {H.OUT / 'step3_final.md'}")


if __name__ == "__main__":
    main()
