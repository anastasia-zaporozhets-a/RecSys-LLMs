#!/usr/bin/env python3
"""Step 2 orchestrator: baselines B1/B2, M1 Personalized PageRank, M2 content-based.

Validation protocol (identical for every method):
  fit on block 11-20 interactions, evaluate on block 21-30;
  candidate pool = labelled programmes airing in the 21-30 schedule;
  target = user's top-5 programmes in 21-30 by summed overlap_seconds (ties by id);
  MAP@5 macro-averaged over the 4,799 users with >= 1 watched programme in 21-30;
  repeats allowed; 5 distinct ids per user; popularity fill; popularity for cold start.

Usage:
    .venv/bin/python src/evaluate.py                 # all alphas (~11 min)
    .venv/bin/python src/evaluate.py --alphas 0.3    # single alpha (~1.5 min)
    .venv/bin/python src/evaluate.py --skip-ppr      # baselines + M2 only (~20 s)

Writes out/step2_results.md, out/val_<method>.csv and out/step2_stdout.log content.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import m2_content as M2  # noqa: E402

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)

REPORT: list[str] = []


def log(msg="", flush=True):
    print(msg, flush=flush)
    REPORT.append(str(msg))


def hr(title):
    log("")
    log("=" * 100)
    log(title)
    log("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.3,0.5,0.7,0.85")
    ap.add_argument("--skip-ppr", action="store_true")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    t_script = time.perf_counter()
    hr("STEP 2 - EVALUATION HARNESS, BASELINES, PPR, CONTENT-BASED")
    log(f"Environment : Python {sys.version.split()[0]} {platform.platform()}")
    log(f"numpy {np.__version__} | pandas {pd.__version__} | scipy {__import__('scipy').__version__} | "
        f"scikit-learn {__import__('sklearn').__version__}")
    log(f"Seed        : {RANDOM_SEED} (no stochastic component in this step)")
    log(f"Protocol    : train {H.TRAIN_BLOCKS_VAL} -> eval {H.EVAL_BLOCK_VAL}; "
        f"pool = labelled programmes in the {H.EVAL_BLOCK_VAL} schedule; K=5; repeats allowed; "
        "popularity fill; popularity for cold-start users.")

    inter = H.load_interactions()
    progs = H.load_programmes()
    train_sum = H.get_user_train(inter, H.TRAIN_BLOCKS_VAL)
    targets_top5, targets_all = H.get_eval_targets(inter, H.EVAL_BLOCK_VAL)
    user_train_set, user_train_list = H.build_user_histories(train_sum)
    n_eval = len([u for u in targets_all if targets_all[u]])
    log(f"\n  eval users with >=1 watched programme in {H.EVAL_BLOCK_VAL}: {n_eval:,} "
        f"(expected {H.EXPECTED_EVAL_USERS})")
    assert n_eval == H.EXPECTED_EVAL_USERS, f"eval user count {n_eval} != {H.EXPECTED_EVAL_USERS}"
    log(f"  training users: {len(user_train_set):,} (expected {H.EXPECTED_TRAIN_USERS})")
    assert len(user_train_set) == H.EXPECTED_TRAIN_USERS
    sub = pd.read_csv(H.DATA / "submission.csv", dtype={"user_id": str})
    log(f"  submission users: {len(sub):,} (ids kept as strings)")
    assert len(sub) == H.EXPECTED_SUB_USERS

    pool = H.load_pool(H.EVAL_BLOCK_VAL)
    log(f"  candidate pool: {len(pool):,} labelled programmes airing in {H.EVAL_BLOCK_VAL}")
    cold = set(targets_all.keys()) - set(user_train_set.keys())
    log(f"  cold-start eval users (no training interactions): {len(cold):,}")
    pop = H.build_popularity(train_sum, pool)
    eval_users = list(targets_all.keys())
    rows_table = []

    # ------------------------------------------------------------------ B1
    hr("B1 - POPULARITY BASELINE (distinct training users per programme, pool-restricted)")
    t0 = time.perf_counter()
    pop_recs = {}
    pop_ids = [int(it) for it in pop.index if int(it) in pool]
    for u in eval_users:
        pop_recs[u] = pop_ids[:5]
    t_b1 = time.perf_counter() - t0
    m5, n_u5 = H.map_at_k(pop_recs, targets_top5, 5)
    ma, _ = H.map_at_k(pop_recs, targets_all, 5)
    log(f"  MAP@5 (top-5 target) = {m5:.6f}   MAP@5 (all-watched) = {ma:.6f}   "
        f"users = {n_u5:,}   runtime = {t_b1:.3f}s")
    rows_table.append(("B1_popularity", "distinct training users per programme, pool-restricted",
                       m5, ma, t_b1, 0))

    # ------------------------------------------------------------------ B2
    hr("B2 - PERSONAL HISTORY BASELINE (own training programmes by overlap_seconds)")
    t0 = time.perf_counter()
    hist_recs = {}
    hist_scores = {}
    for u in eval_users:
        lst = []
        for it, _ in user_train_list.get(u, []):
            if it in pool and it not in lst:
                lst.append(it)
            if len(lst) == 5:
                break
        hist_scores[u] = [(it, 0.0) for it in lst]
    hist_recs, fb_b2 = H.make_recs(hist_scores, pool, pop, user_train_set, cold, 5)
    t_b2 = time.perf_counter() - t0
    m5, _ = H.map_at_k(hist_recs, targets_top5, 5)
    ma, _ = H.map_at_k(hist_recs, targets_all, 5)
    log(f"  MAP@5 (top-5 target) = {m5:.6f}   MAP@5 (all-watched) = {ma:.6f}   "
        f"fallback users = {fb_b2:,}   runtime = {t_b2:.3f}s")
    rows_table.append(("B2_personal_history", "own training programmes by overlap_seconds, pool-restricted",
                       m5, ma, t_b2, fb_b2))

    # ------------------------------------------------------------------ M1
    ppr_rows = []
    ppr_recs_cache = {}
    ppr_scores_cache = {}
    best = (None, -1.0)
    if not args.skip_ppr:
        hr("M1 - PERSONALIZED PAGERANK (bipartite user-programme graph)")
        log("  edge weight = log1p(seconds/60); restart = own programmes weighted by that weight;")
        log("  R <- alpha * P^T R + V; tol 1e-6 on max column L1 change; max_iter 30; chunk 500 users.")
        user_list = sorted(user_train_set.keys())
        item_list = sorted(set(train_sum["tv_show_id"].astype(int).tolist()))
        user2idx = {u: i for i, u in enumerate(user_list)}
        item2idx = {it: i for i, it in enumerate(item_list)}
        for alpha in alphas:
            t0 = time.perf_counter()
            scores = H.ppr_batched(eval_users, user2idx, item2idx, user_list, item_list,
                                 user_train_list, user_train_set, pool, alpha)
            recs, fb = H.make_recs(scores, pool, pop, user_train_set, cold, 5)
            m5, _ = H.map_at_k(recs, targets_top5, 5)
            ma, _ = H.map_at_k(recs, targets_all, 5)
            dt = time.perf_counter() - t0
            log(f"  alpha={alpha}: MAP@5 (top-5 target) = {m5:.6f}   MAP@5 (all-watched) = {ma:.6f}   "
                f"fallback users = {fb:,}   runtime = {dt:.1f}s")
            ppr_rows.append((f"M1_PPR(alpha={alpha})",
                             "log1p(sec/60) edges, own-programme restart, tol 1e-6, max_iter 30",
                             m5, ma, dt, fb))
            ppr_recs_cache[alpha] = recs
            ppr_scores_cache[alpha] = scores
            if m5 > best[1]:
                best = (alpha, m5)
        if best[0] is not None:
            log(f"  selected alpha = {best[0]} (MAP@5 top-5 target = {best[1]:.6f})")
    rows_table.extend(ppr_rows)

    # ------------------------------------------------------------------ M2
    hr("M2 - CONTENT-BASED (category + genres one-hot, TF-IDF of normalised title)")
    t0 = time.perf_counter()
    m2res, m2pool = M2.run(inter, progs, H.TRAIN_BLOCKS_VAL, H.EVAL_BLOCK_VAL, targets_top5,
                            targets_all, user_train_set, user_train_list, cold, pop, eval_users)
    t_m2 = time.perf_counter() - t0
    for tag, d in m2res.items():
        log(f"  {tag}: MAP@5 (top-5 target) = {d['top5']:.6f}   MAP@5 (all-watched) = {d['allw']:.6f}   "
            f"fallback users = {d['fallback']:,}")
        rows_table.append((tag, "cat+genres one-hot & title TF-IDF, seconds-weighted user profile, cosine",
                           d["top5"], d["allw"], t_m2, d["fallback"]))

    # ------------------------------------------------------------------ M3
    # Hybrid: reciprocal-rank fusion of the three single-method rankings.
    # Weights are a small grid searched on the validation hold-out only.
    if not args.skip_ppr and best[0] is not None:
        hr("M3 - HYBRID (reciprocal-rank fusion: PPR + personal history + popularity)")
        log("  weights (PPR, history, popularity) are selected on the 21-30 hold-out;")
        log("  every component is trained on 11-20 only; the 31-42 schedule is not touched.")
        ppr_scores_best = ppr_scores_cache[best[0]]
        # each component ranking is truncated to its top-CAND_CUT candidates
        CAND_CUT = 200
        hist_rank = {u: [it for it, _ in user_train_list.get(u, []) if it in pool][:CAND_CUT]
                     for u in eval_users}
        pop_rank = [int(it) for it in pop.index if int(it) in pool][:CAND_CUT]
        grid = []
        for w_ppr in (1.0, 2.0):
            for w_hist in (0.0, 1.0, 2.0):
                for w_pop in (0.0, 0.5, 1.0):
                    grid.append((w_ppr, w_hist, w_pop))
        best_h = (None, -1.0, None, None)
        for (wp, wh, wq) in grid:
            t0 = time.perf_counter()
            fused = {}
            for u in eval_users:
                ppr_rank = [it for it, _ in ppr_scores_best.get(u, [])][:200]
                fused[u] = H.rank_fusion([ppr_rank, hist_rank.get(u, []), pop_rank],
                                         [wp, wh, wq], k=5)
            recs_h, fb_h = H.make_recs(fused, pool, pop, user_train_set, cold, 5)
            m5h, _ = H.map_at_k(recs_h, targets_top5, 5)
            mah, _ = H.map_at_k(recs_h, targets_all, 5)
            dt = time.perf_counter() - t0
            log(f"  w=({wp}, {wh}, {wq}): MAP@5 (top-5 target) = {m5h:.6f}  "
                f"MAP@5 (all-watched) = {mah:.6f}  runtime = {dt:.1f}s")
            if m5h > best_h[1]:
                best_h = ((wp, wh, wq), m5h, mah, recs_h, fb_h)
        (wp, wh, wq), m5h, mah, recs_h, fb_h = best_h
        log(f"  selected hybrid weights = (PPR {wp}, history {wh}, popularity {wq}); "
            f"MAP@5 (top-5 target) = {m5h:.6f}")
        rows_table.append((f"M3_hybrid(PPR_a{best[0]},w={wp}/{wh}/{wq})",
                           "RRF(k=60) of PPR + personal history + popularity rankings",
                           m5h, mah, dt, fb_h))
        H.save_val("hybrid", recs_h, pool, pop)
        H.log("  out/val_hybrid.csv")

    # ------------------------------------------------------------- table
    hr("VALIDATION RESULTS (train 11-20 -> eval 21-30, K=5)")
    log("method | params | MAP@5 (top-5 target) | MAP@5 (all-watched) | runtime | fallback users")
    log("--- | --- | --- | --- | --- | ---")
    for name, params, m5, ma, dt, fb in rows_table:
        log(f"{name} | {params} | {m5:.6f} | {ma:.6f} | {dt:.3f}s | {fb}")
    log(f"\nevaluated users: {n_eval:,} (macro-average over users with >= 1 watched programme in {H.EVAL_BLOCK_VAL})")
    if best[0] is not None:
        log(f"selected PPR configuration: alpha={best[0]}, MAP@5 (top-5 target)={best[1]:.6f}")

    # ---------------------------------------------------------- save preds
    hr("SAVING VALIDATION PREDICTIONS")
    H.save_val("popularity", pop_recs, pool, pop)
    H.log("  out/val_popularity.csv")
    H.save_val("personal_history", hist_recs, pool, pop)
    H.log("  out/val_personal_history.csv")
    for alpha in alphas:
        recs = ppr_recs_cache.get(alpha)
        if recs is not None:
            H.save_val(f"ppr_alpha{alpha}", recs, pool, pop)
            H.log(f"  out/val_ppr_alpha{alpha}.csv")
    for tag, d in m2res.items():
        H.save_val(tag.replace("+", "_").replace(" ", "_"), d["recs"], m2pool, pop)
        H.log(f"  out/val_{tag.replace('+', '_').replace(' ', '_')}.csv")

    log(f"\ntotal runtime: {time.perf_counter() - t_script:.1f}s")
    (H.OUT / "step2_results.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"\nWrote {H.OUT / 'step2_results.md'}")


ppr_recs_cache = {}

if __name__ == "__main__":
    main()
