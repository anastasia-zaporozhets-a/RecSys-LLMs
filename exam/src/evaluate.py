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


def ppr_batched(eval_users, user2idx, item2idx, user_list, item_list,
                user_train_list, user_train_set, pool, alpha,
                chunk_size=500, max_iter=30, tol=1e-6):
    """Personalized PageRank on the user-programme bipartite graph.

    Edge weight w(u,i) = log1p(seconds/60).  Restart vector = the user's own
    programmes weighted by that edge weight (normalised), scaled by (1-alpha).
    Power iteration: R <- alpha * P^T R + V, chunked over users.
    """
    t0 = time.perf_counter()
    n_u, n_i = len(user_list), len(item_list)
    n = n_u + n_i
    rows, cols, data = [], [], []
    for uid, items in user_train_list.items():
        if uid not in user2idx:
            continue
        for it, sec in items:
            if it not in item2idx:
                continue
            w = np.log1p(float(sec) / 60.0)
            u, i = user2idx[uid], item2idx[it]
            rows.append(u); cols.append(n_u + i); data.append(w)
            rows.append(n_u + i); cols.append(u); data.append(w)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    deg = np.array(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    P = (sparse.diags(1.0 / deg) @ A).tocsr()
    PT = P.T.tocsr()
    users = [u for u in eval_users if u in user2idx and user_train_set.get(u)]
    log(f"  PPR alpha={alpha}: graph n={n} (users={n_u}, items={n_i}), "
        f"scored users={len(users):,}, chunk={chunk_size}, max_iter={max_iter}, tol={tol}")
    out = {}
    for c0 in range(0, len(users), chunk_size):
        c1 = min(c0 + chunk_size, len(users))
        chunk = users[c0:c1]
        tc = time.perf_counter()
        V = np.zeros((n, len(chunk)))
        for j, uid in enumerate(chunk):
            s = sum(np.log1p(float(sec) / 60.0) for it, sec in user_train_list[uid] if it in item2idx)
            if s == 0:
                V[user2idx[uid], j] = 1.0
            else:
                for it, sec in user_train_list[uid]:
                    if it in item2idx:
                        V[n_u + item2idx[it], j] += np.log1p(float(sec) / 60.0) / s
        V *= (1.0 - alpha)
        R = V.copy()
        for it in range(max_iter):
            R_new = alpha * (PT @ R) + V
            diff = np.max(np.sum(np.abs(R_new - R), axis=0))
            R = R_new
            if diff < tol:
                log(f"    chunk {c0 // chunk_size + 1} ({len(chunk)} users) converged at iter {it + 1}, "
                    f"diff={diff:.2e}, {time.perf_counter() - tc:.1f}s")
                break
        else:
            log(f"    chunk {c0 // chunk_size + 1} ({len(chunk)} users) finished {max_iter} iters, "
                f"{time.perf_counter() - tc:.1f}s")
        items_scores = R[n_u:n_u + n_i, :]
        for j, uid in enumerate(chunk):
            col = items_scores[:, j]
            scored = [(int(it), float(col[k])) for k, it in enumerate(item_list) if int(it) in pool]
            scored.sort(key=lambda x: (-x[1], x[0]))
            out[uid] = scored
    log(f"  PPR alpha={alpha} runtime: {time.perf_counter() - t0:.1f}s")
    return out


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
            scores = ppr_batched(eval_users, user2idx, item2idx, user_list, item_list,
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
