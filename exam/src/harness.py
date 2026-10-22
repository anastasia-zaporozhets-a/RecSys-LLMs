#!/usr/bin/env python3
"""Shared evaluation harness for the sweet.tv exam (step 2).

Protocol (identical for every method):
  * validation: fit on block 11-20 interactions, evaluate on 21-30;
    final: fit on 11-20 + 21-30, predict 31-42.
  * target per user = the user's top-5 programmes in the evaluation block ranked
    by summed overlap_seconds (ties by tv_show_id).
  * MAP@5 = Kaggle definition, macro-averaged over users with >= 1 watched
    programme in the evaluation block.
  * candidate pool = labelled programmes airing in the evaluation block schedule.
  * repeats are allowed; every list has exactly 5 distinct ids: method ranking
    first, then filled from the popularity baseline; cold-start users get the
    popularity baseline (same fallback everywhere).
  * user_id is kept as a STRING everywhere (ids exceed 2**63).
"""
from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

EXAM = Path(__file__).resolve().parents[1]
DATA = EXAM / "data"
OUT = EXAM / "out"

TRAIN_BLOCKS_VAL = ["11-20"]
EVAL_BLOCK_VAL = "21-30"
TRAIN_BLOCKS_FINAL = ["11-20", "21-30"]
EVAL_BLOCK_FINAL = "31-42"

EXPECTED_EVAL_USERS = 4799   # users with >= 1 watched programme in 21-30
EXPECTED_TRAIN_USERS = 4834  # users with >= 1 watched programme in 11-20
EXPECTED_SUB_USERS = 1260


def log(msg="", flush=True):
    print(msg, flush=flush)


def load_interactions():
    df = pd.read_csv(OUT / "interactions.csv",
                     dtype={"user_id": str, "tv_show_id": str, "block": str})
    df["tv_show_id"] = df["tv_show_id"].astype(np.int64)
    df["overlap_seconds"] = df["overlap_seconds"].astype(np.int64)
    df["airings"] = df["airings"].astype(np.int64)
    return df


def load_programmes():
    return pd.read_csv(OUT / "programmes.csv", dtype={"tv_show_id": "int64"})


def load_pool(block):
    """Labelled programmes airing in the given schedule block (pool)."""
    pool = set()
    with open(DATA / f"export_arh_{block}-final.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["tv_show_id"] != "0":
                pool.add(int(row["tv_show_id"]))
    return pool


def load_airtime(block):
    """Total scheduled seconds per programme in a block (labelled rows only)."""
    df = pd.read_csv(DATA / f"export_arh_{block}-final.csv",
                     usecols=["tv_show_id", "duration"], dtype={"tv_show_id": "int64"})
    df = df[df["tv_show_id"] != 0]
    return df.groupby("tv_show_id")["duration"].sum().astype(np.int64)


def get_user_train(inter_df, train_blocks):
    train = inter_df[inter_df["block"].isin(train_blocks)].copy()
    s = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
    return s.sort_values(["user_id", "overlap_seconds", "tv_show_id"],
                          ascending=[True, False, True])


def get_eval_targets(inter_df, eval_block):
    """top-5 target set and full watched set per user in the evaluation block."""
    ev = inter_df[inter_df["block"] == eval_block]
    g = ev.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
    g = g.sort_values(["user_id", "overlap_seconds", "tv_show_id"],
                      ascending=[True, False, True])
    top5, allw = {}, {}
    for uid, grp in g.groupby("user_id"):
        allw[uid] = set(grp["tv_show_id"].tolist())
        top5[uid] = grp["tv_show_id"].tolist()[:5]
    return top5, allw


def ap_at_k(predicted, actual_set, k=5):
    if not actual_set:
        return 0.0
    score, hits = 0.0, 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual_set:
            hits += 1.0
            score += hits / (i + 1.0)
    return score / min(len(actual_set), k)


def map_at_k(pred_lists, targets, k=5):
    users = [u for u in pred_lists if u in targets and len(targets[u]) >= 1]
    if not users:
        return 0.0, 0
    return sum(ap_at_k(pred_lists[u], targets[u], k) for u in users) / len(users), len(users)


def build_popularity(train_sum, pool):
    """B1: number of distinct training users per programme, restricted to the pool."""
    per_item = train_sum.groupby("tv_show_id")["user_id"].nunique()
    return per_item[per_item.index.isin(pool)].sort_values(ascending=False)


def build_user_histories(train_sum):
    sets, lists = defaultdict(set), defaultdict(list)
    for uid, it, sec in train_sum[["user_id", "tv_show_id", "overlap_seconds"]].itertuples(index=False):
        sets[uid].add(int(it))
        lists[uid].append((int(it), float(sec)))
    for uid in lists:
        lists[uid].sort(key=lambda x: (-x[1], x[0]))
    return sets, lists


def make_recs(method_scores, pool, popularity_ordered, user_train_set, cold_start_users, k=5):
    """Method ranking first, popularity fill, popularity for cold-start users."""
    recs, fallback = {}, 0
    for uid in set(method_scores.keys()) | set(cold_start_users):
        if uid in cold_start_users or not user_train_set.get(uid):
            lst = [int(it) for it in popularity_ordered.index if int(it) in pool][:k]
            recs[uid] = lst
            fallback += 1
            continue
        ranked, seen = [], set()
        for it, _ in method_scores.get(uid, []):
            it = int(it)
            if it in seen:
                continue
            seen.add(it)
            ranked.append(it)
            if len(ranked) == k:
                break
        if len(ranked) < k:
            for it in popularity_ordered.index:
                it = int(it)
                if it in seen:
                    continue
                seen.add(it)
                ranked.append(it)
                if len(ranked) == k:
                    break
        recs[uid] = ranked[:k]
    return recs, fallback


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


def rank_fusion(user_rankings, weights, k=5, rrf_k=60.0):
    """Reciprocal-rank fusion of several per-user ranked id lists.

    user_rankings: list of lists of item ids (best first), same order as weights.
    Returns the fused top-k as [(item, fused_score)].
    """
    fused = {}
    for lst, w in zip(user_rankings, weights):
        if w == 0:
            continue
        for rank, it in enumerate(lst):
            fused[it] = fused.get(it, 0.0) + w / (rrf_k + rank + 1.0)
    ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:k]


def save_val(name, recs, pool, popularity_ordered):
    path = OUT / f"val_{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id"] + [f"tv_show_id_{i}" for i in range(1, 6)])
        for uid in sorted(recs.keys()):
            lst = list(dict.fromkeys(int(x) for x in recs[uid][:5]))
            if len(lst) < 5:
                for it in popularity_ordered.index:
                    it = int(it)
                    if it in lst or it not in pool:
                        continue
                    lst.append(it)
                    if len(lst) == 5:
                        break
            w.writerow([uid] + lst[:5])
    return path
