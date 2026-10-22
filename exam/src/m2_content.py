#!/usr/bin/env python3
"""M2: content-based recommender (fully vectorised) + its share of the harness.

Item features (sparse, rows L2-normalised):
  * one-hot tv_show_category
  * one-hot of the three genre fields (union of all genre values per programme)
  * TF-IDF of the normalised title (episode/season suffixes stripped)

User profile: U = W @ X with W[user, item] = training overlap seconds,
rows L2-normalised.  Score = cosine similarity U[u] . X_pool[i].

Variants: M2 alone, and M2 with the user's own history as a tie-breaker
(+1e-3 * normalised seconds to already-watched items).

Run standalone:  .venv/bin/python src/m2_content.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

CHUNK = 500
TIE_BONUS = 1e-3


def norm_title(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"[,.:;!?()\[\]\"']", " ", t)
    t = re.sub(r"\b(sezon|season|s\d+|эп\.?|серия|series)\b.*$", " ", t)
    t = re.sub(r"\d+\s*эп.*$", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_item_matrix(progs: pd.DataFrame):
    info = {}
    for pid, title, cat, gens in progs[["tv_show_id", "title", "category", "genres"]].itertuples(index=False):
        info[int(pid)] = (str(title), "" if pd.isna(cat) else str(cat),
                          "" if pd.isna(gens) else str(gens))
    pids = sorted(info.keys())
    idx = {p: i for i, p in enumerate(pids)}
    cats = sorted({info[p][1] for p in pids})
    genres = sorted({g.strip() for p in pids for g in info[p][2].split(";") if g.strip()})
    c_of = {c: i for i, c in enumerate(cats)}
    g_of = {g: i for i, g in enumerate(genres)}
    off = len(cats)
    rows, cols, vals = [], [], []
    for p in pids:
        i = idx[p]
        if info[p][1] in c_of:
            rows.append(i); cols.append(c_of[info[p][1]]); vals.append(1.0)
        for g in info[p][2].split(";"):
            g = g.strip()
            if g in g_of:
                rows.append(i); cols.append(off + g_of[g]); vals.append(1.0)
    X_cg = sparse.csr_matrix((vals, (rows, cols)), shape=(len(pids), off + len(genres)))
    X_t = TfidfVectorizer(max_features=2000, min_df=2).fit_transform(
        [norm_title(info[p][0]) for p in pids])
    X = normalize(sparse.hstack([X_cg, X_t], format="csr"), norm="l2", axis=1)
    return pids, idx, X


def run(inter, progs, train_blocks, eval_block, targets_top5, targets_all,
        user_train_set, user_train_list, cold_start, popularity_ordered, eval_users,
        chunk=CHUNK):
    pool = H.load_pool(eval_block)
    pids, pid_idx, X = build_item_matrix(progs)
    pool_rows = np.array([pid_idx[p] for p in sorted(pool) if p in pid_idx], dtype=np.int64)
    pool_ids = np.array([p for p in sorted(pool) if p in pid_idx], dtype=np.int64)
    Xp = X[pool_rows]
    H.log(f"  items={len(pids):,}, features={X.shape[1]:,}, pool={len(pool_ids):,}")

    users = [u for u in eval_users if u in user_train_set and user_train_set.get(u)]
    wrow, wcol, wval = [], [], []
    uidx = {u: i for i, u in enumerate(users)}
    for u in users:
        for it, sec in user_train_list.get(u, []):
            if it in pid_idx:
                wrow.append(uidx[u]); wcol.append(pid_idx[it]); wval.append(float(sec))
    W = sparse.csr_matrix((wval, (wrow, wcol)), shape=(len(users), len(pids)))
    U = normalize(W @ X, norm="l2", axis=1)
    H.log(f"  user profiles: {U.shape[0]:,} x {U.shape[1]:,}")

    def top5(scores):
        k = min(5, scores.shape[0])
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part], kind="stable")]
        return [(int(pool_ids[i]), float(scores[i])) for i in order]

    def score_all(tiebreak: bool):
        t0 = time.perf_counter()
        out = {}
        for c0 in range(0, len(users), chunk):
            c1 = min(c0 + chunk, len(users))
            H.log(f"    chunk {c0 // chunk + 1} ({c1 - c0} users), elapsed {time.perf_counter() - t0:.1f}s")
            S = (U[c0:c1] @ Xp.T).toarray()
            for j, u in enumerate(users[c0:c1]):
                row = S[j].copy()
                if tiebreak:
                    hist = user_train_list.get(u, [])
                    tot = sum(s for _, s in hist) or 1.0
                    for it, sec in hist:
                        r = pid_idx.get(it)
                        if r is not None:
                            k = int(np.searchsorted(pool_rows, r)) if pool_rows.size else -1
                            if k >= 0 and k < len(pool_rows) and pool_rows[k] == r:
                                row[k] += TIE_BONUS * (sec / tot)
                out[u] = top5(row)
        H.log(f"    scoring done in {time.perf_counter() - t0:.1f}s")
        return out

    res = {}
    for tag, tb in (("M2_content", False), ("M2_content+hist_tiebreak", True)):
        scores = score_all(tb)
        recs, fb = H.make_recs(scores, pool, popularity_ordered, user_train_set, cold_start, 5)
        m5, n = H.map_at_k(recs, targets_top5, 5)
        ma, _ = H.map_at_k(recs, targets_all, 5)
        res[tag] = dict(top5=m5, allw=ma, fallback=fb, recs=recs)
        H.log(f"  {tag}: MAP@5(top-5 target)={m5:.6f}  MAP@5(all-watched)={ma:.6f}  fallback={fb}")
    return res, pool


def main():
    H.log("M2 content-based (standalone)")
    inter = H.load_interactions()
    progs = H.load_programmes()
    train_sum = H.get_user_train(inter, H.TRAIN_BLOCKS_VAL)
    top5, allw = H.get_eval_targets(inter, H.EVAL_BLOCK_VAL)
    sets, lists = H.build_user_histories(train_sum)
    cold = set(allw.keys()) - set(sets.keys())
    pop = H.build_popularity(train_sum, H.load_pool(H.EVAL_BLOCK_VAL))
    res, pool = run(inter, progs, H.TRAIN_BLOCKS_VAL, H.EVAL_BLOCK_VAL, top5, allw,
                    sets, lists, cold, pop, list(allw.keys()))
    for tag, d in res.items():
        p = H.save_val(tag.replace("+", "_").replace(" ", "_"), d["recs"], pool, pop)
        H.log(f"  wrote {p}")


if __name__ == "__main__":
    main()
