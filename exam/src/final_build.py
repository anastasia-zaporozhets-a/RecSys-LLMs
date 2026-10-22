#!/usr/bin/env python3
"""Shared final-refit pipeline (refit on 11-30, predict weeks 31-42).

Used by src/step3_final.py and src/step3_tune.py so that every submitted file
is produced by exactly the same code path, with the same fallback rules and
the same checks.

Final protocol:
  * training interactions = blocks 11-20 + 21-30 (weeks 11-30);
  * candidate pool = labelled programmes that air in the 31-42 schedule
    (the test schedule is used only to know which programmes air where/when);
  * exactly 5 distinct ids per user: method ranking first, popularity fill
    (popularity computed on weeks 11-30, restricted to the pool), and the
    popularity baseline for cold-start users;
  * repeats (already-watched programmes) are allowed.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

import harness as H

SMOOTH_SECONDS = 3600.0  # +1 h smoothing on the airtime ratio


def airtime_ratio(eval_block: str, train_blocks: list[str], pool: set[int]) -> dict[int, float]:
    """ratio_i = (airtime_eval_i + 1h) / (airtime_train_i + 1h).

    airtime = total labelled seconds of the programme in that block's schedule.
    """
    a_eval = H.load_airtime(eval_block)
    a_train = None
    for b in train_blocks:
        ab = H.load_airtime(b)
        a_train = ab if a_train is None else a_train.add(ab, fill_value=0)
    a_train = a_train.reindex(sorted(pool)).fillna(0.0)
    a_eval = a_eval.reindex(sorted(pool)).fillna(0.0)
    ratio = (a_eval.to_numpy() + SMOOTH_SECONDS) / (a_train.to_numpy() + SMOOTH_SECONDS)
    return {int(pid): float(r) for pid, r in zip(sorted(pool), ratio)}


def apply_beta(scores: list[tuple[int, float]], ratio: dict[int, float],
               beta: float, keep: int = 200) -> list[tuple[int, float]]:
    """Multiply each score by ratio**beta and re-rank (ties by tv_show_id).

    Vectorised; only the top-`keep` adjusted items are returned, which is all
    the ranking downstream can use (5 slots + a popularity fill).
    """
    if not scores:
        return []
    if beta == 0.0:
        return list(scores[:keep])
    ids = np.fromiter((int(p) for p, _ in scores), dtype=np.int64, count=len(scores))
    sc = np.fromiter((float(v) for _, v in scores), dtype=np.float64, count=len(scores))
    r = np.array([ratio.get(int(p), 1.0) for p in ids], dtype=np.float64)
    adj = sc * (r ** beta)
    k = min(keep, len(ids))
    part = np.argpartition(-adj, k - 1)[:k]
    # deterministic order: adjusted score desc, then tv_show_id asc
    order = part[np.lexsort((ids[part], -adj[part]))]
    return [(int(ids[i]), float(adj[i])) for i in order]


def hybrid_ranking(history: list[tuple[int, float]], ppr: list[tuple[int, float]],
                   n_history: int = 5) -> list[tuple[int, float]]:
    """'B2 history first, PPR fills the rest'."""
    out, seen = [], set()
    for pid, sc in history[:n_history]:
        if pid not in seen:
            seen.add(pid)
            out.append((pid, sc))
    for pid, sc in ppr:
        if pid not in seen:
            seen.add(pid)
            out.append((pid, sc))
    return out


def build_final(method: str, alpha: float, beta: float, w_ppr: float = 1.0,
                 w_hist: float = 0.0, w_pop: float = 0.0, n_history: int = 5,
                 log=print, eval_block: str = H.EVAL_BLOCK_FINAL,
                 train_blocks: list[str] | None = None,
                 ppr_scores: dict[str, list[tuple[int, float]]] | None = None,
                 ) -> tuple[dict[str, list[int]], dict]:
    """Refit on weeks 11-30 and produce the top-5 lists for the 1,260 submission users."""
    train_blocks = train_blocks or H.TRAIN_BLOCKS_FINAL
    inter = H.load_interactions()
    progs = H.load_programmes()
    train_sum = H.get_user_train(inter, train_blocks)
    user_train_set, user_train_list = H.build_user_histories(train_sum)
    pool = H.load_pool(eval_block)
    ratio = airtime_ratio(eval_block, train_blocks, pool)
    sub = pd.read_csv(H.DATA / "submission.csv", dtype={"user_id": str})
    sub_users = sub["user_id"].tolist()
    cold = {u for u in sub_users if not user_train_set.get(u)}
    pop = H.build_popularity(train_sum, pool)
    log(f"  training pairs: {len(train_sum):,} over {len(user_train_set):,} users; "
        f"pool: {len(pool):,}; cold-start submission users: {len(cold)}")
    log(f"  airtime ratio: min {min(ratio.values()):.3f} median "
        f"{float(np.median(list(ratio.values()))):.3f} max {max(ratio.values()):.3f}")

    t0 = time.perf_counter()
    if method in ("ppr", "hybrid", "hybrid_rrf") and ppr_scores is None:
        user_list = sorted(user_train_set.keys())
        item_list = sorted(set(train_sum["tv_show_id"].astype(int).tolist()))
        user2idx = {u: i for i, u in enumerate(user_list)}
        item2idx = {it: i for i, it in enumerate(item_list)}
        ppr_scores = H.ppr_batched(sub_users, user2idx, item2idx, user_list, item_list,
                                   user_train_list, user_train_set, pool, alpha)
        log(f"  PPR alpha={alpha} fitted in {time.perf_counter() - t0:.1f}s")

    history = {u: [(it, sc) for it, sc in user_train_list.get(u, []) if it in pool]
               for u in sub_users}
    method_scores = {}
    for u in sub_users:
        if method == "ppr":
            s = apply_beta(ppr_scores.get(u, []), ratio, beta)
        elif method == "history":
            s = apply_beta([(it, sc) for it, sc in history[u]], ratio, beta)
        elif method == "hybrid":
            s = hybrid_ranking(apply_beta(history[u], ratio, beta),
                               apply_beta(ppr_scores.get(u, []), ratio, beta), n_history)
        elif method == "hybrid_rrf":
            CAND = 200
            pop_rank = [int(it) for it in pop.index if int(it) in pool][:CAND]
            s = H.rank_fusion([[it for it, _ in ppr_scores.get(u, [])][:CAND],
                               [it for it, _ in history[u]][:CAND],
                               pop_rank], [w_ppr, w_hist, w_pop], k=5)
        elif method == "popularity":
            s = []
        else:
            raise ValueError(method)
        method_scores[u] = s
    recs, fallback = H.make_recs(method_scores, pool, pop, user_train_set, cold, 5)
    log(f"  fallback users served by the popularity baseline: {fallback}")
    return recs, {"pool": pool, "popularity": pop, "cold": cold, "user_train_set": user_train_set,
                  "user_train_list": user_train_list, "train_sum": train_sum, "progs": progs,
                  "ppr_scores": ppr_scores}


def check_and_write(recs: dict[str, list[int]], out_path: Path, pool: set[int],
                    popularity, sub_path: Path | None = None, log=print) -> dict:
    """Write the Kaggle-format file and run the standard checks."""
    sub = pd.read_csv(H.DATA / "submission.csv", dtype={"user_id": str})
    sub_users = sub["user_id"].tolist()
    rows = []
    for u in sub_users:
        lst = list(dict.fromkeys(int(x) for x in recs.get(u, [])[:5]))
        if len(lst) < 5:
            for it in popularity.index:
                it = int(it)
                if it in lst or it not in pool:
                    continue
                lst.append(it)
                if len(lst) == 5:
                    break
        rows.append((u, lst))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "tv_show_id"])
        for u, lst in rows:
            w.writerow([u, " ".join(str(x) for x in lst[:5])])
    ids_all = [i for _, lst in rows for i in lst]
    checks = {
        "rows": len(rows),
        "template_rows": len(sub_users),
        "five_ids_every_row": all(len(lst) == 5 for _, lst in rows),
        "distinct_ids_every_row": all(len(set(lst)) == 5 for _, lst in rows),
        "all_ids_in_pool": all(i in pool for i in ids_all),
        "order_matches_template": [u for u, _ in rows] == sub_users,
        "zero_ids": sum(1 for _, lst in rows if 0 in lst),
        "distinct_programmes_used": len(set(ids_all)),
        "pool_size": len(pool),
    }
    log(f"  wrote {out_path}")
    for k, v in checks.items():
        log(f"    {k}: {v}")
    assert checks["five_ids_every_row"] and checks["distinct_ids_every_row"]
    assert checks["all_ids_in_pool"] and checks["order_matches_template"]
    assert checks["zero_ids"] == 0
    return checks
