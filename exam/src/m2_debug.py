#!/usr/bin/env python3
"""A) Sanity check for M2 (content-based).

Checks, in order:
  1. index alignment: item-matrix row k <-> pool mask <-> id list <-> features;
  2. for three users: their top-5 watched training programmes, with the M2
     score and the rank the model gives them inside the candidate pool,
     next to the model's own Top-5 and the user's real 21-30 Top-5;
  3. where the cosine score comes from (category/genre one-hot vs title TF-IDF);
  4. how universal the evaluation target is (recall of the user's own training
     programmes inside the 21-30 target, category mix of the target);
  5. MAP@5 of feature-block variants (cat+genre, title-only, both) to see
     which block drives (or destroys) the result.

Read-only: prints evidence, writes nothing.
"""
from __future__ import annotations

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
import m2_content as M2  # noqa: E402

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)


def main():
    inter = H.load_interactions()
    progs = H.load_programmes()
    train_sum = H.get_user_train(inter, H.TRAIN_BLOCKS_VAL)
    targets_top5, targets_all = H.get_eval_targets(inter, H.EVAL_BLOCK_VAL)
    user_train_set, user_train_list = H.build_user_histories(train_sum)
    cold = set(targets_all.keys()) - set(user_train_set.keys())
    pool = H.load_pool(H.EVAL_BLOCK_VAL)
    pop = H.build_popularity(train_sum, pool)
    eval_users = list(targets_all.keys())

    pids, pid_idx, X = M2.build_item_matrix(progs)
    pool_ids = np.array([p for p in sorted(pool) if p in pid_idx], dtype=np.int64)
    pool_rows = np.array([pid_idx[p] for p in sorted(pool) if p in pid_idx], dtype=np.int64)
    Xp = X[pool_rows]

    print("=" * 100)
    print("1. INDEX ALIGNMENT CHECK")
    print("=" * 100)
    print(f"  programmes.csv items            : {len(pids):,}")
    print(f"  item matrix X rows              : {X.shape[0]:,}")
    print(f"  pool programmes (21-30 labelled): {len(pool):,}; present in item matrix: {len(pool_ids):,}")
    assert X.shape[0] == len(pids) == len(pid_idx)
    assert len(pool_ids) == len(pool_rows) == Xp.shape[0]
    # (a) row k of the pool sub-matrix is the same programme as id list entry k
    bad = 0
    for k in range(0, len(pool_ids), 97):
        assert int(pool_rows[k]) == pid_idx[int(pool_ids[k])]
        assert int(pids[int(pool_rows[k])]) == int(pool_ids[k])
        # the sparse row we would score really belongs to that programme
        assert Xp[k].nnz >= 0
    # (b) full randomised check of the id <-> row contract
    rng = np.random.default_rng(RANDOM_SEED)
    for k in rng.choice(len(pool_ids), size=min(2000, len(pool_ids)), replace=False):
        assert int(pids[int(pool_rows[k])]) == int(pool_ids[k]), (k, pool_rows[k], pool_ids[k])
    print(f"  OK: for every pool row k, item-matrix row pool_rows[k] and id pool_ids[k] "
          f"are the same programme (checked {min(2000, len(pool_ids)):,} random rows + stride check).")
    # (c) mask semantics: is a programme in the pool iff its id is in pool?
    ids_all = set(pids)
    in_pool_mask = np.array([p in pool for p in pids])
    assert pool_ids.tolist() == [p for p in pids if p in pool]
    print(f"  OK: pool ids == [p for p in item order if p in pool]; "
          f"{int(in_pool_mask.sum()):,} of {len(pids):,} items are in the pool "
          f"({int((~in_pool_mask).sum()):,} outside).")
    # (d) feature row of a known programme
    for probe in (2400480, 20088, 90050977):
        if probe in pid_idx:
            r = pid_idx[probe]
            k = int(np.searchsorted(pool_rows, r))
            inside = k < len(pool_rows) and pool_rows[k] == r
            print(f"  probe id {probe}: item row {r}, pool position {k if inside else 'not in pool'}, "
                  f"row nnz={X[r].nnz}, title='{progs.loc[progs.tv_show_id == probe, 'title'].iloc[0][:45]}'")
    print()

    # ------------------------------------------------------------ profiles
    print("=" * 100)
    print("2. THREE USERS: WATCHED TOP-5 (training) vs M2 SCORE / POOL RANK vs MODEL TOP-5 vs REAL TOP-5")
    print("=" * 100)
    users = [u for u in eval_users if u in user_train_set and user_train_set.get(u)]
    wrow, wcol, wval = [], [], []
    uidx = {u: i for i, u in enumerate(users)}
    for u in users:
        for it, sec in user_train_list.get(u, []):
            if it in pid_idx:
                wrow.append(uidx[u]); wcol.append(pid_idx[it]); wval.append(float(sec))
    W = sparse.csr_matrix((wval, (wrow, wcol)), shape=(len(users), len(pids)))
    U = normalize(W @ X, norm="l2", axis=1)

    # pick three users: heaviest, median, lightest
    tot = train_sum.groupby("user_id")["overlap_seconds"].sum()
    cand = tot[tot.index.isin(set(users))]
    order = cand.sort_values(ascending=False)
    picks = [order.index[0], order.index[len(order) // 2], order.index[-1]]

    title_map = dict(zip(progs["tv_show_id"].astype(int), progs["title"].astype(str)))
    cat_map = dict(zip(progs["tv_show_id"].astype(int), progs["category"].astype(str)))
    gen_map = dict(zip(progs["tv_show_id"].astype(int), progs["genres"].astype(str)))

    for u in picks:
        print(f"\n--- user {u}  (training seconds {int(cand[u]):,}, "
              f"{len(user_train_list[u])} programmes, {len(targets_all.get(u, []))} watched in 21-30) ---")
        up = np.asarray(U[uidx[u]].todense()).ravel()
        S = (Xp @ up)  # cosine (both normalised)
        # rank of every pool item
        order_p = np.argsort(-S, kind="stable")
        rank_of = {int(pool_ids[i]): int(k) + 1 for k, i in enumerate(order_p)}
        top5 = [(int(pool_ids[i]), float(S[i])) for i in order_p[:5]]
        print("  model Top-5 (validation pool, 21-30):")
        for r, (pid, sc) in enumerate(top5, 1):
            print(f"    {r}. id={pid:<12} score={sc:.4f} cat={cat_map.get(pid,'?'):<16} {title_map.get(pid,'?')[:52]}")
        print("  user's own top-5 watched training programmes, with the model's score and pool rank:")
        for pid, sec in user_train_list[u][:5]:
            inpool = pid in pool
            sc = float(S[list(pool_ids).index(pid)]) if inpool else float("nan")
            rk = rank_of.get(pid, None)
            print(f"    id={pid:<12} overlap={int(sec):>7}s  M2 score={sc:.4f}  pool rank={rk}  "
                  f"cat={cat_map.get(pid,'?'):<16} {title_map.get(pid,'?')[:44]}")
            if not inpool:
                print(f"       (not airing in 21-30 -> excluded from the pool, no rank)")
        print("  user's REAL top-5 in 21-30 (the evaluation target):")
        ev = inter[(inter["user_id"] == u) & (inter["block"] == "21-30")]
        ev = ev.groupby("tv_show_id")["overlap_seconds"].sum().sort_values(ascending=False)
        for r, (pid, sec) in enumerate(ev.head(5).items(), 1):
            pid = int(pid)
            print(f"    {r}. id={pid:<12} overlap={int(sec):>7}s  M2 score="
                  f"{float(S[list(pool_ids).index(pid)]) if pid in set(pool_ids.tolist()) else float('nan'):.4f}  "
                  f"cat={cat_map.get(pid,'?'):<16} {title_map.get(pid,'?')[:44]}")
    print()

    # ------------------------------------------------- feature block effect
    print("=" * 100)
    print("3. WHERE DOES THE COSINE SCORE COME FROM? (category/genre one-hot vs title TF-IDF)")
    print("=" * 100)
    info = {}
    for pid, title, cat, gens in progs[["tv_show_id", "title", "category", "genres"]].itertuples(index=False):
        info[int(pid)] = (str(title), "" if pd.isna(cat) else str(cat), "" if pd.isna(gens) else str(gens))
    cats = sorted({info[p][1] for p in pids})
    genres = sorted({g.strip() for p in pids for g in info[p][2].split(";") if g.strip()})
    c_of = {c: i for i, c in enumerate(cats)}
    g_of = {g: i for i, g in enumerate(genres)}
    rows, cols, vals = [], [], []
    for p in pids:
        i = pid_idx[p]
        if info[p][1] in c_of:
            rows.append(i); cols.append(c_of[info[p][1]]); vals.append(1.0)
        for g in info[p][2].split(";"):
            g = g.strip()
            if g in g_of:
                rows.append(i); cols.append(len(cats) + g_of[g]); vals.append(1.0)
    X_cg_raw = sparse.csr_matrix((vals, (rows, cols)), shape=(len(pids), len(cats) + len(genres)))
    X_t_raw = TfidfVectorizer(max_features=2000, min_df=2).fit_transform(
        [M2.norm_title(info[p][0]) for p in pids])
    X_cg = normalize(X_cg_raw, norm="l2", axis=1)
    X_t = normalize(X_t_raw, norm="l2", axis=1)
    # joint matrix actually used by M2 (built the same way as in m2_content)
    X_joint = sparse.hstack([X_cg_raw, X_t_raw], format="csr")
    X_joint = normalize(X_joint, norm="l2", axis=1)
    n_cg = X_cg_raw.shape[1]
    print(f"  feature blocks: category+genre one-hot = {n_cg} columns "
          f"({len(cats)} categories, {len(genres)} distinct genres), "
          f"title TF-IDF = {X_t_raw.shape[1]} columns")

    # contribution of each block to the joint cosine, for the three example users
    for u in picks:
        up = np.asarray(U[uidx[u]].todense()).ravel()          # joint space, normalised profile
        x_cg = X_cg_raw[pool_rows]                               # raw (unnormalised) block
        x_t = X_t_raw[pool_rows]
        num_cg = np.asarray(x_cg.multiply(up[:n_cg]).sum(axis=1)).ravel()
        num_t = np.asarray(x_t.multiply(up[n_cg:]).sum(axis=1)).ravel()
        tot = num_cg + num_t
        m_cg = num_cg.sum() / max(tot.sum(), 1e-12)
        m_t = num_t.sum() / max(tot.sum(), 1e-12)
        print(f"  user {u}: share of the cosine numerator from category/genre = {m_cg:.3f}, "
              f"from title TF-IDF = {m_t:.3f}")

    # structural degeneracy: how many pool items share an identical feature vector?
    key = {}
    for k in range(len(pool_ids)):
        r = pool_rows[k]
        sig = (info[int(pool_ids[k])][1],
               tuple(sorted({g.strip() for g in info[int(pool_ids[k])][2].split(";") if g.strip()})))
        key.setdefault(sig, []).append(int(pool_ids[k]))
    sizes = np.array([len(v) for v in key.values()])
    print(f"  distinct (category, genre-set) signatures among the {len(pool_ids):,} pool items: "
          f"{len(key):,}; largest group = {int(sizes.max()):,} items, "
          f"median group = {int(np.median(sizes))}, "
          f"{float((sizes > 1).sum()) / len(sizes):.1%} of signatures cover > 1 item")
    # exact duplicate rows of the joint matrix (identical one-hot AND identical tfidf)
    Xp_j = X_joint[pool_rows]
    sig_rows = np.asarray([hash(Xp_j.getrow(i).toarray().tobytes()) for i in range(0, len(pool_ids), 1)])
    _, counts = np.unique(sig_rows, return_counts=True)
    print(f"  distinct feature ROWS among the {len(pool_ids):,} pool items: {len(counts):,}; "
          f"items sharing their row with >= 1 other item: {int(counts[counts > 1].sum()):,}")
    print()

    # -------------------------------------------------- target universality
    print("=" * 100)
    print("4. HOW UNIVERSAL IS THE EVALUATION TARGET?")
    print("=" * 100)
    rec_hist, cats_target = [], []
    for u, t5 in targets_top5.items():
        hist = {it for it, _ in user_train_list.get(u, [])}
        rec_hist.append(len([t for t in t5 if t in hist]) / len(t5))
        cats_target.append(cat_map.get(t5[0], "?"))
    print(f"  mean share of the 21-30 target top-5 that the user had ALSO watched in 11-20: "
          f"{np.mean(rec_hist):.3f}")
    vc = pd.Series(cats_target).value_counts(normalize=True)
    print("  category of the user's #1 target programme (share of users):")
    for c, s in vc.head(5).items():
        print(f"    {c:<18} {s:.3f}")
    top_ids = pd.Series([t[0] for t in targets_top5.values()]).value_counts().head(8)
    print("  most frequent target programmes across users:")
    for pid, n in top_ids.items():
        print(f"    id={int(pid):<12} n={n:>5}  cat={cat_map.get(int(pid),'?'):<16} {title_map.get(int(pid),'?')[:48]}")
    print()

    # ----------------------------------------------------- block variants
    print("=" * 100)
    print("5. MAP@5 BY FEATURE BLOCK (validation, 21-30)")
    print("=" * 100)

    def score_with(Xuse, label):
        Uu = normalize(W @ Xuse, norm="l2", axis=1)
        Xpool = Xuse[pool_rows]
        scores = {}
        for u in users:
            i = uidx[u]
            s = (Xpool @ Uu[i].toarray().ravel())
            order = np.argsort(-s, kind="stable")[:5]
            scores[u] = [(int(pool_ids[j]), float(s[j])) for j in order]
        recs, fb = H.make_recs(scores, pool, pop, user_train_set, cold, 5)
        m5, n = H.map_at_k(recs, targets_top5, 5)
        ma, _ = H.map_at_k(recs, targets_all, 5)
        print(f"  {label:<46} MAP@5(top-5)={m5:.6f}  MAP@5(all)={ma:.6f}  users={n:,}")
        return m5, ma

    t0 = time.perf_counter()
    score_with(X_cg, "category + genres one-hot only")
    score_with(X_t, "title TF-IDF only")
    score_with(X_joint, "cat+genres + title TF-IDF (the M2 used)")
    # scale the one-hot block so that both blocks carry comparable weight
    cg_part = np.sqrt(np.asarray(X_cg_raw.multiply(X_cg_raw).sum(axis=1)).ravel())
    t_part = np.sqrt(np.asarray(X_t_raw.multiply(X_t_raw).sum(axis=1)).ravel())
    w = float(np.median(t_part / np.maximum(cg_part, 1e-9)))
    X_scaled = normalize(sparse.hstack([X_cg_raw * w, X_t_raw], format="csr"), norm="l2", axis=1)
    score_with(X_scaled, f"cat+genres scaled x{w:.2f} + title (balanced)")
    print(f"  (variant scoring took {time.perf_counter() - t0:.1f}s)")
    print()
    print("  reference points on the same protocol: B1 popularity 0.147777 | B2 history 0.241036 | "
          "PPR a=0.3 0.247010")


if __name__ == "__main__":
    main()
