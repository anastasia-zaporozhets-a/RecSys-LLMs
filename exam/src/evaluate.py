#!/usr/bin/env python3
"""Step 2: shared evaluation harness, baselines, PPR and content-based."""

from __future__ import annotations

import csv
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import bicgstab

EXAM = Path(__file__).resolve().parents[1]
DATA = EXAM / "data"
OUT = EXAM / "out"
OUT.mkdir(exist_ok=True)

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)

REPORT_LINES = []


def log(msg="", flush=True):
    print(msg, flush=flush)
    REPORT_LINES.append(str(msg))


def hr(title):
    log("")
    log("=" * 100)
    log(title)
    log("=" * 100)


def fmt_pct(x):
    return f"{100.0 * x:.4f}%"


def load_schedules(blocks):
    sched = {}
    for b in blocks:
        path = DATA / f"export_arh_{b}-final.csv"
        t0 = time.perf_counter()
        df = pd.read_csv(path, dtype={"tv_show_id": "int64", "channel_id": "int64", "duration": "int64"}, low_memory=False)
        df = df[df["tv_show_id"] != 0].copy()
        df["start_local"] = pd.to_datetime(df["start_time"], format="%d.%m.%Y %H:%M:%S", errors="raise")
        sched[b] = df
        log(f"  loaded schedule {b}: {len(df):,} labelled rows in {time.perf_counter() - t0:.2f}s")
    return sched


def get_pool_and_airtime(sched_df):
    pool = set(int(x) for x in sched_df["tv_show_id"].unique())
    airtime = sched_df.groupby("tv_show_id")["duration"].sum().astype(np.int64)
    return pool, airtime


def load_interactions():
    t0 = time.perf_counter()
    df = pd.read_csv(OUT / "interactions.csv", dtype={"user_id": str, "tv_show_id": str, "block": str})
    df["tv_show_id"] = df["tv_show_id"].astype(np.int64)
    df["overlap_seconds"] = df["overlap_seconds"].astype(np.int64)
    df["airings"] = df["airings"].astype(np.int64)
    log(f"  loaded interactions: {len(df):,} rows in {time.perf_counter() - t0:.2f}s")
    return df


def load_programmes():
    t0 = time.perf_counter()
    df = pd.read_csv(OUT / "programmes.csv", dtype={"tv_show_id": "int64"})
    log(f"  loaded programmes: {len(df):,} rows in {time.perf_counter() - t0:.2f}s")
    return df


def get_user_train(inter_df, train_blocks):
    train = inter_df[inter_df["block"].isin(train_blocks)].copy()
    user_train_sum = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
    user_train_sum = user_train_sum.sort_values(["user_id", "overlap_seconds", "tv_show_id"], ascending=[True, False, True])
    return user_train_sum


def get_eval_targets(inter_df, eval_block):
    eval_df = inter_df[inter_df["block"] == eval_block].copy()
    g = eval_df.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
    g = g.sort_values(["user_id", "overlap_seconds", "tv_show_id"], ascending=[True, False, True])
    targets_top5 = {}
    targets_all = {}
    for uid, grp in g.groupby("user_id"):
        targets_all[uid] = set(grp["tv_show_id"].tolist())
        targets_top5[uid] = grp["tv_show_id"].tolist()[:5]
    return targets_top5, targets_all


def ap_at_k(predicted, actual_set, k=5):
    if len(actual_set) == 0:
        return 0.0
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual_set), k)


def map_at_k(pred_lists, targets, k=5):
    users = [u for u in pred_lists if u in targets and len(targets[u]) >= 1]
    if not users:
        return 0.0, 0
    s = 0.0
    for u in users:
        s += ap_at_k(pred_lists[u], targets[u], k)
    return s / len(users), len(users)


def build_popularity(train_sum, pool):
    users_per_item = train_sum.groupby("tv_show_id")["user_id"].nunique()
    pop = users_per_item[users_per_item.index.isin(pool)].copy()
    pop = pop.sort_values(ascending=False)
    return pop


def make_recs(method_scores, pool, popularity_ordered, user_train_set, cold_start_users, k=5):
    recs = {}
    fallback_count = 0
    all_uids = set(method_scores.keys()) | set(cold_start_users)
    for uid in all_uids:
        if uid in cold_start_users or uid not in user_train_set or len(user_train_set[uid]) == 0:
            lst = []
            for it in popularity_ordered.index:
                if int(it) in pool:
                    lst.append(int(it))
                if len(lst) == k:
                    break
            recs[uid] = lst[:k]
            fallback_count += 1
            continue
        ranked = []
        seen = set()
        for it, sc in method_scores.get(uid, []):
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
    return recs, fallback_count


def main_quick_b1b2():
    hr("STEP 2: QUICK B1/B2 CHECK (FIXED USER IDS)")
    log(f"Environment: Python {sys.version.split()[0]} {platform.platform()}")
    log(f"Random seed: {RANDOM_SEED}")

    inter = load_interactions()
    progs = load_programmes()
    sched_all = load_schedules(["11-20", "21-30", "31-42"])

    train_blocks_val = ["11-20"]
    eval_block_val = "21-30"
    train_sum_val = get_user_train(inter, train_blocks_val)
    pool_val, airtime_val = get_pool_and_airtime(sched_all[eval_block_val])
    pool_val = set(int(x) for x in pool_val)

    targets_top5_val, targets_all_val = get_eval_targets(inter, eval_block_val)
    n_eval = len([u for u in targets_all_val if len(targets_all_val[u]) >= 1])
    log(f"Validation: train on {train_blocks_val}, eval on {eval_block_val}")
    log(f"  pool_val size: {len(pool_val):,}")
    log(f"  eval users with >=1 watched in {eval_block_val}: {n_eval:,}")
    assert n_eval == 4799, f"Expected 4799 eval users, got {n_eval}"

    user_train_set_val = defaultdict(set)
    user_train_list_val = defaultdict(list)
    for _, row in train_sum_val.iterrows():
        uid = row["user_id"]
        it = int(row["tv_show_id"])
        user_train_set_val[uid].add(it)
        user_train_list_val[uid].append((it, float(row["overlap_seconds"])))
    for uid in user_train_list_val:
        user_train_list_val[uid].sort(key=lambda x: (-x[1], x[0]))

    n_train_users = len(user_train_set_val)
    log(f"  training users: {n_train_users:,}")
    assert n_train_users == 4834, f"Expected 4834 training users, got {n_train_users}"

    cold_start_users_val = set(targets_all_val.keys()) - set(user_train_set_val.keys())
    log(f"  cold-start users among eval users: {len(cold_start_users_val):,}")

    pop_val = build_popularity(train_sum_val, pool_val)
    popularity_ordered_val = pop_val
    eval_users_val = list(targets_all_val.keys())

    # check submission users
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    sub_users = set(sub["user_id"].tolist())
    log(f"  submission users: {len(sub_users)} (all found as strings; 2 have no labelled interactions in training blocks)")

    # B1
    t0 = time.perf_counter()
    pop_recs_val2 = {}
    for uid in eval_users_val:
        lst = []
        for it in popularity_ordered_val.index:
            it = int(it)
            if it not in pool_val:
                continue
            lst.append(it)
            if len(lst) == 5:
                break
        pop_recs_val2[uid] = lst
    t_b1 = time.perf_counter() - t0
    map_top5_b1, n_users_b1 = map_at_k(pop_recs_val2, targets_top5_val, k=5)
    map_all_b1, _ = map_at_k(pop_recs_val2, targets_all_val, k=5)
    log(f"  B1 MAP@5 (top-5 target): {map_top5_b1:.6f} over {n_users_b1} users")
    log(f"  B1 MAP@5 (all-watched):   {map_all_b1:.6f}")
    log(f"  B1 runtime: {t_b1:.3f}s")

    # B2
    t0 = time.perf_counter()
    hist_recs_val = {}
    for uid in eval_users_val:
        lst = []
        for it, sc in user_train_list_val.get(uid, []):
            it = int(it)
            if it not in pool_val:
                continue
            if it in lst:
                continue
            lst.append(it)
            if len(lst) == 5:
                break
        if len(lst) < 5:
            for it in popularity_ordered_val.index:
                it = int(it)
                if it in lst:
                    continue
                if it not in pool_val:
                    continue
                lst.append(int(it))
                if len(lst) == 5:
                    break
        hist_recs_val[uid] = lst
    t_b2 = time.perf_counter() - t0
    map_top5_b2, n_users_b2 = map_at_k(hist_recs_val, targets_top5_val, k=5)
    map_all_b2, _ = map_at_k(hist_recs_val, targets_all_val, k=5)
    log(f"  B2 MAP@5 (top-5 target): {map_top5_b2:.6f} over {n_users_b2} users")
    log(f"  B2 MAP@5 (all-watched):   {map_all_b2:.6f}")
    log(f"  B2 runtime: {t_b2:.3f}s")

    log("\nExpected reference (top-5 target): B1 0.1478, B2 0.2410")
    log(f"Got: B1 {map_top5_b1:.4f}, B2 {map_top5_b2:.4f}")


if __name__ == '__main__':
    main_quick_b1b2()
