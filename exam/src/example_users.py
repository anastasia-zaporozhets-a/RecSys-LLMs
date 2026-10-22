#!/usr/bin/env python3
"""Example users: the same two users across every method (validation + final).

Also validates out/submission_final.csv (Kaggle format) and prints the two
users' actual 21-30 viewing so the lists can be compared with real behaviour.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

HEAVY = "13571199046623737884"   # most training seconds among the 1,260 submission users
LIGHT = "8150816946177670084"    # least training seconds among submission users with 21-30 activity
COLD = "16561743585370579180"    # submission user with no training interaction


def read_val(path):
    df = pd.read_csv(path, dtype=str)
    cols = [c for c in df.columns if c.startswith("tv_show_id")]
    out = {}
    for _, r in df.iterrows():
        out[r["user_id"]] = [int(x) for x in r[cols].tolist()]
    return out


def main():
    progs = H.load_programmes()
    title = dict(zip(progs["tv_show_id"].astype(int), progs["title"].astype(str)))
    cat = dict(zip(progs["tv_show_id"].astype(int), progs["category"].astype(str)))
    inter = H.load_interactions()

    files = {
        "B1_popularity": H.OUT / "val_popularity.csv",
        "B2_personal_history": H.OUT / "val_personal_history.csv",
        "M1_PPR(alpha=0.3)": H.OUT / "val_ppr_alpha0.3.csv",
        "M2_content": H.OUT / "val_M2_content.csv",
        "M2_content+hist_tiebreak": H.OUT / "val_M2_content_hist_tiebreak.csv",
        "M3_hybrid(RRF)": H.OUT / "val_hybrid.csv",
    }
    vals = {}
    for name, p in files.items():
        if p.exists():
            vals[name] = read_val(p)
        else:
            print(f"  [missing] {p}")

    final = read_val(H.OUT / "test_ppr.csv") if (H.OUT / "test_ppr.csv").exists() else {}

    for uid, label in ((HEAVY, "HEAVY viewer"), (LIGHT, "LIGHT viewer"), (COLD, "COLD-START user")):
        print("=" * 110)
        print(f"{label}: user_id={uid}")
        tr = inter[(inter["user_id"] == uid) & (inter["block"] == "11-20")]
        tr = tr.groupby("tv_show_id")["overlap_seconds"].sum().sort_values(ascending=False)
        print(f"  training (11-20) programmes: {len(tr)}, total seconds {int(tr.sum()):,}")
        for name, d in vals.items():
            if uid not in d:
                continue
            print(f"\n  {name} -> Top-5 (validation, pool = programmes airing in 21-30):")
            for r, pid in enumerate(d[uid], 1):
                print(f"    {r}. id={pid:<12} cat={cat.get(pid, '?'):<16} title={title.get(pid, '?')[:58]}")
        if uid in final:
            print(f"\n  M1_PPR(alpha=0.3) FINAL (refit on 11-20+21-30, pool = 31-42):")
            for r, pid in enumerate(final[uid], 1):
                print(f"    {r}. id={pid:<12} cat={cat.get(pid, '?'):<16} title={title.get(pid, '?')[:58]}")
        ev = inter[(inter["user_id"] == uid) & (inter["block"] == "21-30")]
        ev = ev.groupby("tv_show_id")["overlap_seconds"].sum().sort_values(ascending=False)
        print(f"\n  ACTUAL watched in 21-30 (target): {len(ev)} programmes, {int(ev.sum()):,} s")
        for r, (pid, sec) in enumerate(ev.head(5).items(), 1):
            print(f"    {r}. id={int(pid):<12} cat={cat.get(int(pid), '?'):<16} "
                  f"title={title.get(int(pid), '?')[:58]}  overlap={int(sec)}s")
        print()

    # ---- submission sanity check -----------------------------------------
    print("=" * 110)
    print("SUBMISSION CHECK: out/submission_final.csv")
    pool = H.load_pool("31-42")
    with open(H.OUT / "submission_final.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    hdr, body = rows[0], rows[1:]
    sub = pd.read_csv(H.DATA / "submission.csv", dtype={"user_id": str})
    print(f"  header: {hdr}; rows: {len(body)} (template rows: {len(sub)})")
    ok_fmt = ok_pool = ok_distinct = True
    ids_all = []
    for u, toks in body:
        lst = [int(x) for x in toks.split()]
        ids_all += lst
        if len(lst) != 5:
            ok_fmt = False
        if len(set(lst)) != 5:
            ok_distinct = False
        if any(x not in pool for x in lst):
            ok_pool = False
    print(f"  exactly 5 ids per row: {ok_fmt}")
    print(f"  5 distinct ids per row: {ok_distinct}")
    print(f"  all ids in the 31-42 labelled pool: {ok_pool}")
    print(f"  user order identical to template: "
          f"{[r[0] for r in body] == sub['user_id'].tolist()}")
    print(f"  distinct programmes recommended: {len(set(ids_all)):,} of {len(pool):,} in pool")
    print(f"  rows with a 0 id: {sum(1 for r in body if '0' in r[1].split())}")


if __name__ == "__main__":
    main()
