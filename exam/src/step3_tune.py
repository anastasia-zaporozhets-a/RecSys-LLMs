#!/usr/bin/env python3
"""B) Validation-only tuning for the final strategy, then the final build.

Grid (validation only: fit on 11-20, evaluate on 21-30, pool = programmes
airing in 21-30; the 31-42 schedule is not used for any selection):

  * PPR alpha in {0.1, 0.2} (plus 0.3 from the previous run for reference);
  * airtime-ratio adjustment beta in {0, 0.5, 1} applied to B2 (personal
    history) and to the best PPR, where
        ratio_i = (labelled airtime of programme i in the evaluation block
                   + 1 h) / (labelled airtime of i in the training block + 1 h);
  * hybrid "B2 history first, PPR fills the rest" (with the same beta);
  * hybrid RRF (PPR + history + popularity) as the third fusion variant.

The best configuration by validation MAP@5 (top-5 target) is then refitted on
weeks 11-30 and written to exam/submission_final_v2.csv with the standard
checks.

Usage:
    .venv/bin/python src/step3_tune.py                 # full grid + final build
    .venv/bin/python src/step3_tune.py --no-final      # grid only
    .venv/bin/python src/step3_tune.py --alphas 0.1    # fewer PPR settings
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import final_build as FB  # noqa: E402

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
REPORT: list[str] = []


def log(msg="", flush=True):
    print(msg, flush=flush)
    REPORT.append(str(msg))


def hr(t):
    log("")
    log("=" * 104)
    log(t)
    log("=" * 104)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.1,0.2,0.3",
                    help="PPR alphas for the grid; 0.3 is the value selected in step 2, "
                         "recomputed with the same code path so the comparison is fair")
    ap.add_argument("--betas", default="0,0.5,1")
    ap.add_argument("--no-final", action="store_true")
    ap.add_argument("--out", default=str(H.EXAM / "submission_final_v2.csv"))
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    betas = [float(b) for b in args.betas.split(",")]

    t_script = time.perf_counter()
    hr("B) VALIDATION TUNING (train 11-20 -> eval 21-30; 31-42 NOT used)")
    log(f"Environment : Python {sys.version.split()[0]} {platform.platform()}; numpy {np.__version__}; pandas {pd.__version__}")
    log(f"Seed        : {RANDOM_SEED}")
    log(f"Grid        : PPR alpha {alphas} x beta {betas}; hybrid variants")

    inter = H.load_interactions()
    progs = H.load_programmes()
    train_sum = H.get_user_train(inter, H.TRAIN_BLOCKS_VAL)
    targets_top5, targets_all = H.get_eval_targets(inter, H.EVAL_BLOCK_VAL)
    user_train_set, user_train_list = H.build_user_histories(train_sum)
    n_eval = len([u for u in targets_all if targets_all[u]])
    assert n_eval == H.EXPECTED_EVAL_USERS, n_eval
    assert len(user_train_set) == H.EXPECTED_TRAIN_USERS
    pool = H.load_pool(H.EVAL_BLOCK_VAL)
    cold = set(targets_all.keys()) - set(user_train_set.keys())
    pop = H.build_popularity(train_sum, pool)
    eval_users = list(targets_all.keys())
    ratio = FB.airtime_ratio(H.EVAL_BLOCK_VAL, H.TRAIN_BLOCKS_VAL, pool)
    log(f"  eval users {n_eval:,}; pool {len(pool):,}; cold start {len(cold):,}")
    log(f"  airtime ratio (eval 21-30 vs train 11-20, +1h smoothing): "
        f"min {min(ratio.values()):.3f} median {float(np.median(list(ratio.values()))):.3f} "
        f"max {max(ratio.values()):.3f}")

    rows = []

    def record(name, params, recs, fb, dt):
        m5, n = H.map_at_k(recs, targets_top5, 5)
        ma, _ = H.map_at_k(recs, targets_all, 5)
        rows.append(dict(method=name, params=params, map_top5=m5, map_all=ma, runtime=dt, fallback=fb))
        log(f"  {name:<46} MAP@5(top-5)={m5:.6f}  MAP@5(all)={ma:.6f}  fallback={fb}  {dt:.1f}s")
        return m5, ma, recs

    # ------------------------------------------------------------- B2 + beta
    hr("B2 - PERSONAL HISTORY WITH AIRTIME-RATIO ADJUSTMENT")
    for beta in betas:
        t0 = time.perf_counter()
        scores = {u: FB.apply_beta([(it, sc) for it, sc in user_train_list.get(u, []) if it in pool],
                                   ratio, beta) for u in eval_users}
        recs, fb = H.make_recs(scores, pool, pop, user_train_set, cold, 5)
        record(f"B2_history(beta={beta})",
               f"train overlap_seconds x ratio^{beta}, pool-restricted",
               recs, fb, time.perf_counter() - t0)

    # ------------------------------------------------------------ PPR + beta
    hr("M1 - PERSONALIZED PAGERANK WITH AIRTIME-RATIO ADJUSTMENT")
    user_list = sorted(user_train_set.keys())
    item_list = sorted(set(train_sum["tv_show_id"].astype(int).tolist()))
    user2idx = {u: i for i, u in enumerate(user_list)}
    item2idx = {it: i for i, it in enumerate(item_list)}
    ppr_cache = {}
    for alpha in alphas:
        t0 = time.perf_counter()
        log(f"  PPR alpha={alpha} ...")
        ppr_cache[alpha] = H.ppr_batched(eval_users, user2idx, item2idx, user_list, item_list,
                                          user_train_list, user_train_set, pool, alpha)
        for beta in betas:
            t1 = time.perf_counter()
            scores = {u: FB.apply_beta(ppr_cache[alpha].get(u, []), ratio, beta) for u in eval_users}
            recs, fb = H.make_recs(scores, pool, pop, user_train_set, cold, 5)
            record(f"M1_PPR(alpha={alpha},beta={beta})",
                   f"log1p(sec/60) edges, own-programme restart, x ratio^{beta}",
                   recs, fb, time.perf_counter() - t0)
            log(f"      (scoring for this beta only: {time.perf_counter() - t1:.1f}s)")

    # ------------------------------------------------- hybrid: history first
    hr("M3 - HYBRID: B2 HISTORY FIRST, PPR FILLS THE REST")
    best_alpha = max(
        [r for r in rows if r["method"].startswith("M1_PPR")],
        key=lambda r: r["map_top5"])["method"]
    best_alpha = float(best_alpha.split("alpha=")[1].split(",")[0])
    log(f"  best PPR alpha from this run: {best_alpha}")
    for n_hist in (2, 3):
        for beta in betas:
            t0 = time.perf_counter()
            scores = {}
            for u in eval_users:
                hist = FB.apply_beta([(it, sc) for it, sc in user_train_list.get(u, []) if it in pool],
                                     ratio, beta)
                ppr = FB.apply_beta(ppr_cache[best_alpha].get(u, []), ratio, beta)
                scores[u] = FB.hybrid_ranking(hist, ppr, n_history=n_hist)
            recs, fb = H.make_recs(scores, pool, pop, user_train_set, cold, 5)
            record(f"M3_hybrid_hist_first(alpha={best_alpha},beta={beta},n_hist={n_hist})",
                   f"B2 history takes the first {n_hist} slots, PPR fills the rest",
                   recs, fb, time.perf_counter() - t0)

    # --------------------------------------------- hybrid RRF (rank fusion)
    hr("M3 - HYBRID: RECIPROCAL-RANK FUSION (PPR + HISTORY + POPULARITY)")
    CAND = 200
    pop_rank = [int(it) for it in pop.index if int(it) in pool][:CAND]
    for wp, wh, wq in ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.5), (1.0, 0.0, 0.5),
                       (2.0, 1.0, 0.5), (1.0, 2.0, 0.5)):
        t0 = time.perf_counter()
        scores = {}
        for u in eval_users:
            ppr_rank = [it for it, _ in ppr_cache[best_alpha].get(u, [])][:CAND]
            hist_rank = [it for it, _ in user_train_list.get(u, []) if it in pool][:CAND]
            scores[u] = H.rank_fusion([ppr_rank, hist_rank, pop_rank], [wp, wh, wq], k=5)
        recs, fb = H.make_recs(scores, pool, pop, user_train_set, cold, 5)
        record(f"M3_rrf(a={best_alpha},w={wp}/{wh}/{wq})",
               "RRF(k=60) of PPR + history + popularity rankings",
               recs, fb, time.perf_counter() - t0)

    # ------------------------------------------------------------- summary
    hr("ONE TABLE - ALL VALIDATION CONFIGURATIONS (best by MAP@5 top-5 target)")
    df = pd.DataFrame(rows).sort_values("map_top5", ascending=False)
    df.to_csv(H.OUT / "step3_tune_grid.csv", index=False)
    log(df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    best = df.iloc[0]
    log(f"\n  BEST: {best['method']}  MAP@5(top-5)={best['map_top5']:.6f}  "
        f"MAP@5(all)={best['map_all']:.6f}  fallback={int(best['fallback'])}")
    def ref(prefix, default="n/a"):
        hit = df.loc[df["method"].str.startswith(prefix), "map_top5"]
        return f"{hit.iloc[0]:.6f}" if len(hit) else default
    log(f"  reference: B1 popularity 0.147777 | B2 history (beta=0) {ref('B2_history(beta=0.0)')} | "
        f"PPR alpha=0.3 beta=0 {ref('M1_PPR(alpha=0.3,beta=0.0)')}")

    # ---------------------------------------------------------- final build
    if not args.no_final:
        hr("FINAL BUILD: refit on weeks 11-30, predict weeks 31-42")
        m = best["method"]
        wp, wh, wq, n_hist = 1.0, 0.0, 0.0, 5
        if m.startswith("B2_history"):
            method, alpha, beta = "history", 0.3, float(m.split("beta=")[1].rstrip(")"))
        elif m.startswith("M1_PPR"):
            method = "ppr"
            alpha = float(m.split("alpha=")[1].split(",")[0])
            beta = float(m.split("beta=")[1].rstrip(")"))
        elif m.startswith("M3_hybrid_hist_first"):
            method = "hybrid"
            alpha = float(m.split("alpha=")[1].split(",")[0])
            beta = float(m.split("beta=")[1].split(")")[0])
            n_hist = int(m.split("n_hist=")[1].rstrip(")"))
        else:
            method = "hybrid_rrf"
            alpha = best_alpha
            beta = 0.0
            w = m.split("w=")[1].rstrip(")")
            wp, wh, wq = [float(x) for x in w.split("/")]
        log(f"  chosen configuration: {m} -> method={method}, alpha={alpha}, beta={beta}")
        recs, info = FB.build_final(method, alpha, beta,
                                    w_ppr=wp, w_hist=wh, w_pop=wq, n_history=n_hist, log=log)
        out_path = Path(args.out)
        checks = FB.check_and_write(recs, out_path, info["pool"], info["popularity"], log=log)
        log(f"  checks: {checks}")

    log(f"\nTotal runtime: {time.perf_counter() - t_script:.1f}s")
    (H.OUT / "step3_tune.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"\nWrote {H.OUT / 'step3_tune.md'}")


if __name__ == "__main__":
    main()
