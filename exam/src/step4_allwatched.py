#!/usr/bin/env python3
"""Step 4: tune against all programmes watched in the validation block.

Selection policy, decided after comparing validation rankings with Kaggle:
select configurations by MAP@5 against ALL programmes each user watched in
weeks 21-30. The previous top-5-by-overlap target remains in the output only as
a diagnostic. Kaggle ranked PPR, popularity, history+airtime, history in three
of the same four positions as the all-watched metric, while the top-5 target
put popularity last.

Validation fits on weeks 11-20 and uses only the programme pool airing in
weeks 21-30. After selection, the top three configurations are independently
refit on weeks 11-30 and applied to the pool airing in weeks 31-42. PPR vectors
are cached by alpha in both phases; ppr_batched uses chunks of 500 users.

Writes:
  out/step4_allwatched_grid.csv
  out/step4_allwatched.md
  submission_v3_1.csv, submission_v3_2.csv, submission_v3_3.csv

Usage:
  .venv/bin/python src/step4_allwatched.py
"""
from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_build as FB  # noqa: E402
import harness as H  # noqa: E402


RANDOM_SEED = 12345
PPR_ALPHAS = (0.5, 0.7, 0.85, 0.95)
RRF_ALPHAS = (0.5, 0.7)
POPULARITY_WEIGHTS = (0.25, 0.5, 1.0)
THREE_WAY_WEIGHTS = ((1.0, 0.5, 0.5), (1.0, 1.0, 0.5))
RRF_K = 60.0
CANDIDATES_PER_RANKING = 200
REPORT: list[str] = []


def log(message: str = "", flush: bool = True) -> None:
    print(message, flush=flush)
    REPORT.append(str(message))


def heading(title: str) -> None:
    log()
    log("=" * 110)
    log(title)
    log("=" * 110)


def config_name(kind: str, alpha: float, weights: tuple[float, ...] | None = None) -> str:
    if kind == "ppr":
        return f"PPR(alpha={alpha})"
    weight_text = "/".join(str(weight) for weight in weights or ())
    return f"RRF_{kind}(alpha={alpha},weights={weight_text})"


def main() -> None:
    np.random.seed(RANDOM_SEED)
    started = time.perf_counter()
    heading("STEP 4 - SELECT BY VALIDATION MAP@5 AGAINST ALL WATCHED PROGRAMMES")
    log(f"Environment : Python {sys.version.split()[0]} {platform.platform()}; "
        f"numpy {np.__version__}; pandas {pd.__version__}")
    log(f"Seed        : {RANDOM_SEED}")
    log("Decision    : map_all is the selection metric; map_top5 is diagnostic only.")
    log("Kaggle      : PPR a=0.3 0.30721/0.27476 public/private; popularity "
        "0.29788/0.29442; history+airtime 0.28189/0.25260; history 0.28079/0.25013.")

    inter = H.load_interactions()
    train_sum = H.get_user_train(inter, H.TRAIN_BLOCKS_VAL)
    targets_top5, targets_all = H.get_eval_targets(inter, H.EVAL_BLOCK_VAL)
    user_train_set, user_train_list = H.build_user_histories(train_sum)
    pool = H.load_pool(H.EVAL_BLOCK_VAL)
    popularity = H.build_popularity(train_sum, pool)
    eval_users = list(targets_all)
    cold = set(eval_users) - set(user_train_set)
    assert len(eval_users) == H.EXPECTED_EVAL_USERS
    assert len(user_train_set) == H.EXPECTED_TRAIN_USERS
    log(f"Validation  : {len(eval_users):,} users; pool {len(pool):,}; cold start {len(cold):,}")

    user_list = sorted(user_train_set)
    item_list = sorted(set(train_sum["tv_show_id"].astype(int)))
    user2idx = {user: index for index, user in enumerate(user_list)}
    item2idx = {item: index for index, item in enumerate(item_list)}
    pop_rank = [int(item) for item in popularity.index if int(item) in pool][:CANDIDATES_PER_RANKING]
    history_rank = {
        user: [item for item, _ in user_train_list.get(user, []) if item in pool][
            :CANDIDATES_PER_RANKING
        ]
        for user in eval_users
    }

    rows: list[dict] = []
    configs: dict[str, dict] = {}
    ppr_rank_cache: dict[float, dict[str, list[int]]] = {}

    def record(name: str, config: dict, recs: dict[str, list[int]], fallback: int,
               runtime: float) -> None:
        map_all, users_all = H.map_at_k(recs, targets_all, 5)
        map_top5, users_top5 = H.map_at_k(recs, targets_top5, 5)
        assert users_all == users_top5 == H.EXPECTED_EVAL_USERS
        rows.append({
            "method": name,
            "map_all": map_all,
            "map_top5": map_top5,
            "runtime": runtime,
            "fallback": fallback,
        })
        configs[name] = config
        log(f"  {name:<57} MAP@5(all)={map_all:.6f}  "
            f"MAP@5(top-5)={map_top5:.6f}  fallback={fallback}  {runtime:.1f}s")

    heading("PPR")
    for alpha in PPR_ALPHAS:
        config_started = time.perf_counter()
        log(f"  Computing PPR alpha={alpha} ...")
        ppr_scores = H.ppr_batched(
            eval_users, user2idx, item2idx, user_list, item_list,
            user_train_list, user_train_set, pool, alpha, chunk_size=500,
        )
        if alpha in RRF_ALPHAS:
            ppr_rank_cache[alpha] = {
                user: [item for item, _ in ranking[:CANDIDATES_PER_RANKING]]
                for user, ranking in ppr_scores.items()
            }
        recs, fallback = H.make_recs(
            ppr_scores, pool, popularity, user_train_set, cold, 5,
        )
        name = config_name("ppr", alpha)
        record(name, {"kind": "ppr", "alpha": alpha}, recs, fallback,
               time.perf_counter() - config_started)
        del ppr_scores

    heading("RRF(k=60) - PPR + POPULARITY")
    for alpha in RRF_ALPHAS:
        for pop_weight in POPULARITY_WEIGHTS:
            config_started = time.perf_counter()
            scores = {}
            for user in eval_users:
                ppr_rank = ppr_rank_cache[alpha].get(user, [])
                scores[user] = H.rank_fusion(
                    [ppr_rank, pop_rank], [1.0, pop_weight], k=5, rrf_k=RRF_K,
                )
            recs, fallback = H.make_recs(scores, pool, popularity, user_train_set, cold, 5)
            weights = (1.0, pop_weight)
            name = config_name("ppr_pop", alpha, weights)
            record(name, {"kind": "rrf", "alpha": alpha, "weights": (1.0, 0.0, pop_weight)},
                   recs, fallback, time.perf_counter() - config_started)

    heading("RRF(k=60) - PPR + HISTORY + POPULARITY")
    for alpha in RRF_ALPHAS:
        for weights in THREE_WAY_WEIGHTS:
            config_started = time.perf_counter()
            scores = {}
            for user in eval_users:
                ppr_rank = ppr_rank_cache[alpha].get(user, [])
                scores[user] = H.rank_fusion(
                    [ppr_rank, history_rank[user], pop_rank], list(weights), k=5, rrf_k=RRF_K,
                )
            recs, fallback = H.make_recs(scores, pool, popularity, user_train_set, cold, 5)
            name = config_name("ppr_hist_pop", alpha, weights)
            record(name, {"kind": "rrf", "alpha": alpha, "weights": weights},
                   recs, fallback, time.perf_counter() - config_started)

    heading("ALL CONFIGURATIONS - SORTED BY MAP@5 AGAINST ALL WATCHED PROGRAMMES")
    results = pd.DataFrame(rows).sort_values(
        ["map_all", "map_top5", "method"], ascending=[False, False, True], ignore_index=True,
    )
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    results.to_csv(H.OUT / "step4_allwatched_grid.csv", index=False)
    log(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    heading("FINAL BUILDS - TOP 3 REFIT ON WEEKS 11-30")
    final_ppr_cache: dict[float, dict[str, list[tuple[int, float]]]] = {}
    for rank, row in results.head(3).iterrows():
        name = str(row["method"])
        config = configs[name]
        alpha = float(config["alpha"])
        output = H.EXAM / f"submission_v3_{rank + 1}.csv"
        log(f"  {output.name}: rank {rank + 1}, {name}, validation MAP@5(all)={row['map_all']:.6f}")

        if config["kind"] == "ppr":
            method = "ppr"
            weights = (1.0, 0.0, 0.0)
        else:
            method = "hybrid_rrf"
            weights = tuple(config["weights"])

        recs, info = FB.build_final(
            method, alpha=alpha, beta=0.0,
            w_ppr=weights[0], w_hist=weights[1], w_pop=weights[2],
            ppr_scores=final_ppr_cache.get(alpha), log=log,
        )
        if alpha not in final_ppr_cache:
            final_ppr_cache[alpha] = {
                user: ranking[:CANDIDATES_PER_RANKING]
                for user, ranking in info["ppr_scores"].items()
            }
        checks = FB.check_and_write(
            recs, output, info["pool"], info["popularity"], log=log,
        )
        log(f"  {output.name} checks: {checks}")

    log(f"\nTotal runtime: {time.perf_counter() - started:.1f}s")
    report_path = H.OUT / "step4_allwatched.md"
    report_path.write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
