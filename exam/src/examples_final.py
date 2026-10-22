#!/usr/bin/env python3
"""Validate v3 submissions and print/write out/examples_final.md.

Reproduce validation predictions for three users only, using the full 11-20
training graph. No final refit or full validation-grid rerun is performed.
Usage: .venv/bin/python src/examples_final.py
"""
from __future__ import annotations

import hashlib
import html

import pandas as pd

import harness as H
import m2_content as M2
from example_users import HEAVY, LIGHT, COLD

USERS = ((HEAVY, "Heavy"), (LIGHT, "Light"), (COLD, "Cold start"))
KAGGLE = (
    ("submission_v3_1.csv (designated final)", 0.33173, 0.31829),
    ("submission_v3_2.csv", 0.32686, 0.31413),
    ("submission_v3_3.csv", 0.32539, 0.31550),
    ("PPR alpha 0.3", 0.30721, 0.27476),
    ("Popularity", 0.29788, 0.29442),
    ("History + airtime (final_v2)", 0.28189, 0.25260),
    ("History", 0.28079, 0.25013),
)


def validate_submissions() -> list[str]:
    template = pd.read_csv(H.DATA / "submission.csv", dtype=str)
    pool = H.load_pool(H.EVAL_BLOCK_FINAL)
    lines = ["## Submission validation", "",
             "All files use the 31–42 airing pool; user IDs are read as strings.", ""]
    for rank in (1, 2, 3):
        path = H.EXAM / f"submission_v3_{rank}.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert frame.columns.tolist() == ["user_id", "tv_show_id"], path
        assert len(frame) == len(template) == H.EXPECTED_SUB_USERS, path
        assert frame["user_id"].is_unique, path
        assert frame["user_id"].tolist() == template["user_id"].tolist(), path
        lists = [list(map(int, value.split())) for value in frame["tv_show_id"]]
        assert all(len(row) == len(set(row)) == 5 for row in lists), path
        assert all(item in pool and item != 0 for row in lists for item in row), path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.extend([
            f"- `{path.name}`: **PASS** — {len(frame):,} rows, template order, unique users, "
            "five distinct nonzero in-pool IDs per row.",
            f"  SHA-256: `{digest}`",
        ])
    return lines


def main() -> None:
    checks = validate_submissions()
    grid = pd.read_csv(H.OUT / "step4_allwatched_grid.csv")
    winner = grid.sort_values("rank").iloc[0]
    assert winner["method"] == "RRF_ppr_pop(alpha=0.5,weights=1.0/0.5)"
    inter = H.load_interactions()
    progs = H.load_programmes()
    train = H.get_user_train(inter, H.TRAIN_BLOCKS_VAL)
    histories, history_lists = H.build_user_histories(train)
    top5, all_watched = H.get_eval_targets(inter, H.EVAL_BLOCK_VAL)
    pool = H.load_pool(H.EVAL_BLOCK_VAL)
    popularity = H.build_popularity(train, pool)
    users = [user for user, _ in USERS]
    cold = {user for user in users if not histories.get(user)}
    assert cold == {COLD}

    def recommend(scores):
        return H.make_recs(scores, pool, popularity, histories, cold, 5)[0]

    methods = {
        "Popularity": recommend({user: [] for user in users}),
        "History": recommend({
            user: [(item, score) for item, score in history_lists.get(user, []) if item in pool]
            for user in users
        }),
    }
    user_list = sorted(histories)
    item_list = sorted(set(train["tv_show_id"].astype(int)))
    user_index = {user: index for index, user in enumerate(user_list)}
    item_index = {item: index for index, item in enumerate(item_list)}
    ppr = {}
    for alpha in (0.3, 0.5):
        ppr[alpha] = H.ppr_batched(
            users, user_index, item_index, user_list, item_list,
            history_lists, histories, pool, alpha, chunk_size=500,
        )
    methods["PPR alpha 0.3"] = recommend(ppr[0.3])
    content, _ = M2.run(
        inter, progs, H.TRAIN_BLOCKS_VAL, H.EVAL_BLOCK_VAL, top5, all_watched,
        histories, history_lists, cold, popularity, users,
    )
    methods["Content-based M2"] = content["M2_content"]["recs"]
    pop_rank = [int(item) for item in popularity.index][:200]
    methods["v3_1: PPR 0.5 + popularity"] = recommend({
        user: H.rank_fusion(
            [[item for item, _ in ppr[0.5].get(user, [])[:200]], pop_rank],
            [1.0, 0.5], k=5, rrf_k=60.0,
        ) for user in users
    })
    for predictions in methods.values():
        for user in users:
            assert len(predictions[user]) == len(set(predictions[user])) == 5
            assert set(predictions[user]) <= pool
        assert predictions[COLD] == methods["Popularity"][COLD]

    metadata = progs.set_index("tv_show_id")

    def cell(item):
        if item is None:
            return "—"
        title = html.escape(str(metadata.loc[item, "title"])).replace("|", "&#124;")
        category = html.escape(str(metadata.loc[item, "category"])).replace("|", "&#124;")
        return f"`{item}`<br>{title}<br>({category})"

    lines = [
        "# Final submission and validation examples", "",
        "**Designated final submission: `submission_v3_1.csv`.**", "",
        "## Exact winning configuration", "",
        f"- Configuration: `{winner['method']}`.",
        "- Reciprocal-rank fusion: `k=60`, one-based ranks, top 200 items from each input ranking.",
        "- PPR: alpha **0.5**, weight **1.0**. Bipartite user–programme graph, edge weight "
        "`log1p(overlap_seconds / 60)`, row-normalized transitions; restart on the user's own "
        "programmes weighted by those edges. `R <- 0.5 * P.T @ R + V`, with "
        "`V = 0.5 * normalized_restart`; max 30 iterations, tolerance 1e-6, chunks of 500.",
        "- Popularity: weight **0.5**, ranked by distinct training viewers, restricted to the airing pool.",
        "- History weight **0.0**; no airtime adjustment (beta **0.0**).",
        "- Fused score: `1 / (60 + rank_PPR) + 0.5 / (60 + rank_popularity)`; "
        "an absent item contributes zero for that input. Score ties use ascending programme ID.",
        "- Exactly five distinct IDs; popularity fill and popularity-only cold-start fallback; "
        "already-watched programmes allowed.",
        f"- Validation MAP@5, **all watched**: **{winner['map_all']:.16f}**.",
        f"- Validation MAP@5, **top-five-by-time target**: **{winner['map_top5']:.16f}**.",
        "- Validation fits on 11–20, predicts the 21–30 airing pool, and averages over 4,799 users. "
        "Selection uses all-watched MAP@5. Final submissions refit on 11–30 and use the 31–42 airing pool.",
        "", "## Kaggle results", "", "User-reported public/private scores (screenshots saved).", "",
        "| Submission / method | Public | Private |", "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {public:.5f} | {private:.5f} |" for name, public, private in KAGGLE)
    lines.extend(["", *checks, "", "## Side-by-side validation examples", "",
                  "All recommendations below are trained on 11–20 and restricted to programmes airing in "
                  "21–30. Actual Top-5 is ranked by 21–30 overlap seconds (ties by programme ID). "
                  "These are validation predictions, not rows from the 31–42 submission.", ""])
    for user, label in USERS:
        watched = top5.get(user, [])
        seconds = sum(score for _, score in history_lists.get(user, []))
        lines.extend([
            f"### {label}: `{user}`", "",
            f"11–20 history: {len(histories.get(user, set()))} programmes, {int(seconds):,} seconds. "
            f"21–30 actual: {len(all_watched.get(user, set()))} programmes.", "",
        ])
        if user in cold:
            lines.extend(["Cold start in 11–20: all five methods use the same popularity fallback.", ""])
        if not watched:
            lines.extend(["No recorded 21–30 viewing: the actual Top-5 is unavailable, and this user "
                          "is excluded from validation MAP@5 (which requires a nonempty target).", ""])
        lines.extend([
            "| Rank | " + " | ".join(methods) + " | Actual Top-5 watched in 21–30 |",
            "|---:|" + "---|" * (len(methods) + 1),
        ])
        for rank in range(5):
            cells = [cell(predictions[user][rank]) for predictions in methods.values()]
            cells.append(cell(watched[rank] if rank < len(watched) else None))
            lines.append(f"| {rank + 1} | " + " | ".join(cells) + " |")
        lines.append("")
    report = "\n".join(lines).rstrip() + "\n"
    path = H.OUT / "examples_final.md"
    path.write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
