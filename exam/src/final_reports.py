#!/usr/bin/env python3
"""Validate all submission artifacts and regenerate the final two Markdown reports.

Run from exam/: .venv/bin/python src/final_reports.py
Uses saved submissions/validation metrics. Only three M2 user profiles are
scored; no LTR model fitting, tuning, PPR refitting or raw-log join is run.
The examples are FINAL 31-42 predictions, with actual 21-30 and 11-30 viewing
shown as historical context, not as future ground truth. No 31-42 viewing exists.
"""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import hashlib
import html
import json
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

import harness as H
import m2_content as M2
from example_users import HEAVY, LIGHT, COLD, read_val
from step7_refit_all_together import Tee

USERS = ((HEAVY, "Heavy"), (LIGHT, "Light"), (COLD, "Cold start"))
FINAL = "submission_v6_3.csv"
# Public/private values reported by the user; None means not reported.
KAGGLE = {
    "submission_ppr_alpha0.3.csv": (.30721, .27476),
    "submission_popularity.csv": (.29788, .29442),
    "submission_history.csv": (.28079, .25013),
    "submission_final_v2.csv": (.28189, .25260),
    "submission_v3_1.csv": (.33173, .31829), "submission_v3_2.csv": (.32686, .31413),
    "submission_v3_3.csv": (.32539, .31550),
    "submission_v4_1.csv": (.35383, .33450), "submission_v4_2.csv": (.34898, .32602),
    "submission_v4_3.csv": (None, None),
    "submission_v5_ltr.csv": (.39205, .37705), "submission_v5_blend.csv": (.37766, .36082),
    "submission_v6_1.csv": (.40131, .38529), "submission_v6_2.csv": (None, None),
    FINAL: (.41907, .40253),
    "submission_v7_1.csv": (.41615, .40078), "submission_v7_2.csv": (.41519, .39924),
    "submission_v8_ens.csv": (.41669, .39991),
}


def md(value):
    return html.escape(str(value), quote=False).replace("|", "&#124;").replace("\n", " ")


def number(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def load_json(name):
    return json.loads((H.OUT / name).read_text(encoding="utf-8"))


def validate_all(pool):
    template = pd.read_csv(H.DATA / "submission.csv", dtype=str).user_id.tolist()
    paths = sorted(H.EXAM.glob("submission*.csv"))
    paths.extend(sorted(H.OUT.glob("submission*.csv")))
    result = {}
    for path in paths:
        key = str(path.relative_to(H.EXAM))
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert frame.columns.tolist() == ["user_id", "tv_show_id"], key
        assert len(frame) == len(template) == H.EXPECTED_SUB_USERS, key
        assert frame.user_id.is_unique and frame.user_id.tolist() == template, key
        lists = [list(map(int, value.split())) for value in frame.tv_show_id]
        assert all(len(row) == len(set(row)) == 5 and 0 not in row and set(row) <= pool for row in lists), key
        result[key] = {"predictions": dict(zip(template, lists)), "rows": len(frame),
                       "programmes": len({item for row in lists for item in row}),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"VALIDATED {key}: {len(frame):,} ordered unique users; five distinct in-pool IDs", flush=True)
    assert set(KAGGLE) <= set(result), "A recorded submission is missing from disk"
    alias = "out/submission_final.csv"
    if alias in result:
        assert result[alias]["predictions"] == result["submission_ppr_alpha0.3.csv"]["predictions"]
    return result


def content_examples(progs, users, histories, lists, pool, popularity):
    """Same plain M2 recipe/top-k code as m2_content.py, scored for three users."""
    ids, index, matrix = M2.build_item_matrix(progs)
    pool_ids = np.array([item for item in sorted(pool) if item in index], dtype=np.int64)
    pool_matrix = matrix[[index[item] for item in pool_ids]]
    active = [user for user in users if histories.get(user)]
    row, col, seconds = [], [], []
    for user_row, user in enumerate(active):
        for item, score in lists.get(user, []):
            if item in index:
                row.append(user_row)
                col.append(index[item])
                seconds.append(score)
    weights = sparse.csr_matrix((seconds, (row, col)), shape=(len(active), len(ids)))
    profiles = normalize(weights @ matrix, norm="l2", axis=1)
    scores = (profiles @ pool_matrix.T).toarray()
    ranking = {}
    for user, score in zip(active, scores):
        part = np.argpartition(-score, 4)[:5]
        order = part[np.argsort(-score[part], kind="stable")]
        ranking[user] = [(int(pool_ids[i]), float(score[i])) for i in order]
    cold = {user for user in users if not histories.get(user)}
    return H.make_recs(ranking, pool, popularity, histories, cold, 5)[0]


def write_examples(artifacts, inter, progs, pool):
    train = H.get_user_train(inter, H.TRAIN_BLOCKS_FINAL)
    histories, lists = H.build_user_histories(train)
    eval_top5, eval_all = H.get_eval_targets(inter, H.EVAL_BLOCK_VAL)
    popularity = H.build_popularity(train, pool)
    users = [user for user, _ in USERS]
    methods = {
        "Popularity": artifacts["submission_popularity.csv"]["predictions"],
        "History": artifacts["submission_history.csv"]["predictions"],
        "PPR alpha 0.3": artifacts["submission_ppr_alpha0.3.csv"]["predictions"],
        "Content-based M2": content_examples(progs, users, histories, lists, pool, popularity),
        "v6_3 (designated final)": artifacts[FINAL]["predictions"],
    }
    metadata = progs.set_index("tv_show_id")
    assert metadata.index.is_unique

    def cell(item, seconds=None):
        if item is None:
            return "—"
        value = f"`{item}`<br>{md(metadata.loc[item, 'title'])}<br>({md(metadata.loc[item, 'category'])})"
        return value + (f"<br>{int(seconds):,} s" if seconds is not None else "")

    saved = load_json("step7_v6_3_final_configs.json")[0]
    lines = ["# Final recommendation examples", "",
             f"**Designated final: `{FINAL}` — private 0.40253 / public 0.41907.**", "",
             "## Protocol and model", "",
             "All five recommendation columns below predict **31–42**, using **11–30** training viewing "
             "and the 31–42 airing pool. Popularity, history, PPR and v6_3 come directly from the saved "
             "final submission files. Plain M2 uses the same catalogue features and seconds-weighted "
             "11–30 profiles, scored only for these three users.", "",
             "The actual 21–30 and whole-training 11–30 columns are **historical context**, not ground "
             "truth for 31–42. In particular, 21–30 is part of the final model's training history. "
             "These tables are not held-out validation predictions, and they are not used to calculate MAP@5.", "",
             "Actual Top-5 means highest summed qualifying watched seconds, ties by programme ID; "
             "historical lists are not restricted to the future airing pool. The audited watched-pair rule "
             "is at least 120 seconds and at least 5% of summed touched-airing duration per block.", "",
             "The heavy/light labels refer to the original 11–20 examples. Cold start means no qualifying "
             "training programme history; such users receive the same popularity fallback for every method.", "",
             "v6_3: size 200/source; weekly rates; stacked folds 11–20 → 21–30 and 11–25 → 26–30; "
             "channel/time/season/recency features; binary all-watched LambdaRank; five-seed raw-score mean. "
             f"Saved seed/tree counts: `{json.dumps(saved['seed_iterations'], sort_keys=True)}`.", ""]
    eval_seconds = H.get_user_train(inter, [H.EVAL_BLOCK_VAL]).set_index(["user_id", "tv_show_id"]).overlap_seconds
    for user, label in USERS:
        history = lists.get(user, [])
        actual = eval_top5.get(user, [])
        lines.extend([f"## {label}: `{user}`", "",
                      f"Whole 11–30 history: **{len(history):,} programmes, {int(sum(sec for _, sec in history)):,} seconds**. "
                      f"21–30 actual: **{len(eval_all.get(user, set())):,} programmes**.", ""])
        for predictions in methods.values():
            assert len(predictions[user]) == len(set(predictions[user])) == 5
            assert set(predictions[user]) <= pool
        if not history:
            assert all(predictions[user] == methods["Popularity"][user] for predictions in methods.values())
            lines.extend(["All methods use the popularity fallback. No recorded qualifying viewing exists "
                          "in 21–30 or 11–30, so both actual-watch columns are empty.", ""])
        lines.extend(["| Rank | " + " | ".join(methods) + " | Actual Top-5: 21–30 | Actual Top-5: whole 11–30 history |",
                      "|---:|" + "---|" * (len(methods) + 2)])
        for rank in range(5):
            values = [cell(predictions[user][rank]) for predictions in methods.values()]
            item = actual[rank] if rank < len(actual) else None
            values.append(cell(item, eval_seconds.loc[(user, item)] if item is not None else None))
            item, seconds = history[rank] if rank < len(history) else (None, None)
            values.append(cell(item, seconds))
            lines.append(f"| {rank + 1} | " + " | ".join(values) + " |")
        lines.append("")
        print(f"EXAMPLES {label} {user}: {len(history)} training programmes; final v6_3={methods['v6_3 (designated final)'][user]}", flush=True)
    path = H.OUT / "examples_final.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def metric_registry(inter):
    top5, allw = H.get_eval_targets(inter, H.EVAL_BLOCK_VAL)
    result = {}
    def register(file, method, scope, all_score=None, top_score=None, fold5=None, mean=None, source=""):
        result[file] = dict(method=method, scope=scope, all_score=all_score, top_score=top_score,
                            fold5=fold5, mean=mean, source=source)

    for file, val, method in (
        ("submission_popularity.csv", "val_popularity.csv", "Popularity: distinct training viewers"),
        ("submission_history.csv", "val_personal_history.csv", "Personal history by watched seconds"),
        ("submission_ppr_alpha0.3.csv", "val_ppr_alpha0.3.csv", "PPR alpha 0.3"),
    ):
        predictions = read_val(H.OUT / val)
        ma, n = H.map_at_k(predictions, allw, 5)
        mt, nt = H.map_at_k(predictions, top5, 5)
        assert n == nt == H.EXPECTED_EVAL_USERS
        register(file, method, "Full validation (4,799)", ma, mt, source=f"out/{val}")
    registry = pd.read_csv(H.OUT / "step3_tune_grid.csv").set_index("method")
    row = registry.loc["B2_history(beta=0.5)"]
    register("submission_final_v2.csv", "History × airtime ratio^0.5", "Full validation (4,799)",
             float(row.map_all), float(row.map_top5), source="out/step3_tune_grid.csv")
    step4 = pd.read_csv(H.OUT / "step4_allwatched_grid.csv").set_index("rank")
    for rank in (1, 2, 3):
        row = step4.loc[rank]
        register(f"submission_v3_{rank}.csv", str(row.method), "Full validation (4,799)",
                 float(row.map_all), float(row.map_top5), source="out/step4_allwatched_grid.csv")
    step5 = pd.read_csv(H.OUT / "step5_exposure_grid.csv").set_index("method")
    for config in load_json("step5_exposure_final_configs.json"):
        row = step5.loc[config["method"]]
        register(config["file"], config["method"], "Full validation (4,799)",
                 float(row.map_all), float(row.map_top5), source="out/step5_exposure_grid.csv")
    row = pd.read_csv(H.OUT / "step6_heldout.csv").set_index("method").loc["LTR_heldout"]
    register("submission_v5_ltr.csv", "Binary LambdaRank; K50/source; 1 seed; 127 trees", "User holdout (960)",
             float(row.all_watched), float(row.top5_seconds), source="out/step6_heldout.csv")
    register("submission_v5_blend.csv", "RRF(k=60), v5_ltr + v4_1, weights 1/1", "Not measured",
             source="src/step6_ltr.py (no held-out blend measurement)")
    step7 = pd.read_csv(H.OUT / "step7_ablations.csv").set_index("recipe")
    for config in load_json("step7_final_configs.json") + load_json("step7_v6_3_final_configs.json"):
        row, params = step7.loc[config["recipe"]], config["config"]
        method = (f"LambdaRank {config['recipe']}; K{params['size']}/source; "
                  f"{'weekly' if params['scale'] else 'raw'} units; "
                  f"{'stacked' if params['stack'] else 'single'} fold; {len(params['seeds'])} seed(s)")
        register(config["file"], method, "User holdouts (960 / 946)", float(row.map_all_fold10),
                 fold5=float(row.map_all_fold5), mean=float((row.map_all_fold10 + row.map_all_fold5) / 2),
                 source="out/step7_ablations.csv")
    for row in load_json("step8_final_configs.json"):
        config, params = row["config"], row["config"]["params"]
        method = (f"LambdaRank {config['label']}; all Step 8 exposures; leaves={params['num_leaves']}, "
                  f"lr={params['learning_rate']}, min_child={params['min_child_samples']}; 5 seeds")
        register(row["file"], method, "User holdouts (960 / 946)", row["map_fold10"],
                 fold5=row["map_fold5"], mean=row["mean_map"], source="out/step8_final_configs.json")
    register("submission_v8_ens.csv", "Equal RRF(k=60), v6_3 + v7_1 + v7_2; post-Kaggle choice",
             "Not measured", source="out/ensemble_final.md")
    result["out/submission_final.csv"] = {**result["submission_ppr_alpha0.3.csv"],
                                           "method": "Archived prediction-identical alias of PPR alpha 0.3"}
    return result


def write_summary(artifacts, registry, inter):
    confirmed = sum(any(value is not None for value in scores) for file, scores in KAGGLE.items() if file in artifacts)
    primary_files = sum(not file.startswith("out/") for file in artifacts)
    lines = ["# Final experiment summary", "",
             f"## Designated final: `{FINAL}`", "",
             "**Kaggle private 0.40253 / public 0.41907.** The final decision retains v6_3. "
             "v8_ens (private 0.39991 / public 0.41669), v7_1 (private 0.40078 / public 0.41615) "
             "and v7_2 (private 0.39924 / public 0.41519) did not beat it.", "",
             "## Total Kaggle submissions", "",
             f"**{confirmed} documented Kaggle submissions with reported scores**, counting each scored file once. "
             f"There are {primary_files} generated primary submission files. v4_3 and v6_2 have no reported "
             "Kaggle result/upload confirmation, so they are excluded from that count. The archived PPR "
             "alias is also not an additional submission.", "",
             "v6_3 is the all_together recipe: 200 candidates/source; weekly-scaled features; stacked folds; "
             "channel/time/season/recency features; binary LambdaRank; five-seed raw-score averaging. "
             "Seed/tree counts: 12345/150, 12346/112, 12347/132, 12348/173, 12349/275, refit on 100%.", "",
             "## Full submission table", "",
             "The single table follows experiment/file-generation chronology (initial PPR refit, baselines, "
             "history+airtime, then v3 through v8). Exact Kaggle upload timestamps are unavailable. "
             "The archived PPR alias is placed beside its canonical file.", "",
             "All MAP values below are **validation** measurements, not scores on final 31–42 predictions. "
             "The scope column distinguishes full validation (4,799 users) from the 960/946-user holdouts. "
             "Full-validation and user-holdout results are not interchangeable. The mean column is exactly "
             "`(fold10 + fold5) / 2`, not a user-count-weighted average. The top-five-by-time target is diagnostic.", "",
             "**— means unavailable/not reported, never zero.** The v5 blend and v8 ensemble have no saved "
             "held-out validation measurement. Kaggle scores are user-reported; scores not supplied for "
             "v4_3 and v6_2 are left blank. The archived PPR alias shares one result, not a separate trial.", "",
             "| File | Method | Validation scope | MAP@5 all / fold10 | MAP@5 fold5 | Mean | MAP@5 top-5 target | Kaggle public | Kaggle private |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    ordered = []
    for name in KAGGLE:
        if name in artifacts:
            ordered.append(name)
        if name == "submission_ppr_alpha0.3.csv" and "out/submission_final.csv" in artifacts:
            ordered.append("out/submission_final.csv")
    ordered.extend(name for name in artifacts if name not in ordered)
    for file in ordered:
        row = registry.get(file, dict(method="Unclassified saved submission", scope="Not measured"))
        public, private = KAGGLE.get("submission_ppr_alpha0.3.csv" if file == "out/submission_final.csv" else file, (None, None))
        label = f"**`{file}`**" if file == FINAL else f"`{file}`"
        values = [label, md(row["method"]), md(row["scope"]), number(row.get("all_score")),
                  number(row.get("fold5")), number(row.get("mean")), number(row.get("top_score")),
                  number(public, 5), number(private, 5)]
        lines.append("| " + " | ".join(values) + " |")
    v6, v71, v72 = registry[FINAL], registry["submission_v7_1.csv"], registry["submission_v7_2.csv"]
    step8 = pd.read_csv(H.OUT / "step8_trials.csv").set_index("name")
    baseline = float(step8.loc["binary_baseline", "mean_map"])
    lines.extend(["", "## Why validation improvements did not change the final decision", "",
                  f"- v6_3 historical mean: **{v6['mean']:.9f}**.",
                  f"- Step 8 retrained binary baseline with mean-MAP early stopping: **{baseline:.9f}**.",
                  f"- v7_1 mean: **{v71['mean']:.9f}**; delta vs Step 8 baseline **{v71['mean'] - baseline:+.9f}**, "
                  f"vs historical v6_3 **{v71['mean'] - v6['mean']:+.9f}**. Kaggle private delta vs v6_3: **-0.00175**.",
                  f"- v7_2 mean: **{v72['mean']:.9f}**; delta vs Step 8 baseline **{v72['mean'] - baseline:+.9f}**, "
                  f"vs historical v6_3 **{v72['mean'] - v6['mean']:+.9f}**. Kaggle private delta vs v6_3: **-0.00329**.",
                  "- v8_ens: no saved validation measurement; Kaggle private **0.39991**, delta vs v6_3 **-0.00262**.",
                  "- The roughly +0.0035 validation improvement did not transfer to Kaggle. Final choice: v6_3.", "",
                  "## Protocol and selection chronology", "",
                  "- Viewing is available only for 11–30. Final feature windows end before 31; the 31–42 "
                  "schedule supplies programme/channel/time airings only. No 31–42 viewing is used.",
                  "- Validation fold10: 11–20 features → 21–30 labels. Fold5: 11–25 features → 26–30 labels. "
                  "The same held-out users are excluded from supervised training in both folds.",
                  "- Early steps selected by top-five-by-time MAP. Step 4 changed selection to all-watched MAP "
                  "after comparing Kaggle ordering. Step 6 used nine public scores only to calibrate the target "
                  "definition; all-watched won. Both Step 8 winning configurations retained binary labels.",
                  "- Step 7 initially selected v6_1/v6_2 using fold10. v6_3 was an additional **post-hoc "
                  "choice after seeing fold5**, using the original fold10-NDCG early-stopping tree counts.",
                  "- Step 8 predeclared the mean of fold10/fold5 all-watched MAP for early stopping and selection. "
                  "These repeatedly used validation folds are selection data, not untouched test estimates.",
                  "- v8 is a file-only equal RRF ensemble **chosen after seeing Kaggle scores**. "
                  "It did not beat v6_3; the designated final is explicitly v6_3.", "",
                  "## Content-based comparison (no Kaggle submission file)", ""])
    actual_top5, actual_all = H.get_eval_targets(inter, H.EVAL_BLOCK_VAL)
    m2 = read_val(H.OUT / "val_M2_content.csv")
    ma, n = H.map_at_k(m2, actual_all, 5)
    mt, _ = H.map_at_k(m2, actual_top5, 5)
    lines.extend([f"Plain M2: full-validation all-watched MAP@5 **{ma:.6f}**, top-five-target MAP@5 "
                  f"**{mt:.6f}**, {n:,} users. Its final-profile examples are in `examples_final.md`.", "",
                  "## Submission checks and fingerprints", "",
                  "Every listed submission passed: 1,260 template-ordered unique users; exactly five distinct, "
                  "nonzero programme IDs per row; all IDs in the 6,636-programme final airing pool.", "",
                  "Fingerprints below identify the archived files exactly:", ""])
    for file in ordered:
        item = artifacts[file]
        lines.append(f"- `{file}`: {item['rows']} rows; {item['programmes']} distinct programmes; SHA-256 `{item['sha256']}`.")
    lines.extend(["", "## Validation metric provenance", ""])
    for file in ordered:
        lines.append(f"- `{file}`: `{registry.get(file, {}).get('source', 'not measured')}`.")
    lines.extend(["", "## Committed experiment logs", ""])
    lines.extend(f"- `{path.relative_to(H.EXAM)}`" for path in sorted(H.OUT.rglob("*.log")))
    lines.extend(["", "The accompanying experiment Markdown reports, metric/configuration tables and "
                  "saved validation prediction exports are archived alongside these logs. "
                  "`out/test_ppr.csv` is an auxiliary multi-column prediction export, not a separate Kaggle submission.", ""])
    path = H.OUT / "final_summary.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def run():
    print(__doc__, flush=True)
    pool = H.load_pool(H.EVAL_BLOCK_FINAL)
    artifacts = validate_all(pool)
    inter, progs = H.load_interactions(), H.load_programmes()
    examples = write_examples(artifacts, inter, progs, pool)
    summary = write_summary(artifacts, metric_registry(inter), inter)
    print(f"Wrote {examples}\nWrote {summary}", flush=True)
    print(f"Validated {len(artifacts)} submission artifacts. Final: {FINAL}; private 0.40253 / public 0.41907.", flush=True)


def main():
    H.OUT.mkdir(exist_ok=True)
    with (H.OUT / "final_reports.log").open("w", encoding="utf-8", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            run()


if __name__ == "__main__":
    main()
