#!/usr/bin/env python3
"""Deadline-aware coverage experiment: B first, optional budgeted C second.

Run from exam/: .venv/bin/python -u src/step9_coverage.py --deadline 15:00
Live log: out/step9_stdout.log. Default timezone Europe/Kyiv, TODAY, reserving
10 minutes for the user's Kaggle upload. The script writes files, not uploads.

IDEA CREDIT: the per-airing coverage definition and clipped airtime multiplier
were described by classmate Vladimir Kuznetsov. No code or files were shared.

A unions raw sessions globally per user/channel before coverage. Full airing
duration is the denominator; numerator is clipped to each feature/label window.
n50/n80 thresholds use exact integer comparisons. mean coverage is conditional
on positively viewed airings (unseen pairs zero); unconditional mean is cached.
tv_show_id=0 is never an item. No 31-42 viewing is used.

B scores all target-pool items with positive historical n50, not just cached
LTR candidates: (n50 / training_weeks) * sqrt(clip(target_weekly_airings /
training_weekly_airings, .25, 4)). No smoothing. Train supply=0,target>0 gives
multiplier 2; both zero gives 1. Airing supply counts any positive overlap with
the window. Popularity fills short/empty lists. Saved v6_3 holdout scores and
frequency scores are compared on the SAME users under all-watched MAP@5.
Write submission_v9_freq.csv before starting C.

C keeps the v6_3 size-200/weekly/stacked/five-seed recipe, adds four coverage
features, and compares binary all-watched versus >=one 80%-covered airing
labels. Evaluation/early stopping ALWAYS uses mean-fold ALL-WATCHED MAP@5;
the changed labels affect fitting only. Original user holdout is preserved.
Best label variant is refit on 100%, producing submission_v9_ltr.csv.

C is skipped if its conservative estimate exceeds 40 minutes or time left
before the upload reserve. Observed fit times update the estimate; per-tree
callbacks enforce the absolute deadline and C's wall-clock limit. Partial
ensembles are never written. If C is skipped, B remains ready for submission.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import sparse

import final_build as FB
import harness as H
import step5_exposure as E
import step6_ltr as S
import step7_ltr_plus as P
import step7_feature_cache as C
from step7_refit_all_together import Tee
import step8_exposure_features as F
import step8_ltr_graded as G
import step9_coverage_features as V


def log(message=""):
    print(message, flush=True)


class DeadlineStop(RuntimeError):
    pass


class SkipC(RuntimeError):
    pass


class Budget:
    def __init__(self, deadline="15:00", timezone="Europe/Kyiv", reserve=10, c_minutes=40,
                 now=None, clock=None):
        self.now = now or (lambda: datetime.now(ZoneInfo(timezone)))
        self.clock = clock or time.monotonic
        hour, minute = map(int, deadline.split(":"))
        self.deadline = self.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
        self.work_end = self.deadline - timedelta(minutes=reserve)
        self.c_limit = min(c_minutes, 40) * 60
        self.c_started = None

    def remaining(self):
        return (self.work_end - self.now()).total_seconds()

    def check(self, stage):
        if self.remaining() <= 0:
            raise DeadlineStop(f"Stopped at {stage}: upload reserve begins at {self.work_end.isoformat()}")

    def begin_c(self, estimate):
        if estimate > self.c_limit or estimate > self.remaining():
            raise SkipC(f"Estimated C {estimate / 60:.1f} min exceeds available "
                        f"{min(self.c_limit, self.remaining()) / 60:.1f} min (C cap <=40 min)")
        self.c_started = self.clock()

    def check_c(self, stage, estimate_remaining=0):
        elapsed = self.clock() - self.c_started
        available = min(self.c_limit - elapsed, self.remaining())
        if available <= 0 or estimate_remaining > available:
            raise SkipC(f"C skipped/stopped at {stage}: remaining estimate {estimate_remaining / 60:.1f} min, "
                        f"available {available / 60:.1f} min; elapsed C {elapsed / 60:.1f} min")


@dataclass
class Context:
    name: str
    users: list
    pool: set
    popularity: pd.Series
    targets: dict
    training_weeks: int
    target_weeks: int


def load_context(root, name, fp):
    metadata = C.inspect_cache(root / name, fp, "fold")
    assert metadata is not None
    with np.load(root / name / "aligned_arrays.npz", allow_pickle=False) as arrays:
        users = arrays["users"].tolist()
        ptr, ids = arrays["target_ptr"], arrays["target_item_ids"]
        targets = {user: set(ids[ptr[i]:ptr[i + 1]].tolist()) for i, user in enumerate(users)}
        return Context(name, users, set(arrays["pool"].tolist()),
                       pd.Series(arrays["popularity_values"], index=arrays["popularity_ids"]), targets,
                       metadata["training_weeks"], metadata["target_weeks"])


def target_supply(root, fp, budget):
    path = root / "target31_supply"
    if C.inspect_cache(path, fp, "step9_supply") is None:
        # Schedule only. It is never joined to viewing from 31-42.
        frame = V.read_airings(["31-42"], E.epoch(E.FINAL_START), E.epoch(E.FINAL_END), budget.check, log)
        counts = frame.groupby("tv_show_id").size()
        with C.staged_cache(path) as folder:
            np.savez_compressed(folder / "supply.npz", items=counts.index.to_numpy(), counts=counts.to_numpy())
            C.write_manifest(folder, {"idea_credit": V.CREDIT, "usage": "target airings only; no future viewing"}, fp, "step9_supply")
    with np.load(path / "supply.npz", allow_pickle=False) as arrays:
        return pd.Series(arrays["counts"], index=arrays["items"])


def frequency(context, pairs, train_supply, next_supply):
    relevant = pairs.reset_index()
    relevant = relevant.loc[(relevant.n50 > 0) & relevant.tv_show_id.isin(context.pool)].copy()
    relevant["score"] = relevant.n50 / context.training_weeks * V.multiplier(
        relevant.tv_show_id.map(train_supply).fillna(0).to_numpy(),
        relevant.tv_show_id.map(next_supply).fillna(0).to_numpy(), context.training_weeks, context.target_weeks)
    relevant = relevant.sort_values(["user_id", "score", "tv_show_id"], ascending=[True, False, True])
    personal = relevant.groupby("user_id").tv_show_id.apply(lambda items: items.iloc[:5].tolist()).to_dict()
    pop = context.popularity.index.astype(int).tolist()
    result = {}
    for user in context.users:
        row = list(personal.get(user, []))
        for item in pop:
            if len(row) == 5:
                break
            if item not in row:
                row.append(item)
        assert len(row) == len(set(row)) == 5 and set(row) <= context.pool
        result[user] = row
    return result


def evaluate_frequency(contexts, pairs, supplies, heldout):
    row = {"method": "coverage_frequency", "evaluation_target": "all-watched"}
    for context, training, target in zip(contexts, ("train10", "train15"), ("label10", "label5")):
        recs = frequency(context, pairs[training], supplies[training], supplies[target])
        recs = {user: recs[user] for user in context.users if user in heldout}
        score, count = H.map_at_k(recs, context.targets, 5)
        assert count == len(set(context.users) & heldout)
        row[f"map_{context.name}"] = score
        row[f"users_{context.name}"] = count
    row["mean_map"] = (row["map_fold10"] + row["map_fold5"]) / 2
    old = pd.read_csv(H.OUT / "step7_ablations.csv").set_index("recipe").loc["all_together"]
    reference = {"method": "v6_3_saved_holdout", "evaluation_target": "all-watched",
                 "map_fold10": float(old.map_all_fold10), "map_fold5": float(old.map_all_fold5),
                 "mean_map": float((old.map_all_fold10 + old.map_all_fold5) / 2),
                 "users_fold10": int(old.users_fold10), "users_fold5": int(old.users_fold5)}
    assert row["users_fold10"] == reference["users_fold10"] and row["users_fold5"] == reference["users_fold5"]
    table = pd.DataFrame([reference, row])
    table.to_csv(H.OUT / "step9_frequency_validation.csv", index=False)
    log("B: SAME-USER HOLDOUT MAP@5, ALL-WATCHED\n" + table.to_string(index=False))


def estimate_c():
    prior = H.OUT / "step7_v6_3_refit.log"
    matches = re.findall(r"Completed in ([0-9.]+)s", prior.read_text(encoding="utf-8")) if prior.exists() else []
    # Prior total included feature rebuilding and five full refits. Three times
    # that measured total plus prep is a deliberately conservative 15-fit bound.
    return 3 * float(matches[-1]) + 60 if matches else 3600.0


def append_features(data, pairs, train_supply, next_supply):
    values = data.rows.join(pairs[["n50", "n80", "mean_coverage"]], on=["user_id", "tv_show_id"]).fillna(0)
    extra = np.column_stack((values.n50.to_numpy() / data.training_weeks,
                             values.n80.to_numpy() / data.training_weeks, values.mean_coverage.to_numpy(),
                             V.multiplier(data.rows.tv_show_id.map(train_supply).fillna(0).to_numpy(),
                                          data.rows.tv_show_id.map(next_supply).fillna(0).to_numpy(),
                                          data.training_weeks, data.target_weeks))).astype(np.float32)
    assert np.isfinite(extra).all()
    data.base = sparse.hstack([data.base, sparse.csr_matrix(extra)], format="csr", dtype=np.float32)
    data.names = data.names + V.FEATURE_NAMES


def ltr(contexts, pairs, supplies, root, fp, base_root, base_fp, train, heldout, budget, args):
    import lightgbm as lgb
    estimate = estimate_c()
    log(f"C conservative initial estimate: {estimate / 60:.1f} min; hard cap {budget.c_limit / 60:g} min")
    budget.begin_c(estimate)
    datasets = F.load_base(base_root, base_fp)
    label_windows = ("label10", "label5")
    for data, training, target in zip(datasets, ("train10", "train15", "train20"), ("label10", "label5", "target31")):
        budget.check_c("coverage feature construction")
        append_features(data, pairs[training], supplies[training], supplies[target])
    for data, label_window in zip(datasets[:2], label_windows):
        label_pairs = V.load_pairs(root, label_window, fp)
        values = data.rows.join(label_pairs[["n80"]], on=["user_id", "tv_show_id"]).fillna(0)
        data.labels["coverage80"] = (values.n80.to_numpy() >= 1).astype(np.int8)
        log(f"C {data.name}: all-watched positives={int(data.labels['binary'].sum()):,}; "
            f"coverage80 positives={int(data.labels['coverage80'].sum()):,}; fixed cached candidates")
    folds, final = datasets[:2], datasets[2]
    train_idx = [np.flatnonzero(data.rows.user_id.isin(train)) for data in folds]
    valid_idx = [np.flatnonzero(data.rows.user_id.isin(heldout)) for data in folds]
    tx = sparse.vstack([data.base[idx] for data, idx in zip(folds, train_idx)], format="csr")
    vx = sparse.vstack([data.base[idx] for data, idx in zip(folds, valid_idx)], format="csr")
    tg = [group for data, idx in zip(folds, train_idx) for group in S.query_groups(data.rows.iloc[idx])]
    vg = [group for data, idx in zip(folds, valid_idx) for group in S.query_groups(data.rows.iloc[idx])]
    assert all(set(data.rows.iloc[idx].user_id).isdisjoint(heldout) for data, idx in zip(folds, train_idx))
    metric = G.MeanFoldMAP(folds, valid_idx)
    timings, results = [], []
    completed = 0
    model_fp = {"coverage": fp, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "max_trees": args.max_trees, "split_sha256": hashlib.sha256((base_root / "user_split.json").read_bytes()).hexdigest()}
    namespace = hashlib.sha256(json.dumps(model_fp, sort_keys=True).encode()).hexdigest()[:16]
    for label in ("binary", "coverage80"):
        ty = np.concatenate([data.labels[label][idx] for data, idx in zip(folds, train_idx)])
        vy = np.concatenate([data.labels[label][idx] for data, idx in zip(folds, valid_idx)])
        scores, iterations = np.zeros(vx.shape[0]), {}
        for seed in P.SEEDS:
            budget.check_c(f"{label} seed {seed}")
            cache = root / "ltr_seeds" / namespace / label / str(seed)
            metadata = C.inspect_cache(cache, model_fp, "step9_seed")
            if metadata is not None:
                with np.load(cache / "heldout.npz", allow_pickle=False) as arrays:
                    prediction = arrays["scores"]
                trees = metadata["trees"]
                log(f"  reused {label} seed {seed}, trees={trees}")
            else:
                begin = time.monotonic()
                def guard(env):
                    budget.check_c(f"{label} seed {seed}, iteration {env.iteration + 1}")
                model = lgb.LGBMRanker(**{**P.model_params(seed, args.max_trees), "metric": "None", "label_gain": [0, 1]})
                log(f"  fitting coverage LTR {label}, seed {seed}")
                model.fit(tx, ty, group=tg, eval_set=[(vx, vy)], eval_group=[vg], eval_at=[5],
                          eval_metric=metric, feature_name=folds[0].names,
                          callbacks=[guard, lgb.early_stopping(50, first_metric_only=True), lgb.log_evaluation(25)])
                assert set(model.evals_result_["valid_0"]) == {"mean_fold_map5"}
                trees = max(int(model.best_iteration_), 1)
                prediction = model.booster_.predict(vx, num_iteration=trees)
                timings.append(time.monotonic() - begin)
                with C.staged_cache(cache) as folder:
                    np.savez_compressed(folder / "heldout.npz", scores=prediction)
                    C.write_manifest(folder, {"trees": trees, "label": label, "seed": seed}, model_fp, "step9_seed")
            assert len(prediction) == len(scores) and np.isfinite(prediction).all()
            scores += prediction / len(P.SEEDS)
            iterations[str(seed)] = trees
            completed += 1
            if timings:
                remaining_estimate = max(timings) * (15 - completed) * 1.35
                log(f"  updated C remaining estimate {remaining_estimate / 60:.1f} min")
                budget.check_c("updated timing projection", remaining_estimate)
        fold_scores = metric.values(scores)
        result = {"label": label, "map_fold10": fold_scores[0], "map_fold5": fold_scores[1],
                  "mean_map": float(np.mean(fold_scores)), "iterations": iterations}
        results.append(result)
        log("C VALIDATION: " + json.dumps(result))
        pd.DataFrame(results).to_csv(H.OUT / "step9_ltr_validation.csv", index=False)
    best = sorted(results, key=lambda row: (-row["mean_map"], row["label"]))[0]
    log("C selected by mean-fold ALL-WATCHED MAP: " + json.dumps(best))
    x = sparse.vstack([data.base for data in folds], format="csr")
    y = np.concatenate([data.labels[best["label"]] for data in folds])
    groups = [group for data in folds for group in S.query_groups(data.rows)]
    final_scores = np.zeros(len(final.rows))
    model_dir = H.OUT / "step9_models"
    model_dir.mkdir(exist_ok=True)
    gains = np.zeros(len(final.names))
    for seed in P.SEEDS:
        budget.check_c(f"full refit seed {seed}")
        def guard(env):
            budget.check_c(f"full refit seed {seed}, iteration {env.iteration + 1}")
        model = lgb.LGBMRanker(**{**P.model_params(seed, best["iterations"][str(seed)]), "metric": "None", "label_gain": [0, 1]})
        model.fit(x, y, group=groups, feature_name=final.names, callbacks=[guard])
        final_scores += model.booster_.predict(final.base) / len(P.SEEDS)
        gains += model.booster_.feature_importance(importance_type="gain") / len(P.SEEDS)
        model.booster_.save_model(str(model_dir / f"seed{seed}.txt"))
    budget.check_c("complete five-seed output")
    recs = G.predictions(final, np.arange(len(final.rows)), final_scores)
    FB.check_and_write(recs, H.EXAM / "submission_v9_ltr.csv", final.phase.pool, final.phase.popularity, log=log)
    importance = pd.DataFrame({"feature": final.names, "mean_gain": gains}).sort_values("mean_gain", ascending=False)
    importance.to_csv(H.OUT / "step9_importances.csv", index=False)
    log("C FEATURE IMPORTANCES\n" + importance.to_string(index=False))
    return best


def run(args):
    budget = Budget(args.deadline, args.timezone, args.reserve_minutes, args.c_max_minutes)
    state = {"idea_credit": V.CREDIT, "deadline": budget.deadline.isoformat(), "work_end": budget.work_end.isoformat(),
             "frequency_ready": False, "C": "not started", "designated_final_so_far": "submission_v6_3.csv"}
    log(__doc__)
    log(V.CREDIT)
    log(f"Current time {budget.now().isoformat()}; deadline {budget.deadline.isoformat()}; upload reserve begins {budget.work_end.isoformat()}")
    log("Coverage means: positively viewed airings; all-airing mean also cached. Supply=0 policies: target>0 -> multiplier2; both0 ->1.")
    try:
        budget.check("startup")
        base_fp = C.source_fingerprint()
        fp = V.fingerprint(base_fp)
        contexts = [load_context(args.base_cache, name, base_fp) for name in ("fold10", "fold5", "final")]
        split = json.loads((args.base_cache / "user_split.json").read_text())
        train, heldout = set(split["train_users"]), set(split["heldout_users"])
        assert not train & heldout
        V.ensure_coverage(args.cache_dir, fp, budget.check, log)
        pairs = {name: V.load_pairs(args.cache_dir, name, fp) for name in ("train10", "train15", "train20")}
        supplies = {name: V.load_supply(args.cache_dir, name, fp) for name in V.WINDOWS}
        supplies["target31"] = target_supply(args.cache_dir, fp, budget)
        recs = frequency(contexts[2], pairs["train20"], supplies["train20"], supplies["target31"])
        FB.check_and_write(recs, H.EXAM / "submission_v9_freq.csv", contexts[2].pool, contexts[2].popularity, log=log)
        state["frequency_ready"] = True
        evaluate_frequency(contexts[:2], pairs, supplies, heldout)
        log("B READY: submission_v9_freq.csv — upload is performed by the user before 15:00")
        if args.skip_ltr:
            state["C"] = "skipped by --skip-ltr"
            log("C SKIPPED by --skip-ltr")
        else:
            try:
                state["C_result"] = ltr(contexts, pairs, supplies, args.cache_dir, fp, args.base_cache,
                                         base_fp, train, heldout, budget, args)
                state["C"] = "completed; submission_v9_ltr.csv ready"
            except SkipC as error:
                state["C"] = f"skipped: {error}"
                log("C SKIPPED: " + str(error))
    except DeadlineStop as error:
        state["stopped"] = str(error)
        log(str(error))
    finally:
        (H.OUT / "step9_result.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return 0 if state["frequency_ready"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", default="15:00")
    parser.add_argument("--timezone", default="Europe/Kyiv")
    parser.add_argument("--reserve-minutes", type=float, default=10)
    parser.add_argument("--c-max-minutes", type=float, default=40)
    parser.add_argument("--max-trees", type=int, default=1000)
    parser.add_argument("--skip-ltr", action="store_true")
    parser.add_argument("--base-cache", type=Path, default=H.OUT / "cache" / "step7")
    parser.add_argument("--cache-dir", type=Path, default=H.OUT / "cache" / "step9")
    args = parser.parse_args(argv)
    assert 0 <= args.reserve_minutes and 0 <= args.c_max_minutes <= 40 and args.max_trees > 0
    H.OUT.mkdir(exist_ok=True)
    with (H.OUT / "step9_stdout.log").open("w", encoding="utf-8", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
