#!/usr/bin/env python3
"""Step 8: graded labels/exposure additions/LightGBM grid from cached v6_3.

Run from exam/: .venv/bin/python -u src/step8_ltr_graded.py
Prints progress and logs live to out/step8_stdout.log. Five seeds 12345..12349
for EVERY evaluated recipe. Fixed size-200 candidates, weekly scaling, stacked
folds and v6_3's existing new features. Never use viewing from weeks 31-42.

PREDECLARED SELECTION: unweighted MEAN of fold10 and fold5 holdout all-watched
MAP@5, used both for early stopping and for choosing recipes. The same held-out
users are excluded from both training folds. Graded labels change the ranking
objective; evaluation always uses binary all-watched ground truth. This is a
validation-selection experiment, not an untouched test estimate.

Staged, bounded search (not an exhaustive Cartesian product):
  1. Binary baseline; airings/day quartiles separately; each new exposure group
     separately; all new exposures. Channel share is an already-present control.
  2. Eight hyperparameter settings on the binary baseline (one factor: grid).
  3. Pick the best label and exposure setting from stage 1, then test their
     combination with ALL eight grid settings; report the best combination.
The best two distinct configurations across all stages are selected by mean
MAP@5 and refitted on 100% at their per-seed early-stopped tree counts.

Grid: leaves 31/63, rate .05/.1, min child 20/50. Baseline preserves min child 30.
LambdaRank, eval_at=5, max 1000 trees, 50-round early stopping on mean-fold MAP.
The saved Step 7 v6_3 scores are printed as historical reference only: its old
tree counts were chosen on fold10 NDCG, whereas all Step 8 fits stop on mean MAP.

Step 7 caches are immutable. New features/labels and completed per-seed holdout
fits are cached under out/cache/step8 for reuse/resumption. New time exposures
require one additional raw TRAINING-only log pass; no PPR/candidate rebuild.
Writes out/step8_* tables, models, config/importance reports and final files
submission_v7_1.csv and submission_v7_2.csv. Current final remains v6_3.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

import harness as H
import final_build as FB
import step6_ltr as S
import step7_ltr_plus as P
import step7_feature_cache as C
from step7_refit_all_together import Tee
import step8_exposure_features as F

BASE_PARAMS = {"num_leaves": 31, "learning_rate": .05, "min_child_samples": 30}
GRID = [dict(zip(BASE_PARAMS, values)) for values in product((31, 63), (.05, .1), (20, 50))]
ALL_GROUPS = tuple(F.GROUPS)


def log(message=""):
    print(message, flush=True)


def spec(label="binary", extras=(), params=None):
    return {"label": label, "extras": sorted(set(extras)), "params": dict(params or BASE_PARAMS),
            "seeds": list(P.SEEDS), "candidate_size": 200}


def identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]


def make_matrix(data, config, indices):
    columns = sorted({col for group in config["extras"] for col in F.GROUPS[group]})
    base = data.base[indices]
    if not columns:
        return base, data.names
    extra = data.additional[indices][:, columns]
    return sparse.hstack([base, extra], format="csr", dtype=np.float32), data.names + [F.ADDITIONAL_NAMES[i] for i in columns]


def predictions(data, indices, scores):
    rows = data.rows.iloc[indices].copy()
    rows["score"] = scores
    rows = rows.sort_values(["user_id", "score", "tv_show_id"], ascending=[True, False, True])
    ranked = {user: list(zip(frame.tv_show_id.astype(int), frame.score.astype(float)))[:200]
              for user, frame in rows.groupby("user_id", sort=False)}
    return S.recommend_subset(data.phase, ranked, set(rows.user_id))


class MeanFoldMAP:
    """Vectorized custom metric with full-target denominators and cold fallback.

    Candidate rows must be contiguous by user and ascending by item ID. Stable
    score sorting thus exactly matches the submission's ID tiebreak. Padded
    matrix columns contain only candidates, never synthetic positive injection.
    """
    def __init__(self, folds, held_indices):
        groups, binary, denominators, cold, fallback, fold_ids = [], [], [], [], [], []
        for fold_id, (data, indices) in enumerate(zip(folds, held_indices)):
            selected = data.rows.iloc[indices]
            pop = data.phase.popularity.index.astype(int).tolist()[:5]
            for user, frame in selected.groupby("user_id", sort=False):
                items = frame.tv_show_id.to_numpy()
                assert len(items) >= 5 and np.all(items[1:] > items[:-1])
                groups.append(len(items))
                binary.append(data.labels["binary"][frame.index.to_numpy()])
                actual = data.targets[user]
                assert actual
                denominators.append(min(len(actual), 5))
                cold.append(user in data.phase.cold)
                fallback.append(H.ap_at_k(pop, set(actual), 5))
                fold_ids.append(fold_id)
        self.groups = np.array(groups, dtype=np.int32)
        self.qid = np.repeat(np.arange(len(groups)), self.groups)
        self.position = np.arange(sum(groups)) - np.repeat(np.cumsum(self.groups) - self.groups, self.groups)
        self.binary = np.zeros((len(groups), max(groups)), dtype=np.int8)
        self.binary[self.qid, self.position] = np.concatenate(binary)
        self.denominators = np.asarray(denominators)
        self.cold, self.fallback, self.fold_ids = np.asarray(cold), np.asarray(fallback), np.asarray(fold_ids)
        assert all(np.any(self.fold_ids == fold_id) for fold_id in range(len(folds)))

    def values(self, scores):
        assert len(scores) == len(self.qid) and np.isfinite(scores).all()
        padded = np.full(self.binary.shape, -np.inf)
        padded[self.qid, self.position] = scores
        top = np.argsort(-padded, axis=1, kind="stable")[:, :5]
        hit = np.take_along_axis(self.binary, top, axis=1)
        ap = (hit * np.cumsum(hit, axis=1) / np.arange(1, 6)).sum(axis=1) / self.denominators
        ap[self.cold] = self.fallback[self.cold]
        return [float(ap[self.fold_ids == fold_id].mean()) for fold_id in range(2)]

    def __call__(self, y_true, y_pred, weight, group):
        # LightGBM passes graded y_true here; the fixed binary truth is retained
        # separately so model selection cannot silently change its target.
        return "mean_fold_map5", float(np.mean(self.values(y_pred))), True


def split_users(root, datasets):
    split = json.loads((root / "user_split.json").read_text(encoding="utf-8"))
    train, heldout = set(split["train_users"]), set(split["heldout_users"])
    assert not train & heldout
    assert set().union(*(set(data.phase.users) for data in datasets[:2])) <= train | heldout
    return train, heldout


def fitting_data(folds, config, train_users, heldout):
    train_x, train_y, train_groups, valid_x, valid_y, valid_groups, held_indices = [], [], [], [], [], [], []
    names = None
    for data in folds:
        train_idx = np.flatnonzero(data.rows.user_id.isin(train_users))
        valid_idx = np.flatnonzero(data.rows.user_id.isin(heldout))
        assert set(data.rows.iloc[train_idx].user_id).isdisjoint(heldout)
        x, current = make_matrix(data, config, train_idx)
        vx, _ = make_matrix(data, config, valid_idx)
        assert names is None or names == current
        names = current
        train_x.append(x)
        valid_x.append(vx)
        train_y.append(data.labels[config["label"]][train_idx])
        valid_y.append(data.labels[config["label"]][valid_idx])
        train_groups.extend(S.query_groups(data.rows.iloc[train_idx]))
        valid_groups.extend(S.query_groups(data.rows.iloc[valid_idx]))
        held_indices.append(valid_idx)
    return (sparse.vstack(train_x, format="csr"), np.concatenate(train_y), train_groups,
            sparse.vstack(valid_x, format="csr"), np.concatenate(valid_y), valid_groups, held_indices, names)


class Experiment:
    def __init__(self, datasets, train_users, heldout, cache, fingerprint):
        self.folds = datasets[:2]
        self.train_users, self.heldout = train_users, heldout
        self.cache, self.fingerprint = cache, fingerprint
        self.namespace = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
        self.rows, self.seen = [], {}

    def trial(self, name, stage, config, note=""):
        import lightgbm as lgb
        key = identity(config)
        if key in self.seen:
            result = dict(self.seen[key])
        else:
            log(f"\nFIT {name}: {json.dumps(config, sort_keys=True)}")
            tx, ty, tg, vx, vy, vg, held_indices, names = fitting_data(
                self.folds, config, self.train_users, self.heldout)
            evaluator = MeanFoldMAP(self.folds, held_indices)
            average = np.zeros(vx.shape[0], dtype=np.float64)
            iterations = {}
            for seed in P.SEEDS:
                path = self.cache / "seed_fits" / self.namespace / key / str(seed)
                metadata = C.inspect_cache(path, self.fingerprint, "step8_seed")
                if metadata is not None:
                    assert metadata["config"] == config and metadata["feature_names"] == names
                    with np.load(path / "heldout.npz", allow_pickle=False) as arrays:
                        scores = arrays["scores"]
                    iteration = metadata["best_iteration"]
                    log(f"  seed {seed}: cached mean-fold early stopping, trees={iteration}")
                else:
                    params = {**P.model_params(seed), **config["params"], "metric": "None",
                              "label_gain": [0, 1, 3, 7, 15]}
                    model = lgb.LGBMRanker(**params)
                    log(f"  seed {seed}: train rows={tx.shape[0]:,}, queries={len(tg):,}, features={len(names)}")
                    model.fit(tx, ty, group=tg, eval_set=[(vx, vy)], eval_group=[vg], eval_at=[5],
                              eval_metric=evaluator, feature_name=names,
                              callbacks=[lgb.early_stopping(50, first_metric_only=True), lgb.log_evaluation(25)])
                    assert set(model.evals_result_["valid_0"]) == {"mean_fold_map5"}, (
                        "Early stopping must monitor only the unweighted mean-fold all-watched MAP@5")
                    iteration = max(int(model.best_iteration_), 1)
                    scores = model.booster_.predict(vx, num_iteration=iteration)
                    with C.staged_cache(path) as folder:
                        np.savez_compressed(folder / "heldout.npz", scores=scores)
                        C.write_manifest(folder, {"config": config, "seed": seed, "best_iteration": iteration,
                                                  "feature_names": names,
                                                  "early_stopping_metric": "mean_fold_map5"},
                                         self.fingerprint, "step8_seed")
                assert len(scores) == len(average) and np.isfinite(scores).all()
                iterations[str(seed)] = int(iteration)
                average += scores / len(P.SEEDS)
            fold_scores = evaluator.values(average)
            offset = 0
            # Cross-check custom early-stopping metric against the shared AP
            # harness, including pool/fill/cold fallback and full relevant sets.
            for index, (data, idx) in enumerate(zip(self.folds, held_indices)):
                recs = predictions(data, idx, average[offset:offset + len(idx)])
                score, count = H.map_at_k(recs, data.targets, 5)
                assert count == len(set(data.phase.users) & self.heldout)
                assert np.isclose(score, fold_scores[index], rtol=0, atol=1e-12)
                offset += len(idx)
            result = {"id": key, "config": config, "iterations": iterations,
                      "map_fold10": fold_scores[0], "map_fold5": fold_scores[1],
                      "mean_map": float(np.mean(fold_scores))}
            self.seen[key] = result
        row = {"name": name, "stage": stage, "id": key, "label": config["label"],
               "exposures": "/".join(config["extras"]) or "v6_3_existing", **config["params"],
               "map_fold10": result["map_fold10"], "map_fold5": result["map_fold5"],
               "mean_map": result["mean_map"], "note": note}
        self.rows.append(row)
        log("  RESULT " + json.dumps(row, sort_keys=True))
        self.table().to_csv(H.OUT / "step8_trials.csv", index=False)
        return result

    def table(self):
        return pd.DataFrame(self.rows).sort_values(["mean_map", "name"], ascending=[False, True], ignore_index=True)


def search(experiment):
    baseline = experiment.trial("binary_baseline", "one_change", spec())
    label_trials = [baseline] + [experiment.trial(f"graded_{name}_only", "one_change", spec(label=name))
                                 for name in ("airings", "days")]
    exposure_trials = [baseline]
    for group in F.GROUPS:
        exposure_trials.append(experiment.trial(f"{group}_only", "one_change", spec(extras=[group])))
    exposure_trials.append(experiment.trial("all_exposures_only", "one_change", spec(extras=ALL_GROUPS)))
    experiment.trial("channel_share_existing_control", "one_change", spec(),
                     "Already present in v6_3; same configuration, no duplicate feature/fit")
    for index, params in enumerate(GRID, 1):
        experiment.trial(f"parameter_grid_{index}", "parameters_only", spec(params=params))
    # Staged combination search is declared before observing results, not an
    # undocumented claim of exhaustive tuning over all label/feature products.
    best_label = sorted(label_trials, key=lambda row: (-row["mean_map"], row["id"]))[0]["config"]["label"]
    best_extra = sorted(exposure_trials, key=lambda row: (-row["mean_map"], row["id"]))[0]["config"]["extras"]
    log(f"\nCOMBINATION GRID: label={best_label}, exposures={best_extra}; all eight parameter settings")
    combinations = [experiment.trial(f"combination_grid_{index}", "combination",
                                     spec(best_label, best_extra, params)) for index, params in enumerate(GRID, 1)]
    best_combo = sorted(combinations, key=lambda row: (-row["mean_map"], row["id"]))[0]
    log("BEST TESTED COMBINATION: " + json.dumps(best_combo, sort_keys=True))
    chosen = sorted(experiment.seen.values(), key=lambda row: (-row["mean_map"], row["id"]))[:2]
    assert len(chosen) == 2
    table = experiment.table()
    table["delta_vs_binary_baseline"] = table.mean_map - baseline["mean_map"]
    table.to_csv(H.OUT / "step8_trials.csv", index=False)
    log("\nONE-CHANGE / GRID / COMBINATION TABLE — UNWEIGHTED MEAN MAP@5\n" + table.to_string(index=False))
    return chosen


def final_fit(datasets, chosen, fingerprint):
    import lightgbm as lgb
    folds, final = datasets[:2], datasets[2]
    model_dir = H.OUT / "step8_models"
    model_dir.mkdir(exist_ok=True)
    manifest = []
    for rank, result in enumerate(chosen, 1):
        config = result["config"]
        log(f"\nFINAL submission_v7_{rank}.csv: {json.dumps(result, sort_keys=True)}")
        matrices, labels, groups, names = [], [], [], None
        for data in folds:
            x, current = make_matrix(data, config, np.arange(len(data.rows)))
            assert names is None or names == current
            names = current
            matrices.append(x)
            labels.append(data.labels[config["label"]])
            groups.extend(S.query_groups(data.rows))
        x, y = sparse.vstack(matrices, format="csr"), np.concatenate(labels)
        final_x, final_names = make_matrix(final, config, np.arange(len(final.rows)))
        assert final_names == names
        scores, gain, splits = np.zeros(len(final.rows)), np.zeros(len(names)), np.zeros(len(names))
        for seed in P.SEEDS:
            trees = result["iterations"][str(seed)]
            model = lgb.LGBMRanker(**{**P.model_params(seed, trees), **config["params"],
                                      "metric": "None", "label_gain": [0, 1, 3, 7, 15]})
            log(f"  refit seed {seed} on 100% of stacked queries, trees={trees}")
            model.fit(x, y, group=groups, feature_name=names)
            scores += model.booster_.predict(final_x) / len(P.SEEDS)
            gain += model.booster_.feature_importance(importance_type="gain") / len(P.SEEDS)
            splits += model.booster_.feature_importance(importance_type="split") / len(P.SEEDS)
            model.booster_.save_model(str(model_dir / f"rank{rank}_seed{seed}.txt"))
        recs = predictions(final, np.arange(len(final.rows)), scores)
        filename = f"submission_v7_{rank}.csv"
        checks = FB.check_and_write(recs, H.EXAM / filename, final.phase.pool, final.phase.popularity, log=log)
        importance = pd.DataFrame({"feature": names, "mean_gain": gain, "mean_split": splits}).sort_values(
            ["mean_gain", "feature"], ascending=[False, True])
        importance.to_csv(H.OUT / f"step8_importances_{rank}.csv", index=False)
        log("FEATURE IMPORTANCES\n" + importance.to_string(index=False))
        manifest.append({**result, "file": filename, "feature_names": names, "checks": checks,
                         "selection_metric": "unweighted mean of fold10/fold5 all-watched MAP@5",
                         "early_stopping_metric": "same mean-fold MAP@5", "fingerprint": fingerprint})
    (H.OUT / "step8_final_configs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run(base_root, cache):
    import lightgbm as lgb
    started = time.perf_counter()
    log(__doc__)
    log(f"LightGBM {lgb.__version__}; all scores/selections use the predeclared equal-weight mean-fold rule")
    old = H.OUT / "step7_ablations.csv"
    if old.exists():
        reference = pd.read_csv(old).set_index("recipe").loc["all_together"]
        log(f"Historical v6_3 reference (original fold10-NDCG tree counts): fold10={reference.map_all_fold10:.8f}, "
            f"fold5={reference.map_all_fold5:.8f}, mean={(reference.map_all_fold10 + reference.map_all_fold5) / 2:.8f}")
    base_fp = C.source_fingerprint()
    feature_fp = F.feature_fingerprint(base_root, base_fp)
    metadata = C.inspect_cache(base_root / "fold10", base_fp, "fold")
    log("CATEGORY ONE-HOT MAPPING: " + json.dumps(metadata["category_onehot_mapping"], ensure_ascii=False))
    datasets = F.load_base(base_root, base_fp)
    F.additions(datasets, base_root, cache / "features", base_fp, feature_fp)
    train, heldout = split_users(base_root, datasets)
    log(f"Fixed user split: {len(train):,} training / {len(heldout):,} heldout; no heldout user fitted in either fold")
    trial_fp = {"features": feature_fp, "experiment_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "user_split_sha256": hashlib.sha256((base_root / "user_split.json").read_bytes()).hexdigest()}
    experiment = Experiment(datasets, train, heldout, cache, trial_fp)
    chosen = search(experiment)
    (H.OUT / "step8_selection.json").write_text(json.dumps({
        "selection_metric": "mean = (fold10 + fold5) / 2, both all-watched MAP@5",
        "search": "predeclared staged one-factor ablations, baseline parameter grid, then combination grid",
        "chosen": chosen, "designated_final_so_far": "submission_v6_3.csv",
    }, indent=2) + "\n", encoding="utf-8")
    final_fit(datasets, chosen, trial_fp)
    log(f"Completed in {time.perf_counter() - started:.1f}s; designated final remains v6_3 pending Kaggle results")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, default=H.OUT / "cache" / "step7")
    parser.add_argument("--cache-dir", type=Path, default=H.OUT / "cache" / "step8")
    args = parser.parse_args(argv)
    H.OUT.mkdir(exist_ok=True)
    with (H.OUT / "step8_stdout.log").open("w", encoding="utf-8", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            run(args.base_cache, args.cache_dir)


if __name__ == "__main__":
    main()
