#!/usr/bin/env python3
"""Validation-only LTR improvements; run from exam/ with python -u.

No public/private LB score is consulted for any Step 7 selection. Labels are
all-watched/binary, the target selected in Step 6. Never read 31-42 viewing.
The 31-42 schedule supplies airings only, after selecting the two recipes.

Recall selection: test 50/100/200 per history/PPR0.5/E48_uniform/popularity.
Attainable means recall at the largest TESTED budget (200); also report the
full airing-pool ceiling. Pick the smallest retaining >=95% of macro recall
on training users in BOTH folds. Save each user's recall at every size.

Independent single-change ablations versus v5: candidates, per-week scaling,
stacked folds, five new features, five-seed bagging; plus all together.
One shared 20% user holdout is excluded from ALL fold training. Reuse the
Step 6 split where present; additional users receive a stable hashed split.
Early stopping uses NDCG@5 on the common 11-20 -> 21-30 holdout. Report MAP@5
there as the PRIMARY selection metric, and on the 26-30 holdout diagnostically.
Retrain the v5 recipe on 80%, never evaluate its already-full-data fitted model.

Counts/seconds use source-appropriate weekly rates: whole-history features
/10,15,20 training weeks; last-four /4, last-two /2; target supply and exposure
/10,5,12 target weeks. H already divides by training weeks. Airtime ratio uses
weekly target/train airtime with +1h/week smoothing. Ranks, shares, indicators
and one-hot categories stay dimensionless. Pop distinct users is /train weeks.

Fold 1: 11-20 features -> 21-30 labels. Fold 2: 11-25 features -> 26-30 labels.
Queries are (fold,user), NOT user across concatenated folds. Short-window
qualifying pairs use the existing >=120s AND >=5% aggregated overlap rule;
original full-block interactions/targets are reused for exact v5 comparability.

Final: refit each winning recipe on 100% of its training queries at each seed's
early-stopped tree count. Average raw scores for seeds 12345..12349 when bagged.
Recompute final features on 11-30 with 31-42 airings, write submission_v6_1.csv
and submission_v6_2.csv with the usual checks. Save ablations, per-user recall,
split, models, feature importances, config manifest and report under out/step7*.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse

import final_build as FB
import harness as H
import step5_exposure as E
import step6_ltr as S

SEEDS = tuple(range(12345, 12350))
SIZES = (50, 100, 200)
MIDPOINT = "2020-06-22"  # Monday of ISO week 26
WINDOWS = {
    "train10": (E.TRAIN_START, E.VALIDATION_START),
    "train15": (E.TRAIN_START, MIDPOINT),
    "train20": (E.TRAIN_START, E.FINAL_START),
    "label5": (MIDPOINT, E.FINAL_START),
}
NEW_NAMES = ["user_item_channel_share", "E168_recent", "airs_first_two_weeks",
             "target_distinct_weeks", "last2weeks_item_seconds"]
REPORT: list[str] = []


def log(message=""):
    print(message, flush=True)
    REPORT.append(str(message))


@dataclass
class WindowStats:
    pairs: pd.DataFrame
    context: pd.DataFrame


def aggregate_windows(schedules):
    """One raw-log pass, vectorized overlap joins, separate clipped windows."""
    schedule = pd.concat(schedules.values(), ignore_index=True)
    assert set(schedule.block) <= {"11-20", "21-30"}
    schedule["airing_id"] = np.arange(len(schedule))
    groups = {channel: frame.sort_values("start").reset_index(drop=True)
              for channel, frame in schedule.groupby("channel_id")}
    broadcasts = {name: [] for name in WINDOWS}
    dates = {name: [] for name in WINDOWS}
    contexts = {name: [] for name in WINDOWS if name.startswith("train")}
    keys = ["user_id", "tv_show_id"]
    for chunk, view in enumerate(pd.read_csv(
            H.DATA / "dataset11-30.csv", chunksize=E.IO_CHUNK,
            usecols=["user_id", "vsetv_id", "start_time", "duraton"],
            dtype={"user_id": str, "vsetv_id": "int64", "duraton": "int64"}), 1):
        view = view.loc[(view.start_time < f"{E.FINAL_START} 00:00:00") & (view.duraton > 0)].copy()
        start = E.parse_time(view.start_time, "%Y-%m-%d %H:%M:%S")
        view["start"] = np.maximum(start, E.epoch(E.TRAIN_START))
        view["end"] = np.minimum(start + view.duraton.to_numpy(), E.epoch(E.FINAL_START))
        view = view.loc[view.end > view.start]
        for name in contexts:
            lower, upper = map(E.epoch, WINDOWS[name])
            seconds = np.maximum(np.minimum(view.end, upper) - np.maximum(view.start, lower), 0)
            part = view[["user_id", "vsetv_id"]].rename(columns={"vsetv_id": "channel_id"}).copy()
            part["seconds"] = seconds.to_numpy()
            contexts[name].append(part.loc[part.seconds > 0].groupby(["user_id", "channel_id"]).seconds.sum())
        n_events = 0
        for channel, frame in view.groupby("vsetv_id", sort=False):
            airing = groups.get(channel)
            if airing is None:
                continue
            source, row, begin, end = S.overlap_pairs(
                frame.start.to_numpy(), frame.end.to_numpy(), airing.start.to_numpy(), airing.end.to_numpy())
            if not len(source):
                continue
            events = pd.DataFrame({"user_id": frame.user_id.to_numpy()[source],
                                   "tv_show_id": airing.tv_show_id.to_numpy()[row],
                                   "airing_id": airing.airing_id.to_numpy()[row],
                                   "duration_sum": airing.duration.to_numpy()[row]})
            n_events += len(events)
            for name, (lower_date, upper_date) in WINDOWS.items():
                lower, upper = E.epoch(lower_date), E.epoch(upper_date)
                starts, ends = np.maximum(begin, lower), np.minimum(end, upper)
                valid = np.flatnonzero(ends > starts)
                if not len(valid):
                    continue
                part = events.iloc[valid].copy()
                st, en = starts[valid], ends[valid]
                recent4 = E.epoch(pd.Timestamp(upper_date) - pd.Timedelta(weeks=4))
                recent2 = E.epoch(pd.Timestamp(upper_date) - pd.Timedelta(weeks=2))
                part["seconds"] = en - st
                part["recent"] = np.maximum(en - np.maximum(st, recent4), 0)
                part["last2"] = np.maximum(en - np.maximum(st, recent2), 0)
                broadcasts[name].append(part.groupby(keys + ["airing_id"])[
                    ["seconds", "recent", "last2", "duration_sum"]].sum())
                day_source, day = S.local_day_segments(st, en)
                touched = part.iloc[day_source][keys].copy()
                touched["day"] = day
                dates[name].append(touched.drop_duplicates())
        log(f"  raw chunk {chunk}: {len(view):,} views, {n_events:,} overlaps; {len(WINDOWS)} windows accumulated")
    result = {}
    for name in WINDOWS:
        per_airing = pd.concat(broadcasts.pop(name)).groupby(level=[0, 1, 2]).sum()
        pairs = per_airing.groupby(level=[0, 1]).sum()
        pairs["airings"] = per_airing.groupby(level=[0, 1]).size()
        days = pd.concat(dates.pop(name), ignore_index=True).drop_duplicates(keys + ["day"])
        pairs["days"] = days.groupby(keys).size()
        context = (pd.concat(contexts.pop(name)).groupby(level=[0, 1]).sum().to_frame("seconds")
                   if name in contexts else pd.DataFrame())
        result[name] = WindowStats(pairs, context)
        log(f"  {name}: {len(pairs):,} user/item pairs, {int(pairs.airings.sum()):,} watched broadcasts")
    return result


def qualifying_interactions(stats):
    pairs = stats.pairs
    valid = pairs.loc[(pairs.seconds >= 120) & (pairs.seconds / pairs.duration_sum >= 0.05)]
    return valid.reset_index()[["user_id", "tv_show_id", "seconds"]].rename(columns={"seconds": "overlap_seconds"})


def clip_schedule(schedule, lower, upper):
    result = schedule.copy()
    result["start"] = np.maximum(result.start, E.epoch(lower))
    result["end"] = np.minimum(result.end, E.epoch(upper))
    return result.loc[result.end > result.start].copy()


def custom_phase(users, train, target, feature_end):
    """The 15 -> 5 week fold, using only that fold's training observations."""
    items = np.array(sorted(target.tv_show_id.unique()), dtype=np.int64)
    channels = np.array(sorted(target.channel_id.unique()), dtype=np.int64)
    lower, upper = E.epoch(MIDPOINT), E.epoch(E.FINAL_START)
    source, hours, seconds = E.split_hours(target.start.to_numpy(), target.end.to_numpy(), lower, upper)
    rows = np.searchsorted(items, target.tv_show_id.to_numpy()[source])
    cols = np.searchsorted(channels, target.channel_id.to_numpy()[source]) * 168 + E.local_slots(hours)
    a168 = sparse.coo_matrix((seconds, (rows, cols)), shape=(len(items), len(channels) * 168)).tocsr()
    projection = E.collapse_projection(len(channels))
    habits = E.build_habits(users, channels, feature_end, projection)
    histories, lists = H.build_user_histories(train)
    pool = set(items.tolist())
    return E.Phase(users, items, pool, histories, lists, train, H.build_popularity(train, pool),
                   {user for user in users if not histories.get(user)}, habits,
                   {168: a168, 48: (a168 @ projection).tocsr()})


def candidate_depth(phase, ppr, exposure, rows):
    depth = np.full(len(rows), 201, dtype=np.int32)
    popularity = [int(item) for item in phase.popularity.index][:200]
    for user, frame in rows.groupby("user_id", sort=False):
        history = [item for item, _ in phase.history_lists.get(user, []) if item in phase.pool][:200]
        ranks = {}
        for ranking in (history, [item for item, _ in ppr.ranking[user]],
                        [item for item, _ in exposure.ranking[user]], popularity):
            for rank, item in enumerate(ranking, 1):
                ranks[item] = min(ranks.get(item, 201), rank)
        depth[frame.index.to_numpy()] = [ranks[item] for item in frame.tv_show_id]
    assert np.all(depth <= 200)
    return depth


def week_coverage(target, start):
    # Include all ISO weeks touched by an airing, not just its start week.
    source, day = S.local_day_segments(target.start.to_numpy(), target.end.to_numpy())
    origin = np.datetime64(start, "D").astype(np.int64)
    touched = pd.DataFrame({"tv_show_id": target.tv_show_id.to_numpy()[source], "week": (day - origin) // 7})
    distinct = touched.drop_duplicates().groupby("tv_show_id").size()
    early_end = E.epoch(pd.Timestamp(start) + pd.Timedelta(weeks=2))
    early = set(target.loc[(target.start < early_end) & (target.end > E.epoch(start))].tv_show_id)
    return distinct, early


def new_features(phase, features, stats, target, target_start):
    rows = features.rows
    channels = np.array(sorted(target.channel_id.unique()), dtype=np.int64)
    user_index = pd.Index(phase.users)
    context = stats.context.reset_index()
    total = stats.context.groupby(level="user_id").seconds.sum()
    relevant = context.user_id.isin(user_index) & context.channel_id.isin(channels)
    context = context.loc[relevant]
    values = context.seconds.to_numpy() / context.user_id.map(total).to_numpy()
    user_channel = sparse.coo_matrix((values, (user_index.get_indexer(context.user_id),
                                             np.searchsorted(channels, context.channel_id))),
                                    shape=(len(user_index), len(channels))).tocsr()
    weighted = target.assign(seconds=target.end - target.start).groupby(["tv_show_id", "channel_id"]).seconds.sum()
    weights = weighted / weighted.groupby(level="tv_show_id").transform("sum")
    index = weights.index
    item_channel = sparse.coo_matrix((weights.to_numpy(),
                                     (np.searchsorted(phase.items, index.get_level_values("tv_show_id")),
                                      np.searchsorted(channels, index.get_level_values("channel_id")))),
                                    shape=(len(phase.items), len(channels))).tocsr()
    weeks, early = week_coverage(target, target_start)
    joined = rows.join(stats.pairs[["last2"]], on=["user_id", "tv_show_id"]).fillna({"last2": 0})
    extra = np.zeros((len(rows), len(NEW_NAMES)), dtype=np.float32)
    extra[:, 2] = rows.tv_show_id.isin(early).to_numpy()
    extra[:, 3] = rows.tv_show_id.map(weeks).fillna(0).to_numpy()
    extra[:, 4] = joined.last2.to_numpy()
    positions_by_user = rows.groupby("user_id", sort=False).indices
    for begin in range(0, len(phase.users), E.USER_CHUNK):
        end = min(begin + E.USER_CHUNK, len(phase.users))
        shares = (user_channel[begin:end] @ item_channel.T).toarray()
        habits = (phase.habits["168_recent"][begin:end] @ phase.airings[168].T).toarray()
        for offset, user in enumerate(phase.users[begin:end]):
            indices = positions_by_user[user]
            positions = np.searchsorted(phase.items, rows.tv_show_id.to_numpy()[indices])
            extra[indices, 0] = shares[offset, positions]
            extra[indices, 1] = habits[offset, positions]
        log(f"  new channel/time features: users {begin + 1:,}-{end:,}")
    assert np.isfinite(extra).all()
    return sparse.csr_matrix(extra)


@dataclass
class Fold:
    name: str
    phase: E.Phase
    features: S.Features
    extra: sparse.csr_matrix
    depth: np.ndarray
    labels: np.ndarray
    targets: dict
    training_weeks: int
    target_weeks: int
    train_airtime: np.ndarray


def prepare_fold(name, phase, stats, train_schedule, target_schedule, target_start,
                 targets, training_weeks, target_weeks, catalogue, categories):
    log(f"\nPREPARE {name}: {training_weeks} training weeks -> {target_weeks} target weeks")
    ppr, exposure = S.ppr_signal(phase, 0.5, feature_values=True), S.exposure_signal(phase)
    features = S.build_features(phase, ppr, exposure, stats.pairs, train_schedule, target_schedule,
                                catalogue, categories, candidate_size=200)
    depth = candidate_depth(phase, ppr, exposure, features.rows)
    extra = new_features(phase, features, stats, target_schedule, target_start)
    if name == "final":
        # Structural placeholder only; this array is never passed to fitting.
        labels = np.zeros(len(features.rows), dtype=np.int32)
        log("  final inference features: no target viewing or relevance labels consumed")
    else:
        labels = S.labels_for(features, targets, "all_watched")
    train_airtime = features.rows.tv_show_id.map(S.schedule_summary(train_schedule).airtime).fillna(0).to_numpy()
    # Model fitting/final fallback needs only histories, popularity and IDs now.
    phase.habits.clear()
    phase.airings.clear()
    return Fold(name, phase, features, extra, depth, labels, targets,
                training_weeks, target_weeks, train_airtime)


def shared_split(users):
    previous = H.OUT / "step6_user_split.json"
    if previous.exists():
        split = json.loads(previous.read_text())
        train, heldout = set(split["train_users"]), set(split["heldout_users"])
        assert not train & heldout
        train, heldout = train & set(users), heldout & set(users)
        for user in sorted(set(users) - train - heldout):
            value = int(hashlib.sha256(f"12345:{user}".encode()).hexdigest()[:16], 16) / 2 ** 64
            (heldout if value < 0.2 else train).add(user)
        log("  Reusing Step 6 user split; new fold-only users are assigned by a seed-12345 stable hash.")
    else:
        train, heldout = S.user_split(users)
    assert train and heldout and train | heldout == set(users) and not train & heldout
    (H.OUT / "step7_user_split.json").write_text(json.dumps(
        {"seed": 12345, "train_users": sorted(train), "heldout_users": sorted(heldout)}, indent=2) + "\n")
    return train, heldout


def select_candidate_size(folds, train_users):
    details, summary = [], []
    for fold in folds:
        for user, frame in fold.features.rows.groupby("user_id", sort=False):
            actual = set(fold.targets[user])
            for size in SIZES:
                candidates = set(frame.tv_show_id.to_numpy()[fold.depth[frame.index] <= size])
                details.append({"fold": fold.name, "user_id": user, "size": size,
                                "split": "train" if user in train_users else "heldout",
                                "recall": len(candidates & actual) / len(actual),
                                "pool_ceiling": len(fold.phase.pool & actual) / len(actual),
                                "candidate_count": len(candidates)})
    frame = pd.DataFrame(details)
    frame.to_csv(H.OUT / "step7_candidate_recall_per_user.csv", index=False)
    for (fold, split, size), group in frame.groupby(["fold", "split", "size"]):
        summary.append({"fold": fold, "split": split, "size": int(size), "users": len(group),
                        "macro_recall": float(group.recall.mean()), "pool_ceiling": float(group.pool_ceiling.mean()),
                        "mean_candidates": float(group.candidate_count.mean())})
    table = pd.DataFrame(summary)
    table.to_csv(H.OUT / "step7_candidate_recall.csv", index=False)
    log("\nCANDIDATE RECALL\n" + table.to_string(index=False))
    training = table.loc[table.split == "train"].set_index(["fold", "size"])
    selected = 200
    for size in SIZES:
        if all(training.loc[(fold.name, size), "macro_recall"] + 1e-12 >=
               0.95 * training.loc[(fold.name, 200), "macro_recall"] for fold in folds):
            selected = size
            break
    log(f"CANDIDATE BUDGET: {selected}/source; >=95% of the 200/source tested ceiling in BOTH training folds.")
    return selected


def scaled_names(names, include_new):
    result = list(names) + (NEW_NAMES if include_new else [])
    replacement = {
        "user_item_seconds": "user_item_seconds_per_week", "user_item_airings": "user_item_airings_per_week",
        "user_item_days": "user_item_days_per_week", "last4weeks_seconds": "last4weeks_seconds_per_week",
        "popularity_users": "popularity_users_per_week", "global_item_seconds": "global_item_seconds_per_week",
        "target_airtime": "target_airtime_per_week", "target_airings": "target_airings_per_week",
        "exposure_score": "exposure_score_per_target_week", "airtime_ratio": "airtime_ratio_weekly",
        "E168_recent": "E168_recent_per_target_week", "target_distinct_weeks": "target_week_coverage",
        "last2weeks_item_seconds": "last2weeks_item_seconds_per_week",
    }
    return [replacement.get(name, name) for name in result]


def design(fold, config, indices):
    base = fold.features.matrix[indices]
    names = fold.features.names + (NEW_NAMES if config["new"] else [])
    matrix = sparse.hstack([base, fold.extra[indices]], format="csr") if config["new"] else base.copy()
    if config["scale"]:
        divisor = np.ones(len(names), dtype=np.float32)
        for feature in ("user_item_seconds", "user_item_airings", "user_item_days", "popularity_users", "global_item_seconds"):
            divisor[names.index(feature)] = fold.training_weeks
        divisor[names.index("last4weeks_seconds")] = 4
        for feature in ("target_airtime", "target_airings", "exposure_score"):
            divisor[names.index(feature)] = fold.target_weeks
        if config["new"]:
            divisor[names.index("E168_recent")] = fold.target_weeks
            divisor[names.index("target_distinct_weeks")] = fold.target_weeks
            divisor[names.index("last2weeks_item_seconds")] = 2
        matrix = matrix.multiply(1.0 / divisor).tocsr()
        target_air = base[:, fold.features.names.index("target_airtime")].toarray().ravel()
        ratio = (target_air / fold.target_weeks + 3600) / (fold.train_airtime[indices] / fold.training_weeks + 3600)
        col = names.index("airtime_ratio")
        matrix = sparse.hstack([matrix[:, :col], sparse.csr_matrix(ratio[:, None]), matrix[:, col + 1:]],
                               format="csr", dtype=np.float32)
        names = scaled_names(fold.features.names, config["new"])
    assert np.isfinite(matrix.data).all()
    return matrix, names


def recipes(size):
    base = {"size": 50, "scale": False, "stack": False, "new": False, "seeds": [12345]}
    return {
        "previous_v5": dict(base),
        "candidates_only": {**base, "size": size},
        "weekly_scaling_only": {**base, "scale": True},
        "stacked_folds_only": {**base, "stack": True},
        "new_features_only": {**base, "new": True},
        "five_seed_bagging_only": {**base, "seeds": list(SEEDS)},
        "all_together": {"size": size, "scale": True, "stack": True, "new": True, "seeds": list(SEEDS)},
    }


def indices_for(fold, config, users=None):
    mask = fold.depth <= config["size"]
    if users is not None:
        mask &= fold.features.rows.user_id.isin(users).to_numpy()
    return np.flatnonzero(mask)


def model_params(seed, trees=1000):
    return dict(objective="lambdarank", metric="ndcg", random_state=seed,
                learning_rate=0.05, num_leaves=31, min_child_samples=30,
                n_estimators=trees, colsample_bytree=0.9, n_jobs=min(8, os.cpu_count() or 1),
                deterministic=True, force_col_wise=True, verbosity=-1, label_gain=[0, 1, 3, 7, 15, 31])


def ranking_from_scores(fold, indices, scores):
    rows = fold.features.rows.iloc[indices].copy()
    rows["score"] = scores
    rows = rows.sort_values(["user_id", "score", "tv_show_id"], ascending=[True, False, True])
    ranking = {user: list(zip(frame.tv_show_id.astype(int), frame.score.astype(float)))[:200]
               for user, frame in rows.groupby("user_id", sort=False)}
    return S.recommend_subset(fold.phase, ranking, set(rows.user_id))


def training_queries(folds, config, users):
    matrices, labels, groups, names = [], [], [], None
    for fold in (folds if config["stack"] else folds[:1]):
        indices = indices_for(fold, config, users)
        matrix, current = design(fold, config, indices)
        if names is not None:
            assert names == current
        names = current
        matrices.append(matrix)
        labels.append(fold.labels[indices])
        # Separate group-size lists implement (fold,user), preventing user
        # queries from being merged across temporal folds.
        groups.extend(S.query_groups(fold.features.rows.iloc[indices]))
        if users is not None:
            assert set(fold.features.rows.iloc[indices].user_id) <= users
    return sparse.vstack(matrices, format="csr"), np.concatenate(labels), groups, names


def evaluate(folds, configs, train_users, heldout):
    import lightgbm as lgb
    seed_cache, results, seed_iterations = {}, [], {}
    for name, config in configs.items():
        log(f"\nABLATION {name}: {json.dumps(config, sort_keys=True)}")
        fit_x, fit_y, group, names = training_queries(folds, config, train_users)
        held_indices = [indices_for(fold, config, heldout) for fold in folds]
        held_matrices = [design(fold, config, idx)[0] for fold, idx in zip(folds, held_indices)]
        accumulators = [np.zeros(len(idx), dtype=np.float64) for idx in held_indices]
        seed_iterations[name] = {}
        for seed in config["seeds"]:
            key = (config["size"], config["scale"], config["stack"], config["new"], seed)
            if key not in seed_cache:
                log(f"  fitting seed {seed}, rows={fit_x.shape[0]:,}, queries={len(group):,}, features={len(names)}")
                model = lgb.LGBMRanker(**model_params(seed))
                model.fit(fit_x, fit_y, group=group,
                          eval_set=[(held_matrices[0], folds[0].labels[held_indices[0]])],
                          eval_group=[S.query_groups(folds[0].features.rows.iloc[held_indices[0]])],
                          eval_at=[5], feature_name=names,
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(25)])
                iteration = max(int(model.best_iteration_), 1)
                seed_cache[key] = ([model.booster_.predict(matrix, num_iteration=iteration)
                                    for matrix in held_matrices], iteration)
            predictions, iteration = seed_cache[key]
            seed_iterations[name][str(seed)] = iteration
            log(f"  seed {seed}: best_iteration={iteration}")
            for accumulator, prediction in zip(accumulators, predictions):
                accumulator += prediction / len(config["seeds"])
        row = {"recipe": name, "size": config["size"], "seeds": len(config["seeds"])}
        for fold, idx, scores in zip(folds, held_indices, accumulators):
            assert np.isfinite(scores).all()
            recs = ranking_from_scores(fold, idx, scores)
            score, count = H.map_at_k(recs, fold.targets, 5)
            assert count == len(set(fold.phase.users) & heldout)
            row[f"map_all_{fold.name}"] = score
            row[f"users_{fold.name}"] = count
        results.append(row)
        log("  HOLDOUT: " + json.dumps(row, sort_keys=True))
        pd.DataFrame(results).to_csv(H.OUT / "step7_ablations.csv", index=False)
        gc.collect()
    table = pd.DataFrame(results).sort_values(["map_all_fold10", "recipe"], ascending=[False, True], ignore_index=True)
    base_score = float(table.loc[table.recipe == "previous_v5", "map_all_fold10"].iloc[0])
    table["delta_vs_previous_v5"] = table.map_all_fold10 - base_score
    log("\nHOLDOUT TABLE — SELECT BY fold10 MAP@5 ONLY\n" + table.to_string(index=False))
    table.to_csv(H.OUT / "step7_ablations.csv", index=False)
    # If the chosen budget is 50, the candidates-only row duplicates baseline.
    # Select two distinct recipes, not two copies of an identical model.
    chosen, signatures = [], set()
    for name in table.recipe:
        signature = json.dumps(configs[name], sort_keys=True)
        if signature not in signatures:
            chosen.append(str(name))
            signatures.add(signature)
        if len(chosen) == 2:
            break
    return chosen, seed_iterations, table


def refit_and_predict(folds, final_fold, chosen, configs, iterations,
                      start_rank=1, manifest_name="step7_final_configs.json"):
    import lightgbm as lgb
    manifest = []
    model_dir = H.OUT / "step7_models"
    model_dir.mkdir(exist_ok=True)
    for rank, name in enumerate(chosen, start_rank):
        config = configs[name]
        log(f"\nFINAL rank={rank}, recipe={name}: {json.dumps(config, sort_keys=True)}")
        fit_x, fit_y, groups, names = training_queries(folds, config, None)
        final_indices = indices_for(final_fold, config)
        final_x, final_names = design(final_fold, config, final_indices)
        assert final_names == names
        scores = np.zeros(len(final_indices), dtype=np.float64)
        gain, splits = np.zeros(len(names)), np.zeros(len(names))
        for seed in config["seeds"]:
            trees = iterations[name][str(seed)]
            log(f"  refit seed {seed} on 100% of queries, trees={trees}")
            model = lgb.LGBMRanker(**model_params(seed, trees))
            model.fit(fit_x, fit_y, group=groups, feature_name=names)
            scores += model.booster_.predict(final_x) / len(config["seeds"])
            gain += model.booster_.feature_importance(importance_type="gain") / len(config["seeds"])
            splits += model.booster_.feature_importance(importance_type="split") / len(config["seeds"])
            model.booster_.save_model(str(model_dir / f"{name}_seed{seed}.txt"))
        assert np.isfinite(scores).all()
        recs = ranking_from_scores(final_fold, final_indices, scores)
        filename = f"submission_v6_{rank}.csv"
        FB.check_and_write(recs, H.EXAM / filename, final_fold.phase.pool,
                           final_fold.phase.popularity, log=log)
        importance = pd.DataFrame({"feature": names, "mean_gain": gain, "mean_split": splits}).sort_values(
            ["mean_gain", "feature"], ascending=[False, True])
        importance.to_csv(H.OUT / f"step7_importances_{rank}.csv", index=False)
        log(f"FEATURE IMPORTANCES {filename}\n" + importance.to_string(index=False))
        manifest.append({"file": filename, "recipe": name, "config": config,
                         "seed_iterations": iterations[name], "feature_names": names})
    (H.OUT / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError) as error:
        raise SystemExit("Install lightgbm; on macOS also install libomp") from error
    start = time.perf_counter()
    H.OUT.mkdir(exist_ok=True)
    log(__doc__)
    interactions = H.load_interactions()
    schedules = {block: S.read_schedule(block) for block in H.TRAIN_BLOCKS_FINAL}
    windows = aggregate_windows(schedules)
    catalogue = H.load_programmes()
    categories = sorted(set(catalogue.category.fillna("__missing__").astype(str)) | {"__missing__"})
    log("CATEGORY ONE-HOT MAPPING: " + json.dumps(dict(enumerate(categories)), ensure_ascii=False))
    _, first_targets = H.get_eval_targets(interactions, H.EVAL_BLOCK_VAL)
    short_labels = qualifying_interactions(windows["label5"])
    second_targets = short_labels.groupby("user_id").tv_show_id.apply(set).to_dict()
    universe = sorted(set(first_targets) | set(second_targets))
    train_users, heldout = shared_split(universe)
    log(f"SHARED USER SPLIT: train={len(train_users):,}, heldout={len(heldout):,}")
    phase1 = E.build_phase(interactions, list(first_targets))
    fold1 = prepare_fold("fold10", phase1, windows["train10"], schedules["11-20"], schedules["21-30"],
                          E.VALIDATION_START, first_targets, 10, 10, catalogue, categories)
    all_schedule = pd.concat(schedules.values(), ignore_index=True)
    train15 = clip_schedule(all_schedule, E.TRAIN_START, MIDPOINT)
    target5 = clip_schedule(schedules["21-30"], MIDPOINT, E.FINAL_START)
    phase2 = custom_phase(sorted(second_targets), qualifying_interactions(windows["train15"]), target5, MIDPOINT)
    fold2 = prepare_fold("fold5", phase2, windows["train15"], train15, target5,
                          MIDPOINT, second_targets, 15, 5, catalogue, categories)
    # Pair/channel aggregates are large; the prepared folds own only feature
    # matrices, labels and fallback state. Retain solely train20 for final use.
    for obsolete in ("train10", "train15", "label5"):
        del windows[obsolete]
    del train15, target5
    gc.collect()
    folds = [fold1, fold2]
    budget = select_candidate_size(folds, train_users)
    configs = recipes(budget)
    chosen, iterations, table = evaluate(folds, configs, train_users, heldout)
    (H.OUT / "step7_selection.json").write_text(json.dumps({
        "candidate_size": budget, "recall_ceiling": "200/source on training users in both folds",
        "selection_metric": "heldout all-watched MAP@5 on fold10 only", "chosen": chosen,
        "recipes": configs, "iterations": iterations,
        "designated_final_so_far": "submission_v5_ltr.csv",
    }, indent=2) + "\n")
    log("SELECTED BEST TWO: " + json.dumps(chosen))
    # Only now read the 31-42 schedule. No 31-42 viewing ever enters any stage.
    users = pd.read_csv(H.DATA / "submission.csv", dtype={"user_id": str}).user_id.tolist()
    final_phase = E.build_phase(interactions, users, final=True)
    target12 = S.read_schedule(H.EVAL_BLOCK_FINAL)
    final_fold = prepare_fold("final", final_phase, windows["train20"], all_schedule, target12,
                               E.FINAL_START, {user: set() for user in users}, 20, 12, catalogue, categories)
    refit_and_predict(folds, final_fold, chosen, configs, iterations)
    log(f"Completed in {time.perf_counter() - start:.1f}s. Current designated final remains v5_ltr pending results.")
    (H.OUT / "step7_ltr_plus.md").write_text("\n".join(REPORT).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
