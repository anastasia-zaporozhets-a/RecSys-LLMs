#!/usr/bin/env python3
"""Target calibration and LightGBM LambdaRank, with training-only features.

Run from exam/:
  .venv/bin/python -m pip install lightgbm
  # macOS runtime dependency, if absent: brew install libomp
  .venv/bin/python -u src/step6_ltr.py

A: Nine submitted configurations, five definitions of relevance, Spearman
against PUBLIC Kaggle scores only. Public-LB scores choose the evaluation
definition; they are never features, labels, sample weights or model targets.
Private scores are not used. v4_1 remains the designated final so far.

Qualifying watched user/item pairs and seconds targets are the existing
harness interactions. Airings, days and mean fractions are reconstructed from
raw view/airing overlap on those pairs. A broadcast is a unique channel/start/
duration/item schedule event; sessions in the same broadcast are combined,
and watched fractions are capped at one. Days are local dates touched by the
overlap. Target ties break by programme ID. Ranked targets have labels 5..1;
all-watched has binary labels 1 for EVERY watched item, 0 otherwise, because
that target has no preference ordering. MAP@5 uses the target's relevant set.

B: Candidates are the union of top-50 history, PPR(0.5), E(48_uniform), and
popularity. No label-positive injection. Features use ONLY 11-20 viewing and
the 21-30 airing schedule. Split users 80/20 with seed 12345; early stopping
on NDCG@5; report held-out MAP@5 versus v4_1 on identical users, including
cold-start fallback. The holdout is isolated from ranker fitting, but target
definition selection in A uses the full calibration population. After honest
holdout scoring, refit on all validation users at the selected iteration.

C: Recompute features using ONLY 11-30 viewing and 31-42 airings. No 31-42
viewing is available or used. Predict with the refitted ranker; also produce
equal-weight RRF(k=60) of its top-200 ranking with the v4_1 top-200 ranking.
Exactly five distinct IDs, pool restriction and existing popularity fallback.
Outputs: out/step6_* reports, model, split, importances and calibration tables;
submission_v5_ltr.csv and submission_v5_blend.csv.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

import final_build as FB
import harness as H
import step5_exposure as E

SEED = 12345
PUBLIC_SCORES = {
    "v4_1": 0.35383, "v4_2": 0.34898,
    "v3_1": 0.33173, "v3_2": 0.32686, "v3_3": 0.32539,
    "PPR_0.3": 0.30721, "popularity": 0.29788,
    "final_v2": 0.28189, "history": 0.28079,
}
TARGET_COLUMNS = {
    "top5_seconds": "seconds", "top5_airings": "airings",
    "top5_days": "days", "top5_mean_fraction": "mean_fraction",
}
NUMERIC_FEATURES = [
    "user_item_seconds", "user_item_airings", "user_item_days", "last4weeks_seconds",
    "ppr_score", "ppr_rank", "exposure_score", "exposure_rank", "popularity_users",
    "global_item_seconds", "target_airtime", "target_airings", "airtime_ratio",
    "user_category_share", "is_new_for_user",
]
REPORT: list[str] = []


def log(message=""):
    print(message, flush=True)
    REPORT.append(str(message))


def read_schedule(block):
    columns = ["channel_id", "start_time", "duration", "tv_show_id"]
    frame = pd.read_csv(H.DATA / f"export_arh_{block}-final.csv", usecols=columns,
                        dtype={"channel_id": "int64", "duration": "int64", "tv_show_id": "int64"})
    frame = frame.loc[(frame.tv_show_id != 0) & (frame.duration > 0)].drop_duplicates().copy()
    lower, upper = {
        "11-20": (E.TRAIN_START, E.VALIDATION_START),
        "21-30": (E.VALIDATION_START, E.FINAL_START),
        "31-42": (E.FINAL_START, E.FINAL_END),
    }[block]
    starts = E.parse_time(frame.start_time, "%d.%m.%Y %H:%M:%S")
    frame["start"] = np.maximum(starts, E.epoch(lower))
    frame["end"] = np.minimum(starts + frame.duration.to_numpy(), E.epoch(upper))
    frame = frame.loc[frame.end > frame.start].copy()
    frame["block"] = block
    log(f"  schedule {block}: {len(frame):,} distinct labelled broadcasts, clipped to [{lower}, {upper})")
    return frame


def overlap_pairs(view_start, view_end, airing_start, airing_end):
    """Vectorized interval join for one channel, including overlapping airings."""
    cumulative_end = np.maximum.accumulate(airing_end)
    lower = np.searchsorted(cumulative_end, view_start, side="right")
    upper = np.searchsorted(airing_start, view_end, side="left")
    count = np.maximum(upper - lower, 0)
    source = np.repeat(np.arange(len(view_start)), count)
    offset = np.repeat(np.cumsum(count) - count, count)
    airing = np.repeat(lower, count) + np.arange(len(source)) - offset
    start = np.maximum(view_start[source], airing_start[airing])
    end = np.minimum(view_end[source], airing_end[airing])
    valid = end > start
    return source[valid], airing[valid], start[valid], end[valid]


def local_day_segments(starts, ends):
    """Sources/local-date ordinals for all dates touched by half-open overlaps."""
    def days(seconds):
        return pd.to_datetime(seconds, unit="s", utc=True).tz_convert(E.TZ).tz_localize(None).to_numpy(
            dtype="datetime64[D]").astype(np.int64)
    first, last = days(starts), days(ends - 1)
    count = last - first + 1
    source = np.repeat(np.arange(len(starts)), count)
    offset = np.repeat(np.cumsum(count) - count, count)
    return source, np.repeat(first, count) + np.arange(len(source)) - offset


def aggregate_view_stats(schedules):
    """Raw 11-30 views only; return per-block user/item stats for features/labels.

    No schedule 31-42 is accepted here. Broadcast aggregation is global across
    CSV chunks, so chunk boundaries cannot inflate watched-airing/day counts.
    """
    assert set(schedules) == {"11-20", "21-30"}
    schedule = pd.concat(schedules.values(), ignore_index=True)
    schedule["airing_id"] = np.arange(len(schedule))
    groups = {}
    for channel, frame in schedule.groupby("channel_id"):
        groups[channel] = frame.sort_values("start").reset_index(drop=True)
    broadcast_parts, day_parts = [], []
    lower, upper = E.epoch(E.TRAIN_START), E.epoch(E.FINAL_START)
    recent11 = E.epoch(pd.Timestamp(E.VALIDATION_START) - pd.Timedelta(weeks=4))
    recent30 = E.epoch(pd.Timestamp(E.FINAL_START) - pd.Timedelta(weeks=4))
    keys = ["block", "user_id", "tv_show_id"]
    columns = ["user_id", "vsetv_id", "start_time", "duraton"]
    for number, views in enumerate(pd.read_csv(
            H.DATA / "dataset11-30.csv", usecols=columns, chunksize=E.IO_CHUNK,
            dtype={"user_id": str, "vsetv_id": "int64", "duraton": "int64"}), 1):
        views = views.loc[(views.start_time < f"{E.FINAL_START} 00:00:00") & (views.duraton > 0)].copy()
        starts = E.parse_time(views.start_time, "%Y-%m-%d %H:%M:%S")
        views["start"] = np.maximum(starts, lower)
        views["end"] = np.minimum(starts + views.duraton.to_numpy(), upper)
        views = views.loc[views.end > views.start]
        overlaps = 0
        for channel, view in views.groupby("vsetv_id", sort=False):
            airing = groups.get(channel)
            if airing is None:
                continue
            source, row, start, end = overlap_pairs(
                view.start.to_numpy(), view.end.to_numpy(), airing.start.to_numpy(), airing.end.to_numpy())
            if not len(source):
                continue
            events = pd.DataFrame({
                "user_id": view.user_id.to_numpy()[source],
                "tv_show_id": airing.tv_show_id.to_numpy()[row],
                "block": airing.block.to_numpy()[row],
                "airing_id": airing.airing_id.to_numpy()[row],
                "seconds": end - start,
                "recent11": np.maximum(end - np.maximum(start, recent11), 0),
                "recent30": np.maximum(end - np.maximum(start, recent30), 0),
            })
            broadcast_parts.append(events.groupby(keys + ["airing_id"], as_index=False)[
                ["seconds", "recent11", "recent30"]].sum())
            day_source, day = local_day_segments(start, end)
            dates = events.iloc[day_source][keys].copy()
            dates["day"] = day
            day_parts.append(dates.drop_duplicates())
            overlaps += len(events)
        log(f"  raw aggregation chunk {number}: {len(views):,} eligible views, {overlaps:,} labelled overlaps")
    broadcasts = pd.concat(broadcast_parts, ignore_index=True).groupby(
        keys + ["airing_id"], as_index=False)[["seconds", "recent11", "recent30"]].sum()
    del broadcast_parts
    durations = schedule.set_index("airing_id").duration
    broadcasts["fraction"] = np.minimum(broadcasts.seconds.to_numpy() /
                                        broadcasts.airing_id.map(durations).to_numpy(), 1.0)
    pairs = broadcasts.groupby(keys).agg(
        seconds=("seconds", "sum"), airings=("airing_id", "size"),
        fraction_sum=("fraction", "sum"), recent11=("recent11", "sum"), recent30=("recent30", "sum"),
    )
    dates = pd.concat(day_parts, ignore_index=True).drop_duplicates(keys + ["day"])
    del day_parts
    pairs["days"] = dates.groupby(keys).size()
    pairs["mean_fraction"] = pairs.fraction_sum / pairs.airings
    result = {block: pairs.xs(block, level="block").copy() for block in schedules}
    for block, frame in result.items():
        log(f"  aggregated {block}: {len(frame):,} user/item pairs; {int(frame.airings.sum()):,} user/broadcast pairs")
    return result


def calibration_targets(interactions, raw_stats):
    eligible = H.get_user_train(interactions, [H.EVAL_BLOCK_VAL])[
        ["user_id", "tv_show_id", "overlap_seconds"]].rename(columns={"overlap_seconds": "seconds"})
    eligible = eligible.join(raw_stats[["airings", "days", "mean_fraction"]],
                             on=["user_id", "tv_show_id"]).fillna(0)
    targets = {}
    for name, column in TARGET_COLUMNS.items():
        ordered = eligible.sort_values(["user_id", column, "tv_show_id"], ascending=[True, False, True])
        targets[name] = ordered.groupby("user_id").tv_show_id.apply(lambda values: list(values.iloc[:5])).to_dict()
    targets["all_watched"] = eligible.groupby("user_id").tv_show_id.apply(set).to_dict()
    return targets


def relevance(items, target, all_watched=False):
    if all_watched:
        actual = set(target)
        return np.array([int(item in actual) for item in items], dtype=np.int32)
    grades = {int(item): 5 - rank for rank, item in enumerate(target[:5])}
    return np.array([grades.get(int(item), 0) for item in items], dtype=np.int32)


@dataclass
class Signal:
    ranking: dict
    scores: np.ndarray | None = None
    ranks: np.ndarray | None = None


def ppr_signal(phase, alpha, feature_values=False):
    users = sorted(phase.histories)
    items = sorted(set(phase.train.tv_show_id.astype(int)))
    raw = H.ppr_batched(
        phase.users, {user: i for i, user in enumerate(users)}, {item: i for i, item in enumerate(items)},
        users, items, phase.history_lists, phase.histories, phase.pool, alpha, chunk_size=500)
    scores = np.zeros((len(phase.users), len(phase.items)), dtype=np.float32) if feature_values else None
    ranks = np.full(scores.shape, len(phase.items) + 1, dtype=np.int32) if feature_values else None
    ranking = {}
    for index, user in enumerate(phase.users):
        row = raw.pop(user, [])
        ranking[user] = row[:200]
        if feature_values and row:
            ids = np.fromiter((item for item, _ in row), dtype=np.int64)
            positions = np.searchsorted(phase.items, ids)
            scores[index, positions] = np.fromiter((score for _, score in row), dtype=np.float32)
            ranks[index, positions] = np.arange(1, len(row) + 1)
    return Signal(ranking, scores, ranks)


def exposure_signal(phase):
    scores = np.zeros((len(phase.users), len(phase.items)), dtype=np.float32)
    ranks = np.full(scores.shape, len(phase.items) + 1, dtype=np.int32)
    ranking = {}
    for start in range(0, len(phase.users), E.USER_CHUNK):
        stop = min(start + E.USER_CHUNK, len(phase.users))
        raw = (phase.habits["48_uniform"][start:stop] @ phase.airings[48].T).toarray()
        scores[start:stop] = raw
        for offset, user in enumerate(phase.users[start:stop]):
            row = raw[offset]
            positions = np.flatnonzero(row > 0)
            order = positions[np.lexsort((phase.items[positions], -row[positions]))]
            ranks[start + offset, order] = np.arange(1, len(order) + 1)
            ranking[user] = [(int(phase.items[pos]), float(row[pos])) for pos in order[:200]]
        log(f"  E48 features: users {start + 1:,}-{stop:,}")
    return Signal(ranking, scores, ranks)


def v3_rankings(phase, ppr, pop_weight=0.5):
    popularity = [int(item) for item in phase.popularity.index][:200]
    return {user: H.rank_fusion([[item for item, _ in ppr.ranking[user]], popularity],
                               [1.0, pop_weight], k=200, rrf_k=60.0) for user in phase.users}


def submitted_predictions(phase, ppr05, exposure):
    """Recreate precisely the nine scored configurations, not v4_3."""
    predictions = {}
    for name, scores in (("popularity", {user: [] for user in phase.users}),
                         ("history", {user: [(item, score) for item, score in phase.history_lists.get(user, [])
                                              if item in phase.pool] for user in phase.users})):
        predictions[name] = E.recommend(phase, scores)[0]
    ratio = FB.airtime_ratio(H.EVAL_BLOCK_VAL, H.TRAIN_BLOCKS_VAL, phase.pool)
    history_bonus = {user: FB.apply_beta(
        [(item, score) for item, score in phase.history_lists.get(user, []) if item in phase.pool], ratio, 0.5)
        for user in phase.users}
    predictions["final_v2"] = E.recommend(phase, history_bonus)[0]
    ppr03 = ppr_signal(phase, 0.3)
    predictions["PPR_0.3"] = E.recommend(phase, ppr03.ranking)[0]
    del ppr03
    ppr07 = ppr_signal(phase, 0.7)
    predictions["v3_2"] = E.recommend(phase, v3_rankings(phase, ppr07, 0.25))[0]
    del ppr07
    v3 = v3_rankings(phase, ppr05)
    predictions["v3_1"] = E.recommend(phase, v3)[0]
    predictions["v3_3"] = E.recommend(phase, v3_rankings(phase, ppr05, 1.0))[0]
    v4 = E.fuse_rankings(exposure.ranking, v3, "rrf", 0.5)
    predictions["v4_1"] = E.recommend(phase, v4)[0]
    predictions["v4_2"] = E.recommend(phase, E.fuse_rankings(exposure.ranking, v3, "rrf", 1.0))[0]
    # The top-five submitted configuration is extended only for the final blend.
    v4_extended = {user: H.rank_fusion(
        [[item for item, _ in exposure.ranking[user]], [item for item, _ in v3[user]]],
        [0.5, 1.0], k=200, rrf_k=60.0) for user in phase.users}
    assert set(predictions) == set(PUBLIC_SCORES)
    return predictions, v4_extended


def calibrate(predictions, targets):
    rows = []
    for method, public in PUBLIC_SCORES.items():
        row = {"method": method, "kaggle_public": public}
        for target, actual in targets.items():
            score, count = H.map_at_k(predictions[method], actual, 5)
            assert count == H.EXPECTED_EVAL_USERS
            row[target] = score
        rows.append(row)
        log(f"  calibration {method}: {json.dumps(row, sort_keys=True)}")
    table = pd.DataFrame(rows)
    correlations = []
    for name in targets:
        rho, pvalue = spearmanr(table[name].to_numpy(), table.kaggle_public.to_numpy())
        correlations.append({"target": name, "spearman_public": float(rho), "pvalue": float(pvalue)})
    correlation = pd.DataFrame(correlations).sort_values(
        ["spearman_public", "target"], ascending=[False, True], na_position="last", ignore_index=True)
    assert np.isfinite(correlation.iloc[0].spearman_public), "All calibration correlations undefined"
    chosen = str(correlation.iloc[0].target)
    table.to_csv(H.OUT / "step6_calibration.csv", index=False)
    correlation.to_csv(H.OUT / "step6_target_correlations.csv", index=False)
    log("\nNINE-CONFIGURATION VALIDATION TABLE\n" + table.to_string(index=False))
    log("\nPUBLIC-LB SPEARMAN CORRELATIONS\n" + correlation.to_string(index=False))
    log(f"SELECTED TARGET: {chosen}. Public scores choose only the evaluation definition; "
        "they do not fit model parameters. Private scores are unused. Ties break by target name.")
    return chosen


def check_reconstruction(predictions, targets):
    """When prior grids exist, verify rebuilt submitted configurations match."""
    sources = {
        "step3_tune_grid.csv": {
            "PPR_0.3": "M1_PPR(alpha=0.3,beta=0.0)",
            "history": "B2_history(beta=0.0)", "final_v2": "B2_history(beta=0.5)",
        },
        "step4_allwatched_grid.csv": {
            "v3_1": "RRF_ppr_pop(alpha=0.5,weights=1.0/0.5)",
            "v3_2": "RRF_ppr_pop(alpha=0.7,weights=1.0/0.25)",
            "v3_3": "RRF_ppr_pop(alpha=0.5,weights=1.0/1.0)",
        },
        "step5_exposure_grid.csv": {
            "v4_1": "rrf_E_48_uniform_v3_wE=0.5", "v4_2": "rrf_E_48_uniform_v3_wE=1.0",
        },
    }
    for filename, methods in sources.items():
        path = H.OUT / filename
        if not path.exists():
            continue
        table = pd.read_csv(path).set_index("method")
        for method, name in methods.items():
            if name not in table.index:
                continue
            for target, column in (("all_watched", "map_all"), ("top5_seconds", "map_top5")):
                actual, _ = H.map_at_k(predictions[method], targets[target], 5)
                assert np.isclose(actual, float(table.loc[name, column]), rtol=0, atol=1e-10), (
                    "Reconstruction differs from the submitted configuration", method, target, actual)
            log(f"  reconstruction {method}: PASS versus {filename}")


def training_stats(raw_stats, final=False):
    blocks = H.TRAIN_BLOCKS_FINAL if final else H.TRAIN_BLOCKS_VAL
    frame = pd.concat([raw_stats[block] for block in blocks]).groupby(level=[0, 1])[
        ["seconds", "airings", "days", "recent11", "recent30"]].sum()
    frame["recent"] = frame.recent30 if final else frame.recent11
    return frame


def schedule_summary(schedule):
    return schedule.assign(airtime=schedule.end - schedule.start).groupby("tv_show_id").agg(
        airtime=("airtime", "sum"), airings=("tv_show_id", "size"))


@dataclass
class Features:
    matrix: sparse.csr_matrix
    rows: pd.DataFrame
    names: list[str]


def build_features(phase, ppr, exposure, stats, train_schedule, target_schedule, catalogue, categories,
                   candidate_size=50):
    """No labels passed here: candidates and every feature are label-independent."""
    category = catalogue.set_index("tv_show_id").category.fillna("__missing__").astype(str)
    cat_index = {name: index for index, name in enumerate(categories)}
    totals = phase.train.groupby("tv_show_id").overlap_seconds.sum()
    train_airtime = schedule_summary(train_schedule).airtime
    supply = schedule_summary(target_schedule)
    cat_seconds = stats.reset_index()
    cat_seconds["category"] = cat_seconds.tv_show_id.map(category).fillna("__missing__")
    cat_seconds = cat_seconds.groupby(["user_id", "category"]).seconds.sum()
    user_seconds = stats.groupby(level="user_id").seconds.sum()
    pop50 = [int(item) for item in phase.popularity.index][:candidate_size]
    # A per-user slice avoids repeated scans of the full pair table.
    user_stats = {user: frame.droplevel("user_id") for user, frame in stats.groupby(level="user_id", sort=False)}
    numeric, rows, onehot_col = [], [], []
    for index, user in enumerate(phase.users):
        history50 = [item for item, _ in phase.history_lists.get(user, []) if item in phase.pool][:candidate_size]
        candidates = sorted(set(history50) | {item for item, _ in ppr.ranking[user][:candidate_size]} |
                            {item for item, _ in exposure.ranking[user][:candidate_size]} | set(pop50))
        positions = np.searchsorted(phase.items, candidates)
        own = user_stats.get(user)
        for item, pos in zip(candidates, positions):
            cat = str(category.get(item, "__missing__"))
            if own is not None and item in own.index:
                record = own.loc[item]
                seconds, airings, days, recent = record.seconds, record.airings, record.days, record.recent
            else:
                seconds = airings = days = recent = 0.0
            target_air = float(supply.airtime.get(item, 0))
            numeric.append([
                seconds, airings, days, recent,
                ppr.scores[index, pos], ppr.ranks[index, pos],
                exposure.scores[index, pos], exposure.ranks[index, pos],
                phase.popularity.get(item, 0), totals.get(item, 0),
                target_air, supply.airings.get(item, 0),
                (target_air + 3600.0) / (float(train_airtime.get(item, 0)) + 3600.0),
                cat_seconds.get((user, cat), 0.0) / max(float(user_seconds.get(user, 0)), 1.0),
                int(item not in phase.histories.get(user, set())),
            ])
            rows.append((user, item))
            onehot_col.append(cat_index.get(cat, cat_index["__missing__"]))
        if (index + 1) % 250 == 0 or index + 1 == len(phase.users):
            log(f"  feature users {index + 1:,}/{len(phase.users):,}, candidates {len(rows):,}")
    values = np.asarray(numeric, dtype=np.float32)
    assert np.isfinite(values).all()
    category_matrix = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32),
                                        (np.arange(len(rows)), onehot_col)),
                                       shape=(len(rows), len(categories)))
    matrix = sparse.hstack([sparse.csr_matrix(values), category_matrix], format="csr", dtype=np.float32)
    names = NUMERIC_FEATURES + [f"category_{index}" for index in range(len(categories))]
    metadata = pd.DataFrame(rows, columns=["user_id", "tv_show_id"])
    assert set(metadata.user_id) == set(phase.users)
    log(f"  candidate feature matrix: {matrix.shape}, nnz={matrix.nnz:,}")
    return Features(matrix, metadata, names)


def labels_for(features, targets, chosen):
    labels = np.zeros(len(features.rows), dtype=np.int32)
    recalls = []
    for user, frame in features.rows.groupby("user_id", sort=False):
        target = targets.get(user, set())
        labels[frame.index.to_numpy()] = relevance(frame.tv_show_id.to_numpy(), target, chosen == "all_watched")
        recalls.append(len(set(frame.tv_show_id) & set(target)) / len(target) if target else 0.0)
    log(f"  candidate recall of selected target: mean {np.mean(recalls):.6f}; "
        f"positive rows {int((labels > 0).sum()):,}/{len(labels):,}; no positive injection")
    return labels


def user_split(users):
    ordered = np.array(sorted(users), dtype=str)
    shuffled = np.random.default_rng(SEED).permutation(ordered)
    count = int(np.ceil(0.2 * len(shuffled)))
    heldout = set(shuffled[:count].tolist())
    training = set(shuffled[count:].tolist())
    assert training and heldout and not training & heldout
    return training, heldout


def query_groups(rows):
    """Group sizes for already contiguous user queries; reject interleaving."""
    users = rows.user_id.to_numpy()
    changes = np.r_[True, users[1:] != users[:-1]]
    heads = np.flatnonzero(changes)
    assert len(set(users[heads])) == len(heads)
    return np.diff(np.r_[heads, len(users)]).tolist()


def predicted_rankings(model, features, rows=None):
    selected = np.arange(len(features.rows)) if rows is None else np.asarray(rows)
    metadata = features.rows.iloc[selected].copy()
    metadata["score"] = model.predict(features.matrix[selected])
    assert np.isfinite(metadata.score).all()
    metadata = metadata.sort_values(["user_id", "score", "tv_show_id"], ascending=[True, False, True])
    return {user: list(zip(frame.tv_show_id.astype(int), frame.score.astype(float)))[:200]
            for user, frame in metadata.groupby("user_id", sort=False)}


def recommend_subset(phase, scores, users):
    return H.make_recs(scores, phase.pool, phase.popularity, phase.histories,
                       set(users) & phase.cold, 5)[0]


def fit_ranker(features, labels, phase, chosen_targets, all_targets, chosen, v4):
    import lightgbm as lgb
    train_users, heldout = user_split(phase.users)
    train_rows = np.flatnonzero(features.rows.user_id.isin(train_users))
    valid_rows = np.flatnonzero(features.rows.user_id.isin(heldout))
    (H.OUT / "step6_user_split.json").write_text(json.dumps({
        "seed": SEED, "train_users": sorted(train_users), "heldout_users": sorted(heldout),
    }, indent=2) + "\n", encoding="utf-8")
    params = dict(objective="lambdarank", metric="ndcg", random_state=SEED,
                  learning_rate=0.05, num_leaves=31, min_child_samples=30,
                  n_estimators=1000, colsample_bytree=0.9,
                  n_jobs=min(8, os.cpu_count() or 1), deterministic=True, force_col_wise=True,
                  verbosity=-1, label_gain=[0, 1, 3, 7, 15, 31])
    model = lgb.LGBMRanker(**params)
    log(f"\nTRAIN LAMBDARANK: {len(train_users):,} train users / {len(heldout):,} held-out users; seed={SEED}")
    model.fit(features.matrix[train_rows], labels[train_rows],
              group=query_groups(features.rows.iloc[train_rows]),
              eval_set=[(features.matrix[valid_rows], labels[valid_rows])],
              eval_group=[query_groups(features.rows.iloc[valid_rows])], eval_at=[5],
              feature_name=features.names,
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(25)])
    rankings = predicted_rankings(model, features, valid_rows)
    predictions = recommend_subset(phase, rankings, heldout)
    reference = {user: v4[user] for user in heldout}
    log(f"  best iteration: {model.best_iteration_}; held-out user IDs saved for reproducibility")
    results = []
    for method, recs in (("LTR_heldout", predictions), ("v4_1_same_users", reference)):
        row = {"method": method}
        for name, targets in (("chosen_target", chosen_targets),
                              ("all_watched", all_targets["all_watched"]),
                              ("top5_seconds", all_targets["top5_seconds"])):
            score, count = H.map_at_k(recs, targets, 5)
            assert count == len(heldout)
            row[name] = score
        results.append(row)
    result = pd.DataFrame(results)
    result.to_csv(H.OUT / "step6_heldout.csv", index=False)
    log(f"\nHELD-OUT MAP@5; selected target={chosen}\n{result.to_string(index=False)}")
    best_iteration = max(int(model.best_iteration_), 1)
    log(f"\nRefitting on all {len(phase.users):,} validation users at {best_iteration} iterations; "
        "features still 11-20, labels still 21-30, no 31-42 outcomes.")
    params["n_estimators"] = best_iteration
    final_model = lgb.LGBMRanker(**params)
    final_model.fit(features.matrix, labels, group=query_groups(features.rows),
                    feature_name=features.names, callbacks=[lgb.log_evaluation(25)])
    final_model.booster_.save_model(str(H.OUT / "step6_ltr_model.txt"))
    importance = pd.DataFrame({
        "feature": features.names,
        "gain": final_model.booster_.feature_importance(importance_type="gain"),
        "split": final_model.booster_.feature_importance(importance_type="split"),
    }).sort_values(["gain", "feature"], ascending=[False, True])
    importance.to_csv(H.OUT / "step6_feature_importances.csv", index=False)
    log("\nFEATURE IMPORTANCES\n" + importance.to_string(index=False))
    return final_model, best_iteration


def validate_available_v4():
    """Validate already generated v4 CSVs without rewriting submitted artifacts."""
    template = pd.read_csv(H.DATA / "submission.csv", dtype=str)
    pool = H.load_pool(H.EVAL_BLOCK_FINAL)
    for rank in (1, 2, 3):
        path = H.EXAM / f"submission_v4_{rank}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str)
        assert frame.columns.tolist() == ["user_id", "tv_show_id"]
        assert len(frame) == len(template) == H.EXPECTED_SUB_USERS
        assert frame.user_id.is_unique and frame.user_id.tolist() == template.user_id.tolist()
        lists = [list(map(int, value.split())) for value in frame.tv_show_id]
        assert all(len(row) == len(set(row)) == 5 and set(row) <= pool and 0 not in row for row in lists)
        log(f"  {path.name}: PASS, {len(frame):,} ordered users, five distinct in-pool nonzero IDs")


def main():
    # Dependency check before doing expensive I/O or computing model signals.
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError) as error:
        raise SystemExit("Install dependency: .venv/bin/python -m pip install lightgbm; "
                         "on macOS also install the OpenMP runtime: brew install libomp") from error
    started = time.perf_counter()
    H.OUT.mkdir(exist_ok=True)
    log(__doc__)
    log("A) RECONSTRUCT OBSERVATIONS AND CALIBRATE THE TARGET")
    interactions = H.load_interactions()
    schedules = {block: read_schedule(block) for block in H.TRAIN_BLOCKS_FINAL}
    raw_stats = aggregate_view_stats(schedules)
    targets = calibration_targets(interactions, raw_stats[H.EVAL_BLOCK_VAL])
    users = list(targets["all_watched"])
    phase = E.build_phase(interactions, users)
    ppr05 = ppr_signal(phase, 0.5, feature_values=True)
    exposure = exposure_signal(phase)
    predictions, _ = submitted_predictions(phase, ppr05, exposure)
    check_reconstruction(predictions, targets)
    chosen = calibrate(predictions, targets)
    catalogue = H.load_programmes()
    categories = sorted(set(catalogue.category.fillna("__missing__").astype(str)) | {"__missing__"})
    log("\nCATEGORY ONE-HOT MAPPING\n" + json.dumps(
        {f"category_{index}": name for index, name in enumerate(categories)}, ensure_ascii=False, indent=2))
    log("\nB) BUILD TRAINING-ONLY CANDIDATES AND FEATURES")
    features = build_features(phase, ppr05, exposure, training_stats(raw_stats),
                              schedules["11-20"], schedules["21-30"], catalogue, categories)
    labels = labels_for(features, targets[chosen], chosen)
    model, iterations = fit_ranker(features, labels, phase, targets[chosen], targets,
                                   chosen, predictions["v4_1"])
    (H.OUT / "step6_model_metadata.json").write_text(json.dumps({
        "chosen_target": chosen, "seed": SEED, "best_iteration": iterations,
        "feature_names": features.names, "category_onehot_mapping": dict(enumerate(categories)),
        "calibration": "Nine public-LB scores select only the evaluation definition, not model parameters.",
        "all_watched_labels": "binary 1 for every watched item; no top-five truncation",
        "designated_final_so_far": "submission_v4_1.csv",
    }, indent=2) + "\n", encoding="utf-8")
    names = features.names
    del phase, features, labels, ppr05, exposure, predictions
    gc.collect()

    log("\nC) FINAL FEATURES: 11-30 VIEWS ONLY, 31-42 SCHEDULE ONLY")
    users = pd.read_csv(H.DATA / "submission.csv", dtype={"user_id": str}).user_id.tolist()
    phase = E.build_phase(interactions, users, final=True)
    ppr05, exposure = ppr_signal(phase, 0.5, feature_values=True), exposure_signal(phase)
    target_schedule = read_schedule(H.EVAL_BLOCK_FINAL)
    features = build_features(phase, ppr05, exposure, training_stats(raw_stats, final=True),
                              pd.concat(schedules.values(), ignore_index=True), target_schedule,
                              catalogue, categories)
    assert features.names == names
    ltr = predicted_rankings(model, features)
    ltr_recs, _ = E.recommend(phase, ltr)
    v3 = v3_rankings(phase, ppr05)
    v4 = {user: H.rank_fusion(
        [[item for item, _ in exposure.ranking[user]], [item for item, _ in v3[user]]],
        [0.5, 1.0], k=200, rrf_k=60.0) for user in users}
    blended = {user: H.rank_fusion(
        [[item for item, _ in ltr[user]], [item for item, _ in v4[user]]],
        [1.0, 1.0], k=5, rrf_k=60.0) for user in users}
    blend_recs, _ = E.recommend(phase, blended)
    log("submission_v5_ltr.csv = LambdaRank, selected target=" + chosen)
    FB.check_and_write(ltr_recs, H.EXAM / "submission_v5_ltr.csv", phase.pool, phase.popularity, log=log)
    log("submission_v5_blend.csv = RRF(k=60), LTR/v4_1 weights 1/1, top-200 input rankings")
    FB.check_and_write(blend_recs, H.EXAM / "submission_v5_blend.csv", phase.pool, phase.popularity, log=log)
    log(f"\nCompleted in {time.perf_counter() - started:.1f}s; designated final so far remains v4_1.")
    (H.OUT / "step6_ltr.md").write_text("\n".join(REPORT).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
