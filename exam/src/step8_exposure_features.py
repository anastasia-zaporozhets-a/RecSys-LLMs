"""Step 8 additions to immutable Step 7 candidates/features; persisted separately.

Quartile labels use ALL relevant items, including those outside candidates.
Positive grades are 1..4; ties on a quartile boundary take the lower bucket.
Fold10 airing/day counts are train20 minus train10. This is exact because
local dates are disjoint and the original schedule join clips broadcasts at
the block boundary. Fold5 counts come directly from cached label5 aggregates.

New E168 decays integrate exp2(-age/half_life) over each view/hour overlap.
Habits are divided by effective weighted training weeks; target exposure by
target weeks. Uniform exposures in each target half are divided by that
half's weeks. Raw views include unlabelled context and are clipped separately
to each feature window. No target-window viewing enters a feature.
Candidate rank restricts the cached E48 positive ordering to the fixed
candidate set; zero-exposure items receive candidate_count + 1.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import harness as H
import step5_exposure as E
import step6_ltr as S
import step7_ltr_plus as P
import step7_feature_cache as C

ADDITIONAL_NAMES = ["E168_decay4_per_target_week", "E168_decay8_per_target_week",
                    "E168_first_half_per_week", "E168_second_half_per_week", "E48_candidate_rank"]
GROUPS = {"decay4": [0], "decay8": [1], "halves": [2, 3], "candidate_rank": [4]}
DATES = {"fold10": (E.VALIDATION_START, E.FINAL_START),
         "fold5": (P.MIDPOINT, E.FINAL_START), "final": (E.FINAL_START, E.FINAL_END)}
WEEK_SECONDS = 7 * 86400


def log(message):
    print(message, flush=True)


@dataclass
class Data:
    name: str
    phase: E.Phase
    rows: pd.DataFrame
    base: sparse.csr_matrix
    names: list[str]
    labels: dict
    targets: dict
    training_weeks: int
    target_weeks: int
    additional: sparse.csr_matrix | None = None


def feature_fingerprint(base_root, base_fingerprint):
    sources = {}
    for fold in DATES:
        C.inspect_cache(base_root / fold, base_fingerprint, "fold")
        for filename in ("manifest.json", "base_features.npz", "new_features.npz", "aligned_arrays.npz", "training_pairs.csv.gz"):
            path = base_root / fold / filename
            stat = path.stat()
            sources[f"{fold}/{filename}"] = [stat.st_size, stat.st_mtime_ns]
    for window in ("train10", "train20", "label5"):
        assert C.inspect_cache(base_root / "windows" / window, base_fingerprint, "window_stats") is not None
        path = base_root / "windows" / window / "pairs.csv.gz"
        stat = path.stat()
        sources[f"windows/{window}/pairs.csv.gz"] = [stat.st_size, stat.st_mtime_ns]
    return {"version": 1, "base": base_fingerprint, "cache_sources": sources,
            "step8_feature_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_base(root, base_fingerprint):
    result = []
    config = P.recipes(200)["all_together"]
    for name in DATES:
        fold = C.load_fold(root / name, base_fingerprint)
        matrix, names = P.design(fold, config, np.arange(len(fold.labels)))
        # Retain true histories for fallback, but release unused CF graph input
        # tables/lists and raw matrices after applying the cached rate transform.
        phase = E.Phase(fold.phase.users, fold.phase.items, fold.phase.pool,
                        {u: fold.phase.histories.get(u, set()) for u in fold.phase.users}, {}, pd.DataFrame(),
                        fold.phase.popularity, fold.phase.cold, {}, {})
        result.append(Data(name, phase, fold.features.rows, matrix, names, {"binary": fold.labels},
                           fold.targets, fold.training_weeks, fold.target_weeks))
        log(f"  reused {name}: {matrix.shape[0]:,} cached candidates, {matrix.shape[1]} base features")
        del fold
        gc.collect()
    return result


def read_counts(root, name, base_fingerprint):
    assert C.inspect_cache(root / "windows" / name, base_fingerprint, "window_stats") is not None
    return pd.read_csv(root / "windows" / name / "pairs.csv.gz",
                       usecols=["user_id", "tv_show_id", "airings", "days"],
                       dtype={"user_id": str, "tv_show_id": "int64"}).set_index(["user_id", "tv_show_id"])


def difference_counts(long, short):
    result = long.subtract(short, fill_value=0).fillna(0)
    assert (result.to_numpy() >= 0).all(), "Window counts must be additive at the block boundary"
    assert np.equal(result.to_numpy(), np.floor(result.to_numpy())).all()
    return result.astype(np.int64)


def quartile_grades(actual_items, counts, candidate_items):
    """Tied counts share a bucket; watched items cannot get grade zero."""
    actual_items = sorted(actual_items)
    if not actual_items:
        return np.zeros(len(candidate_items), dtype=np.int8), np.zeros(3)
    values = counts.reindex(actual_items).fillna(0).to_numpy(dtype=np.float64)
    cuts = np.quantile(values, [.25, .5, .75], method="linear")
    grades = 1 + np.searchsorted(cuts, values, side="left")
    mapping = dict(zip(actual_items, grades))
    return np.array([mapping.get(int(item), 0) for item in candidate_items], dtype=np.int8), cuts


def graded_labels(data, counts):
    labels = {"binary": data.labels["binary"].astype(np.int8),
              "airings": np.zeros(len(data.rows), dtype=np.int8), "days": np.zeros(len(data.rows), dtype=np.int8)}
    cuts = {name: np.zeros((len(data.phase.users), 3)) for name in ("airings", "days")}
    by_user = {user: frame.droplevel("user_id") for user, frame in counts.groupby(level="user_id", sort=False)}
    user_index = {user: index for index, user in enumerate(data.phase.users)}
    missing = 0
    for user, frame in data.rows.groupby("user_id", sort=False):
        actual = data.targets[user]
        own = by_user.get(user, pd.DataFrame(columns=["airings", "days"]))
        for name in ("airings", "days"):
            values = own[name]
            grade, cut = quartile_grades(actual, values, frame.tv_show_id.to_numpy())
            labels[name][frame.index.to_numpy()] = grade
            cuts[name][user_index[user]] = cut
        missing += sum(item not in own.index or own.loc[item, "airings"] <= 0 for item in actual)
    for name in ("airings", "days"):
        assert np.array_equal(labels[name] > 0, labels["binary"] > 0)
    log(f"  {data.name}: quartiles use all true watched items; {missing} watched items lack positive "
        "clipped count data (retained in the lowest positive grade)")
    return labels, cuts


def decayed_seconds(starts, ends, anchor, half_life):
    starts, ends = np.asarray(starts), np.asarray(ends)
    assert (ends >= starts).all() and (ends <= anchor).all()
    rate = np.log(2.0) / (half_life * WEEK_SECONDS)
    return np.exp(-rate * (anchor - ends)) * (-np.expm1(-rate * (ends - starts))) / rate


def candidate_rank(data):
    global_rank = data.base[:, data.names.index("exposure_rank")].toarray().ravel()
    score = data.base[:, data.names.index("exposure_score_per_target_week")].toarray().ravel()
    result = np.empty(len(data.rows), dtype=np.float32)
    for _, frame in data.rows.groupby("user_id", sort=False):
        idx = frame.index.to_numpy()
        result[idx] = len(idx) + 1
        positive = idx[score[idx] > 0]
        order = positive[np.argsort(global_rank[positive], kind="stable")]
        result[order] = np.arange(1, len(order) + 1)
    return result


def airing_matrices(data, schedule):
    lower_date, upper_date = DATES[data.name]
    target = P.clip_schedule(schedule, lower_date, upper_date)
    channels = np.array(sorted(target.channel_id.unique()), dtype=np.int64)
    midpoint = pd.Timestamp(lower_date) + (pd.Timestamp(upper_date) - pd.Timestamp(lower_date)) / 2
    lower, middle, upper = E.epoch(lower_date), E.epoch(midpoint), E.epoch(upper_date)
    matrices = []
    for begin, end in ((lower, middle), (middle, upper)):
        source, hours, seconds = E.split_hours(target.start.to_numpy(), target.end.to_numpy(), begin, end)
        row = np.searchsorted(data.phase.items, target.tv_show_id.to_numpy()[source])
        col = np.searchsorted(channels, target.channel_id.to_numpy()[source]) * 168 + E.local_slots(hours)
        matrices.append(sparse.coo_matrix((seconds, (row, col)),
                                          shape=(len(data.phase.items), len(channels) * 168)).tocsr())
    return channels, matrices[0] + matrices[1], matrices[0], matrices[1]


def build_habits(datasets, channels):
    matrices = {data.name: {kind: sparse.csr_matrix((len(data.phase.users), len(channels[data.name]) * 168),
                                                   dtype=np.float64) for kind in ("uniform", "decay4", "decay8")}
                for data in datasets}
    for chunk, frame in enumerate(pd.read_csv(
            H.DATA / "dataset11-30.csv", chunksize=E.IO_CHUNK,
            usecols=["user_id", "vsetv_id", "start_time", "duraton"],
            dtype={"user_id": str, "vsetv_id": "int64", "duraton": "int64"}), 1):
        frame = frame.loc[(frame.start_time < f"{E.FINAL_START} 00:00:00") & (frame.duraton > 0)]
        starts = E.parse_time(frame.start_time, "%Y-%m-%d %H:%M:%S")
        ends = starts + frame.duraton.to_numpy()
        for data in datasets:
            anchor = E.epoch(DATES[data.name][0])
            lower = E.epoch(E.TRAIN_START)
            users = pd.Index(data.phase.users).get_indexer(frame.user_id)
            channel = pd.Index(channels[data.name]).get_indexer(frame.vsetv_id)
            valid = (users >= 0) & (channel >= 0) & (starts < anchor) & (ends > lower)
            source, hours, seconds = E.split_hours(starts[valid], ends[valid], lower, anchor)
            if not len(source):
                continue
            row = users[valid][source]
            col = channel[valid][source] * 168 + E.local_slots(hours)
            begin = np.maximum(np.maximum(starts[valid][source], hours * 3600), lower)
            end = np.minimum(np.minimum(ends[valid][source], (hours + 1) * 3600), anchor)
            shape = matrices[data.name]["uniform"].shape
            matrices[data.name]["uniform"] += sparse.coo_matrix((seconds, (row, col)), shape=shape).tocsr()
            for half_life in (4, 8):
                weights = decayed_seconds(begin, end, anchor, half_life)
                matrices[data.name][f"decay{half_life}"] += sparse.coo_matrix((weights, (row, col)), shape=shape).tocsr()
        log(f"  decay-habit chunk {chunk}: {len(frame):,} raw views, separately clipped to each feature window")
    for data in datasets:
        anchor, lower = E.epoch(DATES[data.name][0]), E.epoch(E.TRAIN_START)
        matrices[data.name]["uniform"] /= data.training_weeks
        for half_life in (4, 8):
            effective = float(decayed_seconds(np.array([lower]), np.array([anchor]), anchor, half_life)[0] / WEEK_SECONDS)
            matrices[data.name][f"decay{half_life}"] /= effective
            log(f"  {data.name} decay{half_life}: normalization {effective:.6f} effective weighted weeks")
    return matrices


def build_exposures(datasets):
    schedules = {"21-30": S.read_schedule("21-30")}
    if any(data.name == "final" for data in datasets):
        schedules["31-42"] = S.read_schedule("31-42")
    airing, channels = {}, {}
    for data in datasets:
        built = airing_matrices(data, schedules["31-42" if data.name == "final" else "21-30"])
        channels[data.name], airing[data.name] = built[0], built[1:]
    habits = build_habits(datasets, channels)
    result = {}
    for data in datasets:
        extra = np.zeros((len(data.rows), len(ADDITIONAL_NAMES)), dtype=np.float32)
        extra[:, 4] = candidate_rank(data)
        positions_by_user = data.rows.groupby("user_id", sort=False).indices
        full, first, second = airing[data.name]
        terms = [(habits[data.name]["decay4"], full, data.target_weeks),
                 (habits[data.name]["decay8"], full, data.target_weeks),
                 (habits[data.name]["uniform"], first, data.target_weeks / 2),
                 (habits[data.name]["uniform"], second, data.target_weeks / 2)]
        for begin in range(0, len(data.phase.users), E.USER_CHUNK):
            end = min(begin + E.USER_CHUNK, len(data.phase.users))
            for column, (habit, air, denominator) in enumerate(terms):
                scores = (habit[begin:end] @ air.T).toarray() / denominator
                for offset, user in enumerate(data.phase.users[begin:end]):
                    idx = positions_by_user[user]
                    items = np.searchsorted(data.phase.items, data.rows.tv_show_id.to_numpy()[idx])
                    extra[idx, column] = scores[offset, items]
            log(f"  {data.name} new exposures: users {begin + 1:,}-{end:,}")
        assert np.isfinite(extra).all()
        result[data.name] = sparse.csr_matrix(extra)
        del habits[data.name]
    return result


def additions(datasets, base_root, root, base_fingerprint, fingerprint):
    missing = []
    for data in datasets:
        path = root / data.name
        metadata = C.inspect_cache(path, fingerprint, "step8_features")
        if metadata is None:
            missing.append(data)
            continue
        assert metadata["feature_names"] == ADDITIONAL_NAMES and metadata["rows"] == len(data.rows)
        data.additional = sparse.load_npz(path / "additional.npz").tocsr()
        with np.load(path / "labels.npz", allow_pickle=False) as arrays:
            assert np.array_equal(arrays["binary"], data.labels["binary"])
            data.labels = {name: arrays[name] for name in ("binary", "airings", "days")}
        log(f"  Step 8 cache hit: {data.name}")
    if missing:
        counts = {}
        if any(data.name == "fold10" for data in missing):
            counts["fold10"] = difference_counts(read_counts(base_root, "train20", base_fingerprint),
                                                 read_counts(base_root, "train10", base_fingerprint))
        if any(data.name == "fold5" for data in missing):
            counts["fold5"] = read_counts(base_root, "label5", base_fingerprint)
        exposures = build_exposures(missing)
        for data in missing:
            if data.name == "final":
                labels = {name: np.zeros(len(data.rows), dtype=np.int8) for name in ("binary", "airings", "days")}
                cuts = {name: np.zeros((len(data.phase.users), 3)) for name in ("airings", "days")}
            else:
                labels, cuts = graded_labels(data, counts[data.name])
            data.additional, data.labels = exposures[data.name], labels
            with C.staged_cache(root / data.name) as folder:
                sparse.save_npz(folder / "additional.npz", data.additional)
                np.savez_compressed(folder / "labels.npz", **labels, airings_cuts=cuts["airings"], days_cuts=cuts["days"])
                C.write_manifest(folder, {"rows": len(data.rows), "feature_names": ADDITIONAL_NAMES,
                                         "quantiles": [.25, .5, .75], "quantile_ties": "lower bucket",
                                         "channel_share": "reused from v6_3 base; no duplicate column"},
                                 fingerprint, "step8_features")
            log(f"  saved Step 8 additions: {root / data.name}")
    for data in datasets:
        assert data.additional.shape == (len(data.rows), len(ADDITIONAL_NAMES))
        assert np.isfinite(data.additional.data).all()
        for name in ("airings", "days"):
            assert np.array_equal(data.labels[name] > 0, data.labels["binary"] > 0)
            assert ((data.labels[name] >= 0) & (data.labels[name] <= 4)).all()
