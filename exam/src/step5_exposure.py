#!/usr/bin/env python3
"""Training-only channel/time habits projected onto the target airing schedule.

Run from exam/: .venv/bin/python -u src/step5_exposure.py

Validation: raw views strictly clipped to ISO weeks 11-20; airings from 21-30.
Final: raw views strictly clipped to 11-30; airings from 31-42. No 31-42
viewing is read or inferred. Schedule metadata supplies only channel/time/item.

Four variants: channel x 168 hour-of-week slots or channel x 48
weekday/weekend-hour slots, each uniform or last-four-training-weeks x2.
H is watched seconds per training week, including unlabelled airtime. A is
labelled target-airing seconds. E = H @ A.T is an exposure affinity (seconds
squared/week), not a calibrated expected-seconds estimate. Dividing by 3600
would not change any ranking. The 48-slot variant pools days without further
day-count normalization. Recency keeps the same unweighted week denominator.

Each variant tests E alone; E * 1.25 for watched items; RRF of E/v3_1 at
weights (0.5/1, 1/1, 2/1); and rank-normalized linear blends at E weights
0.3, 0.5, 0.7. Both fusion families use top-200 input rankings. Linear rank
scores are (201-rank)/200, absent items zero; ties use ascending programme ID.
v3_1 is RRF(k=60) of PPR(alpha=0.5)/popularity at weights 1/0.5. Its ranking
is extended beyond five for fusion, preserving the original top five exactly.

All 33 configurations (including v3_1 reference) are eligible for selection
by all-watched MAP@5; top-five-target MAP@5 is diagnostic. Saves the grid,
novelty counts, report and final configuration manifest under out/, and writes
submission_v4_1.csv, submission_v4_2.csv, submission_v4_3.csv.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse

import final_build as FB
import harness as H

TZ = "Europe/Kyiv"
TRAIN_START = "2020-03-09"
VALIDATION_START = "2020-05-18"
FINAL_START = "2020-07-27"
FINAL_END = "2020-10-19"
IO_CHUNK = 200_000
USER_CHUNK = 250
KEEP = 200
HISTORY_BONUS = 0.25
VARIANTS = ("168_uniform", "168_recent", "48_uniform", "48_recent")
REPORT: list[str] = []


def log(message=""):
    print(message, flush=True)
    REPORT.append(str(message))


def epoch(date):
    return int(pd.Timestamp(date, tz=TZ).timestamp())


def parse_time(series, fmt):
    local = pd.to_datetime(series, format=fmt, errors="raise")
    utc = local.dt.tz_localize(TZ, ambiguous="raise", nonexistent="shift_forward")
    return utc.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[s]").astype("int64").to_numpy()


def split_hours(starts, ends, lower, upper):
    """Vectorized half-open interval splitting; return source, UTC hour, seconds.

    Clip before expansion: no session/airing can contribute outside its window.
    UTC hours are contiguous even across DST; local slot mapping happens later.
    """
    starts = np.maximum(np.asarray(starts, dtype=np.int64), lower)
    ends = np.minimum(np.asarray(ends, dtype=np.int64), upper)
    valid = np.flatnonzero(ends > starts)
    first = starts[valid] // 3600
    counts = (ends[valid] - 1) // 3600 - first + 1
    source = np.repeat(valid, counts)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    hours = np.repeat(first, counts) + np.arange(len(source)) - offsets
    seconds = np.minimum(ends[source], (hours + 1) * 3600) - np.maximum(starts[source], hours * 3600)
    assert int(seconds.sum()) == int((ends[valid] - starts[valid]).sum())
    return source, hours, seconds.astype(np.float64)


def local_slots(hours):
    local = pd.to_datetime(hours * 3600, unit="s", utc=True).tz_convert(TZ)
    return np.asarray(local.dayofweek * 24 + local.hour, dtype=np.int64)


def collapse_projection(n_channels):
    """168 -> 48: Mon-Fri share hourly slots; Sat-Sun share hourly slots."""
    source = np.arange(n_channels * 168)
    weekday = source % 168 // 24
    dest = source // 168 * 48 + (weekday >= 5) * 24 + source % 24
    return sparse.csr_matrix((np.ones(len(source)), (source, dest)),
                             shape=(n_channels * 168, n_channels * 48))


@dataclass
class Phase:
    users: list[str]
    items: np.ndarray
    pool: set[int]
    histories: dict
    history_lists: dict
    train: pd.DataFrame
    popularity: pd.Series
    cold: set[str]
    habits: dict
    airings: dict


def build_airings(block, target_start, target_end):
    """Read only channel, time, duration and item ID from the target schedule."""
    log(f"\nBuilding target-airing matrix from schedule {block} only")
    pool = H.load_pool(block)
    items = np.array(sorted(pool), dtype=np.int64)
    lower, upper = epoch(target_start), epoch(target_end)
    pieces = []
    channels = set()
    columns = ["channel_id", "start_time", "duration", "tv_show_id"]
    for number, frame in enumerate(pd.read_csv(
            H.DATA / f"export_arh_{block}-final.csv", usecols=columns,
            dtype={"channel_id": "int64", "tv_show_id": "int64", "duration": "int64"},
            chunksize=IO_CHUNK), 1):
        frame = frame.loc[(frame.tv_show_id != 0) & (frame.duration > 0)].copy()
        starts = parse_time(frame.start_time, "%d.%m.%Y %H:%M:%S")
        source, hours, seconds = split_hours(starts, starts + frame.duration.to_numpy(), lower, upper)
        channel = frame.channel_id.to_numpy()[source]
        channels.update(channel.tolist())
        pieces.append((frame.tv_show_id.to_numpy()[source], channel, local_slots(hours), seconds))
        log(f"  schedule chunk {number}: {len(frame):,} labelled airings, {len(source):,} hour segments")
    channel_ids = np.array(sorted(channels), dtype=np.int64)
    assert len(channel_ids) > 0
    rows, cols, values = [], [], []
    for item, channel, slot, seconds in pieces:
        rows.append(np.searchsorted(items, item))
        cols.append(np.searchsorted(channel_ids, channel) * 168 + slot)
        values.append(seconds)
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(items), len(channel_ids) * 168), dtype=np.float64,
    ).tocsr()
    projection = collapse_projection(len(channel_ids))
    log(f"  A168 {matrix.shape}, nnz={matrix.nnz:,}; target seconds={matrix.sum():,.0f}")
    return items, pool, channel_ids, {168: matrix, 48: (matrix @ projection).tocsr()}, projection


def build_habits(users, channel_ids, train_end, projection):
    lower, upper = epoch(TRAIN_START), epoch(train_end)
    weeks = (pd.Timestamp(train_end) - pd.Timestamp(TRAIN_START)).days / 7
    recent = epoch(pd.Timestamp(train_end) - pd.Timedelta(weeks=4))
    user_index = pd.Index(users)
    shape = (len(users), len(channel_ids) * 168)
    uniform = sparse.csr_matrix(shape, dtype=np.float64)
    last_four = sparse.csr_matrix(shape, dtype=np.float64)
    total_seconds = ignored_seconds = 0.0
    log(f"\nBuilding habits: [{TRAIN_START}, {train_end}), {weeks:g} weeks; recent cutoff "
        f"{pd.Timestamp(train_end) - pd.Timedelta(weeks=4)}")
    log("  All raw channel-view seconds retained, including tv_show_id=0 airtime; no label join/filter.")
    columns = ["user_id", "vsetv_id", "start_time", "duraton"]
    for number, frame in enumerate(pd.read_csv(
            H.DATA / "dataset11-30.csv", usecols=columns,
            dtype={"user_id": str, "vsetv_id": "int64", "duraton": "int64"},
            chunksize=IO_CHUNK), 1):
        # Filter by timestamp before parsing: no future viewing values enter H.
        frame = frame.loc[(frame.start_time < f"{train_end} 00:00:00") &
                          frame.user_id.isin(user_index) & (frame.duraton > 0)].copy()
        if frame.empty:
            log(f"  raw-view chunk {number}: no eligible training rows")
            continue
        starts = parse_time(frame.start_time, "%Y-%m-%d %H:%M:%S")
        source, hours, seconds = split_hours(starts, starts + frame.duraton.to_numpy(), lower, upper)
        channel = frame.vsetv_id.to_numpy()[source]
        channel_pos = np.searchsorted(channel_ids, channel)
        known = channel_pos < len(channel_ids)
        known[known] &= channel_ids[channel_pos[known]] == channel[known]
        ignored_seconds += seconds[~known].sum()
        source, hours, seconds, channel_pos = source[known], hours[known], seconds[known], channel_pos[known]
        row = user_index.get_indexer(frame.user_id.to_numpy())[source]
        col = channel_pos * 168 + local_slots(hours)
        uniform += sparse.coo_matrix((seconds, (row, col)), shape=shape).tocsr()
        mask = hours * 3600 >= recent
        last_four += sparse.coo_matrix((seconds[mask], (row[mask], col[mask])), shape=shape).tocsr()
        total_seconds += seconds.sum()
        log(f"  raw-view chunk {number}: {len(frame):,} eligible views, {len(source):,} hour segments")
    assert np.isclose(float(uniform.sum()), total_seconds)
    uniform /= weeks
    weighted = uniform + last_four / weeks
    habits = {"168_uniform": uniform, "168_recent": weighted,
              "48_uniform": (uniform @ projection).tocsr(),
              "48_recent": (weighted @ projection).tocsr()}
    log(f"  retained context seconds={total_seconds:,.0f}; seconds on channels without target "
        f"labelled airings={ignored_seconds:,.0f} (cannot contribute exposure)")
    for name, matrix in habits.items():
        log(f"  H {name}: {matrix.shape}, nnz={matrix.nnz:,}, zero-habit users="
            f"{int((matrix.getnnz(axis=1) == 0).sum()):,}")
    return habits


def build_phase(interactions, users, final=False):
    block = H.EVAL_BLOCK_FINAL if final else H.EVAL_BLOCK_VAL
    train_blocks = H.TRAIN_BLOCKS_FINAL if final else H.TRAIN_BLOCKS_VAL
    train_end = FINAL_START if final else VALIDATION_START
    target_end = FINAL_END if final else FINAL_START
    train = H.get_user_train(interactions, train_blocks)
    histories, lists = H.build_user_histories(train)
    items, pool, channels, airings, projection = build_airings(block, train_end, target_end)
    habits = build_habits(users, channels, train_end, projection)
    # Preserve the existing item-history cold-start protocol for every method.
    cold = {user for user in users if not histories.get(user)}
    log(f"  phase {block}: {len(users):,} users, {len(items):,} candidates; "
        f"{len(cold):,} item-history cold users get popularity for all methods")
    return Phase(users, items, pool, histories, lists, train,
                 H.build_popularity(train, pool), cold, habits, airings)


def positive_ranking(row, items, keep=KEEP):
    """Top positive scores, deterministic even when ties cross the cutoff."""
    positions = np.flatnonzero(row > 0)
    if len(positions) > keep:
        threshold = np.partition(row[positions], -keep)[-keep]
        better = positions[row[positions] > threshold]
        tied = positions[row[positions] == threshold]
        positions = np.concatenate((better, tied[:keep - len(better)]))
    order = np.lexsort((items[positions], -row[positions]))
    return [(int(items[index]), float(row[index])) for index in positions[order]]


def exposure_rankings(phase, variant):
    slots = int(variant.split("_")[0])
    habits, airings = phase.habits[variant], phase.airings[slots]
    raw, bonus = {}, {}
    item_index = pd.Index(phase.items)
    for start in range(0, len(phase.users), USER_CHUNK):
        stop = min(start + USER_CHUNK, len(phase.users))
        scores = (habits[start:stop] @ airings.T).toarray()
        for offset, user in enumerate(phase.users[start:stop]):
            row = scores[offset]
            raw[user] = positive_ranking(row, phase.items)
            watched = item_index.get_indexer(list(phase.histories.get(user, set())))
            watched = watched[watched >= 0]
            row[watched] *= 1.0 + HISTORY_BONUS
            bonus[user] = positive_ranking(row, phase.items)
        log(f"  exposure {variant}: scored users {start + 1:,}-{stop:,}; sparse H @ A.T")
    return raw, bonus


def v3_rankings(phase):
    user_list = sorted(phase.histories)
    item_list = sorted(set(phase.train.tv_show_id.astype(int)))
    ppr = H.ppr_batched(
        phase.users, {u: i for i, u in enumerate(user_list)},
        {item: i for i, item in enumerate(item_list)}, user_list, item_list,
        phase.history_lists, phase.histories, phase.pool, 0.5, chunk_size=500,
    )
    popularity = [int(item) for item in phase.popularity.index][:KEEP]
    return {
        user: H.rank_fusion([[item for item, _ in ppr.get(user, [])[:KEEP]], popularity],
                            [1.0, 0.5], k=KEEP, rrf_k=60.0)
        for user in phase.users
    }


def fuse_rankings(exposure, baseline, kind, weight):
    scores = {}
    for user in baseline:
        e_rank = [item for item, _ in exposure.get(user, [])[:KEEP]]
        b_rank = [item for item, _ in baseline[user][:KEEP]]
        if kind == "rrf":
            scores[user] = H.rank_fusion([e_rank, b_rank], [weight, 1.0], k=5, rrf_k=60.0)
        elif kind == "linear":
            combined = {}
            for ranking, factor in ((e_rank, weight), (b_rank, 1.0 - weight)):
                for rank, item in enumerate(ranking):
                    combined[item] = combined.get(item, 0.0) + factor * (KEEP - rank) / KEEP
            scores[user] = sorted(combined.items(), key=lambda entry: (-entry[1], entry[0]))[:5]
        else:
            raise ValueError(kind)
    return scores


def recommend(phase, scores):
    recs, fallback = H.make_recs(scores, phase.pool, phase.popularity,
                                phase.histories, phase.cold, 5)
    assert set(recs) == set(phase.users)
    assert all(len(row) == len(set(row)) == 5 and set(row) <= phase.pool for row in recs.values())
    return recs, fallback


def novelty(recs, histories):
    counts = [sum(item not in histories.get(user, set()) for item in row) for user, row in recs.items()]
    return {"novel_top5_count": int(sum(counts)), "novel_per_user": float(np.mean(counts)),
            "novel_fraction": float(sum(counts) / (5 * len(counts)))}


def main():
    started = time.perf_counter()
    log("STEP 5 — TRAINING-ONLY CHANNEL/TIME EXPOSURE; SELECT BY ALL-WATCHED MAP@5")
    log(__doc__)
    H.OUT.mkdir(exist_ok=True)
    interactions = H.load_interactions()
    target_top5, target_all = H.get_eval_targets(interactions, H.EVAL_BLOCK_VAL)
    assert len(target_all) == H.EXPECTED_EVAL_USERS
    phase = build_phase(interactions, list(target_all))
    baseline = v3_rankings(phase)
    configs, rows = {}, []

    def record(name, config, scores):
        recs, fallback = recommend(phase, scores)
        ma, n = H.map_at_k(recs, target_all, 5)
        mt, nt = H.map_at_k(recs, target_top5, 5)
        assert n == nt == H.EXPECTED_EVAL_USERS
        counts = novelty(recs, phase.histories)
        rows.append({"method": name, "map_all": ma, "map_top5": mt,
                     "fallback": fallback, **counts})
        configs[name] = config
        log(f"  {name}: MAP@5(all)={ma:.6f}, MAP@5(top-5)={mt:.6f}; "
            f"novel {counts['novel_top5_count']:,}/{5 * n:,} "
            f"({counts['novel_per_user']:.3f}/user, {counts['novel_fraction']:.1%}); fallback={fallback}")

    record("v3_1_reference", {"kind": "v3"}, baseline)
    for variant in VARIANTS:
        log(f"\nVALIDATION VARIANT {variant}")
        exposure, bonus = exposure_rankings(phase, variant)
        record(f"E_{variant}", {"kind": "exposure", "variant": variant}, exposure)
        record(f"E_{variant}_history_bonus", {"kind": "bonus", "variant": variant}, bonus)
        for kind, weights in (("rrf", (0.5, 1.0, 2.0)), ("linear", (0.3, 0.5, 0.7))):
            for weight in weights:
                name = f"{kind}_E_{variant}_v3_wE={weight}"
                record(name, {"kind": kind, "variant": variant, "weight": weight},
                       fuse_rankings(exposure, baseline, kind, weight))

    table = pd.DataFrame(rows).sort_values(["map_all", "method"], ascending=[False, True], ignore_index=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.to_csv(H.OUT / "step5_exposure_grid.csv", index=False)
    log("\nVALIDATION TABLE — ALL-WATCHED MAP@5 FIRST; NOVEL = NOT WATCHED BY THIS USER IN TRAINING")
    log(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    chosen = [(str(row.method), configs[str(row.method)], float(row.map_all), float(row.map_top5))
              for row in table.head(3).itertuples()]
    # Persist selection before reading the 31-42 schedule or doing any final fit.
    manifest = [{"file": f"submission_v4_{rank}.csv", "method": name, "config": config,
                 "validation_map_all": ma, "validation_map_top5": mt}
                for rank, (name, config, ma, mt) in enumerate(chosen, 1)]
    manifest_path = H.OUT / "step5_exposure_final_configs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    del phase, baseline, exposure, bonus

    log("\nFINAL REFIT — 11-30 VIEWS ONLY; 31-42 SCHEDULE ONLY FOR AIRINGS")
    users = pd.read_csv(H.DATA / "submission.csv", dtype={"user_id": str}).user_id.tolist()
    phase = build_phase(interactions, users, final=True)
    need_v3 = any(config["kind"] in ("v3", "rrf", "linear") for _, config, _, _ in chosen)
    baseline = v3_rankings(phase) if need_v3 else {}
    exposure_cache = {}
    for entry, (_, config, _, _) in zip(manifest, chosen):
        log(f"\n{entry['file']} <- {entry['method']}; config={json.dumps(config, sort_keys=True)}")
        kind = config["kind"]
        if kind == "v3":
            scores = baseline
        else:
            variant = config["variant"]
            if variant not in exposure_cache:
                exposure_cache[variant] = exposure_rankings(phase, variant)
            exposure, bonus = exposure_cache[variant]
            scores = exposure if kind == "exposure" else bonus if kind == "bonus" else fuse_rankings(
                exposure, baseline, kind, config["weight"])
        recs, _ = recommend(phase, scores)
        FB.check_and_write(recs, H.EXAM / entry["file"], phase.pool, phase.popularity, log=log)
    log(f"\nCompleted in {time.perf_counter() - started:.1f}s")
    (H.OUT / "step5_exposure.md").write_text("\n".join(REPORT).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
