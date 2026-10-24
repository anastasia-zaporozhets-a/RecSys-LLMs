"""Merged-interval airing coverage and clipped weekly-airing multiplier.

Idea credit: the per-airing coverage definition and the clipped airtime
multiplier come from an approach described by classmate Vladimir Kuznetsov.
No code or files were shared.

Only positively covered (user, airing, window) rows are materialized; omitted
rows have zero coverage. Intervals are unioned globally per user/channel,
including across CSV chunks, before joining airings. Coverage uses the FULL
airing duration as denominator, with its numerator clipped to the observation
window. mean_coverage averages positively viewed airings; an additional
all_airings_mean includes unviewed scheduled airings as zeros for audit.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import harness as H
import step5_exposure as E
import step6_ltr as S
import step7_ltr_plus as P
import step7_feature_cache as C

CREDIT = ("Idea credit: the per-airing coverage definition and the clipped airtime multiplier "
          "come from an approach described by classmate Vladimir Kuznetsov. No code or files were shared.")
WINDOWS = {
    "train20": (E.TRAIN_START, E.FINAL_START),
    "train10": (E.TRAIN_START, E.VALIDATION_START),
    "train15": (E.TRAIN_START, P.MIDPOINT),
    "label10": (E.VALIDATION_START, E.FINAL_START),
    "label5": (P.MIDPOINT, E.FINAL_START),
}
FEATURE_NAMES = ["n50_per_training_week", "n80_per_training_week", "mean_airing_coverage", "sqrt_clipped_airing_multiplier"]


def fingerprint(base):
    return {"version": 1, "base_sources": base, "coverage_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def union_intervals(users, channels, starts, ends):
    """Vectorized segmented interval union. IDs are integer codes, never floats."""
    users, channels = np.asarray(users, dtype=np.int32), np.asarray(channels, dtype=np.int64)
    starts, ends = np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)
    valid = ends > starts
    users, channels, starts, ends = users[valid], channels[valid], starts[valid], ends[valid]
    if not len(starts):
        return pd.DataFrame({"user_code": users, "channel_id": channels, "start": starts, "end": ends})
    order = np.lexsort((ends, starts, channels, users))
    users, channels, starts, ends = users[order], channels[order], starts[order], ends[order]
    new_group = np.r_[True, (users[1:] != users[:-1]) | (channels[1:] != channels[:-1])]
    group = np.cumsum(new_group, dtype=np.int64) - 1
    lower = int(starts.min())
    span = int(ends.max()) - lower + 1
    assert int(group[-1] + 1) * span < np.iinfo(np.int64).max
    # Increasing disjoint offsets make a global cumulative max equivalent to
    # segmented cumulative maxima, without carrying ends across user/channels.
    running = np.maximum.accumulate(ends - lower + group * span) - group * span + lower
    new_segment = new_group | np.r_[True, starts[1:] > running[:-1]]
    heads = np.flatnonzero(new_segment)
    return pd.DataFrame({"user_code": users[heads], "channel_id": channels[heads],
                         "start": starts[heads], "end": np.maximum.reduceat(ends, heads)})


def merged_views(root, fp, check, log):
    path = root / "merged_views"
    if C.inspect_cache(path, fp, "step9_views") is not None:
        log("A: loading cached globally merged user/channel intervals")
        with np.load(path / "intervals.npz", allow_pickle=False) as arrays:
            return arrays["users"].tolist(), pd.DataFrame({name: arrays[name] for name in ("user_code", "channel_id", "start", "end")})
    parts, lookup, users = [], {}, []
    lower, upper = E.epoch(E.TRAIN_START), E.epoch(E.FINAL_START)
    for number, frame in enumerate(pd.read_csv(
            H.DATA / "dataset11-30.csv", chunksize=E.IO_CHUNK,
            usecols=["user_id", "vsetv_id", "start_time", "duraton"],
            dtype={"user_id": str, "vsetv_id": "int64", "duraton": "int64"}), 1):
        check("raw-view parsing")
        frame = frame.loc[(frame.start_time < f"{E.FINAL_START} 00:00:00") & (frame.duraton > 0)]
        for user in frame.user_id.unique():
            if user not in lookup:
                lookup[user] = len(users)
                users.append(user)
        start = E.parse_time(frame.start_time, "%Y-%m-%d %H:%M:%S")
        part = pd.DataFrame({"user_code": frame.user_id.map(lookup).to_numpy(dtype=np.int32),
                             "channel_id": frame.vsetv_id.to_numpy(), "start": np.maximum(start, lower),
                             "end": np.minimum(start + frame.duraton.to_numpy(), upper)})
        parts.append(part.loc[part.end > part.start])
        log(f"  raw chunk {number}: {len(frame):,} past-only sessions")
    raw = pd.concat(parts, ignore_index=True)
    del parts
    check("global interval union")
    merged = union_intervals(raw.user_code, raw.channel_id, raw.start, raw.end)
    log(f"  interval union: {len(raw):,} sessions -> {len(merged):,} disjoint user/channel intervals")
    del raw
    check("saving merged intervals")
    with C.staged_cache(path) as folder:
        np.savez_compressed(folder / "intervals.npz", users=np.asarray(users, dtype=str),
                            **{name: merged[name].to_numpy() for name in merged})
        C.write_manifest(folder, {"idea_credit": CREDIT, "viewing_end_exclusive": E.FINAL_START,
                                  "rows": len(merged)}, fp, "step9_views")
    return users, merged


def read_airings(blocks, lower, upper, check, log):
    parts = []
    for block in blocks:
        check(f"schedule {block}")
        frame = pd.read_csv(H.DATA / f"export_arh_{block}-final.csv",
                            usecols=["channel_id", "tv_show_id", "start_time", "duration"],
                            dtype={"channel_id": "int64", "tv_show_id": "int64", "duration": "int64"})
        frame = frame.loc[(frame.tv_show_id != 0) & (frame.duration > 0)].copy()
        frame["start"] = E.parse_time(frame.start_time, "%d.%m.%Y %H:%M:%S")
        frame["end"] = frame.start + frame.duration
        parts.append(frame.loc[(frame.start < upper) & (frame.end > lower),
                               ["channel_id", "tv_show_id", "start", "end", "duration"]])
    result = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["channel_id", "tv_show_id", "start", "duration"]).sort_values(["channel_id", "start"]).reset_index(drop=True)
    result["airing_id"] = np.arange(len(result))
    log(f"  schedules {blocks}: {len(result):,} unique labelled airings; full durations retained")
    return result


def coverage_for_window(views, airings, users, lower, upper, check, log):
    parts = []
    schedule = airings.loc[(airings.tv_show_id != 0) & (airings.duration > 0) &
                           (airings.start < upper) & (airings.end > lower)]
    supply = schedule.groupby("tv_show_id").size().rename("airings")
    view_groups = views.groupby("channel_id", sort=False).indices
    for channel, airing in schedule.groupby("channel_id", sort=False):
        check("per-airing coverage join")
        positions = view_groups.get(channel)
        if positions is None:
            continue
        airing = airing.sort_values("start")
        v = views.iloc[positions]
        v = v.loc[(v.start < upper) & (v.end > lower)]
        for offset in range(0, len(v), E.IO_CHUNK):
            chunk = v.iloc[offset:offset + E.IO_CHUNK]
            source, row, start, end = S.overlap_pairs(
                np.maximum(chunk.start.to_numpy(), lower), np.minimum(chunk.end.to_numpy(), upper),
                airing.start.to_numpy(), airing.end.to_numpy())
            if len(source):
                parts.append(pd.DataFrame({"user_code": chunk.user_code.to_numpy()[source],
                                           "airing_id": airing.airing_id.to_numpy()[row], "covered": end - start}))
            check("coverage join chunk")
    if not parts:
        empty = pd.DataFrame(columns=["user_id", "tv_show_id", "n50", "n80", "mean_coverage", "observed_airings", "all_airings_mean"])
        return empty.set_index(["user_id", "tv_show_id"]), supply, pd.DataFrame()
    per_airing = pd.concat(parts, ignore_index=True).groupby(["user_code", "airing_id"], as_index=False).covered.sum()
    details = airings.set_index("airing_id")
    per_airing["duration"] = per_airing.airing_id.map(details.duration)
    per_airing["tv_show_id"] = per_airing.airing_id.map(details.tv_show_id)
    assert ((per_airing.covered > 0) & (per_airing.covered <= per_airing.duration)).all(), "Union coverage exceeds full airing duration"
    per_airing["coverage"] = per_airing.covered / per_airing.duration
    per_airing["n50"] = (per_airing.covered * 2 >= per_airing.duration).astype(np.int32)
    per_airing["n80"] = (per_airing.covered * 5 >= per_airing.duration * 4).astype(np.int32)
    grouped = per_airing.groupby(["user_code", "tv_show_id"]).agg(
        n50=("n50", "sum"), n80=("n80", "sum"), mean_coverage=("coverage", "mean"),
        observed_airings=("airing_id", "size"), coverage_sum=("coverage", "sum"))
    pairs = grouped.reset_index()
    pairs["user_id"] = np.asarray(users, dtype=object)[pairs.user_code.to_numpy()]
    pairs["all_airings_mean"] = pairs.coverage_sum / pairs.tv_show_id.map(supply)
    pairs = pairs.drop(columns=["user_code", "coverage_sum"]).set_index(["user_id", "tv_show_id"])
    log(f"  {len(per_airing):,} positive user/airing rows; {len(pairs):,} user/item pairs; "
        f"n50={int(pairs.n50.sum()):,}, n80={int(pairs.n80.sum()):,}")
    return pairs, supply, per_airing


def ensure_coverage(root, fp, check, log):
    missing = [name for name in WINDOWS if C.inspect_cache(root / name, fp, "step9_coverage") is None]
    if not missing:
        log("A: all coverage caches reused")
        return
    users, views = merged_views(root, fp, check, log)
    # 31-42 is deliberately not passed to the view/airing join.
    airings = read_airings(["11-20", "21-30"], E.epoch(E.TRAIN_START), E.epoch(E.FINAL_START), check, log)
    for name in missing:
        check(f"coverage window {name}")
        begin, end = map(E.epoch, WINDOWS[name])
        log(f"A window {name}: [{WINDOWS[name][0]}, {WINDOWS[name][1]})")
        pairs, supply, per_airing = coverage_for_window(views, airings, users, begin, end, check, log)
        with C.staged_cache(root / name) as folder:
            pairs.to_csv(folder / "pairs.csv.gz", compression="gzip")
            np.savez_compressed(folder / "supply.npz", items=supply.index.to_numpy(dtype=np.int64), counts=supply.to_numpy())
            np.savez_compressed(folder / "per_airing.npz", users=np.asarray(users, dtype=str),
                                **{col: per_airing[col].to_numpy() for col in per_airing})
            C.write_manifest(folder, {"window": name, "dates": WINDOWS[name], "idea_credit": CREDIT,
                                      "mean_coverage": "mean over positively viewed airings; unseen pairs zero",
                                      "denominator": "full airing duration; numerator clipped to window",
                                      "unmaterialized_user_airing_rows": "zero coverage"}, fp, "step9_coverage")
        log(f"  saved {root / name}")


def load_pairs(root, name, fp):
    assert C.inspect_cache(root / name, fp, "step9_coverage") is not None
    return pd.read_csv(root / name / "pairs.csv.gz", dtype={"user_id": str, "tv_show_id": "int64"}).set_index(["user_id", "tv_show_id"])


def load_supply(root, name, fp):
    assert C.inspect_cache(root / name, fp, "step9_coverage") is not None
    with np.load(root / name / "supply.npz", allow_pickle=False) as arrays:
        return pd.Series(arrays["counts"], index=arrays["items"])


def multiplier(training, target, training_weeks, target_weeks):
    training, target = np.asarray(training, dtype=np.float64), np.asarray(target, dtype=np.float64)
    assert (training >= 0).all() and (target >= 0).all()
    ratio = np.ones(np.broadcast_shapes(training.shape, target.shape))
    np.divide(target / target_weeks, training / training_weeks, out=ratio, where=training > 0)
    ratio = np.where((training == 0) & (target > 0), np.inf, ratio)
    return np.sqrt(np.clip(ratio, .25, 4.0))
