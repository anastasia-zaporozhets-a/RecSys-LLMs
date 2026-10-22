#!/usr/bin/env python3
"""Step 1: read-only audit of the sweet.tv data and the view/schedule join.

No model is fitted here.  The script:
  * audits every input file (schema, dtypes, row counts, time formats, timezone);
  * checks the submission users against the view log;
  * joins views to programmes by channel + interval overlap;
  * reports overlap distributions, proposes a "watched" threshold, and
    quantifies unlabelled (tv_show_id == 0) viewing time;
  * reports programme supply / repeat potential;
  * writes out/interactions.csv and out/programmes.csv.

Run from the exam directory:
    .venv/bin/python src/step1_audit_join.py
"""

from __future__ import annotations

import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scipy
    SCIPY_VERSION = scipy.__version__
except ImportError:  # pragma: no cover
    SCIPY_VERSION = "not installed"

EXAM = Path(__file__).resolve().parents[1]
DATA = EXAM / "data"
OUT = EXAM / "out"
OUT.mkdir(exist_ok=True)

VIEW_FILE = DATA / "dataset11-30.csv"
SCHED_FILES = {
    "11-20": DATA / "export_arh_11-20-final.csv",
    "21-30": DATA / "export_arh_21-30-final.csv",
    "31-42": DATA / "export_arh_31-42-final.csv",
}
SUB_FILE = DATA / "submission.csv"

BLOCKS = ["11-20", "21-30", "31-42"]
TRAIN_BLOCKS = ["11-20", "21-30"]
TZ = "Europe/Kyiv"          # inferred, see audit section
BLOCK_STARTS = {
    "11-20": "2020-03-09 00:00:00",
    "21-30": "2020-05-18 00:00:00",
    "31-42": "2020-07-27 00:00:00",
}
BLOCK_ENDS = {
    "11-20": "2020-05-18 00:00:00",
    "21-30": "2020-07-27 00:00:00",
    "31-42": "2020-10-19 00:00:00",
}

# Proposed after looking at the empirical distributions printed below.
WATCH_MIN_SECONDS = 120.0
WATCH_MIN_FRACTION = 0.05

REPORT: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    REPORT.append(msg)


def hr(title: str) -> None:
    log("")
    log("=" * 90)
    log(title)
    log("=" * 90)


def fmt_int(x) -> str:
    return f"{int(x):,}"


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def local_to_utc_epoch(values: pd.Series, fmt: str) -> np.ndarray:
    """Parse naive local wall-clock time and convert to absolute UTC seconds.

    Europe/Kyiv is used because both files have a one-hour DST jump on
    2020-03-29 (03:00 -> 04:00); see the audit evidence below.
    """
    local = pd.to_datetime(values, format=fmt, errors="raise")
    utc = local.dt.tz_localize(TZ, ambiguous="raise", nonexistent="shift_forward")
    utc = utc.dt.tz_convert("UTC").dt.tz_localize(None)
    return utc.astype("datetime64[s]").astype("int64").to_numpy()


def quantile_table(values, name: str) -> str:
    s = pd.Series(values)
    qs = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    q = s.quantile(qs)
    lines = [f"{name}: n={len(s):,}, mean={s.mean():.3f}, std={s.std():.3f}"]
    lines.append("  quantile      value")
    for label, val in q.items():
        lines.append(f"  {label:>8.2f}  {val:>12.3f}")
    return "\n".join(lines)


def read_schedule(path: Path, block: str) -> pd.DataFrame:
    s = pd.read_csv(
        path,
        dtype={"tv_show_id": "int64", "channel_id": "int64", "duration": "int64"},
        low_memory=False,
    )
    s["block"] = block
    return s


def audit_file_basics(name: str, df: pd.DataFrame, path: Path) -> None:
    log(f"\n{name}: {path.name}")
    log(f"  size on disk : {os.path.getsize(path) / 1e6:.1f} MB")
    log(f"  rows         : {len(df):,}")
    log(f"  columns      : {list(df.columns)}")
    log("  dtypes       : " + ", ".join(f"{c}={df[c].dtype}" for c in df.columns))
    nulls = df.isna().sum()
    if nulls.any():
        log("  null counts  : " + ", ".join(f"{c}={int(nulls[c]):,}" for c in df.columns if nulls[c]))
    log("  first rows   :")
    log(df.head(3).to_string(max_colwidth=55))


def build_schedule_groups(train: pd.DataFrame):
    """Return per-channel sorted arrays for interval overlap queries."""
    groups = {}
    for cid, g in train.groupby("channel_id", sort=False):
        g = g.sort_values("start_utc", kind="mergesort")
        starts = g["start_utc"].to_numpy(dtype=np.int64)
        ends = g["end_utc"].to_numpy(dtype=np.int64)
        cummax = np.maximum.accumulate(ends)
        groups[int(cid)] = (
            starts,
            ends,
            cummax,
            g["tv_show_id"].to_numpy(dtype=np.int64),
            g["block"].to_numpy(dtype=object),
            g["duration"].to_numpy(dtype=np.int64),
            g["tv_show_title"].to_numpy(dtype=object),
            g["start_local_str"].to_numpy(dtype=object),
        )
    return groups


def main() -> None:
    t_script = time.perf_counter()

    hr("STEP 1 - DATA AUDIT AND JOIN (READ-ONLY; NO MODEL FITTING)")
    log("Environment")
    log(f"  Python       : {sys.version.split()[0]} ({platform.platform()})")
    log(f"  numpy        : {np.__version__}")
    log(f"  pandas       : {pd.__version__}")
    log(f"  scipy        : {SCIPY_VERSION}")
    log("  randomness   : none used in this step (no seed needed)")
    log(f"  timezone     : {TZ} (inferred from data; see DST evidence)")
    log(f"  exam root    : {EXAM}")

    # ---------------------------------------------------------------- files
    hr("1. FILE INVENTORY AND SCHEMA")

    t0 = time.perf_counter()
    view = pd.read_csv(
        VIEW_FILE,
        dtype={"user_id": "uint64", "vsetv_id": "int64", "duraton": "int64"},
    )
    log(f"Loaded view log in {time.perf_counter() - t0:.2f} s")
    audit_file_basics("View log", view, VIEW_FILE)

    # Parse view times.  Use duraton as the authoritative elapsed duration:
    # wall-clock stop-start is one hour too large for sessions crossing the
    # spring DST jump.
    view["start_utc"] = local_to_utc_epoch(view["start_time"], "%Y-%m-%d %H:%M:%S")
    view["end_utc"] = view["start_utc"] + view["duraton"].to_numpy(dtype=np.int64)
    start_local = pd.to_datetime(view["start_time"], format="%Y-%m-%d %H:%M:%S")
    stop_local = pd.to_datetime(view["stop_time"], format="%Y-%m-%d %H:%M:%S")
    wall_diff = (stop_local - start_local).dt.total_seconds().to_numpy()
    mismatch = wall_diff - view["duraton"].to_numpy(dtype=np.int64)
    log("  time format  : view log = %Y-%m-%d %H:%M:%S (naive local wall clock)")
    log(f"  start range  : {view['start_time'].min()} .. {view['start_time'].max()}")
    log(f"  stop range   : {view['stop_time'].min()} .. {view['stop_time'].max()}")
    log("  view duration (duraton, seconds):")
    log("    " + quantile_table(view["duraton"], "duraton").replace("\n", "\n    "))
    log(f"  stop_time-start_time == duraton: {int((mismatch == 0).sum()):,} / {len(view):,}")
    log(f"  wall-clock difference exceeds duraton by exactly 3600 s: {int((mismatch == 3600).sum()):,}")
    log("  all such rows cross the spring DST jump (start dates): "
        + str(start_local[mismatch == 3600].dt.date.value_counts().sort_index().to_dict()))
    log("  (those sessions last one real hour longer than their local wall-clock timestamps suggest)")
    log(f"  distinct users: {view['user_id'].nunique():,}; distinct vsetv_id: {view['vsetv_id'].nunique():,}")
    log(f"  duplicate full rows: {int(view.duplicated().sum()):,}")

    # Timezone evidence from the view log: the 03:00 local hour disappears on
    # the spring-forward day.
    mar29 = (start_local >= "2020-03-29") & (start_local < "2020-03-30")
    log("  DST evidence (views by hour on 2020-03-29): "
        + str(start_local[mar29].dt.hour.value_counts().sort_index().to_dict()))
    log("  DST evidence (views by hour on 2020-03-28): "
        + str(start_local[(start_local >= "2020-03-28") & (start_local < "2020-03-29")]
              .dt.hour.value_counts().sort_index().to_dict()))

    # Load schedules (also collect labelled rows for the programme catalogue).
    schedule_frames = {}
    labelled_parts = []
    for block in BLOCKS:
        t0 = time.perf_counter()
        s = read_schedule(SCHED_FILES[block], block)
        log(f"Loaded schedule {block} in {time.perf_counter() - t0:.2f} s")
        audit_file_basics(f"Schedule {block}", s, SCHED_FILES[block])
        s["start_local"] = pd.to_datetime(s["start_time"], format="%d.%m.%Y %H:%M:%S", errors="raise")
        s["start_utc"] = local_to_utc_epoch(s["start_time"], "%d.%m.%Y %H:%M:%S")
        s["end_utc"] = s["start_utc"] + s["duration"].to_numpy(dtype=np.int64)
        s["start_local_str"] = s["start_time"]
        log("  time format  : schedule = %d.%m.%Y %H:%M:%S (naive local wall clock)")
        log(f"  start range  : {s['start_local'].min()} .. {s['start_local'].max()}")
        log(f"  end range    : {pd.to_datetime(s['end_utc'], unit='s').min()} .. "
            f"{pd.to_datetime(s['end_utc'], unit='s').max()} (UTC)")
        iso = s["start_local"].dt.isocalendar()
        wk = iso["week"].astype(int)
        log("  ISO week of airing start (rows): "
            + ", ".join(f"W{k}={v:,}" for k, v in wk.value_counts().sort_index().items()))
        log(f"  channels: {s['channel_id'].nunique():,}; labelled rows: "
            f"{int((s['tv_show_id'] != 0).sum()):,} ({fmt_pct((s['tv_show_id'] != 0).mean())}); "
            f"distinct labelled ids: {s.loc[s['tv_show_id'] != 0, 'tv_show_id'].nunique():,}")
        log(f"  tv_show_id == 0 rows: {int((s['tv_show_id'] == 0).sum()):,} "
            f"({fmt_pct((s['tv_show_id'] == 0).mean())})")
        log(f"  schedule duration quantiles: {s['duration'].quantile([0, .01, .5, .99, 1]).to_dict()}")
        # Per-channel monotonicity check.
        g = s.sort_values(["channel_id", "start_local"], kind="mergesort")
        prev_end = g.groupby("channel_id")["end_utc"].shift()
        gap = (g["start_utc"] - prev_end).to_numpy(dtype=float)
        gap = gap[~np.isnan(gap)]
        log(f"  adjacent same-channel rows: gap=0 {fmt_pct((gap == 0).mean())}, "
            f"gap<0 {fmt_pct((gap < 0).mean())}, median gap {np.median(gap):.0f} s, "
            f"gaps>600 s {int((gap > 600).sum()):,}")
        schedule_frames[block] = s
        lab = s.loc[s["tv_show_id"] != 0, [
            "tv_show_id", "tv_show_title", "tv_show_category",
            "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "block",
        ]].copy()
        labelled_parts.append(lab)

    # Schedule DST evidence.
    s1120 = schedule_frames["11-20"]
    x = s1120[(s1120["start_local"] >= "2020-03-29 03:00") &
              (s1120["start_local"] < "2020-03-29 04:00")]
    log(f"\n  DST evidence (schedule 11-20 rows starting 03:00-03:59 on 2020-03-29): {len(x):,}")
    log("  Example channel 3 rows around the jump "
        "(01:25 + 6000 s of real time lands at 04:05 local):")
    log(s1120[(s1120["channel_id"] == 3) &
              (s1120["start_local"] >= "2020-03-29 01:00") &
              (s1120["start_local"] < "2020-03-29 05:30")][
        ["start_time", "duration", "tv_show_id", "tv_show_title"]
    ].to_string(index=False, max_colwidth=45))

    # --------------------------------------------------------- submission
    hr("2. SUBMISSION USERS VS VIEW LOG")
    sub = pd.read_csv(SUB_FILE)
    audit_file_basics("Submission template", sub, SUB_FILE)
    sub_users = set(int(u) for u in sub["user_id"].unique())
    view_users = set(int(u) for u in view["user_id"].unique())
    log(f"  submission rows: {len(sub):,}; distinct users: {len(sub_users):,}")
    log(f"  template tv_show_id tokens: {sub['tv_show_id'].nunique():,} distinct strings "
        f"(all 1,260 rows are the literal '0 0 0 0 0')")

    local_start = start_local
    in_b1 = (local_start >= pd.Timestamp(BLOCK_STARTS["11-20"])) & (
        local_start < pd.Timestamp(BLOCK_ENDS["11-20"]))
    in_b2 = (local_start >= pd.Timestamp(BLOCK_STARTS["21-30"])) & (
        local_start < pd.Timestamp(BLOCK_ENDS["21-30"]))
    in_test = local_start >= pd.Timestamp(BLOCK_STARTS["31-42"])
    log(f"  views outside weeks 11-30: {int((~in_b1 & ~in_b2).sum()):,}; "
        f"views in test weeks 31-42: {int(in_test.sum()):,}")
    u1 = set(int(u) for u in view.loc[in_b1, "user_id"].unique())
    u2 = set(int(u) for u in view.loc[in_b2, "user_id"].unique())
    log(f"  view-log users: {len(view_users):,}; in 11-20: {len(u1):,}; "
        f"in 21-30: {len(u2):,}; in both: {len(u1 & u2):,}; "
        f"11-20 only: {len(u1 - u2):,}; 21-30 only: {len(u2 - u1):,}")
    log(f"  submission users present anywhere in view log: {len(sub_users & view_users):,} / {len(sub_users):,}")
    log(f"  submission users in 11-20: {len(sub_users & u1):,}; "
        f"in 21-30: {len(sub_users & u2):,}; in both: {len(sub_users & u1 & u2):,}")
    log(f"  submission users 11-20 only: {len(sub_users & (u1 - u2)):,}; "
        f"21-30 only: {len(sub_users & (u2 - u1)):,}; "
        f"no view at all: {len(sub_users - view_users):,}")
    log("  NOTE: the view log ends 2020-07-26 (week 30); the test schedule starts "
        "2020-07-27 (week 31). There is no who-watched-what data for weeks 31-42.")

    # --------------------------------------------------------- channel ids
    hr("CHANNEL ID NAMESPACE: vsetv_id vs channel_id")
    view_channels = set(int(c) for c in view["vsetv_id"].unique())
    sched_channels = set()
    for s in schedule_frames.values():
        sched_channels |= set(int(c) for c in s["channel_id"].unique())
    log(f"  view-log vsetv_id values: {len(view_channels):,}; schedule channel_id values: "
        f"{len(sched_channels):,}")
    log(f"  intersection: {len(view_channels & sched_channels):,}; "
        f"view ids with no schedule: {len(view_channels - sched_channels):,} "
        f"{sorted(view_channels - sched_channels)}; "
        f"schedule ids never viewed: {len(sched_channels - view_channels):,}")
    missing = view["vsetv_id"].isin(list(view_channels - sched_channels))
    log(f"  views on channels absent from the schedule: {int(missing.sum()):,} "
        f"({fmt_pct(missing.mean())} of views), "
        f"{int(view.loc[missing, 'duraton'].sum()):,} s")
    log(f"  id ranges: view {min(view_channels)}..{max(view_channels)}, "
        f"schedule {min(sched_channels)}..{max(sched_channels)}")
    log("  Interpretation: the two fields use the same numeric channel namespace where "
        "the channel exists in the export (197/202 view channels match exactly); 5 view "
        "channels are missing from the schedule files. No remapping is applied.")

    # --------------------------------------------------------- clock check
    hr("SAME CLOCK CHECK: DO VIEWS OVERLAP PROGRAMMES ON THE SAME CHANNEL?")
    # Build training groups now (needed for examples and the full join).
    train = pd.concat([schedule_frames[b] for b in TRAIN_BLOCKS], ignore_index=True)
    train = train.sort_values(["channel_id", "start_utc"], kind="mergesort")
    groups = build_schedule_groups(train)

    example_rows = []
    n_zero_examples = 0
    n_labelled_examples = 0
    for r in view.head(200000).itertuples(index=False):
        arr = groups.get(int(r.vsetv_id))
        if arr is None:
            continue
        starts, ends, cummax, show_ids, blocks, durs, titles, start_strs = arr
        st = int(r.start_utc)
        en = st + int(r.duraton)
        i = int(np.searchsorted(cummax, st, side="right"))
        j = i
        while j < len(starts) and starts[j] < en:
            ov = int(min(en, ends[j]) - max(st, int(starts[j])))
            if ov > 0:
                show = int(show_ids[j])
                if show == 0 and n_zero_examples >= 3:
                    pass
                elif show != 0 and n_labelled_examples >= 3:
                    pass
                else:
                    example_rows.append({
                        "user_id": int(r.user_id),
                        "vsetv_id": int(r.vsetv_id),
                        "view_start_local": r.start_time,
                        "view_stop_local": r.stop_time,
                        "programme_start_local": start_strs[j],
                        "programme_dur_s": int(durs[j]),
                        "tv_show_id": show,
                        "overlap_s": ov,
                        "title": str(titles[j])[:45],
                    })
                    if show == 0:
                        n_zero_examples += 1
                    else:
                        n_labelled_examples += 1
                break
            j += 1
        if n_zero_examples >= 3 and n_labelled_examples >= 3:
            break
    log("  Examples from the first 200,000 log rows (same channel, positive interval overlap):")
    log(pd.DataFrame(example_rows).to_string(index=False, max_colwidth=48))
    log("  These are consistent with one local clock: programme starts and view starts "
        "fall on the same wall-clock scale. Absolute timezone is inferred as Europe/Kyiv.")

    # ---------------------------------------------------------------- join
    hr("3. JOIN: CHANNEL + INTERVAL OVERLAP, KEEP tv_show_id != 0")
    log(f"  Training schedule rows used: {len(train):,} "
        f"(blocks {', '.join(TRAIN_BLOCKS)})")

    t0 = time.perf_counter()
    agg = {}
    ratio_events = []
    total_view_seconds = 0
    no_sched_views = 0
    no_sched_seconds = 0
    views_any_overlap = 0
    views_labelled_overlap = 0
    labelled_overlap_seconds = 0
    zero_overlap_seconds = 0
    sum_overlap_seconds = 0
    overlap_events = 0
    excess_views = 0
    excess_seconds = 0

    starts_by_channel = {cid: a[0] for cid, a in groups.items()}
    ends_by_channel = {cid: a[1] for cid, a in groups.items()}
    cummax_by_channel = {cid: a[2] for cid, a in groups.items()}
    shows_by_channel = {cid: a[3] for cid, a in groups.items()}
    blocks_by_channel = {cid: a[4] for cid, a in groups.items()}
    durs_by_channel = {cid: a[5] for cid, a in groups.items()}

    for r in view.itertuples(index=False):
        cid = int(r.vsetv_id)
        user = int(r.user_id)
        dur = int(r.duraton)
        st = int(r.start_utc)
        en = st + dur
        total_view_seconds += dur

        starts = starts_by_channel.get(cid)
        if starts is None:
            no_sched_views += 1
            no_sched_seconds += dur
            continue

        ends = ends_by_channel[cid]
        cummax = cummax_by_channel[cid]
        show_ids = shows_by_channel[cid]
        blocks = blocks_by_channel[cid]
        durs = durs_by_channel[cid]

        i = int(np.searchsorted(cummax, st, side="right"))
        j = i
        n = len(starts)
        view_overlap = 0
        view_has_labelled = False
        while j < n and starts[j] < en:
            e = ends[j]
            if e > st:
                ov = int(min(en, e) - max(st, int(starts[j])))
                if ov > 0:
                    overlap_events += 1
                    view_overlap += ov
                    show = int(show_ids[j])
                    block = str(blocks[j])
                    sched_dur = int(durs[j])
                    ratio_events.append(ov / sched_dur)
                    if show == 0:
                        zero_overlap_seconds += ov
                    else:
                        labelled_overlap_seconds += ov
                        view_has_labelled = True
                        key = (user, show, block)
                        rec = agg.get(key)
                        if rec is None:
                            agg[key] = [ov, 1, sched_dur]
                        else:
                            rec[0] += ov
                            rec[1] += 1
                            rec[2] += sched_dur
            j += 1

        if view_overlap > 0:
            views_any_overlap += 1
            sum_overlap_seconds += view_overlap
            if view_overlap > dur:
                excess_views += 1
                excess_seconds += view_overlap - dur
        if view_has_labelled:
            views_labelled_overlap += 1

    log(f"  Join loop runtime: {time.perf_counter() - t0:.2f} s")
    log(f"  views: {len(view):,}; total viewing time: {total_view_seconds:,} s "
        f"({total_view_seconds / 86400:.1f} channel-days)")
    log(f"  views with no schedule channel: {no_sched_views:,} "
        f"({fmt_pct(no_sched_views / len(view))}), {no_sched_seconds:,} s")
    log(f"  views with any schedule overlap: {views_any_overlap:,} "
        f"({fmt_pct(views_any_overlap / len(view))})")
    log(f"  views with at least one labelled overlap: {views_labelled_overlap:,} "
        f"({fmt_pct(views_labelled_overlap / len(view))})")
    log(f"  view-airing overlap events: {overlap_events:,}")
    log(f"  sum of overlaps / total viewing time: "
        f"{fmt_pct(sum_overlap_seconds / total_view_seconds)}")
    log(f"  views whose summed overlap exceeds their duraton (overlapping schedule rows): "
        f"{excess_views:,}; excess {excess_seconds:,} s")

    log("\n  Viewing-time allocation across schedule labels:")
    log(f"    tv_show_id != 0 : {labelled_overlap_seconds:,} s "
        f"({fmt_pct(labelled_overlap_seconds / total_view_seconds)} of total viewing time)")
    log(f"    tv_show_id == 0 : {zero_overlap_seconds:,} s "
        f"({fmt_pct(zero_overlap_seconds / total_view_seconds)} of total viewing time)")
    residual = total_view_seconds - labelled_overlap_seconds - zero_overlap_seconds
    log(f"    no schedule overlap: {residual:,} s "
        f"({fmt_pct(residual / total_view_seconds)} of total viewing time)")
    denom_covered = labelled_overlap_seconds + zero_overlap_seconds
    if denom_covered:
        log(f"    among covered time, tv_show_id == 0 share: "
            f"{fmt_pct(zero_overlap_seconds / denom_covered)}")

    inter = pd.DataFrame(
        [(u, s, b, v[0], v[1], v[2]) for (u, s, b), v in agg.items()],
        columns=["user_id", "tv_show_id", "block", "overlap_seconds", "airings", "duration_sum"],
    )
    inter["overlap_fraction"] = inter["overlap_seconds"] / inter["duration_sum"]
    inter["_block_order"] = inter["block"].map({b: i for i, b in enumerate(TRAIN_BLOCKS)})
    inter = inter.sort_values(
        ["user_id", "_block_order", "overlap_seconds", "tv_show_id"],
        ascending=[True, True, False, True],
    ).drop(columns="_block_order").reset_index(drop=True)

    log(f"\n  Aggregated (user, programme, block) records with positive labelled overlap: "
        f"{len(inter):,}")
    log(f"  distinct users in these records: {inter['user_id'].nunique():,}")
    log(f"  distinct labelled programmes touched: {inter['tv_show_id'].nunique():,}")
    log(f"  records per block: "
        + ", ".join(f"{b}={int((inter['block'] == b).sum()):,}" for b in TRAIN_BLOCKS))

    log("\n  Distribution of aggregated overlap_seconds (one record = user x programme x block):")
    log(quantile_table(inter["overlap_seconds"], "overlap_seconds"))
    log("\n  Distribution of aggregated overlap_fraction = overlap_seconds / sum(duration of airings touched):")
    log(quantile_table(inter["overlap_fraction"], "overlap_fraction"))
    log("\n  Distribution of airings touched per record:")
    log(quantile_table(inter["airings"], "airings"))
    log("\n  Distribution of per-view/per-airing overlap / programme duration (events):")
    log(quantile_table(pd.Series(ratio_events), "event_fraction"))

    # Threshold evidence.
    log("\n  Candidate absolute thresholds (surviving records, surviving users):")
    for t in [30, 60, 90, 120, 180, 300, 600, 900, 1800]:
        m = inter["overlap_seconds"] >= t
        log(f"    >= {t:>5} s : {int(m.sum()):>10,} records "
            f"({fmt_pct(m.mean())}), {inter.loc[m, 'user_id'].nunique():>5,} users")
    log("  Candidate fraction thresholds (records, users):")
    for f in [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]:
        m = inter["overlap_fraction"] >= f
        log(f"    >= {f:>4.0%} : {int(m.sum()):>10,} records "
            f"({fmt_pct(m.mean())}), {inter.loc[m, 'user_id'].nunique():>5,} users")
    watched = (inter["overlap_seconds"] >= WATCH_MIN_SECONDS) & (
        inter["overlap_fraction"] >= WATCH_MIN_FRACTION)
    log(f"\n  PROPOSED watched rule: overlap_seconds >= {WATCH_MIN_SECONDS:.0f} s AND "
        f"overlap_fraction >= {WATCH_MIN_FRACTION:.0%}")
    log(f"    watched records: {int(watched.sum()):,} ({fmt_pct(watched.mean())} of positive records)")
    log(f"    watched users: {inter.loc[watched, 'user_id'].nunique():,}")
    log(f"    records dropped: {int((~watched).sum()):,}")
    log(f"    watched records per block: "
        + ", ".join(f"{b}={int(((inter['block'] == b) & watched).sum()):,}" for b in TRAIN_BLOCKS))
    log("    Justification from the quantiles above:")
    log("      - 120 s is 1.3x the p10 of overlap_seconds (91 s) and 0.55x the p25 (220 s); "
        "it removes the shortest zapping/boundary overlaps while still fitting short items.")
    log("      - 5% is 1.7x the p10 of overlap_fraction (1.3%) and 0.48x the median (10.4%); "
        "it removes accidental crossings of long films/airings.")
    log("      - The AND rule is protective in both directions: 5% of a 25-min bulletin is only "
        "75 s, and 120 s of a 2-hour film is only 1.7%; neither alone is a good 'watched' test.")
    log("      - The rule keeps 64.2% of positive records and 4,890/4,895 users; the surviving "
        "set is not dominated by a handful of heavy users.")

    # Submission coverage after join.
    sub_in_inter = sub_users & set(int(u) for u in inter["user_id"].unique())
    sub_watched = sub_users & set(int(u) for u in inter.loc[watched, "user_id"].unique())
    log(f"\n  Submission users with >=1 labelled interaction: {len(sub_in_inter):,} / {len(sub_users):,}")
    log(f"  Submission users with >=1 watched record: {len(sub_watched):,} / {len(sub_users):,}")
    log(f"  Submission users with no labelled interaction (cold start at item level): "
        f"{len(sub_users - sub_in_inter):,}")

    # ---------------------------------------------------------- programmes
    hr("4. PROGRAMME SUPPLY AND REPEAT POTENTIAL")
    labelled = pd.concat(labelled_parts, ignore_index=True)
    show_blocks = labelled.groupby("tv_show_id")["block"].apply(lambda x: set(x))
    per_block = {b: set(labelled.loc[labelled["block"] == b, "tv_show_id"]) for b in BLOCKS}
    for b in BLOCKS:
        log(f"  distinct labelled programmes in {b}: {len(per_block[b]):,} "
            f"(+ the tv_show_id=0 bucket = {len(per_block[b]) + 1:,} distinct id values)")
    log("  Block dates (ISO weeks): 11-20 = Mon 2020-03-09 .. Sun 2020-05-17; "
        "21-30 = Mon 2020-05-18 .. Sun 2020-07-26; "
        "31-42 = Mon 2020-07-27 .. Sun 2020-10-18.")
    log(f"  labelled programmes in training union (11-20 or 21-30): "
        f"{len(per_block['11-20'] | per_block['21-30']):,}")
    log(f"  labelled programmes in TEST schedule 31-42: {len(per_block['31-42']):,}")
    log(f"  11-20 programmes that also air in 21-30: "
        f"{len(per_block['11-20'] & per_block['21-30']):,} / {len(per_block['11-20']):,} "
        f"({fmt_pct(len(per_block['11-20'] & per_block['21-30']) / len(per_block['11-20']))})")
    log(f"  11-20 programmes that also air in TEST 31-42: "
        f"{len(per_block['11-20'] & per_block['31-42']):,} / {len(per_block['11-20']):,}")
    log(f"  21-30 programmes that also air in TEST 31-42: "
        f"{len(per_block['21-30'] & per_block['31-42']):,} / {len(per_block['21-30']):,}")
    log(f"  programmes present in all three blocks: "
        f"{len(per_block['11-20'] & per_block['21-30'] & per_block['31-42']):,}")

    # Per-user repeat potential, based on the proposed watched rule.
    w = inter.loc[watched]
    w1 = w.loc[w["block"] == "11-20"].groupby("user_id")["tv_show_id"].apply(set)
    test_shows = per_block["31-42"]
    next_shows = per_block["21-30"]
    rep21 = w1.apply(lambda s: len(s & next_shows) / len(s) if s else np.nan)
    rep31 = w1.apply(lambda s: len(s & test_shows) / len(s) if s else np.nan)
    log(f"\n  Per-user repeat potential for users with >=1 watched 11-20 programme "
        f"(n={len(w1):,}):")
    log("    fraction of watched 11-20 programmes that air again in 21-30:")
    log("      " + quantile_table(rep21.dropna(), "repeat_fraction_21_30").replace("\n", "\n      "))
    log("    fraction of watched 11-20 programmes that air again in TEST 31-42:")
    log("      " + quantile_table(rep31.dropna(), "repeat_fraction_31_42").replace("\n", "\n      "))
    pair_w1 = int(w.loc[w["block"] == "11-20"].shape[0])
    pair_rep21 = int(w.loc[(w["block"] == "11-20") & w["tv_show_id"].isin(next_shows)].shape[0])
    log(f"  Pooled user-programme pairs: {pair_rep21:,} / {pair_w1:,} "
        f"({fmt_pct(pair_rep21 / pair_w1)}) of watched 11-20 pairs have the programme "
        f"airing again in 21-30.")
    sub_w1 = w1[w1.index.isin(sub_users)]
    log(f"  Submission users with >=1 watched 11-20 programme: {len(sub_w1):,}; "
        f"median repeat fraction into 21-30: {sub_w1.apply(lambda s: len(s & next_shows) / len(s)).median():.3f}")

    # ------------------------------------------------------------- exports
    hr("5. EXPORTS")
    # interactions.csv is the model-ready watched set.  It contains exactly the
    # records that pass the proposed rule above; the 542,287 sub-threshold
    # positive overlaps remain in the audit report but are not exported.
    export_inter = inter.loc[watched, [
        "user_id", "tv_show_id", "block", "overlap_seconds", "airings"
    ]].copy()
    export_inter["overlap_seconds"] = export_inter["overlap_seconds"].astype("int64")
    export_inter["airings"] = export_inter["airings"].astype("int32")
    inter_path = OUT / "interactions.csv"
    t0 = time.perf_counter()
    export_inter.to_csv(inter_path, index=False)
    log(f"  wrote {inter_path} in {time.perf_counter() - t0:.2f} s")
    log(f"    rows={len(export_inter):,} (watched only; all positive overlaps={len(inter):,}), "
        f"size={os.path.getsize(inter_path) / 1e6:.1f} MB")
    log(f"    columns={list(export_inter.columns)}")

    # Programme catalogue: mode title, mode category, union of genres, block set.
    def mode_deterministic(s: pd.Series):
        vc = s.value_counts()
        top = vc[vc == vc.max()].index
        return sorted(top)[0]

    title = labelled.groupby("tv_show_id")["tv_show_title"].agg(mode_deterministic)
    category = labelled.groupby("tv_show_id")["tv_show_category"].agg(mode_deterministic)
    genre_long = labelled[["tv_show_id", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]].melt(
        id_vars="tv_show_id", value_name="genre"
    )
    genre_long = genre_long.dropna(subset=["genre"])
    genre_long["genre"] = genre_long["genre"].astype(str).str.strip()
    genre_long = genre_long[genre_long["genre"] != ""]
    genres = genre_long.groupby("tv_show_id")["genre"].apply(lambda s: ";".join(sorted(set(s))))
    block_set = show_blocks.apply(lambda s: ";".join(b for b in BLOCKS if b in s))
    programmes = pd.concat(
        [title.rename("title"), category.rename("category"), genres.rename("genres"),
         block_set.rename("blocks")],
        axis=1,
    ).reset_index()
    programmes = programmes[["tv_show_id", "title", "category", "genres", "blocks"]]
    programmes = programmes.sort_values("tv_show_id").reset_index(drop=True)
    prog_path = OUT / "programmes.csv"
    t0 = time.perf_counter()
    programmes.to_csv(prog_path, index=False)
    log(f"  wrote {prog_path} in {time.perf_counter() - t0:.2f} s")
    log(f"    rows={len(programmes):,}, size={os.path.getsize(prog_path) / 1e6:.1f} MB")
    log(f"    columns={list(programmes.columns)}")
    n_title_variants = int(
        labelled.groupby("tv_show_id")["tv_show_title"].nunique().gt(1).sum())
    genre_signature = (
        labelled["tv_show_genre_1"].fillna("") + "|" +
        labelled["tv_show_genre_2"].fillna("") + "|" +
        labelled["tv_show_genre_3"].fillna("")
    )
    n_genre_variants = int(
        labelled.assign(g=genre_signature).groupby("tv_show_id").g.nunique().gt(1).sum())
    category_consistent = bool(
        labelled.groupby("tv_show_id")["tv_show_category"].nunique().eq(1).all())
    log(f"    distinct title variants collapsed: "
        f"{n_title_variants:,} shows had >1 title")
    log(f"    shows with >1 genre set across airings: {n_genre_variants:,}")
    log(f"    category is consistent within a show: {category_consistent}")

    # ------------------------------------------------------------- summary
    hr("SUMMARY / UNVERIFIED POINTS")
    log("  * Same clock: yes for the overlap join; both files are naive local wall clock.")
    log(f"  * Absolute timezone: inferred as {TZ} from the DST jump; the files do not state a zone. "
        "If the platform actually used a fixed-offset zone, only the 117 DST-spanning sessions "
        "would be affected.")
    log("  * vsetv_id and channel_id use the same numeric namespace where both exist; "
        "5 view channels have no schedule rows and are treated as uncovered time.")
    log("  * Week labels mean ISO weeks: W11 starts 2020-03-09, W20 ends 2020-05-17; "
        "W21 starts 2020-05-18, W30 ends 2020-07-26; W31 starts 2020-07-27, W42 ends 2020-10-18. "
        "Files include small boundary tails from adjacent weeks (W10/W21, W31, W43).")
    log("  * No model fitting and no use of weeks 31-42 viewing data: there are 0 such views. "
        "The test schedule is used only for programme supply/repeat counts in this step.")
    log("  * tv_show_id == 0 is unlabelled airtime: not an item, not a miss; it is counted only "
        "to quantify viewing context and dropped from interactions.")
    log(f"  * WATCHED rule is a proposal, not a ground truth: >= {WATCH_MIN_SECONDS:.0f} s and "
        f">= {WATCH_MIN_FRACTION:.0%} of the summed aired duration of touched airings. "
        "Sensitivity counts are printed above.")
    log(f"  * interactions.csv contains the {int(watched.sum()):,} watched records only; "
        f"the {int((~watched).sum()):,} positive-but-sub-threshold overlaps are not exported. "
        "programmes.csv contains all 9,671 labelled ids seen in any schedule.")
    log("")
    log(f"Total runtime: {time.perf_counter() - t_script:.2f} s")

    report_path = OUT / "audit_report.md"
    report_path.write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
