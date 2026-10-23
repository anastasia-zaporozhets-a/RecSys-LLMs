#!/usr/bin/env python3
"""Final-only v6_3 build; rebuild/cache missing features and reuse saved trees.

Run from exam/: .venv/bin/python -u src/step7_refit_all_together.py
Logs live to out/step7_v6_3_refit.log and stdout. No candidate-recall search,
ablations, user-split changes or early stopping are run. Caches under
out/cache/step7/{fold10,fold5,final} are reusable by Step 8. Existing feature
caches are loaded, so subsequent runs do not recompute them. Intermediate
window aggregates are also cached to make interrupted rebuilds resumable.

POST-HOC selection: all_together was chosen after inspecting fold5 results,
not by the original fold10 criterion. Tree counts still come from the saved
fold10 early stopping. This is additional validation-informed selection,
not a new untouched-holdout estimate. No Kaggle scores are used by this job.
The current designated final remains submission_v6_1.csv.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import gc
import json
from pathlib import Path
import sys
import time

import pandas as pd

import harness as H
import step5_exposure as E
import step6_ltr as S
import step7_ltr_plus as P
import step7_feature_cache as C


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def selection_plan(out):
    original = json.loads((out / "step7_selection.json").read_text(encoding="utf-8"))
    config = original["recipes"]["all_together"]
    assert config == P.recipes(200)["all_together"], "Expected size-200 all_together recipe"
    iterations = original["iterations"]["all_together"]
    assert set(iterations) == {str(seed) for seed in P.SEEDS}
    assert all(isinstance(trees, int) and trees > 0 for trees in iterations.values())
    table = pd.read_csv(out / "step7_ablations.csv").set_index("recipe")
    all_row, previous = table.loc["all_together"], table.loc["previous_v5"]
    return {
        "file": "submission_v6_3.csv", "recipe": "all_together", "config": config,
        "seed_iterations": iterations,
        "selection_metric": "later fold5 all-watched MAP@5",
        "selection_is_post_hoc": True,
        "selection_note": "Chosen by the user after seeing the fold5 column, not by fold10; "
                          "fold5 is consequently not an untouched holdout for this selection.",
        "selection_rationale": "Later fold5 (26-30) is closer in time to the test weeks 31-42.",
        "tree_count_source": "Saved Step 7 fold10 NDCG@5 early stopping; not retuned on fold5",
        "fold5_all_together": float(all_row.map_all_fold5),
        "fold5_previous_v5": float(previous.map_all_fold5),
        "fold5_delta": float(all_row.map_all_fold5 - previous.map_all_fold5),
        "fold10_all_together": float(all_row.map_all_fold10),
        "fold10_previous_v5": float(previous.map_all_fold10),
        "designated_final_so_far": "submission_v6_1.csv",
    }


def obtain_folds(cache, fingerprint):
    """Load complete caches first, reconstruct just the missing folds otherwise."""
    cache = Path(cache)
    required = ("fold10", "fold5", "final")
    missing = [name for name in required if C.inspect_cache(cache / name, fingerprint, "fold") is None]
    if not missing:
        print("CACHE HIT: all three feature folds; no source data or signals will be recomputed", flush=True)
        snapshot_split(cache)
        return [C.load_fold(cache / name, fingerprint) for name in required]
    print("MISSING FEATURE CACHES: " + ", ".join(missing), flush=True)
    schedules = {block: S.read_schedule(block) for block in H.TRAIN_BLOCKS_FINAL}
    need_windows = set()
    if "fold10" in missing:
        need_windows.add("train10")
    if "fold5" in missing:
        need_windows.update(("train15", "label5"))
    if "final" in missing:
        need_windows.add("train20")
    missing_windows = [name for name in sorted(need_windows) if C.inspect_cache(
        cache / "windows" / name, fingerprint, "window_stats") is None]
    if missing_windows:
        print("Rebuilding raw window aggregates once; no model-selection stage is run", flush=True)
        windows = P.aggregate_windows(schedules)
        for name, stats in windows.items():
            path = cache / "windows" / name
            if C.inspect_cache(path, fingerprint, "window_stats") is None:
                C.save_window(path, name, stats, fingerprint)
                print(f"SAVED WINDOW CACHE: {path}", flush=True)
        del windows
        gc.collect()
    interactions = H.load_interactions()
    catalogue = H.load_programmes()
    categories = sorted(set(catalogue.category.fillna("__missing__").astype(str)) | {"__missing__"})
    mapping = dict(enumerate(categories))
    all_schedule = pd.concat(schedules.values(), ignore_index=True)
    for name in required:
        path = cache / name
        if name not in missing:
            print(f"CACHE HIT: {path}", flush=True)
            continue
        if name == "fold10":
            stats = C.load_window(cache / "windows" / "train10", fingerprint)
            _, targets = H.get_eval_targets(interactions, H.EVAL_BLOCK_VAL)
            phase = E.build_phase(interactions, list(targets))
            fold = P.prepare_fold(name, phase, stats, schedules["11-20"], schedules["21-30"],
                                   E.VALIDATION_START, targets, 10, 10, catalogue, categories)
        elif name == "fold5":
            stats = C.load_window(cache / "windows" / "train15", fingerprint)
            label_stats = C.load_window(cache / "windows" / "label5", fingerprint)
            targets = P.qualifying_interactions(label_stats).groupby("user_id").tv_show_id.apply(set).to_dict()
            del label_stats
            training = P.clip_schedule(all_schedule, E.TRAIN_START, P.MIDPOINT)
            target = P.clip_schedule(schedules["21-30"], P.MIDPOINT, E.FINAL_START)
            phase = P.custom_phase(sorted(targets), P.qualifying_interactions(stats), target, P.MIDPOINT)
            fold = P.prepare_fold(name, phase, stats, training, target, P.MIDPOINT,
                                   targets, 15, 5, catalogue, categories)
            del training, target
        else:
            stats = C.load_window(cache / "windows" / "train20", fingerprint)
            users = pd.read_csv(H.DATA / "submission.csv", dtype={"user_id": str}).user_id.tolist()
            targets = {user: set() for user in users}
            phase = E.build_phase(interactions, users, final=True)
            target = S.read_schedule(H.EVAL_BLOCK_FINAL)
            fold = P.prepare_fold(name, phase, stats, all_schedule, target, E.FINAL_START,
                                   targets, 20, 12, catalogue, categories)
            del target
        C.save_fold(path, fold, fingerprint, category_mapping=mapping)
        print(f"SAVED FEATURE CACHE: {path}", flush=True)
        del fold, phase, stats, targets
        gc.collect()
    # Persist the original user split alongside reusable features for Step 8.
    snapshot_split(cache)
    return [C.load_fold(cache / name, fingerprint) for name in required]


def snapshot_split(cache):
    split = H.OUT / "step7_user_split.json"
    destination = Path(cache) / "user_split.json"
    if split.exists() and not destination.exists():
        destination.write_text(split.read_text(encoding="utf-8"), encoding="utf-8")


def run(cache):
    import lightgbm  # dependency/runtime check before expensive feature rebuilding
    started = time.perf_counter()
    plan = selection_plan(H.OUT)
    print(__doc__, flush=True)
    print("POST-HOC LATER-FOLD SELECTION:\n" + json.dumps(plan, indent=2), flush=True)
    print(f"LightGBM {lightgbm.__version__}; saved seed/tree counts are fixed, no early stopping", flush=True)
    fingerprint = C.source_fingerprint()
    plan["feature_cache"] = str(cache)
    plan["source_fingerprint"] = fingerprint
    plan_path = H.OUT / "step7_v6_3_selection.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    fold10, fold5, final = obtain_folds(cache, fingerprint)
    P.refit_and_predict(
        [fold10, fold5], final, ["all_together"], {"all_together": plan["config"]},
        {"all_together": plan["seed_iterations"]}, start_rank=3,
        manifest_name="step7_v6_3_final_configs.json",
    )
    print(f"Completed in {time.perf_counter() - started:.1f}s; v6_1 remains the designated final", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=H.OUT / "cache" / "step7")
    args = parser.parse_args(argv)
    H.OUT.mkdir(exist_ok=True)
    log_path = H.OUT / "step7_v6_3_refit.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            run(args.cache_dir)


if __name__ == "__main__":
    main()
