"""Portable, non-pickle Step 7 feature caches for reuse by later experiments.

Fold caches contain CSR matrices, aligned numeric arrays, string user IDs,
full training pairs, relevant sets (including out-of-pool items) and fallback
state. load_fold() returns a Step 7 Fold usable by design(), training_queries()
and ranking_from_scores(). Habit/airing matrices are intentionally not stored:
their already-computed candidate features are in the base/extra CSR matrices.
Source fingerprints use input size/mtime plus feature-code SHA-256 digests.
Writes are staged then atomically renamed; incomplete/stale caches are refused.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy import sparse

import harness as H
import step5_exposure as E
import step6_ltr as S
import step7_ltr_plus as P

VERSION = 1


def source_fingerprint():
    inputs = [H.DATA / "dataset11-30.csv", H.DATA / "submission.csv",
              H.OUT / "interactions.csv", H.OUT / "programmes.csv"]
    inputs.extend(H.DATA / f"export_arh_{block}-final.csv" for block in ("11-20", "21-30", "31-42"))
    records = {}
    for path in inputs:
        stat = path.stat()
        records[str(path.relative_to(H.EXAM))] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    code = {}
    for path in [Path(module.__file__) for module in (H, E, S, P)] + [Path(__file__)]:
        code[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"version": VERSION, "inputs": records, "feature_code_sha256": code}


def inspect_cache(path, fingerprint, kind):
    path = Path(path)
    if not path.exists():
        return None
    manifest = path / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Incomplete cache at {path}; use a new --cache-dir or remove that incomplete entry")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("version") != VERSION or metadata.get("kind") != kind:
        raise ValueError(f"Unsupported cache schema at {path}")
    if metadata.get("source_fingerprint") != fingerprint:
        raise ValueError(f"Stale cache at {path}; source fingerprint changed. Use a new --cache-dir")
    return metadata


@contextmanager
def staged_cache(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing cache: {path}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        yield temporary
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def write_manifest(folder, metadata, fingerprint, kind):
    metadata = {**metadata, "version": VERSION, "kind": kind, "source_fingerprint": fingerprint}
    (folder / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_fold(fold):
    rows = fold.features.rows
    n = len(rows)
    assert rows.index.equals(pd.RangeIndex(n))
    assert fold.features.matrix.shape == (n, len(fold.features.names))
    assert fold.extra.shape == (n, len(P.NEW_NAMES))
    assert fold.depth.shape == fold.labels.shape == fold.train_airtime.shape == (n,)
    assert np.isfinite(fold.features.matrix.data).all() and np.isfinite(fold.extra.data).all()
    assert np.isfinite(fold.train_airtime).all()
    assert np.isin(fold.labels, [0, 1]).all() and ((fold.depth >= 1) & (fold.depth <= 200)).all()
    assert all(isinstance(user, str) for user in fold.phase.users)
    assert rows.user_id.drop_duplicates().tolist() == fold.phase.users
    assert not rows.duplicated(["user_id", "tv_show_id"]).any()
    assert set(rows.tv_show_id) <= fold.phase.pool == set(fold.phase.items)
    assert set(fold.targets) == set(fold.phase.users)
    if fold.name == "final":
        assert not fold.labels.any() and not any(fold.targets.values())
    assert sum(S.query_groups(rows)) == n


def save_fold(path, fold, fingerprint, category_mapping=None):
    validate_fold(fold)
    users = fold.phase.users
    codes = pd.Index(users).get_indexer(fold.features.rows.user_id)
    assert (codes >= 0).all()
    target_lists = [sorted(fold.targets[user]) for user in users]
    target_ptr = np.r_[0, np.cumsum([len(items) for items in target_lists])].astype(np.int64)
    target_ids = np.array([item for items in target_lists for item in items], dtype=np.int64)
    with staged_cache(path) as folder:
        sparse.save_npz(folder / "base_features.npz", fold.features.matrix, compressed=True)
        sparse.save_npz(folder / "new_features.npz", fold.extra, compressed=True)
        np.savez_compressed(
            folder / "aligned_arrays.npz", users=np.asarray(users, dtype=str), row_user_codes=codes.astype(np.int32),
            row_item_ids=fold.features.rows.tv_show_id.to_numpy(dtype=np.int64),
            depth=fold.depth, labels=fold.labels, train_airtime=fold.train_airtime,
            pool=fold.phase.items.astype(np.int64), cold=np.asarray(sorted(fold.phase.cold), dtype=str),
            popularity_ids=fold.phase.popularity.index.to_numpy(dtype=np.int64),
            popularity_values=fold.phase.popularity.to_numpy(),
            target_ptr=target_ptr, target_item_ids=target_ids,
        )
        fold.phase.train.to_csv(folder / "training_pairs.csv.gz", index=False, compression="gzip")
        write_manifest(folder, {
            "fold": fold.name, "rows": len(fold.features.rows), "users": len(users),
            "training_weeks": fold.training_weeks, "target_weeks": fold.target_weeks,
            "base_feature_names": fold.features.names, "new_feature_names": P.NEW_NAMES,
            "popularity_name": fold.phase.popularity.name,
            "popularity_index_name": fold.phase.popularity.index.name,
            "category_onehot_mapping": category_mapping or {},
            "candidate_cap_per_source": 200,
            "target": "all-watched; binary labels; final labels are unused zero placeholders",
            "feature_role": "base raw-unit features plus new features; apply step7_ltr_plus.design for weekly rates",
            "habit_and_airing_matrices_cached": False,
        }, fingerprint, "fold")


def load_fold(path, fingerprint):
    path = Path(path)
    metadata = inspect_cache(path, fingerprint, "fold")
    if metadata is None:
        raise FileNotFoundError(path)
    assert metadata["new_feature_names"] == P.NEW_NAMES
    with np.load(path / "aligned_arrays.npz", allow_pickle=False) as arrays:
        users = arrays["users"].tolist()
        codes = arrays["row_user_codes"]
        assert ((codes >= 0) & (codes < len(users))).all()
        rows = pd.DataFrame({"user_id": np.asarray(users, dtype=object)[codes], "tv_show_id": arrays["row_item_ids"]})
        train = pd.read_csv(path / "training_pairs.csv.gz", dtype={"user_id": str, "tv_show_id": "int64"})
        histories, history_lists = H.build_user_histories(train)
        popularity = pd.Series(arrays["popularity_values"], index=arrays["popularity_ids"],
                               name=metadata["popularity_name"])
        popularity.index.name = metadata["popularity_index_name"]
        phase = E.Phase(users, arrays["pool"], set(arrays["pool"].tolist()), histories, history_lists,
                        train, popularity, set(arrays["cold"].tolist()), {}, {})
        ptr, ids = arrays["target_ptr"], arrays["target_item_ids"]
        assert len(ptr) == len(users) + 1 and ptr[0] == 0 and ptr[-1] == len(ids)
        assert (np.diff(ptr) >= 0).all()
        targets = {user: set(ids[ptr[index]:ptr[index + 1]].tolist()) for index, user in enumerate(users)}
        features = S.Features(sparse.load_npz(path / "base_features.npz").tocsr(), rows, metadata["base_feature_names"])
        fold = P.Fold(metadata["fold"], phase, features, sparse.load_npz(path / "new_features.npz").tocsr(),
                      arrays["depth"], arrays["labels"], targets, metadata["training_weeks"],
                      metadata["target_weeks"], arrays["train_airtime"])
    assert len(rows) == metadata["rows"] and len(users) == metadata["users"]
    validate_fold(fold)
    return fold


def save_window(path, name, stats, fingerprint):
    with staged_cache(path) as folder:
        stats.pairs.to_csv(folder / "pairs.csv.gz", compression="gzip")
        has_context = not stats.context.empty
        if has_context:
            stats.context.to_csv(folder / "context.csv.gz", compression="gzip")
        write_manifest(folder, {"window": name, "dates": P.WINDOWS[name], "has_context": has_context},
                       fingerprint, "window_stats")


def load_window(path, fingerprint):
    path = Path(path)
    metadata = inspect_cache(path, fingerprint, "window_stats")
    if metadata is None:
        raise FileNotFoundError(path)
    pairs = pd.read_csv(path / "pairs.csv.gz", dtype={"user_id": str, "tv_show_id": "int64"}).set_index(
        ["user_id", "tv_show_id"])
    context = (pd.read_csv(path / "context.csv.gz", dtype={"user_id": str, "channel_id": "int64"}).set_index(
        ["user_id", "channel_id"]) if metadata["has_context"] else pd.DataFrame())
    return P.WindowStats(pairs, context)
