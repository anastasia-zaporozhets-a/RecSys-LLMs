"""Small cache/resume/final-only checks; no real data or long job are used."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import step7_feature_cache as C
import step7_refit_all_together as R
import step7_ltr_plus as P
from test_step7_ltr_plus import tiny_fold

FINGERPRINT = {"synthetic_source": "v1"}
SAVED_TREES = dict(zip(map(str, P.SEEDS), [150, 112, 132, 173, 275]))


def final_fold(users):
    fold = tiny_fold("final", users, 20, 12)
    fold.labels[:] = 0
    fold.targets = {user: set() for user in users}
    return fold


def selection_files(out):
    source = {"recipes": P.recipes(200), "iterations": {"all_together": SAVED_TREES}}
    (out / "step7_selection.json").write_text(json.dumps(source))
    pd.DataFrame([
        {"recipe": "all_together", "map_all_fold10": .7310917245370371, "map_all_fold5": .6401752995066948},
        {"recipe": "previous_v5", "map_all_fold10": .7317445023148148, "map_all_fold5": .6176885130373503},
    ]).to_csv(out / "step7_ablations.csv", index=False)


class FeatureCacheTests(unittest.TestCase):
    def test_roundtrip_preserves_design_targets_ids_and_fallback(self):
        users = ["18417531283109304442", "16561743585370579180"]
        fold = tiny_fold("fold10", users)
        fold.phase.train = fold.phase.train.iloc[:1].copy()
        fold.phase.histories, fold.phase.history_lists = C.H.build_user_histories(fold.phase.train)
        fold.phase.cold = {users[1]}
        fold.phase.popularity.name = "viewers"
        fold.phase.popularity.index.name = "tv_show_id"
        fold.targets[users[0]].add(999999)  # relevant but outside the airing pool
        config = P.recipes(200)["all_together"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold10"
            C.save_fold(path, fold, FINGERPRINT, {0: "News"})
            loaded = C.load_fold(path, FINGERPRINT)
            pd.testing.assert_frame_equal(loaded.features.rows, fold.features.rows)
            pd.testing.assert_frame_equal(loaded.phase.train, fold.phase.train)
            pd.testing.assert_series_equal(loaded.phase.popularity, fold.phase.popularity)
            self.assertEqual(loaded.targets, fold.targets)
            self.assertEqual(loaded.phase.cold, fold.phase.cold)
            a, names_a = P.design(fold, config, np.arange(len(fold.labels)))
            b, names_b = P.design(loaded, config, np.arange(len(loaded.labels)))
            self.assertEqual(names_a, names_b)
            np.testing.assert_array_equal(a.toarray(), b.toarray())
            recs = P.ranking_from_scores(loaded, np.arange(len(loaded.labels)), np.arange(len(loaded.labels)))
            self.assertEqual(recs[users[1]], loaded.phase.popularity.index.tolist())
            with np.load(path / "aligned_arrays.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["users"].dtype.kind, "U")
                self.assertEqual(arrays["users"].tolist(), users)

    def test_stale_incomplete_and_invalid_caches_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold = tiny_fold("fold10", ["u"])
            C.save_fold(root / "fold10", fold, FINGERPRINT)
            with self.assertRaisesRegex(ValueError, "Stale cache"):
                C.load_fold(root / "fold10", {"synthetic_source": "v2"})
            (root / "incomplete").mkdir()
            with self.assertRaisesRegex(ValueError, "Incomplete cache"):
                C.load_fold(root / "incomplete", FINGERPRINT)
            fold.labels = np.array([1])
            with self.assertRaises(AssertionError):
                C.save_fold(root / "invalid", fold, FINGERPRINT)
            self.assertFalse((root / "invalid").exists())

    def test_window_aggregate_roundtrip_and_atomic_publish(self):
        pairs = pd.DataFrame({"user_id": ["18417531283109304442"], "tv_show_id": [1],
                              "seconds": [400], "airings": [1], "days": [1], "recent": [300],
                              "last2": [100], "duration_sum": [600]}).set_index(["user_id", "tv_show_id"])
        context = pd.DataFrame({"user_id": ["18417531283109304442"], "channel_id": [3],
                                "seconds": [500]}).set_index(["user_id", "channel_id"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, frame in (("train10", context), ("label5", pd.DataFrame())):
                C.save_window(root / name, name, P.WindowStats(pairs, frame), FINGERPRINT)
                loaded = C.load_window(root / name, FINGERPRINT)
                pd.testing.assert_frame_equal(loaded.pairs, pairs)
                pd.testing.assert_frame_equal(loaded.context, frame)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with C.staged_cache(root / "interrupted") as staging:
                    (staging / "partial").write_text("partial")
                    raise RuntimeError("interrupted")
            self.assertFalse((root / "interrupted").exists())
            self.assertFalse(any(path.name.startswith(".interrupted-") for path in root.iterdir()))

    def test_complete_cache_load_never_rebuilds_features(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(C.H, "OUT", Path(directory)):
            cache = Path(directory) / "cache"
            for fold in (tiny_fold("fold10", ["u"]), tiny_fold("fold5", ["u"], 15, 5), final_fold(["u"])):
                C.save_fold(cache / fold.name, fold, FINGERPRINT)
            with patch.object(C.H, "load_interactions", side_effect=AssertionError("source data read")), \
                    patch.object(P, "aggregate_windows", side_effect=AssertionError("recompute")), \
                    patch.object(R.S, "read_schedule", side_effect=AssertionError("schedule read")):
                loaded = R.obtain_folds(cache, FINGERPRINT)
            self.assertEqual([fold.name for fold in loaded], ["fold10", "fold5", "final"])

    def test_posthoc_plan_uses_saved_tree_counts_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            selection_files(out)
            plan = R.selection_plan(out)
            self.assertEqual(plan["seed_iterations"], SAVED_TREES)
            self.assertEqual(plan["config"], P.recipes(200)["all_together"])
            self.assertTrue(plan["selection_is_post_hoc"])
            self.assertIn("after seeing the fold5 column", plan["selection_note"])
            self.assertIn("not by fold10", plan["selection_note"])
            self.assertAlmostEqual(plan["fold5_delta"], .0224867864693445)
            self.assertEqual(plan["designated_final_so_far"], "submission_v6_1.csv")

    def test_cached_final_only_run_preserves_v6_1_v6_2_and_original_metadata(self):
        try:
            import lightgbm as lgb
        except (ImportError, OSError):
            self.skipTest("LightGBM runtime not installed")
        users = ["18417531283109304442", "16561743585370579180"] + [f"u{index}" for index in range(8)]
        original_params = P.model_params
        def small_params(seed, trees=1000):
            return {**original_params(seed, min(trees, 3)), "n_jobs": 1, "min_child_samples": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out, data = root / "out", root / "data"
            out.mkdir()
            data.mkdir()
            selection_files(out)
            (out / "step7_final_configs.json").write_text('["original"]')
            (root / "submission_v6_1.csv").write_text("original one\n")
            (root / "submission_v6_2.csv").write_text("original two\n")
            pd.DataFrame({"user_id": users[:2], "tv_show_id": ["0 0 0 0 0"] * 2}).to_csv(data / "submission.csv", index=False)
            cache = out / "cache" / "step7"
            for fold in (tiny_fold("fold10", users), tiny_fold("fold5", users, 15, 5), final_fold(users[:2])):
                C.save_fold(cache / fold.name, fold, FINGERPRINT)
            with patch.object(C.H, "OUT", out), patch.object(C.H, "DATA", data), patch.object(C.H, "EXAM", root), \
                    patch.object(C.H, "EXPECTED_SUB_USERS", 2), \
                    patch.object(C, "source_fingerprint", return_value=FINGERPRINT), \
                    patch.object(P, "model_params", side_effect=small_params), \
                    patch.object(P, "evaluate", side_effect=AssertionError("ablations rerun")), \
                    patch.object(P, "select_candidate_size", side_effect=AssertionError("recall rerun")), \
                    patch.object(P, "aggregate_windows", side_effect=AssertionError("features recomputed")), \
                    patch.object(lgb, "early_stopping", side_effect=AssertionError("early stopping rerun")), \
                    patch("sys.stdout", new=io.StringIO()):
                R.main(["--cache-dir", str(cache)])
            self.assertEqual((root / "submission_v6_1.csv").read_text(), "original one\n")
            self.assertEqual((root / "submission_v6_2.csv").read_text(), "original two\n")
            self.assertEqual((out / "step7_final_configs.json").read_text(), '["original"]')
            result = pd.read_csv(root / "submission_v6_3.csv", dtype=str)
            self.assertEqual(result.user_id.tolist(), users[:2])
            self.assertTrue(all(len(set(value.split())) == 5 for value in result.tv_show_id))
            log = (out / "step7_v6_3_refit.log").read_text()
            self.assertIn("POST-HOC LATER-FOLD SELECTION", log)
            self.assertIn("after seeing the fold5 column", log)
            self.assertIn("CACHE HIT", log)
            manifest = json.loads((out / "step7_v6_3_final_configs.json").read_text())
            self.assertEqual(manifest[0]["file"], "submission_v6_3.csv")
            self.assertEqual(manifest[0]["seed_iterations"], SAVED_TREES)
            self.assertEqual(len(list((out / "step7_models").glob("all_together_seed*.txt"))), 5)


if __name__ == "__main__":
    unittest.main()
