"""Synthetic Step 8 checks only; no real feature extraction/training job."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import step8_exposure_features as F
import step8_ltr_graded as G
from test_step7_ltr_plus import tiny_fold


def data(name, users, training_weeks=10, target_weeks=10):
    fold = tiny_fold(name, users, training_weeks, target_weeks)
    if name == "final":
        fold.labels[:] = 0
        fold.targets = {user: set() for user in users}
    matrix, names = G.P.design(fold, G.P.recipes(200)["all_together"], np.arange(len(fold.labels)))
    labels = {"binary": fold.labels.astype(np.int8), "airings": fold.labels.astype(np.int8) * 4,
              "days": fold.labels.astype(np.int8) * 3}
    return F.Data(name, fold.phase, fold.features.rows, matrix, names, labels, fold.targets,
                  training_weeks, target_weeks, sparse.csr_matrix(np.ones((len(fold.labels), 5), dtype=np.float32)))


class GradedLTRTests(unittest.TestCase):
    def test_quartiles_keep_ties_and_use_full_target(self):
        values = pd.Series([1, 1, 2, 3, 4, 8], index=range(1, 7))
        labels, cuts = F.quartile_grades(set(range(1, 7)), values, [1, 2, 3, 4, 5, 6, 99])
        np.testing.assert_array_equal(labels, [1, 1, 2, 3, 4, 4, 0])
        np.testing.assert_allclose(cuts, [1.25, 2.5, 3.75])
        labels, _ = F.quartile_grades({1, 2, 3}, pd.Series([1, 1, 1], index=[1, 2, 3]), [1, 2, 3, 4])
        np.testing.assert_array_equal(labels, [1, 1, 1, 0])
        _, full_cuts = F.quartile_grades({1, 2, 999}, pd.Series({1: 1, 2: 2, 999: 100}), [1, 2])
        self.assertGreater(full_cuts[-1], 2)  # out-of-candidate target affects quantiles

    def test_counts_difference_and_decay_integral(self):
        index = pd.MultiIndex.from_tuples([("u", 1), ("u", 2)], names=["user_id", "tv_show_id"])
        total = pd.DataFrame({"airings": [8, 4], "days": [6, 3]}, index=index)
        before = pd.DataFrame({"airings": [3], "days": [2]}, index=index[:1])
        difference = F.difference_counts(total, before)
        np.testing.assert_array_equal(difference.to_numpy(), [[5, 4], [4, 3]])
        anchor = 100 * F.WEEK_SECONDS
        width = 3600
        recent = F.decayed_seconds(np.array([anchor - width]), np.array([anchor]), anchor, 4)[0]
        old = F.decayed_seconds(np.array([anchor - 4 * F.WEEK_SECONDS - width]),
                                np.array([anchor - 4 * F.WEEK_SECONDS]), anchor, 4)[0]
        self.assertAlmostEqual(old / recent, .5)
        self.assertGreater(recent, 0)
        self.assertLess(recent, width)
        for weeks in (10, 15, 20):
            weighted = F.decayed_seconds(np.array([anchor - weeks * F.WEEK_SECONDS]), np.array([anchor]), anchor, 4)[0]
            effective_weeks = weighted / F.WEEK_SECONDS
            self.assertAlmostEqual(weighted / effective_weeks, F.WEEK_SECONDS)

    @patch.object(F, "log")
    def test_raw_habits_exclude_each_target_window_and_future_views(self, _log):
        datasets = [data("fold10", ["u"]), data("fold5", ["u"], 15, 5), data("final", ["u"], 20, 12)]
        raw = pd.DataFrame({"user_id": ["u"] * 4, "vsetv_id": [3] * 4,
                            "start_time": ["2020-05-17 09:00:00", "2020-05-25 09:00:00",
                                           "2020-07-26 09:00:00", "2020-07-27 00:00:00"],
                            "duraton": [3600, 3600, 3600, 999999]})
        with patch.object(F.pd, "read_csv", return_value=iter([raw])):
            habits = F.build_habits(datasets, {name: np.array([3]) for name in F.DATES})
        for dataset, total in zip(datasets, [3600, 7200, 10800]):
            self.assertAlmostEqual(habits[dataset.name]["uniform"].sum() * dataset.training_weeks, total)
            self.assertGreater(habits[dataset.name]["decay4"].sum(), 0)
            self.assertTrue(np.isfinite(habits[dataset.name]["decay8"].data).all())

    def test_airing_halves_sum_to_full_and_candidate_rank_is_local(self):
        dataset = data("fold10", ["u"])
        starts = [F.E.epoch("2020-06-21 23:30:00"), F.E.epoch("2020-05-18 09:00:00")]
        schedule = pd.DataFrame({"channel_id": [3, 3], "tv_show_id": [1, 2],
                                 "start": starts, "end": np.array(starts) + 3600})
        channels, full, first, second = F.airing_matrices(dataset, schedule)
        self.assertEqual(channels.tolist(), [3])
        np.testing.assert_array_equal((first + second).toarray(), full.toarray())
        self.assertEqual(first.sum(), 5400)
        self.assertEqual(second.sum(), 1800)
        matrix = dataset.base.toarray()
        matrix[:, dataset.names.index("exposure_rank")] = [20, 1, 30, 8, 100]
        matrix[:, dataset.names.index("exposure_score_per_target_week")] = [1, 1, 1, 1, 0]
        dataset.base = sparse.csr_matrix(matrix)
        np.testing.assert_array_equal(F.candidate_rank(dataset), [3, 1, 4, 2, 6])

    def test_mean_map_metric_matches_harness_with_cold_fallback_and_ties(self):
        folds = [data("fold10", ["a", "b"]), data("fold5", ["a"], 15, 5)]
        folds[0].phase.cold = {"b"}
        folds[1].targets["a"] = {1, 999, 1000, 1001, 1002, 1003}
        folds[1].labels["binary"][:5] = [1, 0, 0, 0, 0]
        indices = [np.arange(len(fold.rows)) for fold in folds]
        metric = G.MeanFoldMAP(folds, indices)
        scores = np.array([0] * 5 + [0, 1, 2, 3, 4] + [0, 1, 2, 3, 4], dtype=float)
        expected = []
        offset = 0
        for fold, idx in zip(folds, indices):
            recs = G.predictions(fold, idx, scores[offset:offset + len(idx)])
            expected.append(G.H.map_at_k(recs, fold.targets, 5)[0])
            offset += len(idx)
        np.testing.assert_allclose(metric.values(scores), expected)
        name, mean, higher = metric(np.full(len(scores), 4), scores, None, None)
        self.assertEqual(name, "mean_fold_map5")
        self.assertTrue(higher)
        self.assertAlmostEqual(mean, sum(expected) / 2)
        self.assertNotAlmostEqual(mean, (2 * expected[0] + expected[1]) / 3)

    def test_stacked_fit_excludes_heldout_users_and_uses_graded_labels(self):
        folds = [data("fold10", ["a", "b"]), data("fold5", ["a", "b"], 15, 5)]
        config = G.spec("airings", ["decay4", "halves"])
        tx, ty, tg, vx, vy, vg, held_idx, names = G.fitting_data(folds, config, {"a"}, {"b"})
        self.assertEqual(tg, [5, 5])
        self.assertEqual(vg, [5, 5])
        self.assertEqual(tx.shape[0], 10)
        self.assertEqual(vx.shape[0], 10)
        self.assertEqual(ty.tolist(), [4, 4, 0, 0, 0] * 2)
        self.assertEqual(len(names), len(folds[0].names) + 3)
        for fold, idx in zip(folds, held_idx):
            self.assertEqual(set(fold.rows.iloc[idx].user_id), {"b"})

    @patch.object(F, "log")
    def test_cached_additions_reuse_without_raw_features(self, _log):
        datasets = [data("fold10", ["u"]), data("fold5", ["u"], 15, 5), data("final", ["u"], 20, 12)]
        fingerprint = {"test": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in datasets:
                with G.C.staged_cache(root / dataset.name) as folder:
                    sparse.save_npz(folder / "additional.npz", dataset.additional)
                    np.savez_compressed(folder / "labels.npz", **dataset.labels)
                    G.C.write_manifest(folder, {"rows": len(dataset.rows), "feature_names": F.ADDITIONAL_NAMES},
                                       fingerprint, "step8_features")
            with patch.object(F, "build_exposures", side_effect=AssertionError("raw features recomputed")):
                F.additions(datasets, root, root, {}, fingerprint)

    @patch.object(F, "log")
    def test_missing_additions_build_only_new_features_and_persist_labels(self, _log):
        datasets = [data("fold10", ["u"]), data("fold5", ["u"], 15, 5), data("final", ["u"], 20, 12)]
        def schedule(dates):
            starts = [F.E.epoch(value) for value in dates]
            return pd.DataFrame({"channel_id": [3] * len(starts), "tv_show_id": range(1, len(starts) + 1),
                                 "start": starts, "end": np.array(starts) + 3600})
        schedules = {"21-30": schedule(["2020-05-18 09:00:00", "2020-06-22 09:00:00"]),
                     "31-42": schedule(["2020-07-27 09:00:00", "2020-09-07 09:00:00"])}
        raw = pd.DataFrame({"user_id": ["u"] * 3, "vsetv_id": [3] * 3,
                            "start_time": ["2020-04-13 09:00:00", "2020-06-15 09:00:00", "2020-07-27 00:00:00"],
                            "duraton": [3600, 3600, 999999]})
        index = pd.MultiIndex.from_tuples([("u", 1), ("u", 2)], names=["user_id", "tv_show_id"])
        counts = {"train10": pd.DataFrame({"airings": [1, 1], "days": [1, 1]}, index=index),
                  "train20": pd.DataFrame({"airings": [4, 4], "days": [3, 3]}, index=index),
                  "label5": pd.DataFrame({"airings": [3, 3], "days": [2, 2]}, index=index)}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(F, "read_counts", side_effect=lambda root, name, fp: counts[name]), \
                patch.object(F.S, "read_schedule", side_effect=lambda block: schedules[block]), \
                patch.object(F.pd, "read_csv", return_value=iter([raw])):
            root = Path(directory)
            F.additions(datasets, root, root, {}, {"test": 2})
            self.assertGreater(datasets[0].additional.toarray()[0, 2], 0)
            self.assertGreater(datasets[0].additional.toarray()[1, 3], 0)
            for dataset in datasets:
                self.assertTrue((root / dataset.name / "additional.npz").exists())
                self.assertTrue((root / dataset.name / "labels.npz").exists())
                self.assertTrue(np.isfinite(dataset.additional.data).all())
                self.assertTrue(np.array_equal(dataset.labels["airings"] > 0, dataset.labels["binary"] > 0))

    @patch.object(G, "log")
    def test_search_uses_mean_rather_than_fold10_for_both_choices(self, _log):
        class ToyExperiment:
            def __init__(self):
                self.rows, self.seen = [], {}

            def trial(self, name, stage, config, note=""):
                # Binary has the higher fold10 score; airings wins by mean.
                fold10, fold5 = (.9, .1) if config["label"] == "binary" else (.8, .8)
                if config["label"] == "days":
                    fold5 = .2
                gain = .005 * len(config["extras"]) + (.002 if config["params"]["num_leaves"] == 63 else 0)
                fold10 += gain
                fold5 += gain
                key = G.identity(config)
                result = {"id": key, "config": config, "iterations": {}, "map_fold10": fold10,
                          "map_fold5": fold5, "mean_map": (fold10 + fold5) / 2}
                self.seen[key] = result
                self.rows.append({"name": name, "stage": stage, "mean_map": result["mean_map"]})
                return result

            def table(self):
                return pd.DataFrame(self.rows)
        with tempfile.TemporaryDirectory() as directory, patch.object(G.H, "OUT", Path(directory)):
            experiment = ToyExperiment()
            chosen = G.search(experiment)
            self.assertEqual(len(chosen), 2)
            self.assertTrue(all(row["config"]["label"] == "airings" for row in chosen))
            self.assertEqual(len([row for row in experiment.rows if row["stage"] == "parameters_only"]), 8)
            self.assertEqual(len([row for row in experiment.rows if row["stage"] == "combination"]), 8)

    @patch.object(G, "log")
    def test_tiny_trial_resume_and_five_seed_final_fit(self, _log):
        try:
            import lightgbm  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("LightGBM runtime not installed")
        users = [f"u{index:02}" for index in range(20)]
        datasets = [data("fold10", users), data("fold5", users, 15, 5), data("final", users[:2], 20, 12)]
        first = G.spec(params={"num_leaves": 3, "learning_rate": .1, "min_child_samples": 1})
        second = G.spec("airings", F.GROUPS, {"num_leaves": 3, "learning_rate": .1, "min_child_samples": 1})
        original = G.P.model_params
        def small_params(seed, trees=1000):
            return {**original(seed, min(trees, 3)), "n_jobs": 1}
        with tempfile.TemporaryDirectory() as directory, patch.object(G.H, "OUT", Path(directory)), \
                patch.object(G.P, "model_params", side_effect=small_params), patch("sys.stdout", new=io.StringIO()):
            root = Path(directory)
            experiment = G.Experiment(datasets, set(users[:16]), set(users[16:]), root / "cache", {"test": 1})
            results = [experiment.trial("binary", "test", first), experiment.trial("graded", "test", second)]
            self.assertTrue(all(len(result["iterations"]) == 5 for result in results))
            for result in results:
                self.assertAlmostEqual(result["mean_map"], (result["map_fold10"] + result["map_fold5"]) / 2)
            resumed = G.Experiment(datasets, set(users[:16]), set(users[16:]), root / "cache", {"test": 1})
            with patch.object(G.P, "model_params", side_effect=AssertionError("cached fit recomputed")):
                cached = resumed.trial("binary_cached", "test", first)
            self.assertEqual(cached["iterations"], results[0]["iterations"])
            self.assertEqual(cached["mean_map"], results[0]["mean_map"])
            with patch.object(G.FB, "check_and_write", return_value={"rows": 2}) as writer:
                G.final_fit(datasets, results, {"test": 1})
                self.assertEqual(writer.call_count, 2)
                self.assertEqual([call.args[1].name for call in writer.call_args_list], ["submission_v7_1.csv", "submission_v7_2.csv"])
                self.assertTrue(all(len(set(row)) == 5 for call in writer.call_args_list for row in call.args[0].values()))
            self.assertTrue((root / "step8_final_configs.json").exists())
            self.assertEqual(len(list((root / "step8_models").glob("*.txt"))), 10)


if __name__ == "__main__":
    unittest.main()
