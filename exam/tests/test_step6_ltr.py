"""Small synthetic Step 6 checks; never invoke its dataset/training job."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import step6_ltr as S


class LTRTests(unittest.TestCase):
    def test_overlapping_airings_and_false_interval_candidates(self):
        source, airing, start, end = S.overlap_pairs(
            np.array([25]), np.array([35]), np.array([0, 10, 30]), np.array([100, 20, 40]))
        np.testing.assert_array_equal(source, [0, 0])
        np.testing.assert_array_equal(airing, [0, 2])
        np.testing.assert_array_equal(start, [25, 30])
        np.testing.assert_array_equal(end, [35, 35])

    def test_distinct_days_use_local_clock_and_exclude_midnight_endpoint(self):
        start = S.E.epoch("2020-03-09 23:30:00")
        source, days = S.local_day_segments(np.array([start, start]), np.array([start + 3600, start + 1800]))
        np.testing.assert_array_equal(source, [0, 0, 1])
        self.assertEqual(days[1], days[0] + 1)
        self.assertEqual(days[2], days[0])

    @patch.object(S, "log")
    def test_global_airing_and_day_aggregation_across_csv_chunks(self, _log):
        user = "18417531283109304442"
        start = S.E.epoch("2020-03-09 23:30:00")
        late = S.E.epoch("2020-06-29 08:00:00")
        schedules = {
            "11-20": pd.DataFrame({"channel_id": [3], "tv_show_id": [1], "start": [start],
                                    "end": [start + 3600], "duration": [3600], "block": ["11-20"]}),
            "21-30": pd.DataFrame({"channel_id": [3], "tv_show_id": [2], "start": [late],
                                    "end": [late + 1800], "duration": [1800], "block": ["21-30"]}),
        }
        def view(time, duration):
            return pd.DataFrame({"user_id": [user], "vsetv_id": [3], "start_time": [time], "duraton": [duration]})
        chunks = [view("2020-03-09 23:30:00", 1800), view("2020-03-10 00:00:00", 1800),
                  view("2020-03-09 23:30:00", 3600), view("2020-06-29 08:00:00", 1800),
                  view("2020-07-27 00:00:00", 999999)]
        with patch.object(S.pd, "read_csv", return_value=iter(chunks)):
            stats = S.aggregate_view_stats(schedules)
        row = stats["11-20"].loc[(user, 1)]
        self.assertEqual(row.seconds, 7200)
        self.assertEqual(row.airings, 1)
        self.assertEqual(row.days, 2)
        self.assertEqual(row.mean_fraction, 1.0)
        self.assertEqual(row.recent11, 0)
        self.assertEqual(stats["21-30"].loc[(user, 2)].recent30, 1800)
        self.assertEqual(S.training_stats(stats).loc[(user, 1)].seconds, 7200)
        self.assertNotIn((user, 2), S.training_stats(stats).index)
        self.assertEqual(S.training_stats(stats, final=True).loc[(user, 2)].recent, 1800)

    def test_target_orders_and_binary_all_watched_labels(self):
        interactions = pd.DataFrame({"user_id": ["u"] * 6, "tv_show_id": range(1, 7),
                                     "block": ["21-30"] * 6, "overlap_seconds": [600, 500, 400, 300, 200, 100]})
        raw = pd.DataFrame({"user_id": ["u"] * 6, "tv_show_id": range(1, 7),
                            "airings": range(1, 7), "days": range(1, 7),
                            "mean_fraction": [.9, .8, .7, .6, .5, .4]}).set_index(["user_id", "tv_show_id"])
        targets = S.calibration_targets(interactions, raw)
        self.assertEqual(targets["top5_seconds"]["u"], [1, 2, 3, 4, 5])
        self.assertEqual(targets["top5_airings"]["u"], [6, 5, 4, 3, 2])
        self.assertEqual(targets["top5_days"]["u"], [6, 5, 4, 3, 2])
        self.assertEqual(targets["top5_mean_fraction"]["u"], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(S.relevance([1, 5, 6, 99], targets["top5_seconds"]["u"]), [5, 1, 0, 0])
        np.testing.assert_array_equal(S.relevance([1, 5, 6, 99], targets["all_watched"]["u"], True), [1, 1, 1, 0])

    @patch.object(S, "log")
    def test_candidate_union_and_training_only_feature_values(self, _log):
        items = np.arange(1, 121)
        train = pd.DataFrame({"user_id": ["a"], "tv_show_id": [1], "overlap_seconds": [100]})
        phase = S.E.Phase(["a", "b"], items, set(items), {"a": {1}}, {"a": [(1, 100)]}, train,
                           pd.Series(np.arange(50, 0, -1), index=np.arange(71, 121)), {"b"}, {}, {})
        scores = np.ones((2, 120), dtype=np.float32)
        ranks = np.tile(np.arange(1, 121), (2, 1))
        ppr = S.Signal({"a": [(item, 1.) for item in range(1, 51)], "b": []}, scores * 7, ranks)
        exposure = S.Signal({"a": [(item, 1.) for item in range(51, 101)], "b": []}, scores * 9, ranks)
        stats = pd.DataFrame({"user_id": ["a"], "tv_show_id": [1], "seconds": [100], "airings": [1],
                              "days": [1], "recent": [20]}).set_index(["user_id", "tv_show_id"])
        schedule = pd.DataFrame({"tv_show_id": items, "start": np.zeros(120), "end": np.full(120, 3600)})
        catalogue = pd.DataFrame({"tv_show_id": items, "category": ["News"] * 120})
        features = S.build_features(phase, ppr, exposure, stats, schedule, schedule, catalogue,
                                     ["News", "__missing__"])
        self.assertEqual(len(features.rows[features.rows.user_id == "a"]), 120)
        self.assertEqual(len(features.rows[features.rows.user_id == "b"]), 50)
        row1, row2 = features.matrix[0].toarray()[0], features.matrix[1].toarray()[0]
        self.assertEqual(row1[S.NUMERIC_FEATURES.index("last4weeks_seconds")], 20)
        self.assertEqual(row2[S.NUMERIC_FEATURES.index("user_item_seconds")], 0)
        self.assertEqual(row2[S.NUMERIC_FEATURES.index("is_new_for_user")], 1)
        self.assertEqual(row2[S.NUMERIC_FEATURES.index("ppr_score")], 7)
        self.assertEqual(row2[S.NUMERIC_FEATURES.index("exposure_score")], 9)
        self.assertEqual(row1[S.NUMERIC_FEATURES.index("user_category_share")], 1)

    def test_user_split_and_query_contiguity(self):
        first = S.user_split([str(index) for index in range(10)])
        self.assertEqual(first, S.user_split([str(index) for index in range(10)]))
        self.assertEqual(tuple(map(len, first)), (8, 2))
        self.assertFalse(first[0] & first[1])
        self.assertEqual(S.query_groups(pd.DataFrame({"user_id": ["a", "a", "b", "b", "b"]})), [2, 3])
        with self.assertRaises(AssertionError):
            S.query_groups(pd.DataFrame({"user_id": ["a", "b", "a"]}))

    @patch.object(S, "log")
    def test_calibration_selects_highest_public_score_spearman(self, _log):
        predictions = {method: method for method in S.PUBLIC_SCORES}
        def metric(method, target, k):
            value = S.PUBLIC_SCORES[method]
            return (value if target == "follow" else -value), S.H.EXPECTED_EVAL_USERS
        with patch.object(S.H, "map_at_k", side_effect=metric), patch.object(S.pd.DataFrame, "to_csv"):
            target = S.calibrate(predictions, {"follows_public": "follow", "opposes_public": "oppose"})
        self.assertEqual(target, "follows_public")

    @patch.object(S, "log")
    def test_tiny_heldout_fit_refit_and_model_artifacts(self, _log):
        try:
            import lightgbm  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("LightGBM runtime not installed")
        users = [f"u{index:02}" for index in range(10)]
        rows = pd.DataFrame([(user, item) for user in users for item in range(1, 5)],
                             columns=["user_id", "tv_show_id"])
        matrix = sparse.csr_matrix(np.tile(np.array([[3., 0.], [2., 1.], [1., 0.], [0., 1.]],
                                                     dtype=np.float32), (10, 1)))
        features = S.Features(matrix, rows, ["signal", "other"])
        labels = np.tile([5, 4, 0, 0], 10)
        phase = SimpleNamespace(users=users, pool=set(range(1, 6)),
                                popularity=pd.Series([10, 9, 8, 7, 6], index=range(1, 6)),
                                histories={user: {1} for user in users}, cold=set())
        targets = {user: [1, 2] for user in users}
        all_targets = {"top5_seconds": targets, "all_watched": {user: {1, 2, 3} for user in users}}
        baseline = {user: [1, 2, 3, 4, 5] for user in users}
        with tempfile.TemporaryDirectory() as directory, patch.object(S.H, "OUT", Path(directory)):
            model, iterations = S.fit_ranker(features, labels, phase, targets, all_targets,
                                              "top5_seconds", baseline)
            self.assertGreaterEqual(iterations, 1)
            self.assertTrue((Path(directory) / "step6_ltr_model.txt").exists())
            self.assertTrue((Path(directory) / "step6_user_split.json").exists())
            heldout = pd.read_csv(Path(directory) / "step6_heldout.csv")
            self.assertEqual(heldout.method.tolist(), ["LTR_heldout", "v4_1_same_users"])
            self.assertTrue(np.isfinite(model.predict(features.matrix)).all())

    def test_lightgbm_sparse_lambdarank_dependency_smoke(self):
        try:
            import lightgbm as lgb
        except (ImportError, OSError):
            self.skipTest("LightGBM runtime not installed")
        x = sparse.csr_matrix(np.tile(np.array([[3., 0.], [2., 1.], [1., 0.], [0., 1.]], dtype=np.float32), (4, 1)))
        y = np.tile([3, 2, 1, 0], 4)
        model = lgb.LGBMRanker(objective="lambdarank", n_estimators=3, num_leaves=3,
                               min_child_samples=1, random_state=S.SEED, n_jobs=1, verbosity=-1)
        model.fit(x[:12], y[:12], group=[4, 4, 4], eval_set=[(x[12:], y[12:])],
                  eval_group=[[4]], eval_at=[5])
        self.assertTrue(np.isfinite(model.predict(x)).all())


if __name__ == "__main__":
    unittest.main()
