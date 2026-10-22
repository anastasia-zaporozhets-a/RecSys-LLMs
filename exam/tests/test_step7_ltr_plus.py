"""Small synthetic checks only: never read project viewing data or run main()."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import step7_ltr_plus as P


def tiny_fold(name, users, training_weeks=10, target_weeks=10):
    items = np.arange(1, 6)
    rows = pd.DataFrame([(user, item) for user in users for item in items], columns=["user_id", "tv_show_id"])
    values = np.ones((len(rows), len(P.S.NUMERIC_FEATURES) + 1), dtype=np.float32)
    values[:, 0] = np.tile([50, 40, 30, 20, 10], len(users))
    train = pd.DataFrame({"user_id": users, "tv_show_id": [1] * len(users), "overlap_seconds": [50] * len(users)})
    phase = P.E.Phase(users, items, set(items), {user: {1} for user in users},
                      {user: [(1, 50)] for user in users}, train,
                      pd.Series([5, 4, 3, 2, 1], index=items), set(), {}, {})
    features = P.S.Features(sparse.csr_matrix(values), rows, P.S.NUMERIC_FEATURES + ["category_0"])
    return P.Fold(name, phase, features, sparse.csr_matrix(np.ones((len(rows), 5), dtype=np.float32)),
                  np.ones(len(rows), dtype=np.int32), np.tile([1, 1, 0, 0, 0], len(users)),
                  {user: {1, 2} for user in users}, training_weeks, target_weeks,
                  np.full(len(rows), 10000.))


class LTRPlusTests(unittest.TestCase):
    @patch.object(P, "log")
    def test_raw_windows_clip_future_and_keep_unlabelled_channel_context(self, _log):
        user = "18417531283109304442"
        strings = ["2020-05-17 09:00:00", "2020-05-25 09:00:00", "2020-06-23 09:00:00"]
        times = [P.E.epoch(value) for value in strings]
        schedules = {
            "11-20": pd.DataFrame({"channel_id": [3], "tv_show_id": [1], "duration": [3600],
                                    "start": times[:1], "end": [times[0] + 3600], "block": ["11-20"]}),
            "21-30": pd.DataFrame({"channel_id": [3, 3], "tv_show_id": [2, 3], "duration": [3600, 3600],
                                    "start": times[1:], "end": [value + 3600 for value in times[1:]],
                                    "block": ["21-30", "21-30"]}),
        }
        views = pd.DataFrame({"user_id": [user] * 5, "vsetv_id": [3, 3, 3, 99, 3],
                              "start_time": strings + [strings[1], "2020-07-27 00:00:00"],
                              "duraton": [3600, 3600, 3600, 1800, 999999]})
        with patch.object(P.pd, "read_csv", return_value=iter([views.iloc[:2], views.iloc[2:]])):
            windows = P.aggregate_windows(schedules)
        self.assertEqual(windows["train10"].pairs.seconds.sum(), 3600)
        self.assertEqual(windows["train15"].pairs.seconds.sum(), 7200)
        self.assertEqual(windows["train20"].pairs.seconds.sum(), 10800)
        self.assertEqual(windows["label5"].pairs.index.tolist(), [(user, 3)])
        self.assertEqual(windows["train15"].context.seconds.sum(), 9000)
        self.assertEqual(P.qualifying_interactions(windows["train15"]).tv_show_id.tolist(), [1, 2])

    def test_weekly_rates_transfer_across_window_lengths(self):
        first, final = tiny_fold("fold10", ["u"]), tiny_fold("final", ["u"], 20, 12)
        first.features.matrix[:, first.features.names.index("last4weeks_seconds")] = 400
        final.features.matrix = first.features.matrix.copy()
        for name in ("user_item_seconds", "user_item_airings", "user_item_days", "popularity_users", "global_item_seconds"):
            col = first.features.names.index(name)
            final.features.matrix[:, col] = first.features.matrix[:, col] * 2
        for name in ("target_airtime", "target_airings", "exposure_score"):
            col = first.features.names.index(name)
            final.features.matrix[:, col] = first.features.matrix[:, col] * 1.2
        final.train_airtime = first.train_airtime * 2
        first.extra = sparse.csr_matrix(np.tile([.8, 100, 1, 10, 200], (5, 1)), dtype=np.float32)
        final.extra = sparse.csr_matrix(np.tile([.8, 120, 1, 12, 200], (5, 1)), dtype=np.float32)
        config = {"new": True, "scale": True}
        a, names_a = P.design(first, config, np.arange(5))
        b, names_b = P.design(final, config, np.arange(5))
        self.assertEqual(names_a, names_b)
        np.testing.assert_allclose(a.toarray(), b.toarray(), rtol=1e-6)
        self.assertEqual(a.toarray()[0, names_a.index("last4weeks_seconds_per_week")], 100)
        self.assertEqual(a.toarray()[0, names_a.index("target_week_coverage")], 1)

    @patch.object(P, "log")
    def test_recall_budget_uses_only_training_users_and_both_folds(self, _log):
        folds = []
        for name, count50 in (("fold10", 90), ("fold5", 94)):
            fold = tiny_fold(name, ["a", "b"])
            fold.features.rows = pd.DataFrame([(user, item) for user in ["a", "b"] for item in range(1, 101)],
                                               columns=["user_id", "tv_show_id"])
            fold.phase.pool = set(range(1, 101))
            fold.targets = {user: set(range(1, 101)) for user in ["a", "b"]}
            fold.depth = np.array([50] * count50 + [100] * (95 - count50) + [200] * 5 + [50] * 100)
            folds.append(fold)
        with tempfile.TemporaryDirectory() as directory, patch.object(P.H, "OUT", Path(directory)):
            chosen = P.select_candidate_size(folds, {"a"})
            self.assertEqual(chosen, 100)
            details = pd.read_csv(Path(directory) / "step7_candidate_recall_per_user.csv")
            self.assertEqual(len(details), 12)

    def test_first_two_weeks_and_cross_week_airing(self):
        starts = [P.E.epoch("2020-08-02 23:30:00"), P.E.epoch("2020-08-10 09:00:00")]
        target = pd.DataFrame({"tv_show_id": [1, 2], "start": starts, "end": np.array(starts) + 3600})
        weeks, early = P.week_coverage(target, P.E.FINAL_START)
        self.assertEqual(weeks.loc[1], 2)
        self.assertEqual(early, {1})

    @patch.object(P, "log")
    def test_channel_share_counts_unlabelled_context_and_recent_slot_feature(self, _log):
        start = P.E.epoch("2020-06-22 09:00:00")
        target = pd.DataFrame({"tv_show_id": [1], "channel_id": [3], "start": [start], "end": [start + 3600]})
        phase = tiny_fold("fold5", ["u"]).phase
        phase.items = np.array([1])
        phase.habits = {"168_recent": sparse.csr_matrix(([10.], ([0], [9])), shape=(1, 168))}
        phase.airings = {168: sparse.csr_matrix(([3600.], ([0], [9])), shape=(1, 168))}
        rows = pd.DataFrame({"user_id": ["u"], "tv_show_id": [1]})
        features = P.S.Features(sparse.csr_matrix((1, 1)), rows, ["unused"])
        stats = P.WindowStats(
            pd.DataFrame({"user_id": ["u"], "tv_show_id": [1], "last2": [80]}).set_index(["user_id", "tv_show_id"]),
            pd.DataFrame({"user_id": ["u", "u"], "channel_id": [3, 99], "seconds": [7200, 1800]}).set_index(["user_id", "channel_id"]),
        )
        extra = P.new_features(phase, features, stats, target, P.MIDPOINT).toarray()[0]
        np.testing.assert_allclose(extra, [.8, 36000, 1, 1, 80])

    def test_stacked_queries_keep_fold_user_groups_and_exclude_heldout(self):
        folds = [tiny_fold("fold10", ["a", "b"]), tiny_fold("fold5", ["a", "b"], 15, 5)]
        config = P.recipes(100)["stacked_folds_only"]
        matrix, labels, groups, names = P.training_queries(folds, config, {"a"})
        self.assertEqual(groups, [5, 5])
        self.assertEqual(matrix.shape[0], 10)
        self.assertEqual(len(labels), 10)

    @patch.object(P, "log")
    def test_previous_split_is_shared_across_new_fold_only_users(self, _log):
        with tempfile.TemporaryDirectory() as directory, patch.object(P.H, "OUT", Path(directory)):
            (Path(directory) / "step6_user_split.json").write_text(json.dumps({"train_users": ["a"], "heldout_users": ["b"]}))
            first = P.shared_split(["a", "b", "c", "d"])
            second = P.shared_split(["a", "b", "c", "d"])
            self.assertEqual(first, second)
            self.assertIn("a", first[0])
            self.assertIn("b", first[1])
            self.assertFalse(first[0] & first[1])

    @patch.object(P, "log")
    def test_tiny_evaluation_bagging_and_final_refit(self, _log):
        try:
            import lightgbm  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("LightGBM runtime not installed")
        users = [f"u{index:02}" for index in range(20)]
        folds = [tiny_fold("fold10", users), tiny_fold("fold5", users, 15, 5)]
        final = tiny_fold("final", users[:2], 20, 12)
        configs = {name: config for name, config in P.recipes(100).items()
                   if name in ("previous_v5", "all_together")}
        original = P.model_params
        def small_params(seed, trees=1000):
            return {**original(seed, min(trees, 3)), "n_jobs": 1, "min_child_samples": 1}
        with tempfile.TemporaryDirectory() as directory, patch.object(P.H, "OUT", Path(directory)), \
                patch.object(P, "model_params", side_effect=small_params), \
                patch.object(P.FB, "check_and_write") as writer:
            chosen, iterations, table = P.evaluate(folds, configs, set(users[:16]), set(users[16:]))
            self.assertEqual(len(chosen), 2)
            self.assertEqual(len(iterations["all_together"]), 5)
            self.assertEqual(table.users_fold10.tolist(), [4, 4])
            P.refit_and_predict(folds, final, chosen, configs, iterations)
            self.assertEqual(writer.call_count, 2)
            for call in writer.call_args_list:
                self.assertEqual(set(call.args[0]), set(users[:2]))
                self.assertTrue(all(len(row) == len(set(row)) == 5 for row in call.args[0].values()))
            self.assertTrue((Path(directory) / "step7_final_configs.json").exists())


if __name__ == "__main__":
    unittest.main()
