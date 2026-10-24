"""Fast synthetic coverage/deadline tests; no real data/job is executed."""
from datetime import datetime
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import step9_coverage as J
import step9_coverage_features as V
from test_step8_ltr_graded import data


def noop(*args):
    pass


class CoverageTests(unittest.TestCase):
    def test_union_merges_nested_chained_touching_and_cross_user_channel(self):
        merged = V.union_intervals([0, 0, 0, 0, 0, 1], [3, 3, 3, 3, 4, 3],
                                    [100, 110, 120, 150, 0, 10], [150, 120, 140, 180, 30, 20])
        self.assertEqual(merged[["user_code", "channel_id", "start", "end"]].values.tolist(),
                         [[0, 3, 100, 180], [0, 4, 0, 30], [1, 3, 10, 20]])

    def test_airing_coverage_exact_thresholds_and_unviewed_mean(self):
        views = V.union_intervals([0, 0, 0, 0, 1], [3] * 5, [0, 20, 60, 110, 0], [40, 60, 80, 160, 10])
        airings = pd.DataFrame({"channel_id": [3] * 4, "tv_show_id": [10, 10, 10, 0],
                                "start": [0, 100, 200, 0], "end": [100, 200, 300, 100],
                                "duration": [100] * 4, "airing_id": range(4)})
        users = ["18417531283109304442", "8150816946177670084"]
        pairs, supply, events = V.coverage_for_window(views, airings, users, 0, 300, noop, noop)
        row = pairs.loc[(users[0], 10)]
        self.assertEqual(row.n50, 2)
        self.assertEqual(row.n80, 1)
        self.assertAlmostEqual(row.mean_coverage, .65)
        self.assertAlmostEqual(row.all_airings_mean, 1.3 / 3)
        self.assertEqual(supply.index.tolist(), [10])
        self.assertTrue((events.covered <= events.duration).all())
        self.assertEqual(pairs.loc[(users[1], 10)].n50, 0)
        clipped, _, _ = V.coverage_for_window(views, airings, users, 50, 150, noop, noop)
        self.assertEqual(clipped.loc[(users[0], 10)].n50, 0)
        self.assertAlmostEqual(clipped.loc[(users[0], 10)].mean_coverage, .35)

    def test_global_merge_across_chunks_and_disk_reuse(self):
        user = "18417531283109304442"
        first = pd.DataFrame({"user_id": [user], "vsetv_id": [3], "start_time": ["2020-03-09 00:00:00"], "duraton": [40]})
        second = pd.DataFrame({"user_id": [user, user], "vsetv_id": [3, 3],
                               "start_time": ["2020-03-09 00:00:20", "2020-07-27 00:00:00"], "duraton": [60, 99999]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(V.pd, "read_csv", return_value=iter([first, second])):
                users, merged = V.merged_views(root, {"test": 1}, noop, noop)
            self.assertEqual(users, [user])
            self.assertEqual(len(merged), 1)
            self.assertEqual(int((merged.end - merged.start).sum()), 80)
            with patch.object(V.pd, "read_csv", side_effect=AssertionError("raw recompute")):
                cached_users, cached = V.merged_views(root, {"test": 1}, noop, noop)
            self.assertEqual(cached_users, users)
            pd.testing.assert_frame_equal(cached, merged)

    def test_clipped_sqrt_multiplier_and_frequency_fallback(self):
        value = V.multiplier([10, 10, 10, 0, 0], [0, 10, 100, 4, 0], 10, 10)
        np.testing.assert_allclose(value, [.5, 1, 2, 2, 1])
        context = J.Context("final", ["a", "b"], set(range(1, 7)),
                             pd.Series([6, 5, 4, 3, 2, 1], index=range(1, 7)), {}, 10, 10)
        pairs = pd.DataFrame({"user_id": ["a", "a"], "tv_show_id": [5, 6], "n50": [1, 3]}).set_index(["user_id", "tv_show_id"])
        supply = pd.Series([10] * 6, index=range(1, 7))
        recs = J.frequency(context, pairs, supply, supply)
        self.assertEqual(recs["a"], [6, 5, 1, 2, 3])
        self.assertEqual(recs["b"], [1, 2, 3, 4, 5])

    def test_deadline_reserve_and_40_minute_gate(self):
        moment = [datetime(2026, 10, 24, 14, 0, tzinfo=ZoneInfo("Europe/Kyiv"))]
        elapsed = [0.]
        budget = J.Budget(now=lambda: moment[0], clock=lambda: elapsed[0])
        self.assertEqual(budget.remaining(), 50 * 60)
        with self.assertRaises(J.SkipC):
            budget.begin_c(41 * 60)
        budget.begin_c(30 * 60)
        elapsed[0] = 39 * 60
        with self.assertRaises(J.SkipC):
            budget.check_c("future fits", 120)
        moment[0] = moment[0].replace(hour=14, minute=51)
        with self.assertRaises(J.DeadlineStop):
            budget.check("upload reserve")

    def test_conservative_prior_runtime_estimate(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(J.H, "OUT", Path(directory)):
            self.assertGreater(J.estimate_c(), 40 * 60)
            (Path(directory) / "step7_v6_3_refit.log").write_text("Completed in 618.8s; done\n")
            self.assertAlmostEqual(J.estimate_c(), 1916.4)

    @patch.object(J, "log")
    def test_frequency_table_uses_same_holdout_users_as_saved_reference(self, _log):
        contexts = [J.Context(name, ["a", "b"], set(range(1, 6)),
                               pd.Series([5, 4, 3, 2, 1], index=range(1, 6)),
                               {"a": {1}, "b": {1, 2}}, train_weeks, target_weeks)
                    for name, train_weeks, target_weeks in (("fold10", 10, 10), ("fold5", 15, 5))]
        pairs = pd.DataFrame({"user_id": ["a"], "tv_show_id": [1], "n50": [1]}).set_index(["user_id", "tv_show_id"])
        supplies = {name: pd.Series([10] * 5, index=range(1, 6))
                    for name in ("train10", "train15", "label10", "label5")}
        with tempfile.TemporaryDirectory() as directory, patch.object(J.H, "OUT", Path(directory)):
            root = Path(directory)
            pd.DataFrame([{"recipe": "all_together", "map_all_fold10": .8, "map_all_fold5": .7,
                           "users_fold10": 1, "users_fold5": 1}]).to_csv(root / "step7_ablations.csv", index=False)
            J.evaluate_frequency(contexts, {"train10": pairs, "train15": pairs}, supplies, {"b"})
            table = pd.read_csv(root / "step9_frequency_validation.csv")
            self.assertEqual(table.users_fold10.tolist(), [1, 1])
            self.assertEqual(table.users_fold5.tolist(), [1, 1])
            self.assertAlmostEqual(table.iloc[1].mean_map, 1.)

    @patch.object(J, "log")
    def test_tiny_two_label_ltr_and_five_full_refits(self, _log):
        try:
            import lightgbm  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("LightGBM runtime not installed")
        users = [f"u{index:02}" for index in range(20)]
        datasets = [data("fold10", users), data("fold5", users, 15, 5), data("final", users[:2], 20, 12)]
        index = pd.MultiIndex.from_tuples([(user, item) for user in users for item in range(1, 6)],
                                          names=["user_id", "tv_show_id"])
        pairs = pd.DataFrame({"n50": 1, "n80": 1, "mean_coverage": .9}, index=index)
        supplies = {name: pd.Series([10] * 5, index=range(1, 6))
                    for name in ("train10", "train15", "train20", "label10", "label5", "target31")}
        moment = datetime(2026, 10, 24, 12, 0, tzinfo=ZoneInfo("Europe/Kyiv"))
        budget = J.Budget(now=lambda: moment)
        original = J.P.model_params
        def small(seed, trees=1000):
            return {**original(seed, min(trees, 3)), "n_jobs": 1, "min_child_samples": 1}
        with tempfile.TemporaryDirectory() as directory, patch.object(J.H, "OUT", Path(directory)), \
                patch.object(J.F, "load_base", return_value=datasets), \
                patch.object(J.V, "load_pairs", return_value=pairs), \
                patch.object(J, "estimate_c", return_value=1), patch.object(J.P, "model_params", side_effect=small), \
                patch.object(J.FB, "check_and_write", return_value={"rows": 2}) as writer, \
                patch("sys.stdout", new=io.StringIO()):
            root = Path(directory)
            (root / "user_split.json").write_text(json.dumps({"train_users": users[:16], "heldout_users": users[16:]}))
            result = J.ltr([], {name: pairs for name in ("train10", "train15", "train20")}, supplies,
                           root / "coverage", {"test": 1}, root, {}, set(users[:16]), set(users[16:]),
                           budget, SimpleNamespace(max_trees=3))
            self.assertIn(result["label"], ("binary", "coverage80"))
            self.assertEqual(len(result["iterations"]), 5)
            self.assertEqual(writer.call_args.args[1].name, "submission_v9_ltr.csv")
            self.assertTrue(all(len(set(row)) == 5 for row in writer.call_args.args[0].values()))


if __name__ == "__main__":
    unittest.main()
