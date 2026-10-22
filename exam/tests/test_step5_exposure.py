"""Tiny synthetic checks only; no project datasets or full fitting job are read."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import step5_exposure as E


class ExposureTests(unittest.TestCase):
    def test_clipping_and_half_open_hour_boundaries(self):
        source, hours, seconds = E.split_hours(
            np.array([-600, 1800, 3600, 8000]),
            np.array([600, 5400, 7200, 9000]), 0, 7200,
        )
        np.testing.assert_array_equal(source, [0, 1, 1, 2])
        np.testing.assert_array_equal(hours, [0, 0, 1, 1])
        np.testing.assert_array_equal(seconds, [600, 1800, 1800, 3600])
        source, hours, seconds = E.split_hours(np.array([10]), np.array([20]), 30, 40)
        self.assertEqual(len(source), 0)

    def test_spring_dst_uses_elapsed_seconds_but_local_slots(self):
        start = E.parse_time(pd.Series(["2020-03-29 02:30:00"]), "%Y-%m-%d %H:%M:%S")
        _, hours, seconds = E.split_hours(start, start + 7200,
                                          E.epoch("2020-03-29"), E.epoch("2020-03-30"))
        np.testing.assert_array_equal(seconds, [1800, 3600, 1800])
        np.testing.assert_array_equal(E.local_slots(hours), [6 * 24 + 2, 6 * 24 + 4, 6 * 24 + 5])

    def test_coarse_projection_keeps_channels_and_weekends_separate(self):
        source = sparse.csr_matrix(([1, 2, 4, 8], ([0] * 4, [9, 24 + 9, 5 * 24 + 9, 168 + 9])),
                                   shape=(1, 336))
        projected = (source @ E.collapse_projection(2)).toarray()[0]
        self.assertEqual(projected[9], 3)
        self.assertEqual(projected[24 + 9], 4)
        self.assertEqual(projected[48 + 9], 8)
        self.assertEqual(projected.sum(), 15)

    @patch.object(E, "log")
    def test_habits_exclude_future_views_clip_sessions_and_double_recent(self, _log):
        user = "18417531283109304442"
        views = pd.DataFrame({
            "user_id": [user] * 4,
            "vsetv_id": [3] * 4,
            "start_time": ["2020-03-09 09:00:00", "2020-05-11 09:00:00",
                           "2020-05-17 23:30:00", "2020-05-18 00:00:00"],
            "duraton": [3600, 3600, 3600, 999999],
        })
        with patch.object(E.pd, "read_csv", return_value=iter([views])):
            habits = E.build_habits([user], np.array([3]), E.VALIDATION_START,
                                     E.collapse_projection(1))
        # Raw views do not carry labels: all context is eligible, even if its
        # channel airtime has no programme label in the training schedule.
        self.assertEqual(habits["168_uniform"].sum(), 9000 / 10)
        self.assertEqual(habits["168_recent"].sum(), 14400 / 10)
        self.assertEqual(habits["48_uniform"].sum(), 9000 / 10)
        self.assertEqual(habits["168_uniform"][0, 9], 7200 / 10)

    @patch.object(E, "log")
    def test_target_schedule_includes_new_items_and_clips_block_tails(self, _log):
        schedule = pd.DataFrame({
            "channel_id": [3, 3, 3], "tv_show_id": [99, 0, 100],
            "start_time": ["18.05.2020 09:30:00", "18.05.2020 09:30:00", "27.07.2020 00:00:00"],
            "duration": [3600, 3600, 3600],
        })
        with patch.object(E.pd, "read_csv", return_value=iter([schedule])), \
                patch.object(E.H, "load_pool", return_value={99, 100}):
            items, pool, channels, airings, _ = E.build_airings(
                "21-30", E.VALIDATION_START, E.FINAL_START)
        np.testing.assert_array_equal(items, [99, 100])
        self.assertEqual(airings[168][0, 9], 1800)
        self.assertEqual(airings[168][0, 10], 1800)
        self.assertEqual(airings[168][1].sum(), 0)
        habit = sparse.csr_matrix(([360], ([0], [9])), shape=(1, 168))
        scores = (habit @ airings[168].T).toarray()[0]
        self.assertEqual(E.positive_ranking(scores, items), [(99, 648000.0)])

    def test_tied_cutoff_has_id_tiebreak_and_zero_scores_are_excluded(self):
        self.assertEqual(E.positive_ranking(np.array([0., 1., 2., 1.]),
                                          np.array([10, 20, 30, 40]), keep=2),
                         [(30, 2.0), (20, 1.0)])

    def test_extended_v3_ranking_preserves_original_top_five(self):
        ppr = list(range(1, 201))
        popularity = list(range(180, 380))
        full = E.H.rank_fusion([ppr, popularity], [1., .5], k=200, rrf_k=60)
        original = E.H.rank_fusion([ppr, popularity], [1., .5], k=5, rrf_k=60)
        self.assertEqual(full[:5], original)
        exposure = {"u": [(999, 10.)]}
        combined = E.fuse_rankings(exposure, {"u": full}, "linear", .7)["u"]
        self.assertEqual(combined[0][0], 999)


if __name__ == "__main__":
    unittest.main()
