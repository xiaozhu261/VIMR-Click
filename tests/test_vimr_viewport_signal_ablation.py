import math
import unittest

import torch

from tools.run_vimr_viewport_signal_ablation import (
    default_trajectory,
    make_derangement,
    oracle_bbox_trajectory,
    synthetic_trajectory,
    trajectory_for_mode,
)


class ViewportSignalAblationTests(unittest.TestCase):
    def test_default_trajectory_is_always_global(self) -> None:
        trajectory = default_trajectory(5)
        self.assertEqual(len(trajectory), 5)
        self.assertTrue(all(state == [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0] for state in trajectory))

    def test_synthetic_trajectory_tracks_clicks_after_global_round(self) -> None:
        clicks = [[[100.0, 200.0]], [[300.0, 400.0]], [[500.0, 600.0]]]
        trajectory = synthetic_trajectory(clicks)
        self.assertEqual(trajectory[0][:2], [0.5, 0.5])
        self.assertAlmostEqual(trajectory[1][0], 300.0 / 1024.0)
        self.assertAlmostEqual(trajectory[1][1], 400.0 / 1024.0)
        self.assertAlmostEqual(trajectory[1][2], math.log(2.0))

    def test_shuffled_mode_is_no_self_and_preserves_trajectory_multiset(self) -> None:
        records = [
            {"synthetic_viewports": [[float(index)] * 7], "oracle_viewports": [[-float(index)] * 7]}
            for index in range(9)
        ]
        mapping = make_derangement(len(records), seed=17)
        self.assertEqual(sorted(mapping), list(range(len(records))))
        self.assertTrue(all(index != source for index, source in enumerate(mapping)))
        shuffled = [trajectory_for_mode("shuffled", index, records, mapping, 1) for index in range(9)]
        expected = [record["synthetic_viewports"] for record in records]
        self.assertEqual(sorted(shuffled), sorted(expected))

    def test_oracle_trajectory_uses_gt_bbox_and_stays_fixed(self) -> None:
        target = torch.zeros(1, 1, 256, 256)
        target[:, :, 64:128, 32:96] = 1
        trajectory = oracle_bbox_trajectory(target, 5)
        self.assertEqual(trajectory[0], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(trajectory[1][0], 64.0 / 256.0)
        self.assertAlmostEqual(trajectory[1][1], 96.0 / 256.0)
        self.assertAlmostEqual(math.exp(trajectory[1][2]), 3.2)
        self.assertEqual(trajectory[1][6], 3.0)
        self.assertEqual(trajectory[2], [*trajectory[1][:3], 0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
