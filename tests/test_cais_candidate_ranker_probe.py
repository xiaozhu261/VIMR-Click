import unittest

import numpy as np

from tools.run_cais_candidate_ranker_probe import RankRecord, base_features, choose_candidate, trajectory_features


class CandidateRankerProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.record = RankRecord(
            event_id="event",
            task_id="task",
            click_index=2,
            image_size=(100, 200),
            original_quality=0.7,
            candidate_xy=np.asarray([[100, 50], [120, 50], [100, 60]], dtype=np.float32),
            actual_iou=np.asarray([0.6, 0.7, 0.5], dtype=np.float32),
            predicted_iou=np.asarray([0.7, 0.71, 0.69], dtype=np.float32),
            mask_area=np.asarray([0.2, 0.21, 0.19], dtype=np.float32),
            mask_iou_original=np.asarray([1.0, 0.9, 0.8], dtype=np.float32),
            mask_iou_previous=np.asarray([0.8, 0.82, 0.78], dtype=np.float32),
            distance=np.asarray([0.0, 0.5, 0.5], dtype=np.float32),
            approach=np.asarray([0.2, 0.0], dtype=np.float32),
        )

    def test_feature_shapes_and_alignment(self) -> None:
        self.assertEqual(base_features(self.record).shape, (3, 13))
        trajectory = trajectory_features(self.record, self.record.approach)
        self.assertEqual(trajectory.shape, (3, 4))
        self.assertAlmostEqual(float(trajectory[1, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(trajectory[2, 0]), 0.0, places=5)

    def test_no_op_margin(self) -> None:
        scores = np.asarray([0.0, 0.004, -0.1])
        self.assertEqual(choose_candidate(scores, 0.005), 0)
        self.assertEqual(choose_candidate(scores, 0.001), 1)


if __name__ == "__main__":
    unittest.main()
