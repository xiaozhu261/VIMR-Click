import unittest

import numpy as np

from tools.eval_cais_quality_gated_local_search import (
    SearchRecord,
    candidate_coordinates,
    recent_approach_vector,
    select_index,
    trajectory_alignment,
)


class QualityGatedLocalSearchTest(unittest.TestCase):
    def test_candidate_grid_keeps_original_first_and_clips(self) -> None:
        candidates = candidate_coordinates(1.0, 1.0, 100, 80, [0.1], 8)
        np.testing.assert_allclose(candidates[0], [1.0, 1.0])
        self.assertTrue(np.all(candidates[:, 0] >= 0))
        self.assertTrue(np.all(candidates[:, 0] <= 99))
        self.assertTrue(np.all(candidates[:, 1] >= 0))
        self.assertTrue(np.all(candidates[:, 1] <= 79))

    def test_recent_vector_and_axis_alignment(self) -> None:
        record = {
            "click_x_norm": 0.5,
            "click_y_norm": 0.5,
            "preceding_trajectory": {
                "points": [[-600.0, 0.1, 0.2], [-400.0, 0.3, 0.5], [0.0, 0.5, 0.5]]
            },
        }
        vector = recent_approach_vector(record, 500.0)
        np.testing.assert_allclose(vector, [0.2, 0.0], atol=1e-6)
        candidates = np.asarray([[50, 50], [60, 50], [50, 60], [40, 50]], dtype=np.float32)
        alignment = trajectory_alignment(candidates, candidates[0], (100, 100), vector)
        np.testing.assert_allclose(alignment, [0.0, 1.0, 0.0, 1.0], atol=1e-6)

    def test_no_op_margin_can_reject_weak_candidate(self) -> None:
        record = SearchRecord(
            event_id="event",
            task_id="task",
            click_index=1,
            split="test",
            original_quality=0.5,
            candidate_xy_px=np.asarray([[0, 0], [1, 0]], dtype=np.float32),
            actual_iou=np.asarray([0.5, 0.6], dtype=np.float32),
            predicted_iou=np.asarray([0.5, 0.505], dtype=np.float32),
            mask_area=np.asarray([0.2, 0.2], dtype=np.float32),
            mask_iou_original=np.asarray([1.0, 0.9], dtype=np.float32),
            mask_iou_previous=np.asarray([0.8, 0.8], dtype=np.float32),
            distance=np.asarray([0.0, 0.0], dtype=np.float32),
            real_alignment=np.asarray([0.0, 0.0], dtype=np.float32),
            approach_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(select_index(record, 0.0, 0.0, record.real_alignment, 0.0), 1)
        self.assertEqual(select_index(record, 0.0, 0.0, record.real_alignment, 0.01), 0)


if __name__ == "__main__":
    unittest.main()
