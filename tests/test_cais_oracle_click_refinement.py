import unittest

import numpy as np

from tools.eval_cais_oracle_click_refinement import build_oracle_target


class OracleClickRefinementTest(unittest.TestCase):
    def setUp(self) -> None:
        self.error = np.zeros((24, 24), dtype=bool)
        self.error[5:19, 5:19] = True

    def test_boundary_click_moves_to_safe_interior(self) -> None:
        target = build_oracle_target(
            self.error,
            5,
            12,
            boundary_threshold_px=3,
            safe_margin_px=5,
            max_correction_px=20,
        )
        self.assertTrue(target.hard)
        self.assertTrue(target.eligible)
        self.assertTrue(target.inside_error)
        self.assertGreater(target.correction_distance_px, 0)
        self.assertTrue(self.error[int(target.target_y_px), int(target.target_x_px)])

    def test_easy_interior_click_is_not_modified(self) -> None:
        target = build_oracle_target(
            self.error,
            12,
            12,
            boundary_threshold_px=3,
            safe_margin_px=5,
            max_correction_px=20,
        )
        self.assertFalse(target.hard)
        self.assertFalse(target.eligible)
        self.assertEqual(target.reason, "easy")

    def test_far_outside_click_is_rejected_as_nonlocal(self) -> None:
        target = build_oracle_target(
            self.error,
            0,
            0,
            boundary_threshold_px=3,
            safe_margin_px=5,
            max_correction_px=4,
        )
        self.assertTrue(target.hard)
        self.assertFalse(target.eligible)
        self.assertEqual(target.reason, "nonlocal_correction")


if __name__ == "__main__":
    unittest.main()
