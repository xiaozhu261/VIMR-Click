import unittest

import numpy as np

from tools.visualize_vimr_viewport_ablation import (
    probability_to_original,
    prompt_point_to_original,
    ranking_indices,
    viewport_rect_to_original,
)


class ViewportVisualizerTests(unittest.TestCase):
    def test_probability_unpads_landscape_image(self) -> None:
        probability = np.zeros((256, 256), dtype=np.float32)
        probability[:128, :256] = 1.0
        restored = probability_to_original(probability, 100, 200)
        self.assertEqual(restored.shape, (100, 200))
        self.assertTrue(np.allclose(restored, 1.0))

    def test_prompt_point_maps_from_padded_sam_frame(self) -> None:
        x, y = prompt_point_to_original([512.0, 256.0], 100, 200)
        self.assertAlmostEqual(x, 100.0)
        self.assertAlmostEqual(y, 50.0)

    def test_global_viewport_clips_to_image(self) -> None:
        rect = viewport_rect_to_original([0.5, 0.5, 0.0, 0, 0, 0, 0], 100, 200)
        self.assertEqual(rect, (0.0, 0.0, 200.0, 100.0))

    def test_rankings_find_both_directions(self) -> None:
        annotation_ids = [10, 20, 30]
        baseline = [[0.1], [0.5], [0.9]]
        synthetic = {10: [0.8], 20: [0.4], 30: [0.7]}
        shuffled = {10: [0.2], 20: [0.6], 30: [0.7]}
        oracle = {10: [0.9], 20: [0.9], 30: [0.8]}
        self.assertEqual(
            ranking_indices("synthetic_wins", baseline, synthetic, shuffled, oracle, annotation_ids)[0],
            0,
        )
        self.assertEqual(
            ranking_indices("shuffled_wins", baseline, synthetic, shuffled, oracle, annotation_ids)[0],
            1,
        )
        self.assertEqual(
            ranking_indices("representative", baseline, synthetic, shuffled, oracle, annotation_ids)[0],
            1,
        )


if __name__ == "__main__":
    unittest.main()
