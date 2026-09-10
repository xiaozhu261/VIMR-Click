import unittest

import torch

from tools.eval_vimr_coco_quick import viewport_for_mode
from tools.run_vimr_coco_protocol import aggregate_metrics, selection_direction, summarize_protocol


class VIMRProtocolTests(unittest.TestCase):
    def test_default_viewport_mode_is_information_free_across_rounds(self) -> None:
        first = torch.tensor([[[128.0, 256.0]]])
        second = torch.tensor([[[900.0, 700.0]]])
        previous = torch.tensor([[0.1, 0.2, 1.0, 0.3, 0.4, 0.5, 3.0]])
        expected = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertTrue(torch.equal(viewport_for_mode(first, 0, None, "default"), expected))
        self.assertTrue(torch.equal(viewport_for_mode(second, 4, previous, "default"), expected))

    def test_summary_records_requested_clicks_and_late_regression(self) -> None:
        rows = [
            {"ious": [0.60, 0.70, 0.80, 0.70, 0.90]},
            {"ious": [0.80, 0.85, 0.90, 0.91, 0.89]},
        ]
        summary = summarize_protocol(rows, max_clicks=5)
        self.assertEqual(set(key for key in summary if key.startswith("mIoU")), {"mIoU@1", "mIoU@2", "mIoU@3", "mIoU@5"})
        self.assertAlmostEqual(summary["NoC@85"], 3.5)
        self.assertAlmostEqual(summary["Success@90/5"], 1.0)
        self.assertAlmostEqual(summary["RegressionRate"], 2 / 8)
        self.assertAlmostEqual(summary["RegressionRate@4"], 0.5)
        self.assertAlmostEqual(summary["RegressionRate@5"], 0.5)
        self.assertAlmostEqual(summary["LateRegressionRate@4-5"], 2 / 4)
        self.assertAlmostEqual(summary["FinalSuccess@90/5"], 0.5)
        self.assertAlmostEqual(summary["LostAfterSuccess@90/5"], 0.5)
        self.assertAlmostEqual(summary["LateSevereRegressionRate@4-5>0.05"], 0.25)

    def test_aggregate_uses_sample_standard_deviation(self) -> None:
        aggregate = aggregate_metrics(
            {"mIoU@5": 0.5},
            {42: {"mIoU@5": 0.6}, 43: {"mIoU@5": 0.8}, 44: {"mIoU@5": 0.7}},
        )
        self.assertAlmostEqual(aggregate["mIoU@5"]["vimr_mean"], 0.7)
        self.assertAlmostEqual(aggregate["mIoU@5"]["vimr_std"], 0.1)
        self.assertAlmostEqual(aggregate["mIoU@5"]["delta_mean"], 0.2)

    def test_checkpoint_selection_direction(self) -> None:
        self.assertEqual(selection_direction("mIoU@5"), "max")
        self.assertEqual(selection_direction("Success@90/5"), "max")
        self.assertEqual(selection_direction("NoC@90"), "min")


if __name__ == "__main__":
    unittest.main()
