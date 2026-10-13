import unittest

import numpy as np

from tools.run_cais_positive_click_gate_probe import GateRecord, policy_metrics, stratified_group_folds


class PositiveClickGateProbeTest(unittest.TestCase):
    @staticmethod
    def record(index: int, task_id: str, label: int) -> GateRecord:
        return GateRecord(
            event_id=str(index),
            task_id=task_id,
            click_index=2 + index % 2,
            label=label,
            delta_iou=0.02 if label else 0.0,
            context=np.zeros(5, dtype=np.float32),
            trajectory=np.zeros((6, 8), dtype=np.float32),
        )

    def test_group_folds_do_not_split_tasks(self) -> None:
        records = []
        for task in range(12):
            records.append(self.record(task, f"task-{task}", int(task % 3 == 0)))
            if task < 3:
                records.append(self.record(100 + task, f"task-{task}", 0))
        folds = stratified_group_folds(records, 3, 17)
        task_fold = {}
        for fold_index, indices in enumerate(folds):
            for index in indices:
                task = records[index].task_id
                self.assertNotIn(task, task_fold) if task not in task_fold else self.assertEqual(
                    task_fold[task], fold_index
                )
                task_fold[task] = fold_index
            self.assertEqual({records[index].label for index in indices}, {0, 1})

    def test_policy_selects_highest_scores(self) -> None:
        result = policy_metrics(
            np.asarray([0.1, 0.9, 0.8, 0.2]),
            np.asarray([0, 1, 1, 0]),
            np.asarray([-0.01, 0.04, 0.02, 0.0]),
            0.5,
        )
        self.assertEqual(result["selected"], 2)
        self.assertEqual(result["beneficial_precision"], 1.0)
        self.assertAlmostEqual(result["all_sample_policy_delta"], 0.015)


if __name__ == "__main__":
    unittest.main()
