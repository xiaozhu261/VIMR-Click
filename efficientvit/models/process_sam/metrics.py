from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F


def binary_iou(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    predicted, truth = predicted.bool(), truth.bool()
    union = (predicted | truth).sum()
    return float((predicted & truth).sum() / union) if union else 1.0


def boundary_iou(predicted: torch.Tensor, truth: torch.Tensor, dilation_ratio: float = 0.02) -> float:
    """Inner boundary IoU, with zero image-border padding, at original resolution.

    Definition follows https://github.com/bowenc0221/boundary-iou-api .
    """
    height, width = truth.shape[-2:]
    distance = max(1, int(round(dilation_ratio * (height**2 + width**2) ** 0.5)))

    def boundary(mask):
        mask = mask.float().reshape(1, 1, height, width)
        # A square radius-d erosion equals d successive 3x3 erosions. Explicit
        # zero padding includes boundaries truncated by the original image.
        padded = F.pad(mask, (distance,) * 4, value=0)
        eroded = -F.max_pool2d(-padded, 2 * distance + 1, stride=1)
        return (mask - eroded).bool()

    return binary_iou(boundary(predicted), boundary(truth))


def replay_metrics(rows: list[dict], max_clicks: int = 5) -> dict:
    by_click, by_task = defaultdict(list), defaultdict(dict)
    for row in rows:
        by_click[int(row["click_index"])].append(row["iou"])
        by_task[row["task_run_id"]][int(row["click_index"])] = row["iou"]
    # Compare click budgets on the same cohort of COMPLETE recorded prefixes.
    complete = {task: scores for task, scores in by_task.items() if all(i in scores for i in range(1, max_clicks + 1))}
    result = {f"IoU@{click}": float(np.mean(by_click[click])) if by_click[click] else None for click in (1, 3, 5)}
    result["click_coverage"] = {str(click): len(by_click[click]) for click in range(1, max_clicks + 1)}
    result["complete_prefix_tasks"] = len(complete)
    result["tasks"] = len(by_task)
    result["samples"] = len(rows)
    result["mean_iou"] = float(np.mean([row["iou"] for row in rows])) if rows else None
    result["Boundary IoU"] = float(np.mean([row["boundary_iou"] for row in rows])) if rows else None
    result["latency_ms_mean"] = float(np.mean([row["latency_ms"] for row in rows])) if rows else None
    result["latency_ms_p95"] = float(np.percentile([row["latency_ms"] for row in rows], 95)) if rows else None
    for click in (1, 3, 5):
        result[f"IoU@{click}_complete_prefix"] = (
            float(np.mean([scores[click] for scores in complete.values()])) if complete else None
        )
    first = [
        next((i for i in range(1, max_clicks + 1) if scores[i] >= 0.9), max_clicks + 1) for scores in complete.values()
    ]
    result["NoC@90"] = float(np.mean(first)) if first else None
    result["NoC@90_failures"] = sum(value == max_clicks + 1 for value in first)
    result["NoC_protocol"] = f"recorded pre-mask replay; complete 1..{max_clicks} only; failure={max_clicks + 1}"
    return result


def paired_process_delta(real_rows: list[dict], shuffled_rows: list[dict], seed: int) -> dict:
    real = {row["event_id"]: row for row in real_rows}
    shuffled = {row["event_id"]: row for row in shuffled_rows}
    if set(real) != set(shuffled):
        raise ValueError("E3/E4 require exactly the same evaluation events")
    tasks = defaultdict(list)
    for event, row in real.items():
        tasks[row["task_run_id"]].append(row["iou"] - shuffled[event]["iou"])
    deltas = np.asarray([np.mean(values) for _, values in sorted(tasks.items())], dtype=float)
    if not len(deltas):
        raise ValueError("Paired comparison requires evaluation tasks")
    generator = np.random.default_rng(seed)
    means = deltas[generator.integers(len(deltas), size=(2000, len(deltas)))].mean(axis=1)
    return {
        "tasks": len(deltas),
        "mean_task_iou_delta": float(deltas.mean()),
        "bootstrap_95ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }
