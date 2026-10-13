#!/usr/bin/env python3
"""Summarize the same-seed MAIS-style CAIS memory ablation."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


RUNS = ("native_no_memory", "click_mask_memory", "click_mask_trajectory_memory")


def paired_delta(
    left: dict[str, float], right: dict[str, float], *, seed: int = 9527, iterations: int = 5000
) -> dict[str, Any]:
    task_ids = sorted(set(left) & set(right))
    values = np.asarray([left[key] - right[key] for key in task_ids], dtype=np.float64)
    generator = np.random.default_rng(seed)
    bootstrap = values[generator.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    return {
        "tasks": len(task_ids),
        "mean_delta": float(values.mean()),
        "bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summaries = {
        name: json.loads((args.root / name / "training_summary.json").read_text(encoding="utf-8"))
        for name in RUNS
    }
    rows = {}
    task_ious = {}
    for name, summary in summaries.items():
        test = summary["test"]
        rows[name] = {
            "best_epoch": summary["best_epoch"],
            "best_val_iou": summary["best_val_iou"],
            "test_click_mean_iou": test["mask_iou"],
            "test_task_mean_iou": test["task_level"]["mean_iou"],
            "memory_gate": summary.get("effective_residual_gates", {}).get(
                "interaction_memory_gate"
            ),
        }
        task_ious[name] = test["task_level"]["iou_by_task"]
    comparisons = {
        "click_mask_minus_native": paired_delta(
            task_ious["click_mask_memory"], task_ious["native_no_memory"], seed=9527
        ),
        "trajectory_minus_click_mask": paired_delta(
            task_ious["click_mask_trajectory_memory"], task_ious["click_mask_memory"], seed=9528
        ),
    }
    trajectory_summary = summaries["click_mask_trajectory_memory"]
    sensitivity = trajectory_summary.get("trajectory_sensitivity")
    result = {"runs": rows, "paired_task_deltas": comparisons, "trajectory_sensitivity": sensitivity}
    (args.root / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# CAIS MAIS-style Memory Ablation",
        "",
        "| Run | Best epoch | Val IoU | Test click IoU | Test task IoU | Memory gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in RUNS:
        row = rows[name]
        gate = "-" if row["memory_gate"] is None else f"{row['memory_gate']:.6f}"
        lines.append(
            f"| {name} | {row['best_epoch']} | {row['best_val_iou']:.6f} | "
            f"{row['test_click_mean_iou']:.6f} | {row['test_task_mean_iou']:.6f} | {gate} |"
        )
    lines.extend(["", "## Paired task-level deltas", ""])
    for name, comparison in comparisons.items():
        low, high = comparison["bootstrap_95ci"]
        lines.append(
            f"- {name}: {comparison['mean_delta']:+.6f}, "
            f"95% bootstrap CI [{low:+.6f}, {high:+.6f}] over {comparison['tasks']} tasks."
        )
    (args.root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
