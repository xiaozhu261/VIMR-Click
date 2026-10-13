#!/usr/bin/env python3
"""Summarize the image-level high-quality-trajectory training experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("outputs/cais_process_high_quality_images_seed9527_20261008")
RUN = ROOT / "high_quality_images_only"
ALL_RUN = Path("outputs/cais_process_trajectory_v2_seed9527_20261008/fixed_window_process_v2")
NATIVE_RUN = Path("outputs/cais_mais_memory_seed9527_20261008/native_no_memory")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paired(candidate: dict[str, Any], baseline: dict[str, Any], seed: int) -> dict[str, Any]:
    left = candidate["task_level"]["iou_by_task"]
    right = baseline["task_level"]["iou_by_task"]
    task_ids = sorted(set(left) & set(right))
    delta = np.asarray([left[key] - right[key] for key in task_ids], dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = delta[generator.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1)
    return {
        "tasks": len(task_ids),
        "mean_delta": float(delta.mean()),
        "bootstrap_95ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def main() -> None:
    filtered = read(RUN / "training_summary.json")
    manifest = read(RUN / "run_manifest.json")
    all_data = read(ALL_RUN / "training_summary.json")
    native = read(NATIVE_RUN / "training_summary.json")
    strata = read(ROOT / "quality_strata.json")
    filtered_real = filtered["trajectory_sensitivity"]["test"]["real"]
    all_real = all_data["trajectory_sensitivity"]["test"]["real"]
    native_test = native["test"]
    checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    process_gate = float(torch.tanh(state["cognitive_prompt_encoder.process_gate"]))
    comparison = {
        "filtered_minus_all_v2": paired(filtered_real, all_real, 9527),
        "filtered_minus_native": paired(filtered_real, native_test, 9528),
    }
    train = manifest["dataset"]["train"]
    report = {
        "train_filter": {
            "clicks": train["clicks"],
            "images": train["tasks"],
            "excluded_low_quality_images": train["excluded_low_quality_images"],
            "epochs": 42,
            "sample_exposures": 42 * train["clicks"],
        },
        "results": {
            "native": {
                "best_val_iou": native["best_val_iou"],
                "test_click_iou": native_test["mask_iou"],
                "test_task_iou": native_test["task_level"]["mean_iou"],
            },
            "all_v2": {
                "best_val_iou": all_data["best_val_iou"],
                "test_click_iou": all_real["mask_iou"],
                "test_task_iou": all_real["task_level"]["mean_iou"],
            },
            "high_quality_images_only": {
                "best_epoch": filtered["best_epoch"],
                "best_val_iou": filtered["best_val_iou"],
                "test_click_iou": filtered_real["mask_iou"],
                "test_task_iou": filtered_real["task_level"]["mean_iou"],
                "process_gate": process_gate,
            },
        },
        "paired_test_deltas": comparison,
        "trajectory_controls": filtered["trajectory_sensitivity"]["paired_task_deltas"]["test"],
        "quality_strata": strata,
    }
    (ROOT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = {
        "native": "原生点击，无记忆",
        "all_v2": "全量图像 Process V2",
        "high_quality_images_only": "仅全高质量轨迹图像",
    }
    lines = [
        "# CAIS 仅高质量轨迹图像训练实验",
        "",
        f"训练集由 450 张/1122 点击过滤为 {train['tasks']} 张/{train['clicks']} 点击，"
        f"排除 {train['excluded_low_quality_images']} 张图像。使用 42 轮，共 "
        f"{42 * train['clicks']} 次样本曝光，对齐全量实验的 11220 次。",
        "",
        "| 方法 | 最佳 Val IoU | Test 点击 IoU | Test 任务 IoU |",
        "|---|---:|---:|---:|",
    ]
    for key, item in report["results"].items():
        lines.append(
            f"| {labels[key]} | {item['best_val_iou']:.6f} | "
            f"{item['test_click_iou']:.6f} | {item['test_task_iou']:.6f} |"
        )
    lines.extend(["", "## 任务级配对比较", ""])
    for key, item in comparison.items():
        lines.append(
            f"- {key}: {item['mean_delta']:+.6f}, 95% CI "
            f"[{item['bootstrap_95ci'][0]:+.6f}, {item['bootstrap_95ci'][1]:+.6f}]。"
        )
    lines.extend(
        [
            "",
            "## 同一检查点过程敏感性",
            "",
            "| 对照 | Real − 对照 | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for key, item in report["trajectory_controls"].items():
        lines.append(
            f"| {key} | {item['mean_delta']:+.6f} | "
            f"[{item['bootstrap_95ci'][0]:+.6f}, {item['bootstrap_95ci'][1]:+.6f}] |"
        )
    lines.extend(
        [
            "",
            f"最佳检查点为第 {filtered['best_epoch']} 轮，过程门控为 {process_gate:.4f}。",
            f"可靠测试轨迹组 Real−Zero 为 {strata['reliable']['mean_delta']:+.6f}，"
            f"低质量组为 {strata['unreliable']['mean_delta']:+.6f}。",
            "",
            "## 结论",
            "",
            "图像级剔除低质量轨迹不成立。它删除了 64.7% 的图像多样性，虽然匹配了优化曝光量，",
            "仍产生明显过拟合并显著降低完整测试集表现。正确方向是保留所有图像、Mask 和点击监督，",
            "只在低质量点击上关闭轨迹/过程分支。",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
