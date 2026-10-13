#!/usr/bin/env python3
"""Summarize the process-aware trajectory experiment against prior CAIS controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_delta(
    candidate: dict[str, Any], baseline: dict[str, Any], *, seed: int, iterations: int = 5000
) -> dict[str, Any]:
    candidate_tasks = candidate["task_level"]["iou_by_task"]
    baseline_tasks = baseline["task_level"]["iou_by_task"]
    task_ids = sorted(set(candidate_tasks) & set(baseline_tasks))
    deltas = np.asarray(
        [candidate_tasks[task_id] - baseline_tasks[task_id] for task_id in task_ids],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    sampled = deltas[
        generator.integers(0, len(deltas), size=(iterations, len(deltas)))
    ].mean(axis=1)
    return {
        "tasks": len(task_ids),
        "mean_delta": float(deltas.mean()),
        "bootstrap_95ci": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/cais_process_trajectory_v2_seed9527_20261008"),
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("outputs/cais_mais_memory_seed9527_20261008"),
    )
    args = parser.parse_args()

    v2_root = args.output_root / "fixed_window_process_v2"
    v2 = read_json(v2_root / "training_summary.json")
    legacy = read_json(args.legacy_root / "click_mask_trajectory_memory" / "training_summary.json")
    native = read_json(args.legacy_root / "native_no_memory" / "training_summary.json")
    audit = read_json(args.output_root / "audit" / "trajectory_audit.json")
    quality_strata_path = args.output_root / "quality_strata.json"
    quality_strata = read_json(quality_strata_path) if quality_strata_path.is_file() else None
    v2_real = v2["trajectory_sensitivity"]["test"]["real"]
    legacy_real = legacy["trajectory_sensitivity"]["test"]["real"]
    native_test = native["test"]
    v2_vs_legacy = paired_delta(v2_real, legacy_real, seed=9527)
    v2_vs_native = paired_delta(v2_real, native_test, seed=9528)
    checkpoint = torch.load(v2_root / "best.pt", map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    process_gate = float(torch.tanh(state["cognitive_prompt_encoder.process_gate"]))
    memory_gate = float(torch.tanh(state["interaction_memory_attention.gate"]))

    report = {
        "audit": audit["overall"],
        "experiments": {
            "native_no_memory": {
                "best_val_iou": native["best_val_iou"],
                "test_click_iou": native_test["mask_iou"],
                "test_task_iou": native_test["task_level"]["mean_iou"],
            },
            "legacy_trajectory_memory": {
                "best_val_iou": legacy["best_val_iou"],
                "test_click_iou": legacy_real["mask_iou"],
                "test_task_iou": legacy_real["task_level"]["mean_iou"],
            },
            "fixed_window_process_v2": {
                "best_epoch": v2["best_epoch"],
                "best_val_iou": v2["best_val_iou"],
                "test_click_iou": v2_real["mask_iou"],
                "test_task_iou": v2_real["task_level"]["mean_iou"],
                "process_gate": process_gate,
                "memory_gate": memory_gate,
            },
        },
        "paired_test_deltas": {
            "v2_minus_legacy": v2_vs_legacy,
            "v2_minus_native": v2_vs_native,
            "v2_real_minus_controls": v2["trajectory_sensitivity"]["paired_task_deltas"]["test"],
        },
        "click_index": {
            click_index: {
                "legacy_real": legacy_real["click_index_level"][click_index]["mean_iou"],
                "v2_real": v2_real["click_index_level"][click_index]["mean_iou"],
                "v2_zero": v2["trajectory_sensitivity"]["test"]["zero"]["click_index_level"][click_index][
                    "mean_iou"
                ],
            }
            for click_index in v2_real["click_index_level"]
        },
        "quality_strata": quality_strata,
    }
    (args.output_root / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# CAIS 轨迹/标注过程处理 V2 实验报告",
        "",
        "## 数据审计",
        "",
        f"- 真人点击：{audit['overall']['clicks']} 次。",
        f"- ≤3 点轨迹：{audit['overall']['three_or_fewer_points']['ratio']:.1%}。",
        f"- 静止轨迹：{audit['overall']['static_trajectory']['ratio']:.1%}。",
        f"- 按质量分层判据可视为较完整的轨迹：{audit['overall']['reliable']['ratio']:.1%}。",
        "",
        "## 同预算结果",
        "",
        "| 方法 | 最佳 Val IoU | Test 点击均值 | Test 任务均值 |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "native_no_memory": "原生点击（无记忆）",
        "legacy_trajectory_memory": "旧轨迹 + 点击/Mask 记忆",
        "fixed_window_process_v2": "固定时轴 + 过程摘要 V2",
    }
    for key, metrics in report["experiments"].items():
        lines.append(
            f"| {labels[key]} | {fmt(metrics['best_val_iou'])} | "
            f"{fmt(metrics['test_click_iou'])} | {fmt(metrics['test_task_iou'])} |"
        )
    lines.extend(
        [
            "",
            "## 配对检验",
            "",
            f"- V2 − 旧轨迹：{v2_vs_legacy['mean_delta']:+.6f}，"
            f"95% CI [{v2_vs_legacy['bootstrap_95ci'][0]:+.6f}, "
            f"{v2_vs_legacy['bootstrap_95ci'][1]:+.6f}]。",
            f"- V2 − 原生无记忆：{v2_vs_native['mean_delta']:+.6f}，"
            f"95% CI [{v2_vs_native['bootstrap_95ci'][0]:+.6f}, "
            f"{v2_vs_native['bootstrap_95ci'][1]:+.6f}]。",
            "",
            "| V2 同检查点控制 | Real − 控制（任务 IoU） | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    control_labels = {"real_minus_zero": "Zero", "real_minus_shuffle": "Shuffle", "real_minus_wrong_image": "Wrong-image"}
    for key, item in report["paired_test_deltas"]["v2_real_minus_controls"].items():
        lines.append(
            f"| {control_labels[key]} | {item['mean_delta']:+.6f} | "
            f"[{item['bootstrap_95ci'][0]:+.6f}, {item['bootstrap_95ci'][1]:+.6f}] |"
        )
    lines.extend(
        [
            "",
            f"最佳检查点的过程门控为 {process_gate:.4f}（初始 0.05），记忆门控为 {memory_gate:.4f}。",
            "门控明显打开，说明模型没有完全忽略过程输入；但所有任务级置信区间均跨 0，",
            "因此单种子结果只能视为弱方向性信号，不能宣称稳定提升。",
        ]
    )
    if quality_strata is not None:
        lines.extend(
            [
                "",
                "## 轨迹质量分层",
                "",
                "| 分层 | 点击数 | Real − Zero（任务 IoU） | 95% CI |",
                "|---|---:|---:|---:|",
            ]
        )
        for name in ("reliable", "unreliable"):
            item = quality_strata[name]
            lines.append(
                f"| {name} | {item['clicks']} | {item['mean_delta']:+.6f} | "
                f"[{item['bootstrap_95ci'][0]:+.6f}, {item['bootstrap_95ci'][1]:+.6f}] |"
            )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "固定时间轴和显式采集覆盖信息修复了旧预处理的语义错误，并在验证集产生了 Real 优于 Zero/Shuffle 的方向性差异。",
            "然而 test 上 V2 没有优于旧轨迹或原生点击基线，且配对区间跨 0。质量分层进一步显示：较完整轨迹方向为正，",
            "低质量轨迹方向为负。当前主要瓶颈已从“轨迹被错误拉伸”转为“需要可靠性条件门控，同时补足真实记录和样本量”。",
        ]
    )
    (args.output_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
