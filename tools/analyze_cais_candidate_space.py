#!/usr/bin/env python3
"""Produce presentation-ready ablations and diagnostics from CAIS candidate search."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-results",
        type=Path,
        default=Path("outputs/cais_quality_gated_local_search_20261006/candidate_results.jsonl"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path(
            "outputs/cais_synthetic_click2_click3_300_20260921/alignment/modeling_clicks_strict.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/cais_supplementary_analysis_20261008")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--shuffle-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9527)
    return parser


def task_bootstrap(
    values: np.ndarray, task_ids: list[str], samples: int, seed: int
) -> dict[str, float | int]:
    groups: dict[str, list[int]] = {}
    for index, task_id in enumerate(task_ids):
        groups.setdefault(task_id, []).append(index)
    tasks = sorted(groups)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        chosen = rng.choice(tasks, len(tasks), replace=True)
        indices = np.concatenate([np.asarray(groups[task]) for task in chosen])
        estimates.append(float(values[indices].mean()))
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "samples": len(values),
        "tasks": len(tasks),
    }


def alignment(displacement: np.ndarray, approach: np.ndarray) -> tuple[float, float]:
    denominator = float(np.linalg.norm(displacement) * np.linalg.norm(approach))
    if denominator <= 1e-12:
        return math.nan, math.nan
    signed = float(np.dot(displacement, approach) / denominator)
    return abs(signed), signed


def subgroup(rows: list[dict[str, Any]], selector) -> dict[str, float | int]:
    selected = [row for row in rows if selector(row)]
    gains = np.asarray([row["grid_gain"] for row in selected], dtype=np.float64)
    return {
        "records": len(selected),
        "mean_grid_gain": float(gains.mean()) if len(gains) else math.nan,
        "median_grid_gain": float(np.median(gains)) if len(gains) else math.nan,
        "large_gain_rate": float(np.mean(gains > 0.01)) if len(gains) else math.nan,
    }


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.candidate_results.open("r", encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle if line.strip()]
    with args.dataset_manifest.open("r", encoding="utf-8") as handle:
        metadata = {
            row["event_id"]: row
            for line in handle
            if line.strip() and (row := json.loads(line)).get("source", "human") == "human"
        }

    rows = []
    candidate_counts = (1, 9, 17, 25, 33)
    radius_labels = ("Original", "0.625%", "1.25%", "2.5%", "5.0%")
    for row in source_rows:
        actual = np.asarray(row["actual_iou"], dtype=np.float64)
        predicted = np.asarray(row["predicted_iou"], dtype=np.float64)
        candidates = np.asarray(row["candidate_xy_px"], dtype=np.float64)
        info = metadata[row["event_id"]]
        best_index = int(np.argmax(actual))
        original = float(actual[0])
        grid_gain = float(actual[best_index] - original)
        correlations = np.corrcoef(predicted, actual)[0, 1]
        quality_order = np.argsort(-predicted, kind="stable")
        best_rank = int(np.where(quality_order == best_index)[0][0]) + 1
        displacement = candidates[best_index] - candidates[0]
        displacement[0] /= max(float(info["image_width_px"]), 1.0)
        displacement[1] /= max(float(info["image_height_px"]), 1.0)
        approach = np.asarray(row["approach_vector"], dtype=np.float64)
        absolute_alignment, signed_alignment = alignment(displacement, approach)
        rows.append(
            {
                "event_id": row["event_id"],
                "task_id": row["task_run_id"],
                "click_index": int(row["click_index"]),
                "gated": bool(row["mean_seed_gate"]),
                "original_iou": original,
                "original_quality": float(row["original_quality"]),
                "grid_gain": grid_gain,
                "best_index": best_index,
                "best_rank_by_quality": best_rank,
                "quality_selected_gain": float(actual[int(np.argmax(predicted))] - original),
                "quality_actual_correlation": float(correlations),
                "best_displacement": displacement,
                "approach": approach,
                "absolute_alignment": absolute_alignment,
                "signed_alignment": signed_alignment,
                "budget_gains": [float(actual[:count].max() - original) for count in candidate_counts],
            }
        )

    task_ids = [row["task_id"] for row in rows]
    gated = np.asarray([row["gated"] for row in rows], dtype=bool)
    budget = []
    for budget_index, (count, label) in enumerate(zip(candidate_counts, radius_labels)):
        gains = np.asarray([row["budget_gains"][budget_index] for row in rows])
        budget.append(
            {
                "candidates": count,
                "maximum_radius": label,
                "all_mean_gain": float(gains.mean()),
                "all_median_gain": float(np.median(gains)),
                "all_positive_rate": float(np.mean(gains > 1e-6)),
                "gated_mean_gain": float(gains[gated].mean()),
                "gated_large_gain_rate": float(np.mean(gains[gated] > 0.01)),
                "bootstrap": task_bootstrap(gains, task_ids, args.bootstrap_samples, args.seed + count),
            }
        )

    valid_indices = [
        index
        for index, row in enumerate(rows)
        if np.linalg.norm(row["approach"]) > 1e-8
        and np.linalg.norm(row["best_displacement"]) > 1e-8
    ]
    valid_approaches = [index for index, row in enumerate(rows) if np.linalg.norm(row["approach"]) > 1e-8]
    by_click: dict[int, list[int]] = {}
    for index in valid_approaches:
        by_click.setdefault(rows[index]["click_index"], []).append(index)
    rng = np.random.default_rng(args.seed)
    shuffled_alignment = []
    for index in valid_indices:
        row = rows[index]
        donors = [
            donor
            for donor in by_click.get(row["click_index"], valid_approaches)
            if rows[donor]["task_id"] != row["task_id"]
        ]
        if not donors:
            donors = [donor for donor in valid_approaches if rows[donor]["task_id"] != row["task_id"]]
        values = []
        for _ in range(args.shuffle_repeats):
            donor = int(rng.choice(donors))
            value, _ = alignment(row["best_displacement"], rows[donor]["approach"])
            values.append(value)
        shuffled_alignment.append(float(np.mean(values)))
    real_alignment = np.asarray([rows[index]["absolute_alignment"] for index in valid_indices])
    shuffled_alignment_array = np.asarray(shuffled_alignment)
    alignment_delta = real_alignment - shuffled_alignment_array
    alignment_tasks = [rows[index]["task_id"] for index in valid_indices]
    alignment_result = {
        "valid_records": len(valid_indices),
        "real_mean_absolute_cosine": float(real_alignment.mean()),
        "shuffled_mean_absolute_cosine": float(shuffled_alignment_array.mean()),
        "real_minus_shuffle": task_bootstrap(
            alignment_delta, alignment_tasks, args.bootstrap_samples, args.seed + 50_000
        ),
        "opposite_approach_rate": float(
            np.mean([rows[index]["signed_alignment"] < 0 for index in valid_indices])
        ),
    }
    large_indices = [index for index in valid_indices if rows[index]["grid_gain"] > 0.01]
    if large_indices:
        position = {record_index: offset for offset, record_index in enumerate(valid_indices)}
        large_real = np.asarray([rows[index]["absolute_alignment"] for index in large_indices])
        large_shuffle = np.asarray([shuffled_alignment[position[index]] for index in large_indices])
        alignment_result["large_gain_subset"] = {
            "records": len(large_indices),
            "real_mean_absolute_cosine": float(large_real.mean()),
            "shuffled_mean_absolute_cosine": float(large_shuffle.mean()),
            "real_minus_shuffle": task_bootstrap(
                large_real - large_shuffle,
                [rows[index]["task_id"] for index in large_indices],
                args.bootstrap_samples,
                args.seed + 60_000,
            ),
        }

    correlations = np.asarray(
        [row["quality_actual_correlation"] for row in rows if math.isfinite(row["quality_actual_correlation"])]
    )
    ranking = {
        "mean_within_click_quality_actual_correlation": float(correlations.mean()),
        "median_within_click_quality_actual_correlation": float(np.median(correlations)),
        "quality_top1_matches_oracle": float(
            np.mean([row["best_rank_by_quality"] == 1 for row in rows])
        ),
        "quality_top3_contains_oracle": float(
            np.mean([row["best_rank_by_quality"] <= 3 for row in rows])
        ),
        "quality_selected_mean_gain": float(
            np.mean([row["quality_selected_gain"] for row in rows])
        ),
        "mean_oracle_rank_by_quality": float(
            np.mean([row["best_rank_by_quality"] for row in rows])
        ),
    }

    original_iou_groups = {
        "iou_lt_0.6": subgroup(rows, lambda row: row["original_iou"] < 0.6),
        "iou_0.6_to_0.8": subgroup(rows, lambda row: 0.6 <= row["original_iou"] < 0.8),
        "iou_ge_0.8": subgroup(rows, lambda row: row["original_iou"] >= 0.8),
    }
    click_index_groups = {
        str(click_index): subgroup(rows, lambda row, value=click_index: row["click_index"] == value)
        for click_index in sorted({row["click_index"] for row in rows})
    }

    payload = {
        "config": {
            "candidate_results": str(args.candidate_results),
            "records": len(rows),
            "tasks": len(set(task_ids)),
            "bootstrap_samples": args.bootstrap_samples,
            "shuffle_repeats": args.shuffle_repeats,
        },
        "candidate_budget_ablation": budget,
        "trajectory_direction_alignment": alignment_result,
        "quality_ranking_diagnostic": ranking,
        "subgroups_by_original_iou": original_iou_groups,
        "subgroups_by_click_index": click_index_groups,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(7.0, 4.2))
    counts = [item["candidates"] for item in budget]
    plt.plot(counts, [100 * item["all_mean_gain"] for item in budget], marker="o", label="All hard clicks")
    plt.plot(counts, [100 * item["gated_mean_gain"] for item in budget], marker="s", label="Quality-gated subset")
    plt.xlabel("Candidate budget")
    plt.ylabel("Oracle IoU gain (percentage points)")
    plt.xticks(counts)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "candidate_budget_ablation.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5.5, 4.2))
    values = [
        alignment_result["real_mean_absolute_cosine"],
        alignment_result["shuffled_mean_absolute_cosine"],
    ]
    plt.bar(["Real trajectory", "Shuffled control"], values, color=["#4472C4", "#A5A5A5"])
    plt.ylabel("|cos(best correction, trajectory)|")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(args.output_dir / "trajectory_direction_alignment.png", dpi=180)
    plt.close()

    lines = [
        "# CAIS supplementary candidate-space analysis",
        "",
        "## Candidate-budget / radius ablation (GT upper bound)",
        "",
        "| Candidates | Maximum radius | All mean gain | Gated mean gain | Gated gain > 0.01 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in budget:
        lines.append(
            f"| {item['candidates']} | {item['maximum_radius']} | {item['all_mean_gain']:+.6f} | "
            f"{item['gated_mean_gain']:+.6f} | {item['gated_large_gain_rate']:.2%} |"
        )
    delta = alignment_result["real_minus_shuffle"]
    lines.extend(
        [
            "",
            "## Trajectory-direction mechanism check",
            "",
            f"- Valid corrections: {alignment_result['valid_records']}",
            f"- Real mean |cosine|: {alignment_result['real_mean_absolute_cosine']:.4f}",
            f"- Shuffled mean |cosine|: {alignment_result['shuffled_mean_absolute_cosine']:.4f}",
            f"- Real - Shuffle: {delta['mean']:+.4f}, 95% CI "
            f"[{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}]",
            f"- Best correction opposite to approach direction: {alignment_result['opposite_approach_rate']:.2%}",
            "",
            "## SAM quality-head ranking diagnostic",
            "",
            f"- Mean within-click quality/actual-IoU correlation: "
            f"{ranking['mean_within_click_quality_actual_correlation']:.4f}",
            f"- Quality Top-1 equals oracle point: {ranking['quality_top1_matches_oracle']:.2%}",
            f"- Quality Top-3 contains oracle point: {ranking['quality_top3_contains_oracle']:.2%}",
            f"- Mean oracle candidate rank by SAM quality: {ranking['mean_oracle_rank_by_quality']:.2f}",
            "",
            "## Subgroups by original IoU",
            "",
            "| Subgroup | N | Mean grid gain | Gain > 0.01 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, result in original_iou_groups.items():
        lines.append(
            f"| {name} | {result['records']} | {result['mean_grid_gain']:+.6f} | "
            f"{result['large_gain_rate']:.2%} |"
        )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
