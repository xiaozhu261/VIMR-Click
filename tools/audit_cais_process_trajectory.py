#!/usr/bin/env python3
"""Audit how much of each pre-click interaction trajectory was actually recorded."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def load_records(dataset_root: Path) -> list[dict[str, Any]]:
    path = dataset_root / "alignment" / "modeling_clicks_strict.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip() and json.loads(line).get("source", "human") == "human"
        ]


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def click_xy(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [finite(record.get("click_x_norm")), finite(record.get("click_y_norm"))],
        dtype=np.float64,
    )


def trajectory_row(record: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    trajectory = record.get("preceding_trajectory") or {}
    raw_points = trajectory.get("points") or []
    points = np.asarray(raw_points, dtype=np.float64)
    if points.size == 0:
        points = np.zeros((0, 3), dtype=np.float64)
    elif points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid trajectory shape for {record.get('event_id')}: {points.shape}")
    if len(points):
        order = np.argsort(points[:, 0], kind="stable")
        points = points[order]
        _, unique_time_indices = np.unique(points[:, 0], return_index=True)
        points = points[np.sort(unique_time_indices)]
    xy = points[:, 1:3]
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.zeros(0)
    path_length = float(segment_lengths.sum())
    observed_displacement = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else 0.0
    observed_duration_ms = (
        max(float(points[-1, 0] - points[0, 0]), 0.0)
        if len(points) > 1
        else max(finite(trajectory.get("total_duration_ms")), 0.0)
    )
    current_xy = click_xy(record)
    end_error = float(np.linalg.norm(xy[-1] - current_xy)) if len(xy) else math.nan
    available_interval_ms = math.nan
    previous_click_distance = math.nan
    start_from_previous_error = math.nan
    coverage_ratio = math.nan
    if previous is not None:
        available_interval_ms = max(
            finite(record.get("click_timestamp_ms")) - finite(previous.get("click_timestamp_ms")),
            0.0,
        )
        previous_xy = click_xy(previous)
        previous_click_distance = float(np.linalg.norm(current_xy - previous_xy))
        start_from_previous_error = (
            float(np.linalg.norm(xy[0] - previous_xy)) if len(xy) else math.nan
        )
        if available_interval_ms > 1e-6:
            coverage_ratio = observed_duration_ms / available_interval_ms

    rounded_unique_xy = len({(round(float(x), 6), round(float(y), 6)) for x, y in xy})
    enough_points = len(points) >= 5
    enough_duration = observed_duration_ms >= 150.0
    has_motion = rounded_unique_xy >= 3 and path_length >= 0.005
    reaches_click = math.isfinite(end_error) and end_error <= 0.01
    if previous is None:
        adequate_coverage = enough_duration
    else:
        adequate_coverage = (
            math.isfinite(coverage_ratio)
            and coverage_ratio >= 0.25
            and (
                previous_click_distance < 0.03
                or (
                    math.isfinite(start_from_previous_error)
                    and start_from_previous_error <= max(0.05, previous_click_distance * 0.5)
                )
            )
        )
    reliable = enough_points and enough_duration and has_motion and reaches_click and adequate_coverage
    missing_interclick_motion = bool(
        previous is not None
        and previous_click_distance >= 0.03
        and (
            not has_motion
            or not math.isfinite(start_from_previous_error)
            or start_from_previous_error > max(0.05, previous_click_distance * 0.5)
        )
    )
    return {
        "event_id": record.get("event_id"),
        "task_run_id": record.get("task_run_id"),
        "human_dataset": record.get("human_dataset", record.get("task_id", "unknown")),
        "split": record.get("split"),
        "click_index": int(record.get("click_index", 0)),
        "click_type": int(record.get("click_type", -1)),
        "raw_point_count": int(len(raw_points)),
        "unique_time_count": int(len(points)),
        "unique_xy_count": int(rounded_unique_xy),
        "observed_duration_ms": observed_duration_ms,
        "available_interval_ms": available_interval_ms,
        "coverage_ratio": coverage_ratio,
        "path_length_norm": path_length,
        "observed_displacement_norm": observed_displacement,
        "previous_click_distance_norm": previous_click_distance,
        "start_from_previous_error_norm": start_from_previous_error,
        "end_to_click_error_norm": end_error,
        "enough_points": enough_points,
        "enough_duration": enough_duration,
        "has_motion": has_motion,
        "reaches_click": reaches_click,
        "adequate_coverage": adequate_coverage,
        "missing_interclick_motion": missing_interclick_motion,
        "reliable": reliable,
    }


def percentile(values: list[float], q: float) -> float | None:
    finite_values = [value for value in values if math.isfinite(value)]
    return float(np.quantile(finite_values, q)) if finite_values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {"clicks": 0}
    metric_names = (
        "raw_point_count",
        "unique_xy_count",
        "observed_duration_ms",
        "available_interval_ms",
        "coverage_ratio",
        "path_length_norm",
        "previous_click_distance_norm",
        "start_from_previous_error_norm",
        "end_to_click_error_norm",
    )
    flag_names = (
        "enough_points",
        "enough_duration",
        "has_motion",
        "reaches_click",
        "adequate_coverage",
        "missing_interclick_motion",
        "reliable",
    )
    result: dict[str, Any] = {"clicks": count}
    for name in metric_names:
        values = [finite(row[name], math.nan) for row in rows]
        result[name] = {
            "p10": percentile(values, 0.10),
            "median": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
        }
    for name in flag_names:
        positives = sum(bool(row[name]) for row in rows)
        result[name] = {"count": positives, "ratio": positives / count}
    result["three_or_fewer_points"] = {
        "count": sum(row["raw_point_count"] <= 3 for row in rows),
        "ratio": sum(row["raw_point_count"] <= 3 for row in rows) / count,
    }
    result["static_trajectory"] = {
        "count": sum(row["unique_xy_count"] <= 1 for row in rows),
        "ratio": sum(row["unique_xy_count"] <= 1 for row in rows) / count,
    }
    return result


def group_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "human_dataset": defaultdict(list),
        "split": defaultdict(list),
        "click_index": defaultdict(list),
        "human_dataset_click_index": defaultdict(list),
    }
    for row in rows:
        grouped["human_dataset"][row["human_dataset"]].append(row)
        grouped["split"][row["split"]].append(row)
        grouped["click_index"][row["click_index"]].append(row)
        key = f"{row['human_dataset']}:click_{row['click_index']}"
        grouped["human_dataset_click_index"][key].append(row)
    return {
        dimension: {str(key): summarize(group_rows) for key, group_rows in sorted(groups.items())}
        for dimension, groups in grouped.items()
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CAIS 标注过程轨迹质量审计",
        "",
        "可靠轨迹判据：至少 5 个采样点、150 ms、3 个不同空间位置、路径长度不小于 0.005，",
        "末点接近点击位置；后续点击还要求覆盖至少 25% 的点击间隔，且轨迹起点与上一点击位置相容。",
        "这些阈值用于质量分层，不用于删除原始数据。",
        "",
        "| 分组 | 点击数 | ≤3点 | 静止轨迹 | 疑似丢失跨点击运动 | 可靠轨迹 | 时长中位数(ms) | 覆盖率中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = {"全部真人": report["overall"]}
    groups.update(report["groups"]["human_dataset"])
    for name, item in groups.items():
        lines.append(
            "| {name} | {clicks} | {few:.1%} | {static:.1%} | {missing:.1%} | {reliable:.1%} | "
            "{duration:.1f} | {coverage} |".format(
                name=name,
                clicks=item["clicks"],
                few=item["three_or_fewer_points"]["ratio"],
                static=item["static_trajectory"]["ratio"],
                missing=item["missing_interclick_motion"]["ratio"],
                reliable=item["reliable"]["ratio"],
                duration=item["observed_duration_ms"]["median"] or 0.0,
                coverage=(
                    f"{item['coverage_ratio']['median']:.3f}"
                    if item["coverage_ratio"]["median"] is not None
                    else "-"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 按点击轮次",
            "",
            "| 轮次 | 点击数 | ≤3点 | 静止轨迹 | 疑似丢失跨点击运动 | 可靠轨迹 | 覆盖率中位数 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for click_index, item in report["groups"]["click_index"].items():
        lines.append(
            "| {idx} | {clicks} | {few:.1%} | {static:.1%} | {missing:.1%} | {reliable:.1%} | {coverage} |".format(
                idx=click_index,
                clicks=item["clicks"],
                few=item["three_or_fewer_points"]["ratio"],
                static=item["static_trajectory"]["ratio"],
                missing=item["missing_interclick_motion"]["ratio"],
                reliable=item["reliable"]["ratio"],
                coverage=(
                    f"{item['coverage_ratio']['median']:.3f}"
                    if item["coverage_ratio"]["median"] is not None
                    else "-"
                ),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    records = load_records(args.dataset_root)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record["task_run_id"])].append(record)
    rows: list[dict[str, Any]] = []
    for task_records in by_task.values():
        task_records.sort(key=lambda item: int(item["click_index"]))
        previous = None
        for record in task_records:
            rows.append(trajectory_row(record, previous))
            previous = record

    report = {
        "dataset_root": str(args.dataset_root.resolve()),
        "definitions": {
            "reliable": (
                "points>=5, duration>=150ms, unique_xy>=3, path>=0.005, end_error<=0.01; "
                "later clicks also require coverage>=0.25 and start consistency with prior click"
            ),
            "missing_interclick_motion": (
                "prior-to-current click distance>=0.03 but observed path has no motion or its start is "
                "inconsistent with the prior click"
            ),
        },
        "overall": summarize(rows),
        "groups": group_summaries(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trajectory_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "trajectory_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "trajectory_audit.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


if __name__ == "__main__":
    main()
