#!/usr/bin/env python3
"""Prepare and audit a raw WisdomStore COCO-Click export for CAIS training."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def grouped_split(image_ids: Iterable[int], seed: int = 9527) -> dict[int, str]:
    shuffled = sorted({int(image_id) for image_id in image_ids})
    random.Random(seed).shuffle(shuffled)
    train_count = round(len(shuffled) * 0.70)
    val_count = round(len(shuffled) * 0.15)
    return {
        image_id: "train" if index < train_count else "val" if index < train_count + val_count else "test"
        for index, image_id in enumerate(shuffled)
    }


def resolve_assets(project_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(row)
    resolved["image_ref"] = str((project_root / row["image_ref"]).resolve())
    for key in ("pre_mask_ref", "post_mask_ref"):
        value = row.get(key)
        resolved[key] = str((project_root / "interaction_data" / value).resolve()) if value else None
    return resolved


def validate_assets(rows: list[dict[str, Any]]) -> dict[str, int]:
    issues: Counter[str] = Counter()
    image_sizes: dict[str, tuple[int, int]] = {}
    for row in rows:
        expected = (int(row["image_width_px"]), int(row["image_height_px"]))
        image_path = Path(row["image_ref"])
        if not image_path.is_file():
            issues["missing_image"] += 1
        else:
            key = str(image_path)
            if key not in image_sizes:
                with Image.open(image_path) as image:
                    image_sizes[key] = image.size
            if image_sizes[key] != expected:
                issues["image_size_mismatch"] += 1
        for key in ("pre_mask_ref", "post_mask_ref"):
            mask_path = Path(row[key]) if row.get(key) else None
            if mask_path is None or not mask_path.is_file():
                issues[f"missing_{key}"] += 1
                continue
            with Image.open(mask_path) as mask:
                if mask.size != expected:
                    issues[f"{key}_size_mismatch"] += 1
    return dict(issues)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--raw-export-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9527)
    parser.add_argument("--minimum-final-iou", type=float, default=0.75)
    args = parser.parse_args()

    clicks = read_jsonl(args.raw_export_root / "modeling_clicks.jsonl")
    tasks = read_jsonl(args.raw_export_root / "difficulty_targets.jsonl")
    raw_clicks = read_jsonl(args.project_root / "interaction_data" / "point_interactions.jsonl")
    task_events = read_jsonl(args.project_root / "interaction_data" / "interaction_task_events.jsonl")
    clicks_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clicks:
        clicks_by_run[row["task_run_id"]].append(row)

    invalid_sequence_runs = set()
    negative_inside_runs = set()
    positive_outside_runs = set()
    for run_id, rows in clicks_by_run.items():
        click_indices = [int(row.get("click_index") or 0) for row in rows]
        if len(click_indices) != len(set(click_indices)) or any(
            len(row.get("prompt_history") or []) != int(row.get("click_index") or 0) for row in rows
        ):
            invalid_sequence_runs.add(run_id)
        if any(row.get("click_type") == 0 and row.get("click_inside_ground_truth") for row in rows):
            negative_inside_runs.add(run_id)
        if any(row.get("click_type") == 1 and not row.get("click_inside_ground_truth") for row in rows):
            positive_outside_runs.add(run_id)

    normal_tasks = [
        row
        for row in tasks
        if row.get("termination") in {"accepted", "max_clicks"} and row.get("final_iou") is not None
    ]
    high_quality_tasks = [row for row in normal_tasks if float(row["final_iou"]) >= args.minimum_final_iou]
    excluded_semantic_runs = invalid_sequence_runs | negative_inside_runs | positive_outside_runs
    strict_tasks = [row for row in high_quality_tasks if row["task_run_id"] not in excluded_semantic_runs]
    strict_run_ids = {row["task_run_id"] for row in strict_tasks}
    valid_run_ids = {row["task_run_id"] for row in normal_tasks}

    split_by_image = grouped_split((row["coco_image_id"] for row in normal_tasks), seed=args.seed)

    def prepare_click(row: dict[str, Any], tier: str) -> dict[str, Any]:
        prepared = resolve_assets(args.project_root, row)
        prepared["split"] = split_by_image[int(row["coco_image_id"])]
        prepared["dataset_tier"] = tier
        return prepared

    valid_clicks = [prepare_click(row, "valid") for row in clicks if row["task_run_id"] in valid_run_ids]
    strict_clicks = [prepare_click(row, "strict") for row in clicks if row["task_run_id"] in strict_run_ids]
    asset_issues = validate_assets(strict_clicks)
    if asset_issues:
        raise RuntimeError(f"Strict-set asset validation failed: {asset_issues}")

    starts = Counter(row["task_run_id"] for row in task_events if row.get("action_type") == "interaction_task_started")
    finishes = Counter(
        row["task_run_id"] for row in task_events if row.get("action_type") == "interaction_task_finished"
    )
    linked_event_ids = {row["event_id"] for row in clicks}
    exclusions = []
    for task in tasks:
        run_id = task["task_run_id"]
        reasons = []
        if task.get("termination") not in {"accepted", "max_clicks"}:
            reasons.append(f"termination:{task.get('termination')}")
        if task.get("final_iou") is None:
            reasons.append("missing_final_iou")
        elif float(task["final_iou"]) < args.minimum_final_iou:
            reasons.append("final_iou_below_threshold")
        if run_id in invalid_sequence_runs:
            reasons.append("invalid_click_sequence")
        if run_id in negative_inside_runs:
            reasons.append("negative_click_inside_ground_truth")
        if run_id in positive_outside_runs:
            reasons.append("positive_click_outside_ground_truth")
        if reasons:
            exclusions.append({"task_run_id": run_id, "annotation_id": task.get("annotation_id"), "reasons": reasons})

    split_rows = [{"coco_image_id": image_id, "split": split} for image_id, split in sorted(split_by_image.items())]
    strict_task_split_counts = Counter(split_by_image[int(row["coco_image_id"])] for row in strict_tasks)
    strict_click_split_counts = Counter(row["split"] for row in strict_clicks)
    summary = {
        "schema_version": "2.0",
        "source_project": str(args.project_root.resolve()),
        "raw_counts": {
            "point_interactions": len(raw_clicks),
            "coco_linked_clicks": len(clicks),
            "task_summaries": len(tasks),
            "unlinked_clicks": len(raw_clicks) - len(linked_event_ids),
        },
        "quality": {
            "normal_tasks": len(normal_tasks),
            "high_quality_tasks": len(high_quality_tasks),
            "invalid_sequence_high_quality_tasks": len(
                {row["task_run_id"] for row in high_quality_tasks} & invalid_sequence_runs
            ),
            "negative_inside_high_quality_tasks": len(
                {row["task_run_id"] for row in high_quality_tasks} & negative_inside_runs
            ),
            "positive_outside_high_quality_tasks": len(
                {row["task_run_id"] for row in high_quality_tasks} & positive_outside_runs
            ),
            "missing_task_finish": len(set(starts) - set(finishes)),
            "missing_task_start": len(set(finishes) - set(starts)),
            "strict_asset_issues": asset_issues,
        },
        "strict": {
            "tasks": len(strict_tasks),
            "clicks": len(strict_clicks),
            "unique_images": len({row["coco_image_id"] for row in strict_tasks}),
            "categories": len({row["category_id"] for row in strict_tasks}),
            "annotators": len({row["annotator_id"] for row in strict_tasks}),
            "task_split_counts": dict(strict_task_split_counts),
            "click_split_counts": dict(strict_click_split_counts),
            "positive_clicks": sum(row.get("click_type") == 1 for row in strict_clicks),
            "negative_clicks": sum(row.get("click_type") == 0 for row in strict_clicks),
            "category_task_counts": dict(Counter(row["category_name"] for row in strict_tasks)),
        },
        "selection_rules": {
            "normal": "termination in {accepted,max_clicks} and final_iou exists",
            "high_quality": f"normal and final_iou >= {args.minimum_final_iou}",
            "strict": (
                "high_quality; unique click indices; prompt-history length matches click index; "
                "no negative click inside GT; no positive click outside GT"
            ),
            "split": f"grouped by coco_image_id, 70/15/15, seed {args.seed}",
        },
    }

    write_jsonl(args.output_root / "alignment" / "modeling_clicks_valid.jsonl", valid_clicks)
    write_jsonl(args.output_root / "alignment" / "modeling_clicks_strict.jsonl", strict_clicks)
    write_jsonl(args.output_root / "difficulty" / "difficulty_targets_valid.jsonl", normal_tasks)
    write_jsonl(args.output_root / "difficulty" / "difficulty_targets_high_quality.jsonl", high_quality_tasks)
    write_jsonl(args.output_root / "exclusions" / "excluded_tasks.jsonl", exclusions)
    (args.output_root / "splits").mkdir(parents=True, exist_ok=True)
    (args.output_root / "splits" / "image_splits.json").write_text(
        json.dumps(
            {
                "strategy": "grouped_by_coco_image_id",
                "seed": args.seed,
                "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
                "images": split_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
