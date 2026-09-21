#!/usr/bin/env python3
"""Merge prepared CAIS human datasets without crossing image splits."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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


def split_by_image(rows: list[dict[str, Any]], label: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in rows:
        image_id = int(row["coco_image_id"])
        split = str(row["split"])
        previous = result.setdefault(image_id, split)
        if previous != split:
            raise ValueError(f"{label} image {image_id} occurs in both {previous} and {split}")
    return result


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tasks": len({row["task_run_id"] for row in rows}),
        "clicks": len(rows),
        "images": len({int(row["coco_image_id"]) for row in rows}),
        "positive_clicks": sum(int(row["click_type"]) == 1 for row in rows),
        "negative_clicks": sum(int(row["click_type"]) == 0 for row in rows),
        "task_split_counts": dict(
            Counter(
                next(row["split"] for row in rows if row["task_run_id"] == task_id)
                for task_id in {row["task_run_id"] for row in rows}
            )
        ),
        "click_split_counts": dict(Counter(row["split"] for row in rows)),
        "image_split_counts": dict(
            Counter(split_by_image(rows, "combined").values())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--add-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--add-label", default="added")
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_root}")
    base_rows = read_jsonl(args.base_root / "alignment" / "modeling_clicks_strict.jsonl")
    add_rows = read_jsonl(args.add_root / "alignment" / "modeling_clicks_strict.jsonl")
    base_splits = split_by_image(base_rows, args.base_label)
    add_splits = split_by_image(add_rows, args.add_label)

    for key in ("event_id", "task_run_id"):
        base_values = {row[key] for row in base_rows}
        add_values = {row[key] for row in add_rows}
        overlap = base_values & add_values
        if overlap:
            raise ValueError(f"{key} collision across datasets: {len(overlap)}")

    shared_images = set(base_splits) & set(add_splits)
    split_conflicts = {
        image_id for image_id in shared_images if base_splits[image_id] != add_splits[image_id]
    }
    combined_rows = []
    for rows, label in ((base_rows, args.base_label), (add_rows, args.add_label)):
        for original in rows:
            row = dict(original)
            image_id = int(row["coco_image_id"])
            row["split"] = base_splits[image_id] if image_id in base_splits else add_splits[image_id]
            row["source"] = "human"
            row["human_dataset"] = label
            row["cognitive_supervision"] = True
            combined_rows.append(row)

    combined_splits = split_by_image(combined_rows, "combined")
    write_jsonl(args.output_root / "alignment" / "modeling_clicks_strict.jsonl", combined_rows)
    (args.output_root / "splits").mkdir(parents=True, exist_ok=True)
    (args.output_root / "splits" / "image_splits.json").write_text(
        json.dumps(
            {
                "strategy": "preserve_prepared_image_splits; base wins shared-image conflicts",
                "images": [
                    {"coco_image_id": image_id, "split": split}
                    for image_id, split in sorted(combined_splits.items())
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "1.0",
        "inputs": {
            args.base_label: {"root": str(args.base_root), **describe(base_rows)},
            args.add_label: {"root": str(args.add_root), **describe(add_rows)},
        },
        "overlap": {
            "images": len(shared_images),
            "annotations": len(
                {int(row["annotation_id"]) for row in base_rows}
                & {int(row["annotation_id"]) for row in add_rows}
            ),
            "split_conflicts_resolved_to_base": len(split_conflicts),
        },
        "combined": describe(combined_rows),
        "leakage_check": {
            "one_split_per_image": len(combined_splits)
            == sum(len({int(row["coco_image_id"]) for row in combined_rows if row["split"] == split}) for split in ("train", "val", "test")),
            "event_id_collisions": 0,
            "task_run_id_collisions": 0,
        },
    }
    (args.output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
