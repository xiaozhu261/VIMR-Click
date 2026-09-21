#!/usr/bin/env python3
"""Generate isolated CAIS training clicks from COCO GT and model errors."""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


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


def decode_annotation(annotation: dict[str, Any], height: int, width: int) -> np.ndarray:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return np.asarray(decoded, dtype=bool)


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(value) for value in encoded["size"]], "counts": counts}


def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    prediction_area = int(prediction.sum())
    target_area = int(target.sum())
    iou = intersection / max(union, 1)
    dice = 2 * intersection / max(prediction_area + target_area, 1)
    return {
        "iou": iou,
        "dice": dice,
        "intersection_px": intersection,
        "union_px": union,
        "prediction_area_px": prediction_area,
        "ground_truth_area_px": target_area,
    }


def largest_component_center(mask: np.ndarray) -> tuple[int, int, int] | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == component_index).astype(np.uint8)
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return int(x), int(y), int(stats[component_index, cv2.CC_STAT_AREA])


def choose_correction(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    prefer_negative: bool,
) -> tuple[int, int, int] | None:
    positive = largest_component_center(np.logical_and(target, ~prediction))
    negative = largest_component_center(np.logical_and(prediction, ~target))
    if prefer_negative and negative is not None:
        return negative[0], negative[1], 0
    if positive is None and negative is None:
        return None
    if positive is None:
        return negative[0], negative[1], 0
    if negative is None:
        return positive[0], positive[1], 1
    if negative[2] > positive[2]:
        return negative[0], negative[1], 0
    return positive[0], positive[1], 1


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def select_balanced_annotations(
    annotations: list[dict[str, Any]],
    images: dict[int, dict[str, Any]],
    available_file_names: set[str],
    excluded_image_ids: set[int],
    *,
    count: int,
    minimum_area: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        image = images.get(image_id)
        if (
            image is None
            or image_id in excluded_image_ids
            or annotation.get("iscrowd")
            or float(annotation.get("area", 0.0)) < minimum_area
            or image["file_name"] not in available_file_names
        ):
            continue
        by_category[int(annotation["category_id"])].append(annotation)
    for rows in by_category.values():
        rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    selected_images: set[int] = set()
    categories = sorted(by_category)
    while len(selected) < count:
        progress = False
        for category_id in categories:
            rows = by_category[category_id]
            while rows and int(rows[-1]["image_id"]) in selected_images:
                rows.pop()
            if not rows:
                continue
            annotation = rows.pop()
            selected.append(annotation)
            selected_images.add(int(annotation["image_id"]))
            progress = True
            if len(selected) == count:
                break
        if not progress:
            break
    if len(selected) < count:
        raise RuntimeError(f"Only found {len(selected)} eligible unique-image annotations, requested {count}")
    return selected


def create_montage(records: list[dict[str, Any]], output_root: Path, path: Path, seed: int) -> None:
    final_records: dict[str, dict[str, Any]] = {}
    for record in records:
        final_records[record["task_run_id"]] = record
    rows = list(final_records.values())
    random.Random(seed).shuffle(rows)
    rows = rows[:16]
    tile_width, tile_height = 320, 240
    canvas = Image.new("RGB", (tile_width * 4, tile_height * 4), "white")
    for index, record in enumerate(rows):
        image = np.asarray(Image.open(record["image_ref"]).convert("RGB"))
        target = mask_utils.decode(record["ground_truth_rle"]).astype(bool)
        prediction = np.asarray(Image.open(output_root / record["post_mask_ref"]).convert("L")) > 0
        overlay = image.copy()
        overlay[prediction] = (0.65 * overlay[prediction] + 0.35 * np.array([255, 32, 32])).astype(np.uint8)
        contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (32, 255, 32), 2)
        tile = Image.fromarray(overlay).resize((tile_width, tile_height), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(tile)
        for prompt in record["prompt_history"]:
            x = int(float(prompt["x_norm"]) * tile_width)
            y = int(float(prompt["y_norm"]) * tile_height)
            color = (20, 255, 20) if int(prompt["click_type"]) == 1 else (255, 40, 40)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=1)
        label = f"{record['category_name']}  n={record['click_index']}  IoU={record['post_mask_metrics']['iou']:.3f}"
        draw.rectangle((0, 0, tile_width, 18), fill=(0, 0, 0))
        draw.text((4, 2), label, fill="white")
        canvas.paste(tile, ((index % 4) * tile_width, (index // 4) * tile_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--human-dataset-root", type=Path, required=True)
    parser.add_argument("--exclude-jsonl", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=200)
    parser.add_argument("--max-clicks", type=int, default=5)
    parser.add_argument("--target-iou", type=float, default=0.95)
    parser.add_argument("--negative-target-ratio", type=float, default=0.25)
    parser.add_argument("--minimum-area", type=float, default=1024.0)
    parser.add_argument("--seed", type=int, default=9527)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.instances < 1 or args.max_clicks < 1:
        raise ValueError("instances and max-clicks must be positive")
    if not 0 <= args.negative_target_ratio <= 1 or not 0 < args.target_iou <= 1:
        raise ValueError("ratios and target-iou must be within [0, 1]")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_root}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in coco["images"]}
    categories = {int(row["id"]): row["name"] for row in coco["categories"]}
    available_file_names = {path.name for path in args.image_root.glob("*.jpg")}
    human_path = args.human_dataset_root / "alignment" / "modeling_clicks_strict.jsonl"
    human_records = read_jsonl(human_path)
    excluded_image_ids = {int(row["coco_image_id"]) for row in human_records}
    if args.exclude_jsonl is not None:
        excluded_image_ids.update(int(row["coco_image_id"]) for row in read_jsonl(args.exclude_jsonl))
    selected = select_balanced_annotations(
        coco["annotations"],
        images,
        available_file_names,
        excluded_image_ids,
        count=args.instances,
        minimum_area=args.minimum_area,
        seed=args.seed,
    )

    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    model = create_efficientvit_sam_model(
        "efficientvit-sam-l0",
        pretrained=True,
        weight_url=str(args.checkpoint),
        image_size=512,
    ).to(device)
    model.eval()
    predictor = EfficientViTSamPredictor(model)

    synthetic_records: list[dict[str, Any]] = []
    task_final_ious: list[float] = []
    positive_count = 0
    negative_count = 0
    started = time.perf_counter()
    for task_number, annotation in enumerate(selected, start=1):
        image_info = images[int(annotation["image_id"])]
        image_path = (args.image_root / image_info["file_name"]).resolve()
        image = np.asarray(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        target = decode_annotation(annotation, height, width)
        if not target.any():
            continue
        initial = largest_component_center(target)
        if initial is None:
            continue
        task_run_id = f"synthetic-coco-ann-{int(annotation['id'])}"
        task_mask_root = args.output_root / "synthetic" / "masks" / task_run_id
        previous_prediction = np.zeros_like(target)
        previous_mask_path = task_mask_root / "step_0000.png"
        save_mask(previous_mask_path, previous_prediction)
        points: list[tuple[int, int]] = [(initial[0], initial[1])]
        labels: list[int] = [1]
        low_res_logits = None
        prompt_history: list[dict[str, Any]] = []

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            predictor.set_image(image)
        for click_index in range(1, args.max_clicks + 1):
            x, y = points[-1]
            click_type = labels[-1]
            if click_type == 1:
                positive_count += 1
            else:
                negative_count += 1
            prompt_history.append(
                {
                    "x_norm": x / width,
                    "y_norm": y / height,
                    "prompt_type": "positive" if click_type == 1 else "negative",
                    "click_type": click_type,
                }
            )
            before = mask_metrics(previous_prediction, target)
            inference_started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                masks, quality, low_res_logits = predictor.predict(
                    point_coords=np.asarray(points, dtype=np.float32),
                    point_labels=np.asarray(labels, dtype=np.int64),
                    mask_input=low_res_logits,
                    multimask_output=False,
                    return_logits=False,
                )
            prediction = np.asarray(masks[0], dtype=bool)
            latency_ms = (time.perf_counter() - inference_started) * 1000.0
            after = mask_metrics(prediction, target)
            post_mask_path = task_mask_root / f"step_{click_index:04d}.png"
            save_mask(post_mask_path, prediction)
            record = {
                "event_id": f"synthetic-{int(annotation['id'])}-{click_index:02d}",
                "task_id": "cais_synthetic_coco_error_center",
                "task_run_id": task_run_id,
                "annotator_id": "simulator:error_center",
                "source": "synthetic",
                "synthetic_policy": "largest_component_distance_center_with_negative_quota",
                "cognitive_supervision": False,
                "click_type": click_type,
                "click_index": click_index,
                "prompt_type": "positive" if click_type == 1 else "negative",
                "image_ref": str(image_path),
                "ground_truth_ref": str(args.annotations.resolve()),
                "coco_image_id": int(annotation["image_id"]),
                "annotation_id": int(annotation["id"]),
                "category_id": int(annotation["category_id"]),
                "category_name": categories[int(annotation["category_id"])],
                "image_file_name": image_info["file_name"],
                "image_width_px": width,
                "image_height_px": height,
                "ground_truth_encoding": "coco_rle",
                "ground_truth_rle": encode_mask(target),
                "click_x_norm": x / width,
                "click_y_norm": y / height,
                "click_inside_ground_truth": bool(target[y, x]),
                "preceding_trajectory": {"total_duration_ms": 0.0, "points": []},
                "prompt_history": list(prompt_history),
                "pre_mask_ref": str(previous_mask_path.relative_to(args.output_root)),
                "post_mask_ref": str(post_mask_path.relative_to(args.output_root)),
                "pre_mask_metrics": before,
                "post_mask_metrics": after,
                "iou_gain": float(after["iou"] - before["iou"]),
                "predicted_iou": float(np.asarray(quality).reshape(-1)[0]),
                "inference_latency_ms": latency_ms,
                "seeking_time_ms": 0.0,
                "pre_click_hover_time_ms": 0.0,
                "hard_brake_count": 0,
                "direction_changes_last_500ms": 0,
                "mean_deceleration_norm_per_s2": 0.0,
                "split": "train",
                "dataset_tier": "synthetic",
            }
            synthetic_records.append(record)
            previous_prediction = prediction
            previous_mask_path = post_mask_path
            if after["iou"] >= args.target_iou:
                break
            ratio = negative_count / max(positive_count + negative_count, 1)
            correction = choose_correction(
                target,
                prediction,
                prefer_negative=ratio < args.negative_target_ratio,
            )
            if correction is None:
                break
            points.append((correction[0], correction[1]))
            labels.append(correction[2])
        task_final_ious.append(float(after["iou"]))
        predictor.reset_image()
        if task_number == 1 or task_number % 20 == 0 or task_number == len(selected):
            print(
                f"generated {task_number:04d}/{len(selected):04d} tasks, "
                f"clicks={len(synthetic_records)}, negative={negative_count}, "
                f"mean_final_iou={np.mean(task_final_ious):.4f}, "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    mixed_human_records = []
    for row in human_records:
        row = dict(row)
        row.setdefault("source", "human")
        row.setdefault("cognitive_supervision", True)
        mixed_human_records.append(row)
    mixed_records = mixed_human_records + synthetic_records
    write_jsonl(args.output_root / "synthetic" / "modeling_clicks_synthetic.jsonl", synthetic_records)
    write_jsonl(args.output_root / "alignment" / "modeling_clicks_strict.jsonl", mixed_records)
    create_montage(
        synthetic_records,
        args.output_root,
        args.output_root / "synthetic" / "qa_montage.jpg",
        args.seed,
    )

    click_type_counts = Counter(int(row["click_type"]) for row in synthetic_records)
    category_counts = Counter(row["category_name"] for row in synthetic_records if int(row["click_index"]) == 1)
    human_split_counts = Counter(row["split"] for row in mixed_human_records)
    summary = {
        "schema_version": "1.0",
        "policy": "model-in-the-loop COCO GT error-center simulation",
        "cognitive_policy": "synthetic trajectories are empty and cognitive supervision is disabled",
        "selection": {
            "requested_instances": args.instances,
            "generated_tasks": len(task_final_ious),
            "excluded_human_image_ids": len(excluded_image_ids),
            "unique_images": len({row["coco_image_id"] for row in synthetic_records}),
            "categories": len(category_counts),
            "category_tasks": dict(sorted(category_counts.items())),
        },
        "synthetic": {
            "clicks": len(synthetic_records),
            "positive_clicks": click_type_counts[1],
            "negative_clicks": click_type_counts[0],
            "negative_ratio": click_type_counts[0] / max(len(synthetic_records), 1),
            "mean_clicks_per_task": len(synthetic_records) / max(len(task_final_ious), 1),
            "mean_final_iou": float(np.mean(task_final_ious)),
            "median_final_iou": float(np.median(task_final_ious)),
            "minimum_final_iou": float(np.min(task_final_ious)),
            "target_iou": args.target_iou,
            "max_clicks": args.max_clicks,
        },
        "mixed": {
            "human_clicks": len(mixed_human_records),
            "synthetic_clicks": len(synthetic_records),
            "total_clicks": len(mixed_records),
            "train_clicks": human_split_counts["train"] + len(synthetic_records),
            "val_clicks": human_split_counts["val"],
            "test_clicks": human_split_counts["test"],
            "cognitively_supervised_clicks": len(mixed_human_records),
        },
        "artifacts": {
            "synthetic_jsonl": "synthetic/modeling_clicks_synthetic.jsonl",
            "mixed_jsonl": "alignment/modeling_clicks_strict.jsonl",
            "qa_montage": "synthetic/qa_montage.jpg",
        },
    }
    (args.output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
