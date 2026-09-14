"""Full COCO point-prompt baseline at multiple click checkpoints.

This evaluator encodes every prompted mask as a COCO segmentation result so
that prompt-conditioned mask AP can be reported alongside instance-averaged
mIoU.  Image features are computed once per image and the 1/3/5-click results
are collected from one five-click interaction sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from applications.efficientvit_sam.sam_eval_utils import Clicker, iou
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=ROOT_DIR / "dataset/coco/val2017")
    parser.add_argument(
        "--annotation-json",
        type=Path,
        default=ROOT_DIR / "dataset/coco/annotations/instances_val2017.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt",
    )
    parser.add_argument("--model", default="efficientvit-sam-l0")
    parser.add_argument("--clicks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "outputs/coco_click_baseline_full")
    parser.add_argument("--max-images", type=int, default=0, help="0 evaluates all COCO images")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def summarize_miou(rows: list[dict[str, float]]) -> dict[str, float]:
    groups = {
        "mIoU": rows,
        "mIoUS": [row for row in rows if row["area"] < 32**2],
        "mIoUM": [row for row in rows if 32**2 <= row["area"] < 96**2],
        "mIoUL": [row for row in rows if row["area"] >= 96**2],
    }
    return {
        name: float(np.mean([row["iou"] for row in group])) if group else float("nan")
        for name, group in groups.items()
    }


def encode_coco_result(mask: np.ndarray, annotation: dict, score: float) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return {
        "image_id": int(annotation["image_id"]),
        "category_id": int(annotation["category_id"]),
        "segmentation": rle,
        "score": float(score),
    }


def evaluate_coco_ap(coco_gt: COCO, predictions: list[dict], image_ids: list[int]) -> dict[str, float]:
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "segm")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "mAP": float(evaluator.stats[0] * 100.0),
        "APS": float(evaluator.stats[3] * 100.0),
        "APM": float(evaluator.stats[4] * 100.0),
        "APL": float(evaluator.stats[5] * 100.0),
    }


def save_results(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "click",
        "mIoU",
        "mIoUS",
        "mIoUM",
        "mIoUL",
        "mAP",
        "APS",
        "APM",
        "APL",
        "Params(M)",
        "MACs(G)",
        "DocumentedThroughput(image/s)",
        "MeasuredPipelineThroughput(image/s)",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["table"]:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    clicks = sorted(set(args.clicks))
    if not clicks or clicks[0] < 1:
        raise ValueError("--clicks must contain positive integers")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    coco = COCO(str(args.annotation_json))
    image_records = sorted(coco.loadImgs(coco.getImgIds()), key=lambda record: record["id"])
    if args.max_images > 0:
        image_records = image_records[: args.max_images]
    image_ids = [int(record["id"]) for record in image_records]

    model = create_efficientvit_sam_model(args.model, pretrained=True, weight_url=str(args.checkpoint))
    model = model.to(torch.device(args.device)).eval()
    predictor = EfficientViTSamPredictor(model)
    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6

    miou_rows: dict[int, list[dict[str, float]]] = {click: [] for click in clicks}
    coco_predictions: dict[int, list[dict]] = {click: [] for click in clicks}
    instance_count = 0
    inference_start = time.perf_counter()
    for image_record in tqdm(image_records, desc="COCO multi-click baseline", dynamic_ncols=True):
        image_path = args.image_root / image_record["file_name"]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        predictor.set_image(image)
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_record["id"]]))
        for annotation in annotations:
            if annotation["area"] < 1:
                continue
            target = coco.annToMask(annotation).astype(bool)
            clicker = Clicker(gt_mask=target)
            prediction = np.zeros_like(target)
            point_coords: list[tuple[int, int]] = []
            point_labels: list[int] = []
            for click in range(1, clicks[-1] + 1):
                clicker.make_next_click(prediction)
                point_coords.append(clicker.clicks_list[-1].coords[::-1])
                point_labels.append(int(clicker.clicks_list[-1].is_positive))
                masks, quality_scores, _ = predictor.predict(
                    point_coords=np.asarray(point_coords),
                    point_labels=np.asarray(point_labels),
                    box=None,
                    multimask_output=True,
                )
                selected = int(quality_scores.argmax())
                prediction = masks[selected]
                if click in miou_rows:
                    miou_rows[click].append(
                        {"area": float(annotation["area"]), "iou": iou(prediction, target)}
                    )
                    coco_predictions[click].append(
                        encode_coco_result(prediction, annotation, float(quality_scores[selected]))
                    )
            instance_count += 1
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    measured_throughput = len(image_records) / inference_seconds

    metrics = {}
    for click in clicks:
        print(f"\nCOCOeval for click={click}", flush=True)
        metrics[str(click)] = {
            **summarize_miou(miou_rows[click]),
            **evaluate_coco_ap(coco, coco_predictions[click], image_ids),
        }
        print(json.dumps({"click": click, **metrics[str(click)]}, indent=2), flush=True)

    complexity = {
        "Params(M)": params_m,
        "MACs(G)": 35.0,
        "DocumentedThroughput(image/s)": 762.0,
        "MeasuredPipelineThroughput(image/s)": measured_throughput,
        "documented_protocol": "EfficientViT-SAM README: 512x512, A100 TensorRT fp16, batch 16",
        "measured_protocol": (
            f"{args.device}, PyTorch, full image loading/encoding, all GT instances, "
            f"click checkpoints {clicks} in one {clicks[-1]}-click sequence"
        ),
    }
    table = []
    for click in clicks:
        table.append({"click": click, **metrics[str(click)], **{key: complexity[key] for key in (
            "Params(M)",
            "MACs(G)",
            "DocumentedThroughput(image/s)",
            "MeasuredPipelineThroughput(image/s)",
        )}})
    payload = {
        "protocol": {
            "dataset": "COCO 2017 val",
            "model": args.model,
            "checkpoint": str(args.checkpoint.resolve().relative_to(ROOT_DIR)),
            "images": len(image_records),
            "instances": instance_count,
            "clicks": clicks,
            "prompt_policy": "largest-error corrective click; all clicks accumulated",
            "mask_selection": "highest SAM predicted IoU among multimask outputs",
            "ap_definition": (
                "COCO segmentation AP for GT-instance-conditioned point-prompt masks; "
                "GT category and SAM predicted IoU score are used"
            ),
            "inference_seconds": inference_seconds,
        },
        "metrics": metrics,
        "complexity": complexity,
        "table": table,
    }
    save_results(args.output, payload)
    print("\nFinal table", flush=True)
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()
