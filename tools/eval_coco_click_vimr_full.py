"""Evaluate a trained VIMR checkpoint on the full COCO val2017 click protocol.

The image set, GT instances, corrective-click policy, mIoU area bins and
prompt-conditioned COCO segmentation AP definition match
``eval_coco_click_baseline_full.py``.  The frozen baseline results are loaded
from that evaluator so this run only performs the additional VIMR inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from applications.efficientvit_sam.sam_eval_utils import Clicker, iou
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from tools.eval_coco_click_baseline_full import (
    encode_coco_result,
    evaluate_coco_ap,
    summarize_miou,
)
from tools.eval_vimr_coco_quick import freeze_pretrained_modules, viewport_for_mode
from tools.run_vimr_coco_protocol import build_model, load_vimr_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=ROOT_DIR / "dataset/coco/val2017")
    parser.add_argument(
        "--annotation-json",
        type=Path,
        default=ROOT_DIR / "dataset/coco/annotations/instances_val2017.json",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ROOT_DIR / "assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt",
    )
    parser.add_argument(
        "--vimr-checkpoint",
        type=Path,
        default=ROOT_DIR / "outputs/vimr_coco_full/seed_42/best_vimr_modules.pt",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=ROOT_DIR / "outputs/coco_click_baseline_full/results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "outputs/coco_click_vimr_seed42_full",
    )
    parser.add_argument("--clicks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--max-images", type=int, default=0, help="0 evaluates all COCO images")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--viewport-mode", choices=("synthetic", "default"), default="synthetic")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run_vimr_interaction(
    model: torch.nn.Module,
    predictor: EfficientViTSamPredictor,
    target: np.ndarray,
    clicks: list[int],
    viewport_mode: str,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Run one target-conditioned interaction in the original-image frame."""
    clicker = Clicker(gt_mask=target)
    prediction = np.zeros_like(target, dtype=bool)
    click_events: list[torch.Tensor] = []
    viewports: list[torch.Tensor] = []
    mask_history: list[torch.Tensor] = []
    previous: torch.Tensor | None = None
    masks_at_click: dict[int, np.ndarray] = {}
    scores_at_click: dict[int, float] = {}
    device = predictor.features.device
    dtype = predictor.features.dtype

    for round_index in range(clicks[-1]):
        clicker.make_next_click(prediction)
        click = clicker.clicks_list[-1]
        original_coords = torch.tensor(
            [[click.coords[1], click.coords[0]]], device=device, dtype=dtype
        )
        coords = predictor.apply_coords_torch(original_coords).to(device=device, dtype=dtype).unsqueeze(0)
        labels = torch.tensor(
            [[int(click.is_positive)]], device=device, dtype=torch.long
        )
        viewport = viewport_for_mode(
            coords,
            round_index,
            viewports[-1] if viewports else None,
            viewport_mode,
        )
        event = torch.cat((coords[:, 0] / 1024.0, labels[:, 0:1].to(dtype=dtype)), dim=-1)
        click_events.append(event)
        viewports.append(viewport)
        output = model(
            image_embeddings=predictor.features,
            point_coords=coords,
            point_labels=labels,
            previous_mask=previous,
            viewport_state=viewport,
            click_history=torch.stack(click_events, dim=1),
            viewport_history=torch.stack(viewports, dim=1).squeeze(2),
            mask_history=torch.stack(mask_history, dim=1) if mask_history else None,
            multimask_output=False,
        )
        probability = output["final_mask"]
        full_resolution = model.postprocess_masks(
            probability,
            predictor.input_size,
            predictor.original_size,
        )
        prediction = full_resolution[0, 0].detach().cpu().numpy() >= 0.5
        click_number = round_index + 1
        if click_number in clicks:
            masks_at_click[click_number] = prediction.copy()
            scores_at_click[click_number] = float(output["iou_predictions"][0, 0].detach().cpu())
        mask_history.append(probability.detach())
        previous = probability.detach()

    return masks_at_click, scores_at_click


def effective_parameter_counts(model: torch.nn.Module) -> dict[str, float]:
    base_modules = (model.image_encoder, model.prompt_encoder, model.mask_decoder)
    base = sum(parameter.numel() for module in base_modules for parameter in module.parameters())
    active_vimr = sum(parameter.numel() for parameter in freeze_pretrained_modules(model))
    total = sum(parameter.numel() for parameter in model.parameters())
    model.eval()
    return {
        "Params(M)": (base + active_vimr) / 1e6,
        "BaseParams(M)": base / 1e6,
        "ActiveVIMRParams(M)": active_vimr / 1e6,
        "InstantiatedParams(M)": total / 1e6,
    }


def validate_baseline_payload(payload: dict[str, Any], clicks: list[int], image_count: int) -> None:
    protocol = payload.get("protocol", {})
    if int(protocol.get("images", -1)) != image_count:
        raise ValueError(
            f"Baseline used {protocol.get('images')} images but this evaluation uses {image_count}"
        )
    baseline_clicks = [int(click) for click in protocol.get("clicks", [])]
    if baseline_clicks != clicks:
        raise ValueError(f"Baseline clicks {baseline_clicks} do not match requested clicks {clicks}")


def save_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    metric_names = ["mIoU", "mIoUS", "mIoUM", "mIoUL", "mAP", "APS", "APM", "APL"]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "method",
            "click",
            *metric_names,
            "Params(M)",
            "MACs(G)",
            "MeasuredPipelineThroughput(image/s)",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["table"])
    with (output_dir / "delta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["click", *metric_names])
        writer.writeheader()
        writer.writerows(payload["delta_vimr_minus_baseline"])


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
    baseline_payload = json.loads(args.baseline_results.read_text(encoding="utf-8"))
    validate_baseline_payload(baseline_payload, clicks, len(image_records))

    device = torch.device(args.device)
    model = build_model(args.base_checkpoint, device, seed=42)
    checkpoint = load_vimr_checkpoint(model, args.vimr_checkpoint)
    model.eval()
    predictor = EfficientViTSamPredictor(model)
    parameter_counts = effective_parameter_counts(model)

    miou_rows: dict[int, list[dict[str, float]]] = {click: [] for click in clicks}
    coco_predictions: dict[int, list[dict[str, Any]]] = {click: [] for click in clicks}
    instance_count = 0
    inference_start = time.perf_counter()
    with torch.inference_mode():
        for image_index, image_record in enumerate(image_records, start=1):
            image_path = args.image_root / image_record["file_name"]
            image = np.asarray(Image.open(image_path).convert("RGB"))
            predictor.set_image(image)
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_record["id"]]))
            for annotation in annotations:
                if annotation["area"] < 1:
                    continue
                target = coco.annToMask(annotation).astype(bool)
                predictions, scores = run_vimr_interaction(
                    model,
                    predictor,
                    target,
                    clicks,
                    args.viewport_mode,
                )
                for click in clicks:
                    miou_rows[click].append(
                        {"area": float(annotation["area"]), "iou": iou(predictions[click], target)}
                    )
                    coco_predictions[click].append(
                        encode_coco_result(predictions[click], annotation, scores[click])
                    )
                instance_count += 1
            if (
                image_index == 1
                or image_index % args.progress_every == 0
                or image_index == len(image_records)
            ):
                elapsed = time.perf_counter() - inference_start
                rate = image_index / max(elapsed, 1e-9)
                remaining = (len(image_records) - image_index) / max(rate, 1e-9)
                print(
                    f"VIMR COCO val {image_index}/{len(image_records)} images, "
                    f"{instance_count} instances, {rate:.3f} image/s, "
                    f"ETA {remaining / 60:.1f} min",
                    flush=True,
                )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    measured_throughput = len(image_records) / inference_seconds

    vimr_metrics: dict[str, dict[str, float]] = {}
    for click in clicks:
        print(f"\nCOCOeval for VIMR click={click}", flush=True)
        vimr_metrics[str(click)] = {
            **summarize_miou(miou_rows[click]),
            **evaluate_coco_ap(coco, coco_predictions[click], image_ids),
        }
        print(json.dumps({"click": click, **vimr_metrics[str(click)]}, indent=2), flush=True)

    baseline_metrics = baseline_payload["metrics"]
    baseline_complexity = baseline_payload["complexity"]
    # The EfficientViT-SAM 35G figure is the published end-to-end L0 value.
    # Until an equivalent deployment-engine profile exists for the custom
    # interaction modules, retain it as the shared backbone lower bound and
    # record the VIMR overhead protocol explicitly rather than invent precision.
    vimr_complexity = {
        **parameter_counts,
        "MACs(G)": None,
        "BackboneLowerBoundMACs(G)": float(baseline_complexity["MACs(G)"]),
        "MeasuredPipelineThroughput(image/s)": measured_throughput,
        "measured_protocol": (
            f"{args.device}, PyTorch, full image loading/encoding, all GT instances, "
            f"click checkpoints {clicks} in one {clicks[-1]}-click sequence"
        ),
    }
    metric_names = ["mIoU", "mIoUS", "mIoUM", "mIoUL", "mAP", "APS", "APM", "APL"]
    table: list[dict[str, Any]] = []
    deltas: list[dict[str, float]] = []
    for click in clicks:
        baseline_row = {
            "method": "Baseline",
            "click": click,
            **baseline_metrics[str(click)],
            "Params(M)": baseline_complexity["Params(M)"],
            "MACs(G)": baseline_complexity["MACs(G)"],
            "MeasuredPipelineThroughput(image/s)": baseline_complexity[
                "MeasuredPipelineThroughput(image/s)"
            ],
        }
        vimr_row = {
            "method": "VIMR seed 42",
            "click": click,
            **vimr_metrics[str(click)],
            "Params(M)": vimr_complexity["Params(M)"],
            "MACs(G)": vimr_complexity["MACs(G)"],
            "MeasuredPipelineThroughput(image/s)": measured_throughput,
        }
        table.extend((baseline_row, vimr_row))
        deltas.append(
            {
                "click": click,
                **{
                    metric: float(vimr_metrics[str(click)][metric] - baseline_metrics[str(click)][metric])
                    for metric in metric_names
                },
            }
        )

    payload = {
        "protocol": {
            "dataset": "COCO 2017 val",
            "images": len(image_records),
            "instances": instance_count,
            "clicks": clicks,
            "checkpoint": str(args.vimr_checkpoint.resolve().relative_to(ROOT_DIR)),
            "checkpoint_seed": int(checkpoint["seed"]),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "viewport_mode": args.viewport_mode,
            "prompt_policy": "largest-error corrective click in original-image space",
            "baseline_prompt_state": "all clicks accumulated plus previous-mask prompt",
            "vimr_prompt_state": "current click plus explicit click/view/mask history",
            "mask_selection": "single VIMR output; inherited SAM predicted IoU used as AP score",
            "ap_definition": baseline_payload["protocol"]["ap_definition"],
            "inference_seconds": inference_seconds,
        },
        "baseline_metrics": baseline_metrics,
        "vimr_metrics": vimr_metrics,
        "baseline_complexity": baseline_complexity,
        "vimr_complexity": vimr_complexity,
        "table": table,
        "delta_vimr_minus_baseline": deltas,
    }
    save_outputs(args.output, payload)
    print("\nFinal comparison", flush=True)
    print(json.dumps(table, indent=2), flush=True)
    print("\nVIMR minus baseline", flush=True)
    print(json.dumps(deltas, indent=2), flush=True)


if __name__ == "__main__":
    main()
