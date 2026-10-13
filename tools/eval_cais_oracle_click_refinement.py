#!/usr/bin/env python3
"""Measure the frozen-SAM upper bound of local oracle click refinement."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficientvit.models.cais import COCOClickCAISDataset, hard_iou_score, load_rgb_image
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


@dataclass(frozen=True)
class OracleTarget:
    hard: bool
    eligible: bool
    inside_error: bool
    signed_distance_px: float
    target_x_px: float | None
    target_y_px: float | None
    correction_distance_px: float | None
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/cais_synthetic_click2_click3_300_20260921"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("assets/checkpoints/efficientvit_sam/distilled_model/efficientvit_sam_l0.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/cais_oracle_click_refinement_20261006")
    )
    parser.add_argument("--boundary-threshold-px", type=float, default=3.0)
    parser.add_argument("--safe-margin-px", type=float, default=5.0)
    parser.add_argument(
        "--max-correction-ratio",
        type=float,
        default=0.05,
        help="Maximum local correction as a fraction of the original image long side.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=9527)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def decode_ground_truth(record: dict[str, Any]) -> np.ndarray:
    target = mask_utils.decode(record["ground_truth_rle"])
    if target.ndim == 3:
        target = target.any(axis=2)
    return np.asarray(target, dtype=bool)


def intended_error_mask(record: dict[str, Any]) -> np.ndarray:
    ground_truth = decode_ground_truth(record)
    with Image.open(record["pre_mask_ref"]) as image:
        previous = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    if previous.shape != ground_truth.shape:
        raise ValueError(
            f"Previous mask {previous.shape} != target {ground_truth.shape} for {record['event_id']}"
        )
    click_type = int(record["prompt_history"][-1]["click_type"])
    if click_type == 1:
        return ground_truth & ~previous
    if click_type == 0:
        return previous & ~ground_truth
    raise ValueError(f"Unsupported click type {click_type} for {record['event_id']}")


def _nearest_coordinate(mask: np.ndarray, x: float, y: float) -> tuple[int, int, float]:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        raise ValueError("Cannot find a coordinate in an empty mask")
    distances = np.square(coordinates[:, 1] - x) + np.square(coordinates[:, 0] - y)
    best = int(np.argmin(distances))
    target_y, target_x = coordinates[best]
    return int(target_x), int(target_y), float(math.sqrt(float(distances[best])))


def build_oracle_target(
    error_mask: np.ndarray,
    click_x_px: float,
    click_y_px: float,
    *,
    boundary_threshold_px: float,
    safe_margin_px: float,
    max_correction_px: float,
) -> OracleTarget:
    """Find the nearest safe point in the clicked/nearest intended error component."""
    height, width = error_mask.shape
    pixel_x = int(np.clip(round(click_x_px), 0, width - 1))
    pixel_y = int(np.clip(round(click_y_px), 0, height - 1))
    if not error_mask.any():
        return OracleTarget(False, False, False, -math.inf, None, None, None, "empty_error")

    inside = bool(error_mask[pixel_y, pixel_x])
    labels_count, labels = cv2.connectedComponents(error_mask.astype(np.uint8), connectivity=8)
    if labels_count <= 1:
        return OracleTarget(False, False, inside, -math.inf, None, None, None, "empty_error")

    if inside:
        full_distance = cv2.distanceTransform(error_mask.astype(np.uint8), cv2.DIST_L2, 5)
        signed_distance = float(full_distance[pixel_y, pixel_x])
        hard = signed_distance <= boundary_threshold_px
        component_label = int(labels[pixel_y, pixel_x])
    else:
        nearest_x, nearest_y, outside_distance = _nearest_coordinate(
            error_mask, click_x_px, click_y_px
        )
        signed_distance = -outside_distance
        hard = True
        component_label = int(labels[nearest_y, nearest_x])

    if not hard:
        return OracleTarget(False, False, inside, signed_distance, None, None, None, "easy")

    component = labels == component_label
    component_distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    safe = component_distance >= safe_margin_px
    if not safe.any():
        maximum = float(component_distance.max())
        safe = component_distance >= max(maximum - 1e-6, 0.0)
    target_x, target_y, correction_distance = _nearest_coordinate(
        safe, click_x_px, click_y_px
    )
    if correction_distance > max_correction_px:
        return OracleTarget(
            True,
            False,
            inside,
            signed_distance,
            float(target_x),
            float(target_y),
            correction_distance,
            "nonlocal_correction",
        )
    return OracleTarget(
        True,
        True,
        inside,
        signed_distance,
        float(target_x),
        float(target_y),
        correction_distance,
        "eligible",
    )


class EmbeddingCache:
    def __init__(self, model: torch.nn.Module, device: torch.device, amp: bool) -> None:
        self.model = model
        self.device = device
        self.amp = amp
        self.values: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def get(self, image_path: str) -> torch.Tensor:
        if image_path not in self.values:
            image = load_rgb_image(image_path)
            tensor = self.model.transform(image).unsqueeze(0).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                embedding = self.model.image_encoder(tensor)
            self.values[image_path] = embedding.detach().to("cpu", dtype=torch.float16)
        dtype = torch.float16 if self.amp else torch.float32
        return self.values[image_path].to(self.device, dtype=dtype)


@torch.no_grad()
def predict_iou(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    sample: dict[str, Any],
    point_coords: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> tuple[float, float]:
    labels = sample["point_labels"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    previous = sample["previous_mask"].unsqueeze(0).to(device=device, dtype=embedding.dtype)
    epsilon = torch.finfo(previous.dtype).eps
    mask_prompt = torch.logit(previous.clamp(epsilon, 1.0 - epsilon))
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        sparse, dense = model.prompt_encoder(
            points=(point_coords.unsqueeze(0).to(device), labels), boxes=None, masks=mask_prompt
        )
        logits, quality = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
    actual_iou = float(hard_iou_score(logits.float().sigmoid(), target.float()).reshape(-1)[0])
    predicted_iou = float(quality.float().reshape(-1)[0])
    return actual_iou, predicted_iou


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"samples": 0}
    original = np.asarray([row["original_iou"] for row in rows], dtype=np.float64)
    oracle = np.asarray([row["oracle_iou"] for row in rows], dtype=np.float64)
    delta = oracle - original
    ordered_delta = np.sort(delta)
    trim = int(0.05 * len(ordered_delta))
    trimmed_delta = ordered_delta[trim : len(ordered_delta) - trim] if trim else ordered_delta
    return {
        "samples": len(rows),
        "tasks": len({row["task_run_id"] for row in rows}),
        "original_iou": float(original.mean()),
        "oracle_iou": float(oracle.mean()),
        "mean_delta_iou": float(delta.mean()),
        "median_delta_iou": float(np.median(delta)),
        "trimmed_mean_delta_iou_5pct": float(trimmed_delta.mean()),
        "win_rate": float(np.mean(delta > 1e-6)),
        "tie_rate": float(np.mean(np.abs(delta) <= 1e-6)),
        "harm_rate": float(np.mean(delta < -1e-6)),
        "large_win_rate_delta_gt_0.01": float(np.mean(delta > 0.01)),
        "large_harm_rate_delta_lt_minus_0.01": float(np.mean(delta < -0.01)),
        "minimum_delta_iou": float(delta.min()),
        "maximum_delta_iou": float(delta.max()),
        "mean_correction_distance_px": float(
            np.mean([row["correction_distance_px"] for row in rows])
        ),
    }


def task_bootstrap(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, float | int]:
    if not rows:
        return {"samples": 0}
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(row["task_run_id"], []).append(row["delta_iou"])
    task_ids = sorted(groups)
    observed = float(np.mean([row["delta_iou"] for row in rows]))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        selected = rng.choice(task_ids, size=len(task_ids), replace=True)
        deltas = [delta for task_id in selected for delta in groups[task_id]]
        values.append(float(np.mean(deltas)))
    array = np.asarray(values)
    return {
        "mean_delta_iou": observed,
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "bootstrap_samples": samples,
    }


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    aggregate = payload["results"]["all"]
    bootstrap = payload["bootstrap"]["all"]
    decision = payload["decision"]
    candidate = payload["candidate_audit"]
    lines = [
        "# CAIS oracle local click-refinement experiment",
        "",
        "## Protocol",
        "",
        "Frozen EfficientViT-SAM-L0; all prompt history, click type, previous mask, image features, "
        "and decoder settings are held fixed. Only the current click coordinate is replaced by the "
        "nearest point with a safe interior margin in the corresponding GT error component.",
        "",
        f"- Boundary threshold: {payload['config']['boundary_threshold_px']} px",
        f"- Safe interior margin: {payload['config']['safe_margin_px']} px",
        f"- Maximum correction: {payload['config']['max_correction_ratio']:.1%} of image long side",
        f"- Human clicks audited: {candidate['total_human_clicks']}",
        f"- Hard clicks: {candidate['hard_clicks']}",
        f"- Eligible local corrections: {candidate['eligible_clicks']}",
        "",
        "## Frozen-SAM results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Eligible clicks | {aggregate['samples']} |",
        f"| Original mean IoU | {aggregate['original_iou']:.6f} |",
        f"| Oracle mean IoU | {aggregate['oracle_iou']:.6f} |",
        f"| Mean IoU delta | {aggregate['mean_delta_iou']:+.6f} |",
        f"| Median IoU delta | {aggregate['median_delta_iou']:+.6f} |",
        f"| 5% trimmed mean delta | {aggregate['trimmed_mean_delta_iou_5pct']:+.6f} |",
        f"| Win rate | {aggregate['win_rate']:.2%} |",
        f"| Harm rate | {aggregate['harm_rate']:.2%} |",
        f"| Delta > 0.01 | {aggregate['large_win_rate_delta_gt_0.01']:.2%} |",
        f"| Delta < -0.01 | {aggregate['large_harm_rate_delta_lt_minus_0.01']:.2%} |",
        f"| Mean correction distance | {aggregate['mean_correction_distance_px']:.2f} px |",
        f"| Task-bootstrap 95% CI | [{bootstrap['ci95_low']:+.6f}, {bootstrap['ci95_high']:+.6f}] |",
        "",
        "## Predeclared decision",
        "",
        f"- Mean IoU delta >= 0.02: `{decision['mean_delta_at_least_0.02']}`",
        f"- Win rate >= 60%: `{decision['win_rate_at_least_0.60']}`",
        f"- Bootstrap CI lower bound > 0: `{decision['ci_lower_bound_above_zero']}`",
        f"- Proceed to a learned trajectory click refiner: `{decision['proceed_to_learned_refiner']}`",
        "",
        "## Subgroups",
        "",
        "| Group | N | Mean delta | 95% task-bootstrap CI | Win rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("positive", "negative", "inside_boundary", "outside_error"):
        result = payload["results"][name]
        group_bootstrap = payload["bootstrap"][name]
        if result["samples"]:
            lines.append(
                f"| {name} | {result['samples']} | {result['mean_delta_iou']:+.6f} | "
                f"[{group_bootstrap['ci95_low']:+.6f}, {group_bootstrap['ci95_high']:+.6f}] | "
                f"{result['win_rate']:.2%} |"
            )
    lines.extend([
        "",
        "The overall mean should be interpreted with the median and trimmed mean: a small number "
        "of catastrophic positive-click recoveries can contribute a large fraction of the average gain.",
        "",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.boundary_threshold_px <= 0 or args.safe_margin_px <= 0:
        raise ValueError("Boundary threshold and safe margin must be positive")
    if not 0 < args.max_correction_ratio <= 0.25:
        raise ValueError("max-correction-ratio must be in (0, 0.25]")
    if args.smoke_test:
        args.bootstrap_samples = min(args.bootstrap_samples, 100)
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    model = create_efficientvit_sam_model(
        "efficientvit-sam-l0",
        pretrained=True,
        weight_url=str(args.checkpoint),
        image_size=512,
    ).to(device)
    model.eval()
    cache = EmbeddingCache(model, device, amp)

    audit = Counter()
    rows: list[dict[str, Any]] = []
    stop = False
    for split in ("train", "val", "test"):
        dataset = COCOClickCAISDataset(args.dataset_root, split, source_filter="human")
        for index, record in enumerate(dataset.records):
            audit["total_human_clicks"] += 1
            error_mask = intended_error_mask(record)
            height, width = error_mask.shape
            prompt = record["prompt_history"][-1]
            click_x = float(prompt["x_norm"]) * width
            click_y = float(prompt["y_norm"]) * height
            oracle = build_oracle_target(
                error_mask,
                click_x,
                click_y,
                boundary_threshold_px=args.boundary_threshold_px,
                safe_margin_px=args.safe_margin_px,
                max_correction_px=args.max_correction_ratio * max(height, width),
            )
            audit[f"reason_{oracle.reason}"] += 1
            audit["hard_clicks"] += int(oracle.hard)
            audit["eligible_clicks"] += int(oracle.eligible)
            audit[f"{split}_hard_clicks"] += int(oracle.hard)
            audit[f"{split}_eligible_clicks"] += int(oracle.eligible)
            if not oracle.eligible:
                continue

            sample = dataset[index]
            original_coords = sample["point_coords"].clone()
            corrected_coords = original_coords.clone()
            coordinate_scale = sample["trajectory_coordinate_scale"]
            corrected_coords[-1, 0] = float(oracle.target_x_px) / width * float(coordinate_scale[0])
            corrected_coords[-1, 1] = float(oracle.target_y_px) / height * float(coordinate_scale[1])
            embedding = cache.get(sample["image_path"])
            original_iou, original_predicted_iou = predict_iou(
                model, embedding, sample, original_coords, device, amp
            )
            oracle_iou, oracle_predicted_iou = predict_iou(
                model, embedding, sample, corrected_coords, device, amp
            )
            row = {
                "event_id": sample["event_id"],
                "task_run_id": sample["task_run_id"],
                "split": split,
                "click_index": int(sample["click_index"]),
                "click_type": int(sample["point_labels"][-1]),
                "inside_error": oracle.inside_error,
                "signed_distance_px": oracle.signed_distance_px,
                "correction_distance_px": oracle.correction_distance_px,
                "original_x_px": click_x,
                "original_y_px": click_y,
                "oracle_x_px": oracle.target_x_px,
                "oracle_y_px": oracle.target_y_px,
                "original_iou": original_iou,
                "original_predicted_iou": original_predicted_iou,
                "oracle_iou": oracle_iou,
                "oracle_predicted_iou": oracle_predicted_iou,
                "delta_iou": oracle_iou - original_iou,
                "recorded_post_iou": float(record["post_mask_metrics"]["iou"]),
            }
            rows.append(row)
            if args.smoke_test and len(rows) >= 3:
                stop = True
                break
        if stop:
            break

    if not rows:
        raise RuntimeError("No eligible hard clicks found")
    results = {
        "all": summarize(rows),
        **{split: summarize([row for row in rows if row["split"] == split]) for split in ("train", "val", "test")},
        "positive": summarize([row for row in rows if row["click_type"] == 1]),
        "negative": summarize([row for row in rows if row["click_type"] == 0]),
        "inside_boundary": summarize([row for row in rows if row["inside_error"]]),
        "outside_error": summarize([row for row in rows if not row["inside_error"]]),
    }
    grouped_rows = {
        "all": rows,
        "positive": [row for row in rows if row["click_type"] == 1],
        "negative": [row for row in rows if row["click_type"] == 0],
        "inside_boundary": [row for row in rows if row["inside_error"]],
        "outside_error": [row for row in rows if not row["inside_error"]],
    }
    bootstrap = {
        name: task_bootstrap(group, args.bootstrap_samples, args.bootstrap_seed + offset)
        for offset, (name, group) in enumerate(grouped_rows.items())
    }
    aggregate = results["all"]
    decision = {
        "mean_delta_at_least_0.02": aggregate["mean_delta_iou"] >= 0.02,
        "win_rate_at_least_0.60": aggregate["win_rate"] >= 0.60,
        "ci_lower_bound_above_zero": bootstrap["all"]["ci95_low"] > 0,
    }
    decision["proceed_to_learned_refiner"] = all(decision.values())
    payload = {
        "config": {
            "dataset_root": str(args.dataset_root),
            "checkpoint": str(args.checkpoint),
            "boundary_threshold_px": args.boundary_threshold_px,
            "safe_margin_px": args.safe_margin_px,
            "max_correction_ratio": args.max_correction_ratio,
            "amp": amp,
            "smoke_test": args.smoke_test,
        },
        "candidate_audit": dict(audit),
        "results": results,
        "bootstrap": bootstrap,
        "decision": decision,
    }
    with (args.output_dir / "per_click_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(args.output_dir, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
