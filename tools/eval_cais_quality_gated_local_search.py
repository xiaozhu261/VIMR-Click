#!/usr/bin/env python3
"""Evaluate quality-gated local click search and a trajectory spatial prior.

This is a conditional feasibility experiment over the GT-audited, locally
correctable positive clicks produced by ``eval_cais_oracle_click_refinement``.
Ground truth is used only for evaluation and for fitting selector
hyperparameters inside each training fold; held-out tasks never participate in
that fit.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficientvit.models.cais import COCOClickCAISDataset, hard_iou_score
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from tools.eval_cais_oracle_click_refinement import EmbeddingCache


@dataclass(frozen=True)
class SearchRecord:
    event_id: str
    task_id: str
    click_index: int
    split: str
    original_quality: float
    candidate_xy_px: np.ndarray
    actual_iou: np.ndarray
    predicted_iou: np.ndarray
    mask_area: np.ndarray
    mask_iou_original: np.ndarray
    mask_iou_previous: np.ndarray
    distance: np.ndarray
    real_alignment: np.ndarray
    approach_vector: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/cais_synthetic_click2_click3_300_20260921"),
    )
    parser.add_argument(
        "--oracle-results",
        type=Path,
        default=Path("outputs/cais_oracle_click_refinement_20261006/per_click_results.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("assets/checkpoints/efficientvit_sam/distilled_model/efficientvit_sam_l0.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cais_quality_gated_local_search_20261006"),
    )
    parser.add_argument(
        "--radius-ratios", type=float, nargs="+", default=[0.00625, 0.0125, 0.025, 0.05]
    )
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--gate-fraction", type=float, default=0.20)
    parser.add_argument("--trajectory-window-ms", type=float, default=500.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9527, 1337, 2026])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def candidate_coordinates(
    click_x: float,
    click_y: float,
    width: int,
    height: int,
    radius_ratios: Iterable[float],
    directions: int,
) -> np.ndarray:
    """Return original click plus a clipped polar grid, with duplicates removed."""
    if directions < 4:
        raise ValueError("At least four candidate directions are required")
    coordinates = [(float(click_x), float(click_y))]
    long_side = float(max(width, height))
    for ratio in radius_ratios:
        if ratio <= 0:
            raise ValueError("Candidate radii must be positive")
        radius = float(ratio) * long_side
        for direction in range(directions):
            angle = 2.0 * math.pi * direction / directions
            coordinates.append(
                (
                    float(np.clip(click_x + radius * math.cos(angle), 0.0, width - 1.0)),
                    float(np.clip(click_y + radius * math.sin(angle), 0.0, height - 1.0)),
                )
            )
    unique: list[tuple[float, float]] = []
    keys: set[tuple[int, int]] = set()
    for x_value, y_value in coordinates:
        key = (int(round(x_value * 1000)), int(round(y_value * 1000)))
        if key not in keys:
            keys.add(key)
            unique.append((x_value, y_value))
    return np.asarray(unique, dtype=np.float32)


def recent_approach_vector(record: dict[str, Any], window_ms: float) -> np.ndarray:
    """Return the recent cursor approach in normalized image coordinates."""
    points = (record.get("preceding_trajectory") or {}).get("points") or []
    if len(points) < 2:
        return np.zeros(2, dtype=np.float32)
    trajectory = np.asarray(points, dtype=np.float64)
    trajectory = trajectory[np.argsort(trajectory[:, 0], kind="stable")]
    recent = trajectory[trajectory[:, 0] >= -float(window_ms)]
    if len(recent) < 2:
        recent = trajectory[-2:]
    endpoint = np.asarray(
        [float(record["click_x_norm"]), float(record["click_y_norm"])], dtype=np.float64
    )
    vector = endpoint - recent[0, 1:3]
    if float(np.linalg.norm(vector)) < 1e-5:
        return np.zeros(2, dtype=np.float32)
    return vector.astype(np.float32)


def trajectory_alignment(
    candidate_xy_px: np.ndarray,
    click_xy_px: np.ndarray,
    image_size: tuple[int, int],
    approach_vector: np.ndarray,
) -> np.ndarray:
    """Cosine-squared alignment of candidate displacement and cursor approach axis."""
    height, width = image_size
    displacement = candidate_xy_px.astype(np.float64) - click_xy_px.astype(np.float64)
    displacement[:, 0] /= max(float(width), 1.0)
    displacement[:, 1] /= max(float(height), 1.0)
    direction = np.asarray(approach_vector, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    displacement_norm = np.linalg.norm(displacement, axis=1)
    valid = (displacement_norm > 1e-12) & (direction_norm > 1e-12)
    result = np.zeros(len(candidate_xy_px), dtype=np.float32)
    if np.any(valid):
        cosine = (displacement[valid] @ direction) / (displacement_norm[valid] * direction_norm)
        result[valid] = np.square(cosine).astype(np.float32)
    return result


@torch.no_grad()
def evaluate_candidates(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    sample: dict[str, Any],
    candidate_xy_px: np.ndarray,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(candidate_xy_px)
    coords = sample["point_coords"].unsqueeze(0).repeat(count, 1, 1)
    scale = sample["trajectory_coordinate_scale"]
    height, width = sample["original_size"]
    coords[:, -1, 0] = torch.from_numpy(candidate_xy_px[:, 0]) / width * scale[0]
    coords[:, -1, 1] = torch.from_numpy(candidate_xy_px[:, 1]) / height * scale[1]
    labels = sample["point_labels"].unsqueeze(0).repeat(count, 1)
    previous = sample["previous_mask"].unsqueeze(0).repeat(count, 1, 1, 1)
    previous = previous.to(device=device, dtype=embedding.dtype)
    epsilon = torch.finfo(previous.dtype).eps
    masks = torch.logit(previous.clamp(epsilon, 1.0 - epsilon))
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        sparse, dense = model.prompt_encoder(
            points=(coords.to(device), labels.to(device)), boxes=None, masks=masks
        )
        logits, quality = model.mask_decoder(
            # SAM repeats each image embedding for the sparse-prompt batch internally.
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
    target = sample["target"].unsqueeze(0).expand(count, -1, -1, -1).to(device)
    probabilities = logits.float().sigmoid()
    actual = hard_iou_score(probabilities, target.float()).reshape(-1)
    hard_masks = probabilities >= 0.5
    flat_masks = hard_masks.reshape(count, -1)
    mask_area = flat_masks.float().mean(dim=1)

    original = flat_masks[0].unsqueeze(0).expand(count, -1)
    original_intersection = (flat_masks & original).sum(dim=1).float()
    original_union = (flat_masks | original).sum(dim=1).float().clamp_min(1.0)
    mask_iou_original = original_intersection / original_union

    previous_mask = previous[0] >= 0.5
    previous_flat = previous_mask.reshape(1, -1).expand(count, -1)
    previous_intersection = (flat_masks & previous_flat).sum(dim=1).float()
    previous_union = (flat_masks | previous_flat).sum(dim=1).float().clamp_min(1.0)
    mask_iou_previous = previous_intersection / previous_union
    return (
        actual.cpu().numpy(),
        quality.float().reshape(-1).cpu().numpy(),
        mask_area.cpu().numpy(),
        mask_iou_original.cpu().numpy(),
        mask_iou_previous.cpu().numpy(),
    )


def grouped_folds(records: list[SearchRecord], folds: int, seed: int) -> list[list[int]]:
    tasks: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        tasks.setdefault(record.task_id, []).append(index)
    task_ids = sorted(tasks)
    random.Random(seed).shuffle(task_ids)
    bins: list[list[int]] = [[] for _ in range(folds)]
    for offset, task_id in enumerate(task_ids):
        bins[offset % folds].extend(tasks[task_id])
    if sorted(index for fold in bins for index in fold) != list(range(len(records))):
        raise RuntimeError("Incomplete grouped fold assignment")
    return bins


def select_index(
    record: SearchRecord,
    distance_penalty: float,
    alignment_weight: float,
    alignment: np.ndarray,
    no_op_margin: float = 0.0,
) -> int:
    score = (
        record.predicted_iou
        - float(distance_penalty) * record.distance
        + float(alignment_weight) * alignment
    )
    selected = int(np.argmax(score))
    if selected != 0 and float(score[selected] - score[0]) < float(no_op_margin):
        return 0
    return selected


def fit_selector(
    records: list[SearchRecord], indices: list[int], use_trajectory: bool
) -> tuple[float, float, float]:
    distance_grid = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04)
    alignment_grid = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04) if use_trajectory else (0.0,)
    margin_grid = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.04)
    best: tuple[float, float, float, float, float] | None = None
    for distance_penalty in distance_grid:
        for alignment_weight in alignment_grid:
            for no_op_margin in margin_grid:
                gains = []
                for index in indices:
                    record = records[index]
                    selected = select_index(
                        record,
                        distance_penalty,
                        alignment_weight,
                        record.real_alignment,
                        no_op_margin,
                    )
                    gains.append(float(record.actual_iou[selected] - record.actual_iou[0]))
                objective = float(np.mean(gains))
                harm_rate = float(np.mean(np.asarray(gains) < -1e-6))
                candidate = (
                    objective,
                    -harm_rate,
                    no_op_margin,
                    -alignment_weight,
                    -distance_penalty,
                )
                if best is None or candidate > best:
                    best = candidate
                    best_parameters = (distance_penalty, alignment_weight, no_op_margin)
    return best_parameters


def shuffled_vectors(
    records: list[SearchRecord], test_indices: list[int], seed: int
) -> dict[int, np.ndarray]:
    rng = random.Random(seed)
    donors = test_indices.copy()
    for _ in range(200):
        rng.shuffle(donors)
        if all(records[target].task_id != records[donor].task_id for target, donor in zip(test_indices, donors)):
            break
    return {target: records[donor].approach_vector for target, donor in zip(test_indices, donors)}


def task_bootstrap(
    records: list[SearchRecord], values: np.ndarray, samples: int, seed: int
) -> dict[str, float | int]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.task_id, []).append(index)
    task_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        chosen = rng.choice(task_ids, size=len(task_ids), replace=True)
        indices = np.concatenate([np.asarray(groups[task_id]) for task_id in chosen])
        estimates.append(float(values[indices].mean()))
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_samples": samples,
    }


def summarize_policy(values: np.ndarray, gate: np.ndarray) -> dict[str, float | int]:
    selected = values[gate]
    return {
        "records": int(len(values)),
        "selected": int(gate.sum()),
        "selected_fraction": float(gate.mean()),
        "selected_mean_delta_iou": float(selected.mean()) if len(selected) else math.nan,
        "all_record_policy_delta_iou": float(np.where(gate, values, 0.0).mean()),
        "selected_win_rate": float(np.mean(selected > 1e-6)) if len(selected) else math.nan,
        "selected_harm_rate": float(np.mean(selected < -1e-6)) if len(selected) else math.nan,
    }


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    methods = payload["results"]
    paired = payload["paired_bootstrap"]
    decision = payload["decision"]
    lines = [
        "# CAIS quality-gated local candidate search",
        "",
        "## Question",
        "",
        "On GT-audited, locally correctable positive clicks, can frozen-SAM quality ranking select "
        "a better point from a fixed local grid, and does the real cursor-approach axis improve "
        "selection beyond shuffled or null trajectory controls?",
        "",
        f"- Records: {payload['dataset']['records']} / tasks: {payload['dataset']['tasks']}",
        f"- Candidate budget: {payload['dataset']['mean_candidates']:.1f} points per click",
        f"- Quality gate target fraction: {payload['config']['gate_fraction']:.0%}",
        f"- Evaluation: grouped {payload['config']['folds']}-fold OOF, seeds {payload['config']['seeds']}",
        "",
        "## Quality-gated policy results",
        "",
        "| Method | Selected fraction | Selected mean IoU delta | All-record policy delta | Win rate | Harm rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("quality_raw", "quality_distance", "trajectory_real", "trajectory_shuffle", "trajectory_null", "grid_oracle"):
        result = methods[name]
        lines.append(
            f"| {name} | {result['selected_fraction']:.2%} | "
            f"{result['selected_mean_delta_iou']:+.6f} | "
            f"{result['all_record_policy_delta_iou']:+.6f} | "
            f"{result['selected_win_rate']:.2%} | {result['selected_harm_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Paired task-bootstrap checks",
            "",
            "| Contrast | Mean delta | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for name, label in (
        ("quality_distance_vs_original", "Quality-distance policy vs original"),
        ("real_vs_quality_distance", "Trajectory Real vs quality-distance"),
        ("real_vs_shuffle", "Trajectory Real vs Shuffle"),
        ("real_vs_null", "Trajectory Real vs Null"),
    ):
        result = paired[name]
        lines.append(
            f"| {label} | {result['mean']:+.6f} | "
            f"[{result['ci95_low']:+.6f}, {result['ci95_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Predeclared decision",
            "",
            f"- Local quality search is feasible: `{decision['quality_search_feasible']}`",
            f"- Real trajectory beats Shuffle: `{decision['real_beats_shuffle']}`",
            f"- Real trajectory beats Null: `{decision['real_beats_null']}`",
            f"- Proceed to a learned trajectory-conditioned ranker: `{decision['proceed_to_trajectory_ranker']}`",
            "",
            "This is a conditional feasibility study, not an end-to-end claim: all evaluated records "
            "were identified as hard and locally correctable using GT during the preceding audit. "
            "The quality gate and candidate selectors themselves do not use GT at held-out inference.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.gate_fraction < 1.0:
        raise ValueError("gate-fraction must be in (0, 1)")
    if args.smoke_test:
        args.seeds = args.seeds[:1]
        args.folds = 2
        args.bootstrap_samples = min(args.bootstrap_samples, 100)
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.oracle_results.open("r", encoding="utf-8") as handle:
        oracle_rows = [
            json.loads(line)
            for line in handle
            if line.strip() and int(json.loads(line)["click_type"]) == 1
        ]
    oracle_by_event = {row["event_id"]: row for row in oracle_rows}
    datasets = {
        split: COCOClickCAISDataset(args.dataset_root, split, source_filter="human")
        for split in ("train", "val", "test")
    }
    source_by_event = {
        record["event_id"]: record
        for dataset in datasets.values()
        for record in dataset.records
    }
    sample_by_event = {
        dataset.records[index]["event_id"]: (dataset, index)
        for dataset in datasets.values()
        for index in range(len(dataset))
        if dataset.records[index]["event_id"] in oracle_by_event
    }

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp = device.type == "cuda" and not args.no_amp
    model = create_efficientvit_sam_model(
        "efficientvit-sam-l0",
        pretrained=True,
        weight_url=str(args.checkpoint),
        image_size=512,
    ).to(device)
    model.eval()
    cache = EmbeddingCache(model, device, amp)

    records: list[SearchRecord] = []
    limit = 8 if args.smoke_test else len(oracle_rows)
    for row_index, row in enumerate(oracle_rows[:limit]):
        dataset, index = sample_by_event[row["event_id"]]
        sample = dataset[index]
        source = source_by_event[row["event_id"]]
        height, width = sample["original_size"]
        click_x = float(row["original_x_px"])
        click_y = float(row["original_y_px"])
        candidates = candidate_coordinates(
            click_x, click_y, width, height, args.radius_ratios, args.directions
        )
        actual, predicted, mask_area, mask_iou_original, mask_iou_previous = evaluate_candidates(
            model, cache.get(sample["image_path"]), sample, candidates, device, amp
        )
        displacement = candidates.astype(np.float64) - candidates[0].astype(np.float64)
        max_radius = max(args.radius_ratios) * max(width, height)
        distance = (np.linalg.norm(displacement, axis=1) / max(max_radius, 1e-6)).astype(
            np.float32
        )
        approach = recent_approach_vector(source, args.trajectory_window_ms)
        alignment = trajectory_alignment(
            candidates, candidates[0], (height, width), approach
        )
        records.append(
            SearchRecord(
                event_id=row["event_id"],
                task_id=row["task_run_id"],
                click_index=int(row["click_index"]),
                split=row["split"],
                original_quality=float(predicted[0]),
                candidate_xy_px=candidates,
                actual_iou=actual.astype(np.float32),
                predicted_iou=predicted.astype(np.float32),
                mask_area=mask_area.astype(np.float32),
                mask_iou_original=mask_iou_original.astype(np.float32),
                mask_iou_previous=mask_iou_previous.astype(np.float32),
                distance=distance,
                real_alignment=alignment,
                approach_vector=approach,
            )
        )
        print(
            f"candidates {row_index + 1}/{limit} event={row['event_id']} "
            f"original={actual[0]:.4f} grid_best={actual.max():.4f}",
            flush=True,
        )

    method_contributions: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "quality_raw",
            "quality_distance",
            "trajectory_real",
            "trajectory_shuffle",
            "trajectory_null",
            "grid_oracle",
        )
    }
    gates: list[np.ndarray] = []
    fit_parameters: list[dict[str, Any]] = []
    for seed in args.seeds:
        folds = grouped_folds(records, args.folds, seed)
        gains = {name: np.zeros(len(records), dtype=np.float64) for name in method_contributions}
        gate = np.zeros(len(records), dtype=bool)
        for fold_index, test_indices in enumerate(folds):
            test_set = set(test_indices)
            train_indices = [index for index in range(len(records)) if index not in test_set]
            threshold = float(
                np.quantile(
                    [records[index].original_quality for index in train_indices],
                    args.gate_fraction,
                )
            )
            distance_parameters = fit_selector(records, train_indices, use_trajectory=False)
            trajectory_parameters = fit_selector(records, train_indices, use_trajectory=True)
            shuffled = shuffled_vectors(records, test_indices, seed + 10_000 + fold_index)
            fit_parameters.append(
                {
                    "seed": seed,
                    "fold": fold_index,
                    "gate_threshold": threshold,
                    "quality_distance_penalty": distance_parameters[0],
                    "quality_no_op_margin": distance_parameters[2],
                    "trajectory_distance_penalty": trajectory_parameters[0],
                    "trajectory_alignment_weight": trajectory_parameters[1],
                    "trajectory_no_op_margin": trajectory_parameters[2],
                }
            )
            for index in test_indices:
                record = records[index]
                gate[index] = record.original_quality <= threshold
                original = float(record.actual_iou[0])
                raw_index = select_index(record, 0.0, 0.0, record.real_alignment)
                distance_index = select_index(
                    record,
                    distance_parameters[0],
                    0.0,
                    record.real_alignment,
                    distance_parameters[2],
                )
                real_index = select_index(
                    record,
                    trajectory_parameters[0],
                    trajectory_parameters[1],
                    record.real_alignment,
                    trajectory_parameters[2],
                )
                shuffled_alignment = trajectory_alignment(
                    record.candidate_xy_px,
                    record.candidate_xy_px[0],
                    (
                        int(source_by_event[record.event_id]["image_height_px"]),
                        int(source_by_event[record.event_id]["image_width_px"]),
                    ),
                    shuffled[index],
                )
                shuffle_index = select_index(
                    record,
                    trajectory_parameters[0],
                    trajectory_parameters[1],
                    shuffled_alignment,
                    trajectory_parameters[2],
                )
                null_index = select_index(
                    record,
                    trajectory_parameters[0],
                    trajectory_parameters[1],
                    np.zeros_like(record.real_alignment),
                    trajectory_parameters[2],
                )
                gains["quality_raw"][index] = float(record.actual_iou[raw_index]) - original
                gains["quality_distance"][index] = (
                    float(record.actual_iou[distance_index]) - original
                )
                gains["trajectory_real"][index] = float(record.actual_iou[real_index]) - original
                gains["trajectory_shuffle"][index] = (
                    float(record.actual_iou[shuffle_index]) - original
                )
                gains["trajectory_null"][index] = float(record.actual_iou[null_index]) - original
                gains["grid_oracle"][index] = float(record.actual_iou.max()) - original
        gates.append(gate)
        for name, values in gains.items():
            method_contributions[name].append(np.where(gate, values, 0.0))
        print(
            f"seed={seed} gated={gate.mean():.2%} "
            f"quality_policy={np.where(gate, gains['quality_distance'], 0.0).mean():+.6f} "
            f"real_policy={np.where(gate, gains['trajectory_real'], 0.0).mean():+.6f} "
            f"shuffle_policy={np.where(gate, gains['trajectory_shuffle'], 0.0).mean():+.6f}",
            flush=True,
        )

    gate_stack = np.stack(gates)
    mean_gate = gate_stack.mean(axis=0) > 0.5
    averaged = {
        name: np.stack(values).mean(axis=0) for name, values in method_contributions.items()
    }
    results = {}
    for name, values_by_seed in method_contributions.items():
        contribution_stack = np.stack(values_by_seed)
        selected_values = contribution_stack[gate_stack]
        results[name] = {
            "records": len(records),
            "selected": float(gate_stack.sum() / len(args.seeds)),
            "selected_fraction": float(gate_stack.mean()),
            "selected_mean_delta_iou": float(selected_values.mean()),
            "all_record_policy_delta_iou": float(contribution_stack.mean()),
            "selected_win_rate": float(np.mean(selected_values > 1e-6)),
            "selected_harm_rate": float(np.mean(selected_values < -1e-6)),
        }

    paired_values = {
        "quality_distance_vs_original": averaged["quality_distance"],
        "real_vs_quality_distance": averaged["trajectory_real"] - averaged["quality_distance"],
        "real_vs_shuffle": averaged["trajectory_real"] - averaged["trajectory_shuffle"],
        "real_vs_null": averaged["trajectory_real"] - averaged["trajectory_null"],
    }
    paired = {
        name: task_bootstrap(records, values, args.bootstrap_samples, 70_000 + offset)
        for offset, (name, values) in enumerate(paired_values.items())
    }
    decision = {
        "quality_search_feasible": paired["quality_distance_vs_original"]["ci95_low"] > 0,
        "real_beats_shuffle": paired["real_vs_shuffle"]["ci95_low"] > 0,
        "real_beats_null": paired["real_vs_null"]["ci95_low"] > 0,
    }
    decision["proceed_to_trajectory_ranker"] = all(decision.values())
    payload = {
        "config": {
            "dataset_root": str(args.dataset_root),
            "oracle_results": str(args.oracle_results),
            "checkpoint": str(args.checkpoint),
            "radius_ratios": args.radius_ratios,
            "directions": args.directions,
            "gate_fraction": args.gate_fraction,
            "trajectory_window_ms": args.trajectory_window_ms,
            "folds": args.folds,
            "seeds": args.seeds,
            "amp": amp,
            "smoke_test": args.smoke_test,
        },
        "dataset": {
            "records": len(records),
            "tasks": len({record.task_id for record in records}),
            "mean_candidates": float(np.mean([len(record.actual_iou) for record in records])),
            "valid_trajectory_fraction": float(
                np.mean([np.linalg.norm(record.approach_vector) > 1e-8 for record in records])
            ),
        },
        "fit_parameters": fit_parameters,
        "results": results,
        "paired_bootstrap": paired,
        "decision": decision,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "candidate_results.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(
                json.dumps(
                    {
                        "event_id": record.event_id,
                        "task_run_id": record.task_id,
                        "click_index": record.click_index,
                        "original_quality": record.original_quality,
                        "candidate_xy_px": record.candidate_xy_px.tolist(),
                        "actual_iou": record.actual_iou.tolist(),
                        "predicted_iou": record.predicted_iou.tolist(),
                        "mask_area": record.mask_area.tolist(),
                        "mask_iou_original": record.mask_iou_original.tolist(),
                        "mask_iou_previous": record.mask_iou_previous.tolist(),
                        "distance": record.distance.tolist(),
                        "real_alignment": record.real_alignment.tolist(),
                        "approach_vector": record.approach_vector.tolist(),
                        "mean_seed_gate": bool(mean_gate[index]),
                        **{f"{name}_policy_contribution": float(values[index]) for name, values in averaged.items()},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    write_report(args.output_dir, payload)
    print(json.dumps({"results": results, "paired": paired, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
