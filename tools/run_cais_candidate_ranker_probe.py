#!/usr/bin/env python3
"""Train a leakage-controlled local candidate ranker with trajectory controls."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class RankRecord:
    event_id: str
    task_id: str
    click_index: int
    image_size: tuple[int, int]
    original_quality: float
    candidate_xy: np.ndarray
    actual_iou: np.ndarray
    predicted_iou: np.ndarray
    mask_area: np.ndarray
    mask_iou_original: np.ndarray
    mask_iou_previous: np.ndarray
    distance: np.ndarray
    approach: np.ndarray


class CandidateRanker(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/cais_synthetic_click2_click3_300_20260921"),
    )
    parser.add_argument(
        "--candidate-results",
        type=Path,
        default=Path(
            "outputs/cais_quality_gated_local_search_20261006/candidate_results.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cais_candidate_ranker_probe_20261006"),
    )
    parser.add_argument("--gate-fraction", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9527, 1337, 2026])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-op-margin", type=float, default=0.005)
    parser.add_argument("--target-clip", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def load_records(args: argparse.Namespace) -> list[RankRecord]:
    with (args.dataset_root / "alignment" / "modeling_clicks_strict.jsonl").open(
        "r", encoding="utf-8"
    ) as handle:
        source = {
            row["event_id"]: row
            for line in handle
            if line.strip() and (row := json.loads(line)).get("source", "human") == "human"
        }
    with args.candidate_results.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.smoke_test:
        rows = rows[:30]
    records = []
    for row in rows:
        metadata = source[row["event_id"]]
        records.append(
            RankRecord(
                event_id=row["event_id"],
                task_id=row["task_run_id"],
                click_index=int(row["click_index"]),
                image_size=(
                    int(metadata["image_height_px"]), int(metadata["image_width_px"])
                ),
                original_quality=float(row["original_quality"]),
                candidate_xy=np.asarray(row["candidate_xy_px"], dtype=np.float32),
                actual_iou=np.asarray(row["actual_iou"], dtype=np.float32),
                predicted_iou=np.asarray(row["predicted_iou"], dtype=np.float32),
                mask_area=np.asarray(row["mask_area"], dtype=np.float32),
                mask_iou_original=np.asarray(row["mask_iou_original"], dtype=np.float32),
                mask_iou_previous=np.asarray(row["mask_iou_previous"], dtype=np.float32),
                distance=np.asarray(row["distance"], dtype=np.float32),
                approach=np.asarray(row["approach_vector"], dtype=np.float32),
            )
        )
    if not records:
        raise ValueError("No candidate records found")
    return records


def grouped_folds(records: list[RankRecord], folds: int, seed: int) -> list[list[int]]:
    tasks: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        tasks.setdefault(record.task_id, []).append(index)
    task_ids = sorted(tasks)
    random.Random(seed).shuffle(task_ids)
    result: list[list[int]] = [[] for _ in range(folds)]
    for offset, task_id in enumerate(task_ids):
        result[offset % folds].extend(tasks[task_id])
    return result


def trajectory_features(record: RankRecord, approach: np.ndarray) -> np.ndarray:
    height, width = record.image_size
    displacement = record.candidate_xy.astype(np.float64) - record.candidate_xy[0]
    displacement[:, 0] /= max(float(width), 1.0)
    displacement[:, 1] /= max(float(height), 1.0)
    direction = np.asarray(approach, dtype=np.float64)
    magnitude = float(np.linalg.norm(direction))
    norm = np.linalg.norm(displacement, axis=1)
    signed_cosine = np.zeros(len(displacement), dtype=np.float64)
    valid = (norm > 1e-12) & (magnitude > 1e-12)
    if np.any(valid):
        signed_cosine[valid] = (displacement[valid] @ direction) / (norm[valid] * magnitude)
    return np.stack(
        (
            signed_cosine,
            np.square(signed_cosine),
            signed_cosine * record.distance,
            np.full(len(displacement), min(magnitude * 10.0, 1.0)),
        ),
        axis=1,
    ).astype(np.float32)


def base_features(record: RankRecord) -> np.ndarray:
    displacement = record.candidate_xy - record.candidate_xy[0]
    maximum = max(float(np.linalg.norm(displacement, axis=1).max()), 1e-6)
    dx = displacement[:, 0] / maximum
    dy = displacement[:, 1] / maximum
    return np.stack(
        (
            record.predicted_iou,
            record.predicted_iou - record.predicted_iou[0],
            record.distance,
            dx,
            dy,
            record.mask_area,
            record.mask_area - record.mask_area[0],
            np.abs(record.mask_area - record.mask_area[0]),
            record.mask_iou_original,
            record.mask_iou_previous,
            record.mask_iou_previous - record.mask_iou_previous[0],
            np.full(len(dx), record.original_quality),
            np.full(len(dx), record.click_index / 5.0),
        ),
        axis=1,
    ).astype(np.float32)


def record_features(record: RankRecord, approach: np.ndarray | None) -> np.ndarray:
    base = base_features(record)
    if approach is None:
        return base
    return np.concatenate((base, trajectory_features(record, approach)), axis=1)


def flatten_training(
    records: list[RankRecord], indices: list[int], fused: bool, target_clip: float
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    for index in indices:
        record = records[index]
        features.append(record_features(record, record.approach if fused else None))
        gain = record.actual_iou - record.actual_iou[0]
        targets.append(np.clip(gain, -target_clip, target_clip))
    return np.concatenate(features), np.concatenate(targets).astype(np.float32)


def train_ranker(
    features: np.ndarray,
    targets: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[CandidateRanker, np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.maximum(std, 1e-6)
    normalized = ((features - mean) / std).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(normalized), torch.from_numpy(targets))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    device = torch.device(args.device)
    model = CandidateRanker(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = torch.nn.SmoothL1Loss(beta=0.02)
    for _ in range(args.epochs):
        model.train()
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_features.to(device))
            loss = criterion(prediction, batch_targets.to(device))
            loss.backward()
            optimizer.step()
    return model, mean.astype(np.float32), std.astype(np.float32)


@torch.no_grad()
def predict(
    model: CandidateRanker, features: np.ndarray, mean: np.ndarray, std: np.ndarray, device: str
) -> np.ndarray:
    model.eval()
    normalized = ((features - mean) / std).astype(np.float32)
    return (
        model(torch.from_numpy(normalized).to(torch.device(device))).float().cpu().numpy()
    )


def choose_candidate(scores: np.ndarray, no_op_margin: float) -> int:
    relative = scores - scores[0]
    selected = int(np.argmax(relative))
    if selected != 0 and float(relative[selected]) < no_op_margin:
        return 0
    return selected


def shuffled_approaches(
    records: list[RankRecord], indices: list[int], seed: int
) -> dict[int, np.ndarray]:
    donors = indices.copy()
    rng = random.Random(seed)
    for _ in range(200):
        rng.shuffle(donors)
        if all(records[target].task_id != records[donor].task_id for target, donor in zip(indices, donors)):
            break
    return {target: records[donor].approach for target, donor in zip(indices, donors)}


def task_bootstrap(
    records: list[RankRecord], values: np.ndarray, samples: int, seed: int
) -> dict[str, float | int]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.task_id, []).append(index)
    task_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        chosen = rng.choice(task_ids, len(task_ids), replace=True)
        indices = np.concatenate([np.asarray(groups[task]) for task in chosen])
        estimates.append(float(values[indices].mean()))
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_samples": samples,
    }


def summarize(contribution_stack: np.ndarray, gate_stack: np.ndarray) -> dict[str, float]:
    selected = contribution_stack[gate_stack]
    return {
        "selected_fraction": float(gate_stack.mean()),
        "selected_mean_delta_iou": float(selected.mean()),
        "all_record_policy_delta_iou": float(contribution_stack.mean()),
        "selected_win_rate": float(np.mean(selected > 1e-6)),
        "selected_harm_rate": float(np.mean(selected < -1e-6)),
        "selected_no_op_rate": float(np.mean(np.abs(selected) <= 1e-6)),
    }


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# CAIS learned local candidate-ranker probe",
        "",
        "## Protocol",
        "",
        "A small MLP ranks the same fixed candidate grid using only deployable SAM/mask-change "
        "features. The fused model additionally receives candidate-wise cursor direction features. "
        "All results are task-grouped out-of-fold; Shuffle and Null replace trajectory only at "
        "inference using the same fused checkpoint.",
        "",
        f"- Records/tasks: {payload['dataset']['records']} / {payload['dataset']['tasks']}",
        f"- Seeds: {payload['config']['seeds']}",
        f"- No-op margin: {payload['config']['no_op_margin']}",
        "",
        "## Quality-gated policy",
        "",
        "| Method | Selected mean delta | All-record policy delta | Win | Harm | No-op |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("ranker_base", "ranker_real", "ranker_shuffle", "ranker_null", "grid_oracle"):
        result = payload["results"][name]
        lines.append(
            f"| {name} | {result['selected_mean_delta_iou']:+.6f} | "
            f"{result['all_record_policy_delta_iou']:+.6f} | "
            f"{result['selected_win_rate']:.2%} | {result['selected_harm_rate']:.2%} | "
            f"{result['selected_no_op_rate']:.2%} |"
        )
    lines.extend(["", "## Paired task-bootstrap", "", "| Contrast | Mean | 95% CI |", "|---|---:|---:|"])
    for name, label in (
        ("base_vs_original", "Base ranker vs original"),
        ("real_vs_base", "Real trajectory vs base"),
        ("real_vs_shuffle", "Real trajectory vs Shuffle"),
        ("real_vs_null", "Real trajectory vs Null"),
    ):
        result = payload["paired_bootstrap"][name]
        lines.append(
            f"| {label} | {result['mean']:+.6f} | "
            f"[{result['ci95_low']:+.6f}, {result['ci95_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Learned base selector is feasible: `{payload['decision']['base_selector_feasible']}`",
            f"- Real trajectory beats Shuffle: `{payload['decision']['real_beats_shuffle']}`",
            f"- Real trajectory beats Null: `{payload['decision']['real_beats_null']}`",
            f"- Trajectory has independent value: `{payload['decision']['trajectory_has_independent_value']}`",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.folds = 2
        args.seeds = args.seeds[:1]
        args.epochs = 2
        args.bootstrap_samples = 100
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args)
    contributions: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("ranker_base", "ranker_real", "ranker_shuffle", "ranker_null", "grid_oracle")
    }
    gates = []
    for seed in args.seeds:
        folds = grouped_folds(records, args.folds, seed)
        gains = {name: np.zeros(len(records), dtype=np.float64) for name in contributions}
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
            base_train = flatten_training(records, train_indices, False, args.target_clip)
            fused_train = flatten_training(records, train_indices, True, args.target_clip)
            base_model, base_mean, base_std = train_ranker(
                *base_train, seed + fold_index, args
            )
            fused_model, fused_mean, fused_std = train_ranker(
                *fused_train, seed + 10_000 + fold_index, args
            )
            shuffled = shuffled_approaches(records, test_indices, seed + 20_000 + fold_index)
            for index in test_indices:
                record = records[index]
                gate[index] = record.original_quality <= threshold
                original = float(record.actual_iou[0])
                base_score = predict(
                    base_model, record_features(record, None), base_mean, base_std, args.device
                )
                real_score = predict(
                    fused_model,
                    record_features(record, record.approach),
                    fused_mean,
                    fused_std,
                    args.device,
                )
                shuffle_score = predict(
                    fused_model,
                    record_features(record, shuffled[index]),
                    fused_mean,
                    fused_std,
                    args.device,
                )
                null_score = predict(
                    fused_model,
                    record_features(record, np.zeros(2, dtype=np.float32)),
                    fused_mean,
                    fused_std,
                    args.device,
                )
                chosen = {
                    "ranker_base": choose_candidate(base_score, args.no_op_margin),
                    "ranker_real": choose_candidate(real_score, args.no_op_margin),
                    "ranker_shuffle": choose_candidate(shuffle_score, args.no_op_margin),
                    "ranker_null": choose_candidate(null_score, args.no_op_margin),
                    "grid_oracle": int(np.argmax(record.actual_iou)),
                }
                for name, candidate_index in chosen.items():
                    gains[name][index] = float(record.actual_iou[candidate_index]) - original
        gates.append(gate)
        for name, values in gains.items():
            contributions[name].append(np.where(gate, values, 0.0))
        print(
            f"seed={seed} gate={gate.mean():.2%} "
            f"base={np.where(gate, gains['ranker_base'], 0).mean():+.6f} "
            f"real={np.where(gate, gains['ranker_real'], 0).mean():+.6f} "
            f"shuffle={np.where(gate, gains['ranker_shuffle'], 0).mean():+.6f}",
            flush=True,
        )

    gate_stack = np.stack(gates)
    contribution_stack = {name: np.stack(values) for name, values in contributions.items()}
    averaged = {name: values.mean(axis=0) for name, values in contribution_stack.items()}
    results = {
        name: summarize(values, gate_stack) for name, values in contribution_stack.items()
    }
    contrasts = {
        "base_vs_original": averaged["ranker_base"],
        "real_vs_base": averaged["ranker_real"] - averaged["ranker_base"],
        "real_vs_shuffle": averaged["ranker_real"] - averaged["ranker_shuffle"],
        "real_vs_null": averaged["ranker_real"] - averaged["ranker_null"],
    }
    paired = {
        name: task_bootstrap(records, values, args.bootstrap_samples, 90_000 + offset)
        for offset, (name, values) in enumerate(contrasts.items())
    }
    decision = {
        "base_selector_feasible": paired["base_vs_original"]["ci95_low"] > 0,
        "real_beats_shuffle": paired["real_vs_shuffle"]["ci95_low"] > 0,
        "real_beats_null": paired["real_vs_null"]["ci95_low"] > 0,
    }
    decision["trajectory_has_independent_value"] = (
        decision["real_beats_shuffle"] and decision["real_beats_null"]
    )
    payload = {
        "config": {
            "candidate_results": str(args.candidate_results),
            "gate_fraction": args.gate_fraction,
            "folds": args.folds,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "no_op_margin": args.no_op_margin,
            "target_clip": args.target_clip,
            "smoke_test": args.smoke_test,
        },
        "dataset": {
            "records": len(records),
            "tasks": len({record.task_id for record in records}),
            "mean_candidates": float(np.mean([len(record.actual_iou) for record in records])),
        },
        "results": results,
        "paired_bootstrap": paired,
        "decision": decision,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(args.output_dir, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
