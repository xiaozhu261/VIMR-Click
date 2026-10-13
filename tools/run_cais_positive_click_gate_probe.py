#!/usr/bin/env python3
"""Probe whether trajectory semantics can gate high-benefit positive-click refinement."""

from __future__ import annotations

import argparse
import copy
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

from tools.run_cais_trajectory_probe import (
    TrainNormalizer,
    TrajectoryProbe,
    average_precision,
    roc_auc,
    set_seed,
    trajectory_tensor,
)


@dataclass(frozen=True)
class GateRecord:
    event_id: str
    task_id: str
    click_index: int
    label: int
    delta_iou: float
    context: np.ndarray
    trajectory: np.ndarray


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
        "--output-dir", type=Path, default=Path("outputs/cais_positive_click_gate_probe_20261006")
    )
    parser.add_argument("--benefit-threshold", type=float, default=0.01)
    parser.add_argument("--selection-fraction", type=float, default=0.20)
    parser.add_argument("--window-ms", type=float, default=500.0)
    parser.add_argument("--trajectory-samples", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9527, 1337, 2026])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def load_gate_records(args: argparse.Namespace) -> list[GateRecord]:
    with (args.dataset_root / "alignment" / "modeling_clicks_strict.jsonl").open(
        "r", encoding="utf-8"
    ) as handle:
        source = {
            record["event_id"]: record
            for line in handle
            if line.strip()
            and (record := json.loads(line)).get("source", "human") == "human"
        }
    with args.oracle_results.open("r", encoding="utf-8") as handle:
        oracle = [json.loads(line) for line in handle if line.strip()]

    result = []
    for row in oracle:
        if int(row["click_type"]) != 1:
            continue
        record = source[row["event_id"]]
        image_area = max(float(record["image_width_px"]) * float(record["image_height_px"]), 1.0)
        previous_area = float(record["pre_mask_metrics"].get("prediction_area_px", 0.0)) / image_area
        context = np.asarray(
            [
                float(row["click_index"]) / 5.0,
                2.0 * float(record["click_x_norm"]) - 1.0,
                2.0 * float(record["click_y_norm"]) - 1.0,
                previous_area,
                float(row["original_predicted_iou"]),
            ],
            dtype=np.float32,
        )
        result.append(
            GateRecord(
                event_id=row["event_id"],
                task_id=row["task_run_id"],
                click_index=int(row["click_index"]),
                label=int(float(row["delta_iou"]) > args.benefit_threshold),
                delta_iou=float(row["delta_iou"]),
                context=context,
                trajectory=trajectory_tensor(record, args.window_ms, args.trajectory_samples),
            )
        )
    if not result:
        raise ValueError("No positive-click oracle records were found")
    return result


def stratified_group_folds(records: list[GateRecord], folds: int, seed: int) -> list[list[int]]:
    """Assign complete tasks to folds while balancing positive-bearing and negative-only tasks."""
    if folds < 2:
        raise ValueError("At least two folds are required")
    task_indices: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        task_indices.setdefault(record.task_id, []).append(index)
    if len(task_indices) < folds:
        raise ValueError("Fewer tasks than folds")
    positive_tasks = []
    negative_tasks = []
    for task_id, indices in task_indices.items():
        target = positive_tasks if any(records[index].label for index in indices) else negative_tasks
        target.append(task_id)
    rng = random.Random(seed)
    rng.shuffle(positive_tasks)
    rng.shuffle(negative_tasks)
    fold_tasks: list[list[str]] = [[] for _ in range(folds)]
    for task_list in (positive_tasks, negative_tasks):
        for offset, task_id in enumerate(task_list):
            fold_tasks[offset % folds].append(task_id)
    result = [[index for task_id in task_ids for index in task_indices[task_id]] for task_ids in fold_tasks]
    assigned = [index for fold in result for index in fold]
    if sorted(assigned) != list(range(len(records))):
        raise RuntimeError("Cross-validation fold assignment is incomplete")
    for fold in result:
        labels = {records[index].label for index in fold}
        if labels != {0, 1}:
            raise ValueError("Every held-out fold must contain both classes")
    return result


def arrays(
    records: list[GateRecord], indices: list[int], normalizer: TrainNormalizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [records[index] for index in indices]
    context = np.stack([normalizer.transform_context(record.context) for record in selected])
    trajectory = np.stack(
        [normalizer.transform_trajectory(record.trajectory) for record in selected]
    )
    labels = np.asarray([record.label for record in selected], dtype=np.float32)
    return context, trajectory, labels


def shuffled_trajectories(
    records: list[GateRecord], indices: list[int], trajectory: np.ndarray, seed: int
) -> tuple[np.ndarray, float]:
    rng = random.Random(seed)
    donors = list(range(len(indices)))
    strata: dict[int, list[int]] = {}
    for local_index, record_index in enumerate(indices):
        strata.setdefault(records[record_index].click_index, []).append(local_index)
    for local_indices in strata.values():
        if len(local_indices) < 2:
            continue
        permutation = local_indices.copy()
        for _ in range(100):
            rng.shuffle(permutation)
            if all(
                records[indices[target]].task_id != records[indices[donor]].task_id
                for target, donor in zip(local_indices, permutation)
            ):
                break
        for target, donor in zip(local_indices, permutation):
            donors[target] = donor
    fraction = float(
        np.mean(
            [
                records[indices[target]].task_id != records[indices[donor]].task_id
                for target, donor in enumerate(donors)
            ]
        )
    )
    return trajectory[np.asarray(donors)].copy(), fraction


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    context: np.ndarray,
    trajectory: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    logits = model(
        torch.from_numpy(context).to(device), torch.from_numpy(trajectory).to(device)
    )
    return logits.sigmoid().cpu().numpy()


def train_fixed(
    kind: str,
    seed: int,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> TrajectoryProbe:
    use_context = kind in {"context", "fused"}
    use_trajectory = kind in {"trajectory", "fused"}
    set_seed(seed)
    device = torch.device(args.device)
    model = TrajectoryProbe(use_context, use_trajectory, context_dim=train[0].shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    positives = float(train[2].sum())
    negatives = float(len(train[2]) - positives)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / max(positives, 1.0), device=device)
    )
    dataset = TensorDataset(*(torch.from_numpy(value) for value in train))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, generator=generator)
    for _ in range(args.epochs):
        model.train()
        for context, trajectory, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(context.to(device), trajectory.to(device)), labels.to(device)
            )
            loss.backward()
            optimizer.step()
    return model


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": roc_auc(labels.astype(np.int64), scores),
        "average_precision": average_precision(labels.astype(np.int64), scores),
        "brier": float(np.square(scores - labels).mean()),
    }


def policy_metrics(
    scores: np.ndarray, labels: np.ndarray, delta_iou: np.ndarray, selection_fraction: float
) -> dict[str, float | int]:
    count = max(1, int(math.ceil(len(scores) * selection_fraction)))
    selected = np.argsort(-scores, kind="stable")[:count]
    selected_delta = delta_iou[selected]
    return {
        "selected": count,
        "selected_fraction": count / len(scores),
        "beneficial_precision": float(labels[selected].mean()),
        "selected_mean_oracle_delta": float(selected_delta.mean()),
        "all_sample_policy_delta": float(selected_delta.sum() / len(scores)),
        "selected_harm_rate": float(np.mean(selected_delta < 0)),
    }


def task_bootstrap_auc_delta(
    records: list[GateRecord], left: np.ndarray, right: np.ndarray, samples: int, seed: int
) -> dict[str, float | int]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.task_id, []).append(index)
    task_ids = sorted(groups)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    observed = roc_auc(labels, left) - roc_auc(labels, right)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        chosen = rng.choice(task_ids, size=len(task_ids), replace=True)
        indices = np.concatenate([np.asarray(groups[task_id]) for task_id in chosen])
        delta = roc_auc(labels[indices], left[indices]) - roc_auc(labels[indices], right[indices])
        if math.isfinite(delta):
            values.append(delta)
    array = np.asarray(values)
    return {
        "delta_auc": observed,
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "bootstrap_samples": len(array),
    }


def task_bootstrap_policy_delta(
    records: list[GateRecord],
    left: np.ndarray,
    right: np.ndarray,
    selection_fraction: float,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.task_id, []).append(index)
    task_ids = sorted(groups)
    labels = np.asarray([record.label for record in records], dtype=np.float32)
    delta_iou = np.asarray([record.delta_iou for record in records], dtype=np.float64)

    def policy_delta(indices: np.ndarray) -> float:
        left_gain = policy_metrics(
            left[indices], labels[indices], delta_iou[indices], selection_fraction
        )["all_sample_policy_delta"]
        right_gain = policy_metrics(
            right[indices], labels[indices], delta_iou[indices], selection_fraction
        )["all_sample_policy_delta"]
        return float(left_gain - right_gain)

    all_indices = np.arange(len(records))
    observed = policy_delta(all_indices)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        chosen = rng.choice(task_ids, size=len(task_ids), replace=True)
        indices = np.concatenate([np.asarray(groups[task_id]) for task_id in chosen])
        values.append(policy_delta(indices))
    array = np.asarray(values)
    return {
        "delta_all_sample_policy_iou": observed,
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "bootstrap_samples": len(array),
    }


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def write_report(path: Path, payload: dict[str, Any]) -> None:
    ensemble = payload["ensemble"]
    decision = payload["decision"]
    lines = [
        "# CAIS positive-click trajectory gate probe",
        "",
        "## Question",
        "",
        "Among GT-audited locally correctable positive clicks, can the last 500 ms mouse trajectory "
        "identify corrections whose frozen-SAM oracle IoU gain exceeds 0.01?",
        "",
        f"- Records: {payload['dataset']['records']} / tasks: {payload['dataset']['tasks']}",
        f"- Beneficial corrections: {payload['dataset']['beneficial']} "
        f"({payload['dataset']['positive_rate']:.2%})",
        f"- Evaluation: {payload['config']['folds']}-fold task-grouped OOF, seeds {payload['config']['seeds']}",
        f"- Gate selection fraction: {payload['config']['selection_fraction']:.0%}",
        "",
        "## Seed-ensemble OOF results",
        "",
        "| Input/control | ROC-AUC | AP | Policy IoU gain over all records | Selected benefit precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "quality_only",
        "context",
        "trajectory_real",
        "fused_real",
        "fused_shuffle",
        "fused_null",
    ):
        metric = ensemble[name]["classification"]
        policy = ensemble[name]["policy"]
        lines.append(
            f"| {name} | {metric['roc_auc']:.4f} | {metric['average_precision']:.4f} | "
            f"{policy['all_sample_policy_delta']:+.6f} | {policy['beneficial_precision']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Paired task-bootstrap checks",
            "",
            "| Check | Delta | 95% CI | Pass |",
            "|---|---:|---:|---:|",
            f"| Fused Real - Context AUC | {ensemble['paired']['fused_minus_context_auc']['delta_auc']:+.4f} | "
            f"[{ensemble['paired']['fused_minus_context_auc']['ci95_low']:+.4f}, "
            f"{ensemble['paired']['fused_minus_context_auc']['ci95_high']:+.4f}] | "
            f"{decision['fused_beats_context']} |",
            f"| Fused Real - Shuffle AUC | {ensemble['paired']['real_minus_shuffle_auc']['delta_auc']:+.4f} | "
            f"[{ensemble['paired']['real_minus_shuffle_auc']['ci95_low']:+.4f}, "
            f"{ensemble['paired']['real_minus_shuffle_auc']['ci95_high']:+.4f}] | "
            f"{decision['real_beats_shuffle']} |",
            f"| Fused - Context policy IoU | "
            f"{ensemble['paired']['fused_minus_context_policy']['delta_all_sample_policy_iou']:+.6f} | "
            f"[{ensemble['paired']['fused_minus_context_policy']['ci95_low']:+.6f}, "
            f"{ensemble['paired']['fused_minus_context_policy']['ci95_high']:+.6f}] | "
            f"{decision['fused_policy_beats_context']} |",
            "",
            f"Proceed to learn trajectory offsets: `{decision['proceed_to_offset_learning']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.folds = 2
        args.epochs = 2
        args.bootstrap_samples = 100
        args.seeds = args.seeds[:1]
        args.output_dir = args.output_dir / "smoke"
    if not 0 < args.selection_fraction < 1:
        raise ValueError("selection-fraction must be in (0,1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_gate_records(args)
    labels = np.asarray([record.label for record in records], dtype=np.float32)
    delta_iou = np.asarray([record.delta_iou for record in records], dtype=np.float64)
    device = torch.device(args.device)

    seed_results: dict[str, Any] = {}
    ensemble_values: dict[str, list[np.ndarray]] = {}
    for seed in args.seeds:
        folds = stratified_group_folds(records, args.folds, seed)
        predictions = {
            "context": np.zeros(len(records), dtype=np.float32),
            "trajectory_real": np.zeros(len(records), dtype=np.float32),
            "trajectory_shuffle": np.zeros(len(records), dtype=np.float32),
            "trajectory_null": np.zeros(len(records), dtype=np.float32),
            "fused_real": np.zeros(len(records), dtype=np.float32),
            "fused_shuffle": np.zeros(len(records), dtype=np.float32),
            "fused_null": np.zeros(len(records), dtype=np.float32),
        }
        shuffle_fractions = []
        for fold_index, test_indices in enumerate(folds):
            test_set = set(test_indices)
            train_indices = [index for index in range(len(records)) if index not in test_set]
            normalizer = TrainNormalizer([records[index] for index in train_indices])
            train = arrays(records, train_indices, normalizer)
            test = arrays(records, test_indices, normalizer)
            shuffled, fraction = shuffled_trajectories(
                records, test_indices, test[1], seed + 100_000 + fold_index
            )
            shuffle_fractions.append(fraction)
            for kind_index, kind in enumerate(("context", "trajectory", "fused")):
                model = train_fixed(
                    kind,
                    seed + 10_000 * kind_index + fold_index,
                    train,
                    args,
                )
                real_scores = predict(model, test[0], test[1], device)
                predictions["context" if kind == "context" else f"{kind}_real"][test_indices] = real_scores
                if kind in {"trajectory", "fused"}:
                    predictions[f"{kind}_shuffle"][test_indices] = predict(
                        model, test[0], shuffled, device
                    )
                    predictions[f"{kind}_null"][test_indices] = predict(
                        model, test[0], np.zeros_like(test[1]), device
                    )
        seed_result = {
            name: {
                "classification": metrics(labels, scores),
                "policy": policy_metrics(scores, labels, delta_iou, args.selection_fraction),
            }
            for name, scores in predictions.items()
        }
        seed_result["shuffle_cross_task_fraction"] = float(np.mean(shuffle_fractions))
        seed_results[str(seed)] = seed_result
        for name, scores in predictions.items():
            ensemble_values.setdefault(name, []).append(scores)
        print(
            f"seed={seed} context_auc={seed_result['context']['classification']['roc_auc']:.4f} "
            f"trajectory_auc={seed_result['trajectory_real']['classification']['roc_auc']:.4f} "
            f"fused_auc={seed_result['fused_real']['classification']['roc_auc']:.4f} "
            f"shuffle_auc={seed_result['fused_shuffle']['classification']['roc_auc']:.4f}",
            flush=True,
        )

    ensemble_scores = {
        name: np.stack(values, axis=0).mean(axis=0) for name, values in ensemble_values.items()
    }
    ensemble = {
        name: {
            "classification": metrics(labels, scores),
            "policy": policy_metrics(scores, labels, delta_iou, args.selection_fraction),
        }
        for name, scores in ensemble_scores.items()
    }
    quality_only_scores = np.clip(
        1.0 - np.asarray([record.context[4] for record in records], dtype=np.float32), 0.0, 1.0
    )
    ensemble["quality_only"] = {
        "classification": metrics(labels, quality_only_scores),
        "policy": policy_metrics(
            quality_only_scores, labels, delta_iou, args.selection_fraction
        ),
    }
    ensemble["paired"] = {
        "fused_minus_context_auc": task_bootstrap_auc_delta(
            records,
            ensemble_scores["fused_real"],
            ensemble_scores["context"],
            args.bootstrap_samples,
            301_001,
        ),
        "real_minus_shuffle_auc": task_bootstrap_auc_delta(
            records,
            ensemble_scores["fused_real"],
            ensemble_scores["fused_shuffle"],
            args.bootstrap_samples,
            301_002,
        ),
        "real_minus_null_auc": task_bootstrap_auc_delta(
            records,
            ensemble_scores["fused_real"],
            ensemble_scores["fused_null"],
            args.bootstrap_samples,
            301_003,
        ),
        "fused_minus_context_policy": task_bootstrap_policy_delta(
            records,
            ensemble_scores["fused_real"],
            ensemble_scores["context"],
            args.selection_fraction,
            args.bootstrap_samples,
            302_001,
        ),
        "fused_minus_quality_auc": task_bootstrap_auc_delta(
            records,
            ensemble_scores["fused_real"],
            quality_only_scores,
            args.bootstrap_samples,
            302_002,
        ),
        "fused_minus_quality_policy": task_bootstrap_policy_delta(
            records,
            ensemble_scores["fused_real"],
            quality_only_scores,
            args.selection_fraction,
            args.bootstrap_samples,
            302_003,
        ),
    }
    decision = {
        "fused_beats_context": ensemble["paired"]["fused_minus_context_auc"]["ci95_low"] > 0,
        "real_beats_shuffle": ensemble["paired"]["real_minus_shuffle_auc"]["ci95_low"] > 0,
        "fused_policy_beats_context": ensemble["paired"]["fused_minus_context_policy"]["ci95_low"] > 0,
    }
    decision["proceed_to_offset_learning"] = all(decision.values())
    aggregate = {
        name: mean_std(
            [
                seed_results[str(seed)][name]["classification"]["roc_auc"]
                for seed in args.seeds
            ]
        )
        for name in ensemble_scores
    }
    payload = {
        "config": {
            "dataset_root": str(args.dataset_root),
            "oracle_results": str(args.oracle_results),
            "benefit_threshold": args.benefit_threshold,
            "selection_fraction": args.selection_fraction,
            "window_ms": args.window_ms,
            "trajectory_samples": args.trajectory_samples,
            "folds": args.folds,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "smoke_test": args.smoke_test,
        },
        "dataset": {
            "records": len(records),
            "tasks": len({record.task_id for record in records}),
            "beneficial": int(labels.sum()),
            "positive_rate": float(labels.mean()),
        },
        "seed_results": seed_results,
        "aggregate_auc": aggregate,
        "ensemble": ensemble,
        "decision": decision,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(args.output_dir / "REPORT.md", payload)
    print(json.dumps({"ensemble": ensemble, "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
