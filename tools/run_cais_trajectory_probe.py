"""Test whether human mouse trajectories predict a later corrective click.

This probe is deliberately independent from the mask decoder.  It asks a
smaller question before another CAIS fusion experiment: after clicks 1--4,
does the preceding 500 ms trajectory add information about whether the user
will click again, beyond basic click/mask context?
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ProbeRecord:
    event_id: str
    task_id: str
    split: str
    click_index: int
    label: int
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
        "--output-dir", type=Path, default=Path("outputs/cais_trajectory_probe_20261006")
    )
    parser.add_argument("--window-ms", type=float, default=500.0)
    parser.add_argument("--trajectory-samples", type=int, default=32)
    parser.add_argument("--max-click-index", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9527, 1337, 2026])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def trajectory_tensor(record: dict[str, Any], window_ms: float, sample_count: int) -> np.ndarray:
    """Return fixed-time [dx,dy,vx,vy,speed,valid] samples ending at the click."""
    points = (record.get("preceding_trajectory") or {}).get("points") or []
    if len(points) < 2:
        return np.zeros((6, sample_count), dtype=np.float32)

    values = np.asarray(points, dtype=np.float64)
    values = values[np.argsort(values[:, 0], kind="stable")]
    times, unique = np.unique(values[:, 0], return_index=True)
    values = values[unique]
    keep = times >= -float(window_ms)
    values = values[keep]
    if len(values) < 2:
        return np.zeros((6, sample_count), dtype=np.float32)

    click_x = float(record["click_x_norm"])
    click_y = float(record["click_y_norm"])
    grid_ms = np.linspace(-float(window_ms), 0.0, sample_count)
    valid = grid_ms >= float(values[0, 0])
    dx = np.interp(grid_ms, values[:, 0], values[:, 1]) - click_x
    dy = np.interp(grid_ms, values[:, 0], values[:, 2]) - click_y
    dt_seconds = max(float(window_ms) / max(sample_count - 1, 1) / 1000.0, 1e-6)
    vx = np.gradient(dx, dt_seconds)
    vy = np.gradient(dy, dt_seconds)
    speed = np.sqrt(vx * vx + vy * vy)
    channels = np.stack((dx, dy, vx, vy, speed), axis=0)
    channels[:, ~valid] = 0.0
    return np.concatenate((channels, valid[None].astype(np.float64)), axis=0).astype(np.float32)


def context_vector(record: dict[str, Any], max_click_index: int) -> np.ndarray:
    pre_metrics = record.get("pre_mask_metrics") or {}
    image_area = max(float(record["image_width_px"]) * float(record["image_height_px"]), 1.0)
    return np.asarray(
        [
            float(record["click_index"]) / max(float(max_click_index), 1.0),
            float(record.get("click_type", 1)),
            2.0 * float(record["click_x_norm"]) - 1.0,
            2.0 * float(record["click_y_norm"]) - 1.0,
            float(pre_metrics.get("iou", 0.0)),
            float(pre_metrics.get("prediction_area_px", 0.0)) / image_area,
        ],
        dtype=np.float32,
    )


def load_records(
    dataset_root: Path, window_ms: float, trajectory_samples: int, max_click_index: int
) -> list[ProbeRecord]:
    path = dataset_root / "alignment" / "modeling_clicks_strict.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        raw = [json.loads(line) for line in handle if line.strip()]
    human = [record for record in raw if record.get("source", "human") == "human"]
    task_last_click: dict[str, int] = {}
    task_splits: dict[str, set[str]] = {}
    for record in human:
        task_id = str(record["task_run_id"])
        task_last_click[task_id] = max(task_last_click.get(task_id, 0), int(record["click_index"]))
        task_splits.setdefault(task_id, set()).add(str(record["split"]))
    leaking = {task_id: splits for task_id, splits in task_splits.items() if len(splits) != 1}
    if leaking:
        raise ValueError(f"Tasks cross dataset splits: {list(leaking.items())[:3]}")

    result = []
    for record in human:
        click_index = int(record["click_index"])
        if click_index > max_click_index:
            continue
        task_id = str(record["task_run_id"])
        result.append(
            ProbeRecord(
                event_id=str(record["event_id"]),
                task_id=task_id,
                split=str(record["split"]),
                click_index=click_index,
                label=int(click_index < task_last_click[task_id]),
                context=context_vector(record, max_click_index),
                trajectory=trajectory_tensor(record, window_ms, trajectory_samples),
            )
        )
    return result


class TrainNormalizer:
    def __init__(self, records: list[ProbeRecord]) -> None:
        context = np.stack([record.context for record in records])
        self.context_mean = context.mean(axis=0)
        self.context_std = np.maximum(context.std(axis=0), 1e-6)

        trajectory = np.stack([record.trajectory for record in records])
        valid = trajectory[:, 5].astype(bool)
        values = trajectory[:, :5].transpose(0, 2, 1)[valid]
        if not len(values):
            raise ValueError("No valid human trajectory samples in the training split")
        self.trajectory_median = np.median(values, axis=0)
        lower = np.quantile(values, 0.05, axis=0)
        upper = np.quantile(values, 0.95, axis=0)
        self.trajectory_scale = np.maximum(upper - lower, 1e-6)

    def transform_context(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.context_mean) / self.context_std).astype(np.float32)

    def transform_trajectory(self, value: np.ndarray) -> np.ndarray:
        output = value.copy()
        valid = output[5].astype(bool)
        normalized = (output[:5] - self.trajectory_median[:, None]) / self.trajectory_scale[:, None]
        output[:5] = np.clip(normalized, -5.0, 5.0)
        output[:5, ~valid] = 0.0
        return output.astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "context_mean": self.context_mean.tolist(),
            "context_std": self.context_std.tolist(),
            "trajectory_median": self.trajectory_median.tolist(),
            "trajectory_scale_q95_minus_q05": self.trajectory_scale.tolist(),
        }


class TrajectoryProbe(nn.Module):
    def __init__(self, use_context: bool, use_trajectory: bool, context_dim: int = 6) -> None:
        super().__init__()
        if not use_context and not use_trajectory:
            raise ValueError("At least one probe input is required")
        self.use_context = use_context
        self.use_trajectory = use_trajectory
        feature_dim = 0
        if use_context:
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, 32), nn.ReLU(), nn.LayerNorm(32)
            )
            feature_dim += 32
        if use_trajectory:
            self.trajectory_encoder = nn.Sequential(
                nn.Conv1d(6, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1),
                nn.ReLU(),
            )
            feature_dim += 128
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, context: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        features = []
        if self.use_context:
            features.append(self.context_encoder(context))
        if self.use_trajectory:
            encoded = self.trajectory_encoder(trajectory)
            features.extend((encoded.mean(dim=-1), encoded.amax(dim=-1)))
        return self.head(torch.cat(features, dim=-1)).squeeze(-1)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0)) / comparisons.size)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / positives)


def best_balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = scores >= threshold
        tpr = float(predicted[labels == 1].mean())
        tnr = float((~predicted[labels == 0]).mean())
        value = 0.5 * (tpr + tnr)
        if value > best[0]:
            best = (value, float(threshold))
    return best[1]


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    tpr = float(predicted[labels == 1].mean())
    tnr = float((~predicted[labels == 0]).mean())
    return {
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "brier": float(np.square(scores - labels).mean()),
        "threshold": float(threshold),
    }


def arrays(
    records: list[ProbeRecord], normalizer: TrainNormalizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context = np.stack([normalizer.transform_context(record.context) for record in records])
    trajectory = np.stack(
        [normalizer.transform_trajectory(record.trajectory) for record in records]
    )
    labels = np.asarray([record.label for record in records], dtype=np.float32)
    return context, trajectory, labels


def shuffled_trajectories(
    records: list[ProbeRecord], trajectory: np.ndarray, seed: int
) -> tuple[np.ndarray, list[int]]:
    rng = random.Random(seed)
    donors = list(range(len(records)))
    strata: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        strata.setdefault(record.click_index, []).append(index)
    for indices in strata.values():
        permutation = indices.copy()
        for _ in range(100):
            rng.shuffle(permutation)
            if all(records[target].task_id != records[donor].task_id for target, donor in zip(indices, permutation)):
                break
        else:
            permutation = indices[1:] + indices[:1]
        for target, donor in zip(indices, permutation):
            donors[target] = donor
    return trajectory[np.asarray(donors)].copy(), donors


@torch.no_grad()
def predict(
    model: nn.Module,
    context: np.ndarray,
    trajectory: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    logits = model(
        torch.from_numpy(context).to(device), torch.from_numpy(trajectory).to(device)
    )
    return logits.sigmoid().cpu().numpy()


def train_probe(
    kind: str,
    seed: int,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> tuple[TrajectoryProbe, dict[str, Any]]:
    use_context = kind in {"context", "fused"}
    use_trajectory = kind in {"trajectory", "fused"}
    set_seed(seed)
    device = torch.device(args.device)
    model = TrajectoryProbe(use_context, use_trajectory).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    positives = float(train[2].sum())
    negatives = float(len(train[2]) - positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    dataset = TensorDataset(*(torch.from_numpy(value) for value in train))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, generator=generator)

    best_auc = -math.inf
    best_epoch = 0
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for context, trajectory, labels in loader:
            context = context.to(device)
            trajectory = trajectory.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(context, trajectory), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
        val_scores = predict(model, val[0], val[1], device)
        val_auc = roc_auc(val[2].astype(np.int64), val_scores)
        history.append({"epoch": epoch, "train_loss": total_loss / len(dataset), "val_auc": val_auc})
        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Probe training never produced a checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_auc": best_auc, "history": history}


def task_bootstrap_delta(
    records: list[ProbeRecord],
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.task_id, []).append(index)
    task_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        selected = rng.choice(task_ids, size=len(task_ids), replace=True)
        indices = np.concatenate([np.asarray(groups[task_id]) for task_id in selected])
        delta = roc_auc(labels[indices].astype(np.int64), left[indices]) - roc_auc(
            labels[indices].astype(np.int64), right[indices]
        )
        if math.isfinite(delta):
            deltas.append(delta)
    values = np.asarray(deltas)
    return {
        "delta": roc_auc(labels.astype(np.int64), left) - roc_auc(labels.astype(np.int64), right),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "bootstrap_samples": int(len(values)),
    }


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)
        args.bootstrap_samples = min(args.bootstrap_samples, 50)
        args.seeds = args.seeds[:1]
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(
        args.dataset_root, args.window_ms, args.trajectory_samples, args.max_click_index
    )
    by_split = {split: [record for record in records if record.split == split] for split in ("train", "val", "test")}
    normalizer = TrainNormalizer(by_split["train"])
    data = {split: arrays(split_records, normalizer) for split, split_records in by_split.items()}

    results: dict[str, Any] = {
        "config": {
            **vars(args),
            "dataset_root": str(args.dataset_root),
            "output_dir": str(args.output_dir),
        },
        "dataset": {
            split: {
                "records": len(split_records),
                "tasks": len({record.task_id for record in split_records}),
                "positive_rate": float(np.mean([record.label for record in split_records])),
            }
            for split, split_records in by_split.items()
        },
        "normalizer": normalizer.to_dict(),
        "seeds": {},
    }
    checkpoint_payload: dict[str, Any] = {}
    ensemble_predictions: dict[str, list[np.ndarray]] = {}
    for seed in args.seeds:
        seed_result: dict[str, Any] = {}
        predictions: dict[str, np.ndarray] = {}
        for kind in ("context", "trajectory", "fused"):
            model, training = train_probe(kind, seed, data["train"], data["val"], args)
            device = torch.device(args.device)
            val_scores = predict(model, data["val"][0], data["val"][1], device)
            threshold = best_balanced_threshold(data["val"][2].astype(np.int64), val_scores)
            test_scores = predict(model, data["test"][0], data["test"][1], device)
            predictions[kind] = test_scores
            kind_result: dict[str, Any] = {
                "training": training,
                "test_real": classification_metrics(
                    data["test"][2].astype(np.int64), test_scores, threshold
                ),
            }
            if kind in {"trajectory", "fused"}:
                shuffled, donors = shuffled_trajectories(
                    by_split["test"], data["test"][1], seed + 100_000
                )
                shuffled_scores = predict(model, data["test"][0], shuffled, device)
                null_scores = predict(
                    model, data["test"][0], np.zeros_like(data["test"][1]), device
                )
                predictions[f"{kind}_shuffled"] = shuffled_scores
                predictions[f"{kind}_null"] = null_scores
                kind_result["test_shuffled"] = classification_metrics(
                    data["test"][2].astype(np.int64), shuffled_scores, threshold
                )
                kind_result["test_null"] = classification_metrics(
                    data["test"][2].astype(np.int64), null_scores, threshold
                )
                kind_result["shuffle_cross_task_fraction"] = float(
                    np.mean(
                        [
                            record.task_id != by_split["test"][donor].task_id
                            for record, donor in zip(by_split["test"], donors)
                        ]
                    )
                )
            seed_result[kind] = kind_result
            checkpoint_payload[f"{kind}_seed_{seed}"] = {
                "state_dict": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
                "best_epoch": training["best_epoch"],
            }

        labels = data["test"][2]
        seed_result["paired_task_deltas"] = {
            "fused_minus_context_auc": task_bootstrap_delta(
                by_split["test"], labels, predictions["fused"], predictions["context"],
                args.bootstrap_samples, seed + 200_001,
            ),
            "fused_real_minus_shuffled_auc": task_bootstrap_delta(
                by_split["test"], labels, predictions["fused"], predictions["fused_shuffled"],
                args.bootstrap_samples, seed + 200_002,
            ),
            "fused_real_minus_null_auc": task_bootstrap_delta(
                by_split["test"], labels, predictions["fused"], predictions["fused_null"],
                args.bootstrap_samples, seed + 200_003,
            ),
        }
        results["seeds"][str(seed)] = seed_result
        for name, values in predictions.items():
            ensemble_predictions.setdefault(name, []).append(values)
        print(
            f"seed={seed} context_auc={seed_result['context']['test_real']['roc_auc']:.4f} "
            f"trajectory_auc={seed_result['trajectory']['test_real']['roc_auc']:.4f} "
            f"fused_auc={seed_result['fused']['test_real']['roc_auc']:.4f} "
            f"fused_shuffle_auc={seed_result['fused']['test_shuffled']['roc_auc']:.4f}",
            flush=True,
        )

    results["aggregate"] = {}
    metric_paths = {
        "context_real_auc": ("context", "test_real"),
        "trajectory_real_auc": ("trajectory", "test_real"),
        "trajectory_shuffled_auc": ("trajectory", "test_shuffled"),
        "fused_real_auc": ("fused", "test_real"),
        "fused_shuffled_auc": ("fused", "test_shuffled"),
        "fused_null_auc": ("fused", "test_null"),
    }
    for name, (kind, control) in metric_paths.items():
        results["aggregate"][name] = mean_std(
            [results["seeds"][str(seed)][kind][control]["roc_auc"] for seed in args.seeds]
        )
    for name in ("fused_minus_context_auc", "fused_real_minus_shuffled_auc", "fused_real_minus_null_auc"):
        results["aggregate"][name] = mean_std(
            [results["seeds"][str(seed)]["paired_task_deltas"][name]["delta"] for seed in args.seeds]
        )

    ensemble = {
        name: np.stack(seed_values, axis=0).mean(axis=0)
        for name, seed_values in ensemble_predictions.items()
    }
    ensemble_thresholds = {
        "context": float(
            np.mean(
                [results["seeds"][str(seed)]["context"]["test_real"]["threshold"] for seed in args.seeds]
            )
        ),
        "trajectory": float(
            np.mean(
                [results["seeds"][str(seed)]["trajectory"]["test_real"]["threshold"] for seed in args.seeds]
            )
        ),
        "fused": float(
            np.mean(
                [results["seeds"][str(seed)]["fused"]["test_real"]["threshold"] for seed in args.seeds]
            )
        ),
    }
    test_labels = data["test"][2].astype(np.int64)
    results["ensemble"] = {
        "context_real": classification_metrics(
            test_labels, ensemble["context"], ensemble_thresholds["context"]
        ),
        "trajectory_real": classification_metrics(
            test_labels, ensemble["trajectory"], ensemble_thresholds["trajectory"]
        ),
        "trajectory_shuffled": classification_metrics(
            test_labels, ensemble["trajectory_shuffled"], ensemble_thresholds["trajectory"]
        ),
        "trajectory_null": classification_metrics(
            test_labels, ensemble["trajectory_null"], ensemble_thresholds["trajectory"]
        ),
        "fused_real": classification_metrics(
            test_labels, ensemble["fused"], ensemble_thresholds["fused"]
        ),
        "fused_shuffled": classification_metrics(
            test_labels, ensemble["fused_shuffled"], ensemble_thresholds["fused"]
        ),
        "fused_null": classification_metrics(
            test_labels, ensemble["fused_null"], ensemble_thresholds["fused"]
        ),
        "paired_task_deltas": {
            "fused_minus_context_auc": task_bootstrap_delta(
                by_split["test"], test_labels, ensemble["fused"], ensemble["context"],
                args.bootstrap_samples, 310_001,
            ),
            "fused_real_minus_shuffled_auc": task_bootstrap_delta(
                by_split["test"], test_labels, ensemble["fused"], ensemble["fused_shuffled"],
                args.bootstrap_samples, 310_002,
            ),
            "fused_real_minus_null_auc": task_bootstrap_delta(
                by_split["test"], test_labels, ensemble["fused"], ensemble["fused_null"],
                args.bootstrap_samples, 310_003,
            ),
        },
    }
    success = results["ensemble"]["paired_task_deltas"]
    results["decision"] = {
        "trajectory_semantics_supported": bool(
            success["fused_minus_context_auc"]["ci95_low"] > 0
            and success["fused_real_minus_shuffled_auc"]["ci95_low"] > 0
        ),
        "decision_checks": {
            "trajectory_only_auc_at_least_0.55_heuristic": bool(
                results["ensemble"]["trajectory_real"]["roc_auc"] >= 0.55
            ),
            "fused_beats_context_ci_above_zero": bool(
                success["fused_minus_context_auc"]["ci95_low"] > 0
            ),
            "real_beats_shuffle_ci_above_zero": bool(
                success["fused_real_minus_shuffled_auc"]["ci95_low"] > 0
            ),
            "real_beats_null_ci_above_zero": bool(
                success["fused_real_minus_null_auc"]["ci95_low"] > 0
            ),
        },
    }

    serializable = json.loads(json.dumps(results, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value))
    (args.output_dir / "trajectory_probe_results.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "checkpoints": checkpoint_payload,
            "normalizer": normalizer.to_dict(),
            "config": serializable["config"],
        },
        args.output_dir / "trajectory_probe_checkpoints.pt",
    )
    print(json.dumps(results["aggregate"], indent=2), flush=True)
    print(json.dumps({"ensemble": results["ensemble"], "decision": results["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
