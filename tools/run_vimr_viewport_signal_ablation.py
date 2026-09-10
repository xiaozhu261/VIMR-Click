"""Controlled viewport-signal ablation with frozen, shared click trajectories.

The four trainable variants differ only in the viewport state supplied to a
Viewport-only VIMR model.  Corrective clicks are generated once by the frozen
EfficientViT-SAM baseline and then reused by every mode, epoch, and seed.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from efficientvit.models.vimr import VIMREfficientViTSam, vimr_interactive_loss
from tools.eval_vimr_coco_quick import (
    baseline_round,
    corrective_click,
    decode_mask,
    encode_image,
    freeze_pretrained_modules,
    initial_click,
    load_split,
    target_to_sam,
    viewport_from_click,
)
from tools.run_vimr_coco_protocol import (
    build_model,
    file_sha256,
    load_vimr_checkpoint,
    sample_manifest,
    save_best_checkpoint,
    selection_direction,
    set_vimr_train_mode,
    summarize_protocol,
)

MODES = ("default", "synthetic", "shuffled", "correct")
REPORT_METRICS = (
    "mIoU@1",
    "mIoU@2",
    "mIoU@3",
    "mIoU@5",
    "NoC@85",
    "NoC@90",
    "Success@85/5",
    "Success@90/5",
    "RegressionRate",
    "LateRegressionRate@4-5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=ROOT_DIR / "dataset/coco")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT_DIR / "outputs/vimr_viewport_signal_ablation"
    )
    parser.add_argument("--train-instances", type=int, default=500)
    parser.add_argument("--eval-instances", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-clicks", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--shuffle-seed", type=int, default=90210)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-area-ratio", type=float, default=0.60)
    parser.add_argument("--selection-metric", default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rerun-existing", action="store_true")
    return parser.parse_args()


def mask_iou(probability: torch.Tensor, target: torch.Tensor) -> float:
    prediction = probability >= 0.5
    truth = target >= 0.5
    intersection = (prediction & truth).sum().float()
    union = (prediction | truth).sum().float()
    return float((intersection / union.clamp_min(1)).detach().cpu())


def default_trajectory(max_clicks: int) -> list[list[float]]:
    return [[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(max_clicks)]


def synthetic_trajectory(click_coords: list[list[list[float]]]) -> list[list[float]]:
    states: list[list[float]] = []
    previous = None
    for round_index, coords_list in enumerate(click_coords):
        coords = torch.tensor(coords_list, dtype=torch.float32).reshape(1, 1, 2)
        state = viewport_from_click(coords, round_index, previous)
        states.append(state[0].tolist())
        previous = state
    return states


def oracle_bbox_trajectory(target: torch.Tensor, max_clicks: int, margin: float = 0.8) -> list[list[float]]:
    """GT-bbox oracle camera path used only as an upper-bound diagnostic."""
    foreground = torch.nonzero(target[0, 0] >= 0.5, as_tuple=False)
    if foreground.numel() == 0:
        return default_trajectory(max_clicks)
    y_min, x_min = foreground.amin(dim=0).float()
    y_max, x_max = foreground.amax(dim=0).float()
    height, width = target.shape[-2:]
    cx = float(((x_min + x_max + 1.0) / 2.0 / width).cpu())
    cy = float(((y_min + y_max + 1.0) / 2.0 / height).cpu())
    width_fraction = float(((x_max - x_min + 1.0) / width).cpu())
    height_fraction = float(((y_max - y_min + 1.0) / height).cpu())
    zoom = min(8.0, max(1.0, margin / max(width_fraction, height_fraction)))
    log_zoom = math.log(zoom)
    states = default_trajectory(max_clicks)
    for round_index in range(1, max_clicks):
        states[round_index] = [cx, cy, log_zoom, 0.0, 0.0, 0.0, 0.0]
    if max_clicks > 1:
        states[1][3] = cx - 0.5
        states[1][4] = cy - 0.5
        states[1][5] = log_zoom
        states[1][6] = 3.0  # combined pan + zoom
    return states


def make_derangement(size: int, seed: int) -> list[int]:
    """Return a deterministic no-self source index for every target index."""
    if size < 2:
        return list(range(size))
    order = list(range(size))
    random.Random(seed).shuffle(order)
    mapping = list(range(size))
    for position, target_index in enumerate(order):
        mapping[target_index] = order[(position + 1) % size]
    if any(index == source for index, source in enumerate(mapping)):
        raise AssertionError("Failed to construct a derangement")
    return mapping


def canonical_interaction(
    model: VIMREfficientViTSam,
    sample: dict[str, Any],
    device: torch.device,
    max_clicks: int,
) -> dict[str, Any]:
    target = target_to_sam(decode_mask(sample), device)
    embedding = encode_image(model, sample["path"], device)
    current_coords, current_labels = initial_click(target)
    all_coords: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    click_coords: list[list[list[float]]] = []
    click_labels: list[list[int]] = []
    baseline_ious: list[float] = []
    previous = None
    for round_index in range(max_clicks):
        click_coords.append(current_coords.detach().cpu().tolist()[0])
        click_labels.append(current_labels.detach().cpu().tolist()[0])
        all_coords.append(current_coords)
        all_labels.append(current_labels)
        probability = baseline_round(
            model,
            embedding,
            torch.cat(all_coords, dim=1),
            torch.cat(all_labels, dim=1),
            previous,
        )
        baseline_ious.append(mask_iou(probability, target))
        previous = probability.detach()
        if round_index + 1 < max_clicks:
            current_coords, current_labels = corrective_click(previous, target)
    return {
        "image_id": int(sample["image"]["id"]),
        "annotation_id": int(sample["annotation"]["id"]),
        "category_id": int(sample["annotation"]["category_id"]),
        "click_coords": click_coords,
        "click_labels": click_labels,
        "baseline_ious": baseline_ious,
        "synthetic_viewports": synthetic_trajectory(click_coords),
        "oracle_viewports": oracle_bbox_trajectory(target, max_clicks),
    }


def generate_canonical_interactions(
    model: VIMREfficientViTSam,
    samples: list[dict[str, Any]],
    device: torch.device,
    max_clicks: int,
    progress_every: int,
    prefix: str,
) -> list[dict[str, Any]]:
    model.eval()
    records = []
    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            record = canonical_interaction(model, sample, device, max_clicks)
            records.append(record)
            if index == 1 or index % progress_every == 0 or index == len(samples):
                print(
                    f"{prefix} {index}/{len(samples)} baseline IoU@{max_clicks}="
                    f"{record['baseline_ious'][-1]:.4f}",
                    flush=True,
                )
    return records


def trajectory_for_mode(
    mode: str,
    index: int,
    records: list[dict[str, Any]],
    shuffled_sources: list[int],
    max_clicks: int,
) -> list[list[float]]:
    if mode == "default":
        return default_trajectory(max_clicks)
    if mode == "synthetic":
        return records[index]["synthetic_viewports"]
    if mode == "shuffled":
        return records[shuffled_sources[index]]["synthetic_viewports"]
    if mode == "correct":
        return records[index]["oracle_viewports"]
    raise ValueError(f"Unknown viewport mode: {mode}")


def run_fixed_sequence(
    model: VIMREfficientViTSam,
    embedding: torch.Tensor,
    target: torch.Tensor,
    interaction: dict[str, Any],
    viewport_trajectory: list[list[float]],
    train: bool,
) -> list[float]:
    previous = None
    ious = []
    for coords_list, labels_list, viewport_list in zip(
        interaction["click_coords"], interaction["click_labels"], viewport_trajectory
    ):
        coords = torch.tensor(coords_list, device=target.device, dtype=target.dtype).reshape(1, 1, 2)
        labels = torch.tensor(labels_list, device=target.device, dtype=torch.long).reshape(1, 1)
        viewport = torch.tensor(viewport_list, device=target.device, dtype=target.dtype).reshape(1, 7)
        output = model(
            image_embeddings=embedding,
            point_coords=coords,
            point_labels=labels,
            previous_mask=previous,
            viewport_state=viewport,
            multimask_output=False,
        )
        probability = output["final_mask"]
        if train:
            losses = vimr_interactive_loss(
                probability,
                target,
                previous,
                lambda_safe=0.0,
                lambda_monotonic=0.0,
            )
            (losses["loss"] / len(interaction["click_coords"])).backward()
        ious.append(mask_iou(probability, target))
        previous = probability.detach()
    return ious


def evaluate_fixed(
    model: VIMREfficientViTSam,
    samples: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    shuffled_sources: list[int],
    mode: str,
    device: torch.device,
    max_clicks: int,
    progress_every: int,
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for index, (sample, interaction) in enumerate(zip(samples, interactions), start=1):
            target = target_to_sam(decode_mask(sample), device)
            embedding = encode_image(model, sample["path"], device)
            trajectory = trajectory_for_mode(mode, index - 1, interactions, shuffled_sources, max_clicks)
            ious = run_fixed_sequence(model, embedding, target, interaction, trajectory, train=False)
            rows.append(
                {
                    "image_id": interaction["image_id"],
                    "annotation_id": interaction["annotation_id"],
                    "category_id": interaction["category_id"],
                    "viewport_source_annotation_id": (
                        interactions[shuffled_sources[index - 1]]["annotation_id"]
                        if mode == "shuffled"
                        else interaction["annotation_id"]
                    ),
                    "ious": ious,
                }
            )
            if index == 1 or index % progress_every == 0 or index == len(samples):
                print(f"{prefix} {index}/{len(samples)} IoU@{max_clicks}={ious[-1]:.4f}", flush=True)
    return rows, summarize_protocol(rows, max_clicks)


def train_one_epoch_fixed(
    model: VIMREfficientViTSam,
    optimizer: torch.optim.Optimizer,
    samples: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    shuffled_sources: list[int],
    mode: str,
    device: torch.device,
    epoch: int,
    seed: int,
    max_clicks: int,
    progress_every: int,
) -> dict[str, float]:
    set_vimr_train_mode(model)
    order = list(range(len(samples)))
    random.Random(seed * 10_000 + epoch).shuffle(order)
    grad_norms = []
    final_ious = []
    for step, sample_index in enumerate(order, start=1):
        sample = samples[sample_index]
        interaction = interactions[sample_index]
        target = target_to_sam(decode_mask(sample), device)
        embedding = encode_image(model, sample["path"], device)
        trajectory = trajectory_for_mode(
            mode, sample_index, interactions, shuffled_sources, max_clicks
        )
        optimizer.zero_grad(set_to_none=True)
        ious = run_fixed_sequence(model, embedding, target, interaction, trajectory, train=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
        optimizer.step()
        grad_norms.append(float(grad_norm.detach().cpu()))
        final_ious.append(ious[-1])
        if step == 1 or step % progress_every == 0 or step == len(order):
            print(
                f"mode={mode} seed={seed} epoch={epoch} train={step}/{len(order)} "
                f"IoU@{max_clicks}={ious[-1]:.4f} grad_norm={grad_norms[-1]:.4f}",
                flush=True,
            )
    return {
        "mean_train_iou_at_max_clicks": float(np.mean(final_ious)),
        "mean_gradient_norm": float(np.mean(grad_norms)),
    }


def aggregate_mode(
    baseline: dict[str, float], seed_metrics: dict[int, dict[str, float]]
) -> dict[str, dict[str, float]]:
    aggregate = {}
    for metric, baseline_value in baseline.items():
        values = np.asarray([seed_metrics[seed][metric] for seed in sorted(seed_metrics)], dtype=np.float64)
        aggregate[metric] = {
            "baseline": baseline_value,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "delta_vs_baseline": float(values.mean() - baseline_value),
        }
    return aggregate


def paired_contrast(
    left: dict[int, dict[str, float]], right: dict[int, dict[str, float]]
) -> dict[str, dict[str, Any]]:
    metrics = left[next(iter(left))]
    output = {}
    for metric in metrics:
        values = np.asarray(
            [left[seed][metric] - right[seed][metric] for seed in sorted(left)], dtype=np.float64
        )
        output[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "per_seed": values.tolist(),
        }
    return output


def write_per_instance(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("image_id", "annotation_id", "category_id", "viewport_source_annotation_id", "click", "iou")
        )
        for row in rows:
            for click, iou in enumerate(row["ious"], start=1):
                writer.writerow(
                    (
                        row["image_id"],
                        row["annotation_id"],
                        row["category_id"],
                        row["viewport_source_annotation_id"],
                        click,
                        iou,
                    )
                )


def write_progress(output: Path, **fields: Any) -> None:
    payload = {"updated_at_unix": time.time(), **fields}
    (output / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_report(output: Path, result: dict[str, Any]) -> None:
    baseline = result["baseline"]
    aggregates = result["aggregates"]
    with (output / "viewport_ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "metric", "mean", "std", "delta_vs_baseline"))
        for metric, value in baseline.items():
            writer.writerow(("baseline", metric, value, 0.0, 0.0))
        for mode in MODES:
            for metric, values in aggregates[mode].items():
                writer.writerow((mode, metric, values["mean"], values["std"], values["delta_vs_baseline"]))

    criterion = result["viewport_signal_criterion"]
    lines = [
        "# Viewport signal controlled ablation",
        "",
        "All four trainable rows are Viewport-only VIMR models with identical initialization per seed, "
        "split, schedule, and frozen baseline-derived click coordinates. Only the viewport trajectory changes.",
        "",
        "`correct` is a GT-instance-bbox oracle camera trajectory, not recorded user zoom/pan data; it is an upper bound.",
        "`shuffled` is a no-self instance permutation of complete synthetic trajectories.",
        "",
        "| Mode | " + " | ".join(REPORT_METRICS) + " |",
        "| --- | " + " | ".join("---:" for _ in REPORT_METRICS) + " |",
    ]
    baseline_cells = [f"{baseline[metric]:.6f}" for metric in REPORT_METRICS]
    lines.append("| Frozen baseline | " + " | ".join(baseline_cells) + " |")
    for mode in MODES:
        cells = [
            f"{aggregates[mode][metric]['mean']:.6f} ± {aggregates[mode][metric]['std']:.6f}"
            for metric in REPORT_METRICS
        ]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines.extend(
        (
            "",
            "## Predeclared viewport-signal check",
            "",
            "A positive result requires both paired mIoU@5 effects to be at least 0.01 and positive "
            "for every seed: synthetic − default and synthetic − shuffled.",
            "",
            f"- synthetic − default: {criterion['synthetic_minus_default_mIoU@5']:+.6f}",
            f"- synthetic − shuffled: {criterion['synthetic_minus_shuffled_mIoU@5']:+.6f}",
            f"- Result: **{criterion['verdict']}**",
        )
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_interactions(
    samples: list[dict[str, Any]], interactions: list[dict[str, Any]], max_clicks: int
) -> None:
    if len(samples) != len(interactions):
        raise ValueError("Sample/interaction count mismatch")
    for sample, interaction in zip(samples, interactions):
        if int(sample["annotation"]["id"]) != interaction["annotation_id"]:
            raise ValueError("Interaction manifest does not match split order")
        for key in ("click_coords", "click_labels", "baseline_ious", "synthetic_viewports", "oracle_viewports"):
            if len(interaction[key]) != max_clicks:
                raise ValueError(f"Malformed {key} for annotation {interaction['annotation_id']}")


def main() -> None:
    args = parse_args()
    if args.max_clicks != 5:
        print("Warning: report columns and late-stage metrics are designed for exactly 5 clicks", flush=True)
    if min(args.train_instances, args.eval_instances) < 2:
        print("Warning: shuffled trajectory cannot be a no-self permutation when a split has <2 samples", flush=True)
    selection_metric = args.selection_metric or f"mIoU@{args.max_clicks}"
    direction = selection_direction(selection_metric)
    device = torch.device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    checkpoint_hash = file_sha256(args.checkpoint)

    train_samples, eval_samples = load_split(
        args.coco_root,
        args.train_instances,
        args.eval_instances,
        args.split_seed,
        args.min_area_ratio,
        args.max_area_ratio,
    )
    split = {
        "split_seed": args.split_seed,
        "train": sample_manifest(train_samples),
        "validation": sample_manifest(eval_samples),
    }
    (output / "split_manifest.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    canonical_path = output / "canonical_interactions.json"
    if canonical_path.is_file() and not args.rerun_existing:
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        if (
            canonical.get("max_clicks") != args.max_clicks
            or canonical.get("checkpoint_sha256") != checkpoint_hash
        ):
            raise ValueError("Existing canonical interactions do not match this run; use --rerun-existing")
        train_interactions = canonical["train"]
        eval_interactions = canonical["validation"]
        validate_interactions(train_samples, train_interactions, args.max_clicks)
        validate_interactions(eval_samples, eval_interactions, args.max_clicks)
        print(f"Reusing locked click trajectories: {canonical_path}", flush=True)
    else:
        write_progress(output, stage="generating_canonical_clicks", completed=False)
        baseline_model = build_model(
            args.checkpoint,
            device,
            args.seeds[0],
            lambda_safe=0.0,
            lambda_monotonic=0.0,
            use_viewport=True,
            use_history=False,
            use_residual=False,
            use_safe_update=False,
        )
        freeze_pretrained_modules(baseline_model)
        train_interactions = generate_canonical_interactions(
            baseline_model,
            train_samples,
            device,
            args.max_clicks,
            args.progress_every,
            "canonical train clicks",
        )
        eval_interactions = generate_canonical_interactions(
            baseline_model,
            eval_samples,
            device,
            args.max_clicks,
            args.progress_every,
            "canonical validation clicks",
        )
        canonical = {
            "definition": "Fixed clicks generated once by frozen EfficientViT-SAM with accumulated clicks and previous-mask prompt",
            "max_clicks": args.max_clicks,
            "checkpoint_sha256": checkpoint_hash,
            "train": train_interactions,
            "validation": eval_interactions,
        }
        canonical_path.write_text(json.dumps(canonical, indent=2), encoding="utf-8")
        del baseline_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_shuffle = make_derangement(len(train_samples), args.shuffle_seed)
    eval_shuffle = make_derangement(len(eval_samples), args.shuffle_seed + 1)
    shuffle_manifest = {
        "shuffle_seed": args.shuffle_seed,
        "train_source_indices": train_shuffle,
        "validation_source_indices": eval_shuffle,
        "train_fixed_points": sum(index == source for index, source in enumerate(train_shuffle)),
        "validation_fixed_points": sum(index == source for index, source in enumerate(eval_shuffle)),
    }
    (output / "shuffled_viewport_manifest.json").write_text(
        json.dumps(shuffle_manifest, indent=2), encoding="utf-8"
    )

    baseline_rows = [
        {
            "image_id": item["image_id"],
            "annotation_id": item["annotation_id"],
            "category_id": item["category_id"],
            "ious": item["baseline_ious"],
        }
        for item in eval_interactions
    ]
    baseline_metrics = summarize_protocol(baseline_rows, args.max_clicks)
    mode_seed_metrics: dict[str, dict[int, dict[str, float]]] = {mode: {} for mode in MODES}
    mode_records: dict[str, dict[str, Any]] = {}

    for mode_index, mode in enumerate(MODES, start=1):
        mode_dir = output / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        seed_records = {}
        for seed_index, seed in enumerate(args.seeds, start=1):
            seed_dir = mode_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = seed_dir / "metrics.json"
            if metrics_path.is_file() and not args.rerun_existing:
                seed_record = json.loads(metrics_path.read_text(encoding="utf-8"))
                mode_seed_metrics[mode][seed] = seed_record["metrics"]
                seed_records[str(seed)] = seed_record
                print(f"Reusing completed mode={mode} seed={seed}", flush=True)
                continue

            write_progress(
                output,
                stage="training",
                completed=False,
                mode=mode,
                mode_index=mode_index,
                mode_count=len(MODES),
                seed=seed,
                seed_index=seed_index,
                seed_count=len(args.seeds),
                epoch=0,
                epochs=args.epochs,
            )
            model = build_model(
                args.checkpoint,
                device,
                seed,
                lambda_safe=0.0,
                lambda_monotonic=0.0,
                gate_init_probability=0.1,
                use_viewport=True,
                use_history=False,
                use_residual=False,
                use_safe_update=False,
            )
            trainable = freeze_pretrained_modules(model)
            trainable_count = sum(parameter.numel() for parameter in trainable)
            print(f"mode={mode} seed={seed} trainable parameters={trainable_count:,}", flush=True)
            optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
            best_score = float("inf") if direction == "min" else -float("inf")
            best_epoch = 0
            history = []
            checkpoint_path = seed_dir / "best_vimr_modules.pt"
            for epoch in range(1, args.epochs + 1):
                write_progress(
                    output,
                    stage="training",
                    completed=False,
                    mode=mode,
                    mode_index=mode_index,
                    mode_count=len(MODES),
                    seed=seed,
                    seed_index=seed_index,
                    seed_count=len(args.seeds),
                    epoch=epoch,
                    epochs=args.epochs,
                )
                train_stats = train_one_epoch_fixed(
                    model,
                    optimizer,
                    train_samples,
                    train_interactions,
                    train_shuffle,
                    mode,
                    device,
                    epoch,
                    seed,
                    args.max_clicks,
                    args.progress_every,
                )
                _, metrics = evaluate_fixed(
                    model,
                    eval_samples,
                    eval_interactions,
                    eval_shuffle,
                    mode,
                    device,
                    args.max_clicks,
                    args.progress_every,
                    f"mode={mode} seed={seed} epoch={epoch} validation",
                )
                if selection_metric not in metrics:
                    raise KeyError(f"Unknown selection metric: {selection_metric}")
                history.append({"epoch": epoch, **train_stats, **metrics})
                score = metrics[selection_metric]
                is_better = score < best_score if direction == "min" else score > best_score
                if is_better:
                    best_score = score
                    best_epoch = epoch
                    save_best_checkpoint(
                        checkpoint_path, model, seed, epoch, selection_metric, direction, metrics
                    )
                    print(
                        f"mode={mode} seed={seed} new best epoch={epoch} "
                        f"{selection_metric}={score:.6f}",
                        flush=True,
                    )

            load_vimr_checkpoint(model, checkpoint_path)
            rows, final_metrics = evaluate_fixed(
                model,
                eval_samples,
                eval_interactions,
                eval_shuffle,
                mode,
                device,
                args.max_clicks,
                args.progress_every,
                f"mode={mode} seed={seed} best={best_epoch} final",
            )
            seed_record = {
                "seed": seed,
                "mode": mode,
                "best_epoch": best_epoch,
                "selection_metric": selection_metric,
                "selection_direction": direction,
                "selection_score": best_score,
                "trainable_parameters": trainable_count,
                "epoch_history": history,
                "metrics": final_metrics,
            }
            metrics_path.write_text(json.dumps(seed_record, indent=2), encoding="utf-8")
            write_per_instance(seed_dir / "per_instance.csv", rows)
            mode_seed_metrics[mode][seed] = final_metrics
            seed_records[str(seed)] = seed_record
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        mode_records[mode] = seed_records

    aggregates = {
        mode: aggregate_mode(baseline_metrics, mode_seed_metrics[mode]) for mode in MODES
    }
    contrasts = {
        "synthetic_minus_default": paired_contrast(
            mode_seed_metrics["synthetic"], mode_seed_metrics["default"]
        ),
        "synthetic_minus_shuffled": paired_contrast(
            mode_seed_metrics["synthetic"], mode_seed_metrics["shuffled"]
        ),
        "correct_minus_synthetic": paired_contrast(
            mode_seed_metrics["correct"], mode_seed_metrics["synthetic"]
        ),
    }
    sd = contrasts["synthetic_minus_default"][f"mIoU@{args.max_clicks}"]
    ss = contrasts["synthetic_minus_shuffled"][f"mIoU@{args.max_clicks}"]
    positive_all_seeds = all(value > 0 for value in (*sd["per_seed"], *ss["per_seed"]))
    passes = sd["mean"] >= 0.01 and ss["mean"] >= 0.01 and positive_all_seeds
    criterion = {
        "definition": "Both paired mIoU@5 means >= +0.01 and both contrasts positive for every seed",
        "synthetic_minus_default_mIoU@5": sd["mean"],
        "synthetic_minus_shuffled_mIoU@5": ss["mean"],
        "positive_for_every_seed": positive_all_seeds,
        "passes": passes,
        "verdict": (
            "supports aligned viewport information under this synthetic protocol"
            if passes
            else "inconclusive: independent viewport information not established"
        ),
    }
    protocol = {
        "dataset": "COCO 2017 train split, fixed image-disjoint holdout",
        "train_instances": args.train_instances,
        "validation_instances": args.eval_instances,
        "epochs": args.epochs,
        "max_clicks": args.max_clicks,
        "training_seeds": args.seeds,
        "split_seed": args.split_seed,
        "shuffle_seed": args.shuffle_seed,
        "learning_rate": args.learning_rate,
        "selection_metric": selection_metric,
        "selection_direction": direction,
        "components": {
            "viewport": True,
            "history": False,
            "residual": False,
            "safe_update": False,
        },
        "fixed_click_policy": (
            "One frozen-baseline-derived sequence per instance, shared by all modes, epochs, and seeds"
        ),
        "viewport_modes": {
            "default": "global [0.5,0.5,0,0,0,0,0] at every round",
            "synthetic": "current click-centered policy from viewport_from_click",
            "shuffled": "no-self instance permutation of complete synthetic trajectories",
            "correct": "GT-instance-bbox oracle from round 2 onward; not real user telemetry",
        },
        "checkpoint_sha256": checkpoint_hash,
        "elapsed_seconds": time.time() - started,
    }
    result = {
        "protocol": protocol,
        "baseline": baseline_metrics,
        "modes": mode_records,
        "aggregates": aggregates,
        "paired_contrasts": contrasts,
        "viewport_signal_criterion": criterion,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_report(output, result)
    write_progress(output, stage="complete", completed=True, report=str(output / "report.md"))
    print(json.dumps({"protocol": protocol, "viewport_signal_criterion": criterion}, indent=2), flush=True)


if __name__ == "__main__":
    main()
