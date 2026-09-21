"""Multi-seed COCO holdout training and evaluation protocol for VIMR-Click."""

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from efficientvit.models.vimr import VIMRConfig, VIMREfficientViTSam
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from tools.eval_vimr_coco_quick import (
    decode_mask,
    encode_image,
    freeze_pretrained_modules,
    load_split,
    run_baseline_sequence,
    run_vimr_sequence,
    target_to_sam,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=ROOT_DIR / "dataset/coco")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt",
    )
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "outputs/vimr_coco_protocol")
    parser.add_argument("--train-instances", type=int, default=500) # 训练实例数
    parser.add_argument("--eval-instances", type=int, default=200) # 验证实例数
    parser.add_argument("--epochs", type=int, default=3) # 每个 seed 训练 3 epoch
    parser.add_argument("--max-clicks", type=int, default=5) # 最多交互 5 次
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44]) # 3 个训练 seed
    parser.add_argument("--split-seed", type=int, default=2026) # 固定 train/val 划分
    parser.add_argument("--learning-rate", type=float, default=1e-4) # AdamW 学习率
    parser.add_argument("--lambda-safe", type=float, default=0.2) # Safe Update loss 权重
    parser.add_argument("--lambda-monotonic", type=float, default=0.1) # 单调性约束权重
    parser.add_argument("--gate-init-probability", type=float, default=0.1) # gate 初始概率
    parser.add_argument("--disable-viewport", action="store_true")
    parser.add_argument("--disable-history", action="store_true")
    parser.add_argument("--disable-residual", action="store_true")
    parser.add_argument("--disable-safe-update", action="store_true")
    # synthetic：使用以点击位置为中心的 viewport state
    # default：每轮都给模型一个没有额外信息的全局 viewport state
    parser.add_argument(
        "--viewport-mode",
        choices=("synthetic", "default"),
        default="synthetic",
        help="Use click-centred viewport states or an information-free global state at every round",
    )
    parser.add_argument("--enable-expert-router", action="store_true")
    parser.add_argument("--enable-detail-refinement", action="store_true")
    parser.add_argument("--min-area-ratio", type=float, default=0.01) # 实例最小面积比例
    parser.add_argument("--max-area-ratio", type=float, default=0.60) # 实例最大面积比例
    parser.add_argument("--selection-metric", default=None, help="Defaults to mIoU@max-clicks (higher is better)")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(
    checkpoint: Path,
    device: torch.device,
    seed: int,
    lambda_safe: float = 0.2,
    lambda_monotonic: float = 0.1,
    gate_init_probability: float = 0.1,
    use_viewport: bool = True,
    use_history: bool = True,
    use_residual: bool = True,
    use_safe_update: bool = True,
    enable_expert_router: bool = False,
    enable_detail_refinement: bool = False,
) -> VIMREfficientViTSam:
    set_seed(seed)
    config = VIMRConfig(
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
        lambda_safe=lambda_safe,
        lambda_monotonic=lambda_monotonic,
        gate_init_probability=gate_init_probability,
        use_viewport=use_viewport,
        use_history=use_history,
        use_residual=use_residual,
        use_safe_update=use_safe_update,
        safe_update=use_safe_update,
        enable_expert_router=enable_expert_router,
        enable_detail_refinement=enable_detail_refinement,
    )
    model = create_efficientvit_sam_model(
        "efficientvit-sam-vimr-l0",
        pretrained=True,
        weight_url=str(checkpoint),
        vimr_config=config.to_dict(),
    )
    return model.to(device)


def summarize_protocol(rows: list[dict[str, Any]], max_clicks: int) -> dict[str, float]:
    sequences = np.asarray([row["ious"] for row in rows], dtype=np.float64)
    report_clicks = sorted({1, 2, 3, max_clicks})
    summary = {f"mIoU@{click}": float(sequences[:, click - 1].mean()) for click in report_clicks}
    for threshold in (0.85, 0.90):
        reached = sequences >= threshold
        first = np.where(reached.any(axis=1), reached.argmax(axis=1) + 1, max_clicks + 1)
        summary[f"NoC@{int(threshold * 100)}"] = float(first.mean())
        summary[f"Success@{int(threshold * 100)}/{max_clicks}"] = float(reached.any(axis=1).mean())
    all_regressions = sequences[:, 1:] < sequences[:, :-1] - 1e-6
    summary["RegressionRate"] = float(all_regressions.mean())
    regression_magnitudes = np.maximum(sequences[:, :-1] - sequences[:, 1:], 0.0)
    summary["MeanRegressionMagnitude"] = float(regression_magnitudes.mean())
    for threshold in (0.85, 0.90):
        reached = (sequences >= threshold).any(axis=1)
        final_success = sequences[:, -1] >= threshold
        suffix = int(threshold * 100)
        summary[f"FinalSuccess@{suffix}/{max_clicks}"] = float(final_success.mean())
        summary[f"LostAfterSuccess@{suffix}/{max_clicks}"] = float((reached & ~final_success).mean())
    late_regressions = []
    for click in (4, 5):
        if max_clicks >= click:
            regression = sequences[:, click - 1] < sequences[:, click - 2] - 1e-6
            summary[f"RegressionRate@{click}"] = float(regression.mean())
            late_regressions.append(regression)
    if len(late_regressions) == 2:
        late_regressions_array = np.stack(late_regressions, axis=1)
        late_deltas = sequences[:, 2:4] - sequences[:, 3:5]
        summary["LateRegressionRate@4-5"] = float(late_regressions_array.mean())
        summary["LateMeanRegressionMagnitude@4-5"] = float(np.maximum(late_deltas, 0.0).mean())
        summary["LateSevereRegressionRate@4-5>0.05"] = float((late_deltas > 0.05).mean())
        summary["LateSevereRegressionRate@4-5>0.10"] = float((late_deltas > 0.10).mean())
    return summary


def evaluate_method(
    model: VIMREfficientViTSam,
    samples: list[dict[str, Any]],
    device: torch.device,
    max_clicks: int,
    method: str,
    progress_every: int,
    prefix: str,
    viewport_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if method not in {"baseline", "vimr"}:
        raise ValueError(f"Unsupported method: {method}")
    model.eval()
    rows = []
    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            target = target_to_sam(decode_mask(sample), device)
            embedding = encode_image(model, sample["path"], device)
            if method == "baseline":
                ious = run_baseline_sequence(model, embedding, target, max_clicks)
            else:
                _, ious = run_vimr_sequence(
                    model,
                    embedding,
                    target,
                    max_clicks,
                    train=False,
                    viewport_mode=viewport_mode,
                )
            rows.append(
                {
                    "image_id": sample["image"]["id"],
                    "annotation_id": sample["annotation"]["id"],
                    "category_id": sample["annotation"]["category_id"],
                    "ious": ious,
                }
            )
            if index == 1 or index % progress_every == 0 or index == len(samples):
                print(f"{prefix} {index}/{len(samples)} IoU@{max_clicks}={ious[-1]:.4f}", flush=True)
    return rows, summarize_protocol(rows, max_clicks)


def set_vimr_train_mode(model: VIMREfficientViTSam) -> None:
    model.train()
    model.image_encoder.eval()
    model.prompt_encoder.eval()
    model.mask_decoder.eval()


def train_one_epoch(
    model: VIMREfficientViTSam,
    optimizer: torch.optim.Optimizer,
    samples: list[dict[str, Any]],
    device: torch.device,
    epoch: int,
    seed: int,
    max_clicks: int,
    progress_every: int,
    viewport_mode: str,
) -> dict[str, float]:
    set_vimr_train_mode(model)
    order = list(range(len(samples)))
    random.Random(seed * 10_000 + epoch).shuffle(order)
    grad_norms = []
    final_ious = []
    for step, sample_index in enumerate(order, start=1):
        sample = samples[sample_index]
        target = target_to_sam(decode_mask(sample), device)
        embedding = encode_image(model, sample["path"], device)
        optimizer.zero_grad(set_to_none=True)
        _, ious = run_vimr_sequence(
            model,
            embedding,
            target,
            max_clicks,
            train=True,
            viewport_mode=viewport_mode,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
        optimizer.step()
        grad_norms.append(float(grad_norm.detach().cpu()))
        final_ious.append(ious[-1])
        if step == 1 or step % progress_every == 0 or step == len(order):
            print(
                f"seed={seed} epoch={epoch} train={step}/{len(order)} "
                f"IoU@{max_clicks}={ious[-1]:.4f} grad_norm={grad_norms[-1]:.4f}",
                flush=True,
            )
    return {
        "mean_train_iou_at_max_clicks": float(np.mean(final_ious)),
        "mean_gradient_norm": float(np.mean(grad_norms)),
    }


def vimr_state_dict(model: VIMREfficientViTSam) -> dict[str, torch.Tensor]:
    prefixes = (
        "viewport_encoder.",
        "history_memory.",
        "residual_analyzer.",
        "interaction_fusion.",
        "expert_router.",
        "safe_update.",
        "detail_refinement.",
    )
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if key.startswith(prefixes)}


def save_best_checkpoint(
    path: Path,
    model: VIMREfficientViTSam,
    seed: int,
    epoch: int,
    selection_metric: str,
    direction: str,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "vimr_state_dict": vimr_state_dict(model),
            "seed": seed,
            "epoch": epoch,
            "selection_metric": selection_metric,
            "selection_direction": direction,
            "selection_score": metrics[selection_metric],
            "validation_metrics": metrics,
        },
        path,
    )


def load_vimr_checkpoint(model: VIMREfficientViTSam, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(checkpoint["vimr_state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected best-checkpoint keys: {incompatible.unexpected_keys}")
    return checkpoint


def save_per_instance(
    path: Path,
    baseline_rows: list[dict[str, Any]],
    vimr_rows: list[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_id", "annotation_id", "category_id", "method", "click", "iou"))
        for method, rows in (("baseline", baseline_rows), ("vimr", vimr_rows)):
            for row in rows:
                for click, iou in enumerate(row["ious"], start=1):
                    writer.writerow(
                        (row["image_id"], row["annotation_id"], row["category_id"], method, click, iou)
                    )


def aggregate_metrics(
    baseline: dict[str, float], seed_metrics: dict[int, dict[str, float]]
) -> dict[str, dict[str, float]]:
    aggregate = {}
    for metric, baseline_value in baseline.items():
        values = np.asarray([metrics[metric] for metrics in seed_metrics.values()], dtype=np.float64)
        aggregate[metric] = {
            "baseline": baseline_value,
            "vimr_mean": float(values.mean()),
            "vimr_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "delta_mean": float(values.mean() - baseline_value),
        }
    return aggregate


def selection_direction(metric: str) -> str:
    return "min" if metric.startswith("NoC@") else "max"


def save_aggregate_report(
    output_dir: Path,
    aggregate: dict[str, dict[str, float]],
    protocol: dict[str, Any],
) -> None:
    with (output_dir / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "baseline", "vimr_mean", "vimr_std", "delta_mean"))
        for metric, values in aggregate.items():
            writer.writerow(
                (metric, values["baseline"], values["vimr_mean"], values["vimr_std"], values["delta_mean"])
            )
    lines = [
        "# VIMR-Click multi-seed COCO protocol results",
        "",
        f"- Split: {protocol['train_instances']} train / {protocol['validation_instances']} validation "
        f"(split seed {protocol['split_seed']})",
        f"- Schedule: {protocol['epochs']} epochs, {protocol['max_clicks']} clicks, "
        f"training seeds {protocol['training_seeds']}",
        f"- Best checkpoint: {protocol['selection_direction']} validation "
        f"`{protocol['selection_metric']}` per seed",
        f"- NoC failure value: {protocol['noc_failure_value']}",
        "- Frozen modules: image encoder, prompt encoder, mask decoder",
        "",
        "| Metric | Baseline | VIMR mean | VIMR std | Mean delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, values in aggregate.items():
        lines.append(
            f"| {metric} | {values['baseline']:.6f} | {values['vimr_mean']:.6f} | "
            f"{values['vimr_std']:.6f} | {values['delta_mean']:+.6f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample_manifest(samples: list[dict[str, Any]]) -> list[dict[str, int]]:
    return [
        {
            "image_id": int(sample["image"]["id"]),
            "annotation_id": int(sample["annotation"]["id"]),
            "category_id": int(sample["annotation"]["category_id"]),
        }
        for sample in samples
    ]


def main() -> None:
    args = parse_args()
    if args.max_clicks < 3:
        raise ValueError("max-clicks must be at least 3")
    if args.max_clicks < 5:
        print("Warning: max-clicks < 5 disables the requested click-4/5 late-stage protocol", flush=True)
    selection_metric = args.selection_metric or f"mIoU@{args.max_clicks}"
    direction = selection_direction(selection_metric)
    use_viewport = not args.disable_viewport
    use_history = not args.disable_history
    use_residual = not args.disable_residual
    use_safe_update = not args.disable_safe_update
    enable_expert_router = args.enable_expert_router
    enable_detail_refinement = args.enable_detail_refinement
    device = torch.device(args.device)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
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
    (output_dir / "split_manifest.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    print("Evaluating frozen pretrained baseline once on the fixed validation split", flush=True)
    baseline_model = build_model(
        args.checkpoint,
        device,
        args.seeds[0],
        args.lambda_safe,
        args.lambda_monotonic,
        args.gate_init_probability,
        use_viewport,
        use_history,
        use_residual,
        use_safe_update,
        enable_expert_router,
        enable_detail_refinement,
    )
    freeze_pretrained_modules(baseline_model)
    baseline_rows, baseline_metrics = evaluate_method(
        baseline_model,
        eval_samples,
        device,
        args.max_clicks,
        "baseline",
        args.progress_every,
        "baseline validation",
        args.viewport_mode,
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    seed_metrics: dict[int, dict[str, float]] = {}
    seed_records = {}
    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model = build_model(
            args.checkpoint,
            device,
            seed,
            args.lambda_safe,
            args.lambda_monotonic,
            args.gate_init_probability,
            use_viewport,
            use_history,
            use_residual,
            use_safe_update,
            enable_expert_router,
            enable_detail_refinement,
        )
        trainable = freeze_pretrained_modules(model)
        trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
        print(
            f"seed={seed} protocol trainable VIMR params: {trainable_parameter_count:,}; "
            "image/prompt/mask modules frozen",
            flush=True,
        )
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
        best_score = float("inf") if direction == "min" else -float("inf")
        best_epoch = 0
        epoch_history = []
        checkpoint_path = seed_dir / "best_vimr_modules.pt"
        for epoch in range(1, args.epochs + 1):
            train_stats = train_one_epoch(
                model,
                optimizer,
                train_samples,
                device,
                epoch,
                seed,
                args.max_clicks,
                args.progress_every,
                args.viewport_mode,
            )
            _, validation_metrics = evaluate_method(
                model,
                eval_samples,
                device,
                args.max_clicks,
                "vimr",
                args.progress_every,
                f"seed={seed} epoch={epoch} validation",
                args.viewport_mode,
            )
            if selection_metric not in validation_metrics:
                raise KeyError(f"Unknown selection metric {selection_metric}; got {list(validation_metrics)}")
            record = {"epoch": epoch, **train_stats, **validation_metrics}
            epoch_history.append(record)
            score = validation_metrics[selection_metric]
            is_better = score < best_score if direction == "min" else score > best_score
            if is_better:
                best_score = score
                best_epoch = epoch
                save_best_checkpoint(
                    checkpoint_path,
                    model,
                    seed,
                    epoch,
                    selection_metric,
                    direction,
                    validation_metrics,
                )
                print(f"seed={seed} new best epoch={epoch} {selection_metric}={score:.6f}", flush=True)

        checkpoint = load_vimr_checkpoint(model, checkpoint_path)
        vimr_rows, final_metrics = evaluate_method(
            model,
            eval_samples,
            device,
            args.max_clicks,
            "vimr",
            args.progress_every,
            f"seed={seed} best-epoch={best_epoch} final",
            args.viewport_mode,
        )
        seed_metrics[seed] = final_metrics
        seed_record = {
            "seed": seed,
            "best_epoch": best_epoch,
            "selection_metric": selection_metric,
            "selection_direction": direction,
            "selection_score": checkpoint["selection_score"],
            "trainable_vimr_parameters": trainable_parameter_count,
            "epoch_history": epoch_history,
            "baseline_metrics": baseline_metrics,
            "vimr_metrics": final_metrics,
        }
        seed_records[str(seed)] = seed_record
        (seed_dir / "metrics.json").write_text(json.dumps(seed_record, indent=2), encoding="utf-8")
        save_per_instance(seed_dir / "per_instance.csv", baseline_rows, vimr_rows)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = aggregate_metrics(baseline_metrics, seed_metrics)
    protocol = {
        "dataset": "COCO 2017 train split, fixed image-disjoint holdout",
        "train_instances": args.train_instances,
        "validation_instances": args.eval_instances,
        "epochs": args.epochs,
        "max_clicks": args.max_clicks,
        "training_seeds": args.seeds,
        "split_seed": args.split_seed,
        "learning_rate": args.learning_rate,
        "lambda_safe": args.lambda_safe,
        "lambda_monotonic": args.lambda_monotonic,
        "gate_init_probability": args.gate_init_probability,
        "viewport_mode": args.viewport_mode,
        "components": {
            "viewport": use_viewport,
            "history": use_history,
            "residual": use_residual,
            "safe_update": use_safe_update,
            "expert_router": enable_expert_router,
            "detail_refinement": enable_detail_refinement,
        },
        "selection_metric": selection_metric,
        "selection_direction": direction,
        "base_modules_frozen": True,
        "instance_sampling": "largest eligible non-crowd instance; one instance per image",
        "area_ratio_range": [args.min_area_ratio, args.max_area_ratio],
        "click_policy": (
            "first positive click at the GT distance-transform maximum; subsequent click at maximum "
            "absolute GT-probability error, with label read from GT"
        ),
        "baseline_prompt_policy": "all clicks accumulated plus previous-mask prompt",
        "vimr_prompt_policy": "current click through SAM plus explicit click/view/mask history",
        "noc_failure_value": args.max_clicks + 1,
        "regression_epsilon": 1e-6,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "elapsed_seconds": time.time() - start,
    }
    result = {"protocol": protocol, "baseline": baseline_metrics, "seeds": seed_records, "aggregate": aggregate}
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_aggregate_report(output_dir, aggregate, protocol)
    print(json.dumps({"protocol": protocol, "aggregate": aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
