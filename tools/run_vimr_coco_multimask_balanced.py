"""Phase-2 VIMR training with size-aware sampling and calibrated multimask selection."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from efficientvit.models.vimr import vimr_interactive_loss
from efficientvit.models.vimr.quality import (
    VIMRFinalMaskQualityHead,
    hard_candidate_iou,
    quality_calibration_loss,
    quality_head_state_dict,
    quality_metrics,
)
from tools.eval_vimr_coco_quick import (
    corrective_click,
    decode_mask,
    freeze_pretrained_modules,
    initial_click,
    viewport_for_mode,
)
from tools.run_vimr_coco_protocol import (
    build_model,
    load_vimr_checkpoint,
    set_seed,
    set_vimr_train_mode,
    vimr_state_dict,
)


SIZE_NAMES = ("small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=ROOT_DIR / "dataset/coco")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ROOT_DIR / "assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt",
    )
    parser.add_argument(
        "--init-vimr-checkpoint",
        type=Path,
        default=ROOT_DIR / "outputs/vimr_coco_full/seed_42/best_vimr_modules.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "outputs/vimr_coco_multimask_balanced_seed42",
    )
    parser.add_argument("--train-instances", type=int, default=104760)
    parser.add_argument("--eval-instances", type=int, default=4625)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-clicks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=2027)
    parser.add_argument("--size-ratios", type=float, nargs=3, default=(0.42, 0.34, 0.24))
    parser.add_argument("--max-area-ratio", type=float, default=0.95)
    parser.add_argument("--max-train-instances-per-image", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--quality-learning-rate", type=float, default=2e-4)
    parser.add_argument("--lambda-quality", type=float, default=1.0)
    parser.add_argument("--lambda-ranking", type=float, default=0.2)
    parser.add_argument("--lambda-safe", type=float, default=0.2)
    parser.add_argument("--lambda-monotonic", type=float, default=0.1)
    parser.add_argument("--gate-init-probability", type=float, default=0.1)
    parser.add_argument("--viewport-mode", choices=("synthetic", "default"), default="synthetic")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--amp", choices=("off", "fp16"), default="fp16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def size_name(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def allocate_quotas(total: int, ratios: tuple[float, float, float] | list[float]) -> dict[str, int]:
    values = np.asarray(ratios, dtype=np.float64)
    if np.any(values <= 0):
        raise ValueError("All --size-ratios values must be positive")
    values /= values.sum()
    raw = values * total
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: total - int(counts.sum())]:
        counts[index] += 1
    return dict(zip(SIZE_NAMES, counts.tolist()))


def load_balanced_split(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict]:
    annotation_path = args.coco_root / "annotations/instances_train2017.json"
    image_root = args.coco_root / "train2017"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(record["id"]): record for record in coco["images"]}
    pools: dict[str, list[dict[str, Any]]] = {name: [] for name in SIZE_NAMES}
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0) or not annotation.get("segmentation") or annotation.get("area", 0) < 1:
            continue
        image = images[int(annotation["image_id"])]
        area_ratio = float(annotation["area"]) / float(image["width"] * image["height"])
        if area_ratio > args.max_area_ratio:
            continue
        pools[size_name(float(annotation["area"]))].append(annotation)

    eval_quotas = allocate_quotas(args.eval_instances, args.size_ratios)
    train_quotas = allocate_quotas(args.train_instances, args.size_ratios)
    rng = random.Random(args.split_seed)
    for index, name in enumerate(SIZE_NAMES):
        random.Random(args.split_seed + index).shuffle(pools[name])

    eval_image_ids: set[int] = set()
    eval_annotations: list[tuple[str, dict[str, Any]]] = []
    for name in SIZE_NAMES:
        for annotation in pools[name]:
            image_id = int(annotation["image_id"])
            if image_id in eval_image_ids:
                continue
            image_path = image_root / images[image_id]["file_name"]
            if not image_path.is_file():
                continue
            eval_annotations.append((name, annotation))
            eval_image_ids.add(image_id)
            if sum(group == name for group, _ in eval_annotations) == eval_quotas[name]:
                break
        else:
            raise RuntimeError(f"Unable to fill validation quota for {name}")

    per_image_train_count: dict[int, int] = defaultdict(int)
    train_annotations: list[tuple[str, dict[str, Any]]] = []
    for name in SIZE_NAMES:
        selected = 0
        for annotation in pools[name]:
            image_id = int(annotation["image_id"])
            if image_id in eval_image_ids:
                continue
            if per_image_train_count[image_id] >= args.max_train_instances_per_image:
                continue
            image_path = image_root / images[image_id]["file_name"]
            if not image_path.is_file():
                continue
            train_annotations.append((name, annotation))
            per_image_train_count[image_id] += 1
            selected += 1
            if selected == train_quotas[name]:
                break
        if selected != train_quotas[name]:
            raise RuntimeError(f"Only selected {selected}/{train_quotas[name]} training {name} instances")

    def make_sample(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        group, annotation = item
        image = images[int(annotation["image_id"])]
        return {
            "image": image,
            "annotation": annotation,
            "path": str(image_root / image["file_name"]),
            "size_group": group,
        }

    rng.shuffle(train_annotations)
    rng.shuffle(eval_annotations)
    train_samples = [make_sample(item) for item in train_annotations]
    eval_samples = [make_sample(item) for item in eval_annotations]
    manifest = {
        "split_seed": args.split_seed,
        "size_ratios": dict(zip(SIZE_NAMES, args.size_ratios)),
        "train_quotas": train_quotas,
        "validation_quotas": eval_quotas,
        "validation_image_disjoint": True,
        "max_train_instances_per_image": args.max_train_instances_per_image,
        "train": [
            {
                "image_id": int(sample["image"]["id"]),
                "annotation_id": int(sample["annotation"]["id"]),
                "size_group": sample["size_group"],
            }
            for sample in train_samples
        ],
        "validation": [
            {
                "image_id": int(sample["image"]["id"]),
                "annotation_id": int(sample["annotation"]["id"]),
                "size_group": sample["size_group"],
            }
            for sample in eval_samples
        ],
    }
    return train_samples, eval_samples, manifest


def target_to_sam_balanced(sample: dict[str, Any], device: torch.device) -> torch.Tensor:
    mask = decode_mask(sample)
    height, width = mask.shape
    scale = 256.0 / max(height, width)
    new_height = int(height * scale + 0.5)
    new_width = int(width * scale + 0.5)
    tensor = torch.from_numpy(mask).to(device=device).reshape(1, 1, height, width)
    if sample["size_group"] == "small" and new_height <= height and new_width <= width:
        resized = F.adaptive_max_pool2d(tensor, (new_height, new_width))
    else:
        resized = F.interpolate(tensor, (new_height, new_width), mode="nearest")
    target = F.pad(resized, (0, 256 - new_width, 0, 256 - new_height))
    if target.max() <= 0:
        annotation = sample["annotation"]
        x, y, box_width, box_height = annotation["bbox"]
        center_x = min(255, max(0, int((x + box_width / 2) * scale)))
        center_y = min(255, max(0, int((y + box_height / 2) * scale)))
        target[..., center_y, center_x] = 1.0
    return target


def encode_image(model: torch.nn.Module, path: str, device: torch.device, amp_enabled: bool) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"))
    image_tensor = model.transform(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
        return model.image_encoder(image_tensor).detach()


def run_multimask_sequence(
    model: torch.nn.Module,
    quality_head: VIMRFinalMaskQualityHead,
    embedding: torch.Tensor,
    target: torch.Tensor,
    max_clicks: int,
    viewport_mode: str,
    amp_enabled: bool,
    train: bool,
    scaler: torch.amp.GradScaler | None = None,
    teacher_forcing_probability: float = 0.0,
    selection_rng: random.Random | None = None,
    lambda_quality: float = 1.0,
    lambda_ranking: float = 0.2,
) -> dict[str, Any]:
    coords, labels = initial_click(target)
    click_events: list[torch.Tensor] = []
    viewports: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    previous = None
    ious: list[float] = []
    losses: list[float] = []
    quality_mae: list[float] = []
    quality_top1: list[float] = []

    for round_index in range(max_clicks):
        viewport = viewport_for_mode(
            coords,
            round_index,
            viewports[-1] if viewports else None,
            viewport_mode,
        )
        event = torch.cat((coords[:, 0] / 1024.0, labels[:, 0:1].float()), dim=-1)
        click_events.append(event)
        viewports.append(viewport)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            output = model(
                image_embeddings=embedding,
                point_coords=coords,
                point_labels=labels,
                previous_mask=previous,
                viewport_state=viewport,
                click_history=torch.stack(click_events, dim=1),
                viewport_history=torch.stack(viewports, dim=1).squeeze(2),
                mask_history=torch.stack(masks, dim=1) if masks else None,
                multimask_output=True,
            )
            candidates = output["final_mask"]
            target_quality = hard_candidate_iou(candidates, target)
            predicted_quality = quality_head(output, round_index, max_clicks)
            calibration = quality_calibration_loss(predicted_quality, target_quality)
            candidate_losses = [
                vimr_interactive_loss(
                    candidates[:, candidate : candidate + 1],
                    target,
                    previous,
                    lambda_safe=model.vimr_config.lambda_safe,
                    lambda_monotonic=model.vimr_config.lambda_monotonic,
                )["loss"]
                for candidate in range(candidates.shape[1])
            ]
            stacked_candidate_losses = torch.stack(candidate_losses)
            oracle_index = int(stacked_candidate_losses.detach().argmin().item())
            segmentation_loss = stacked_candidate_losses[oracle_index]
            total_loss = (
                segmentation_loss
                + lambda_quality * calibration["loss_quality"]
                + lambda_ranking * calibration["loss_ranking"]
            ) / max_clicks
        if train:
            if scaler is None:
                total_loss.backward()
            else:
                scaler.scale(total_loss).backward()
        predicted_index = int(predicted_quality.detach().argmax(dim=1).item())
        use_oracle = (
            train
            and selection_rng is not None
            and selection_rng.random() < teacher_forcing_probability
        )
        state_index = oracle_index if use_oracle else predicted_index
        probability = candidates[:, state_index : state_index + 1].detach()
        ious.append(float(target_quality[0, state_index].detach().cpu()))
        losses.append(float(total_loss.detach().cpu()))
        calibration_metrics = quality_metrics(predicted_quality.detach(), target_quality.detach())
        quality_mae.append(calibration_metrics["quality_mae"])
        quality_top1.append(calibration_metrics["quality_top1_accuracy"])
        masks.append(probability)
        previous = probability
        if round_index + 1 < max_clicks:
            coords, labels = corrective_click(previous, target)
    return {
        "ious": ious,
        "mean_loss": float(np.mean(losses)),
        "quality_mae": float(np.mean(quality_mae)),
        "quality_top1_accuracy": float(np.mean(quality_top1)),
    }


def summarize_rows(rows: list[dict[str, Any]], max_clicks: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    report_clicks = sorted({1, 3, max_clicks})
    for click in report_clicks:
        values = [row["ious"][click - 1] for row in rows]
        metrics[f"mIoU@{click}"] = float(np.mean(values))
        for group in SIZE_NAMES:
            group_values = [row["ious"][click - 1] for row in rows if row["size_group"] == group]
            metrics[f"mIoU{group[0].upper()}@{click}"] = float(np.mean(group_values))
    metrics["quality_mae"] = float(np.mean([row["quality_mae"] for row in rows]))
    metrics["quality_top1_accuracy"] = float(
        np.mean([row["quality_top1_accuracy"] for row in rows])
    )
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    quality_head: VIMRFinalMaskQualityHead,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    set_vimr_train_mode(model)
    quality_head.train()
    amp_enabled = args.amp == "fp16" and args.device.startswith("cuda")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[int(sample["image"]["id"])].append(sample)
    image_ids = list(grouped)
    rng = random.Random(args.seed * 10000 + epoch)
    rng.shuffle(image_ids)
    teacher_forcing_probability = 1.0 if args.epochs == 1 else max(
        0.0, 1.0 - (epoch - 1) / (args.epochs - 1)
    )
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    processed = 0
    final_ious: list[float] = []
    losses: list[float] = []
    quality_mae: list[float] = []
    quality_top1: list[float] = []
    for image_id in image_ids:
        image_samples = grouped[image_id]
        rng.shuffle(image_samples)
        embedding = encode_image(model, image_samples[0]["path"], torch.device(args.device), amp_enabled)
        for sample in image_samples:
            target = target_to_sam_balanced(sample, torch.device(args.device))
            optimizer.zero_grad(set_to_none=True)
            sequence = run_multimask_sequence(
                model,
                quality_head,
                embedding,
                target,
                args.max_clicks,
                args.viewport_mode,
                amp_enabled,
                train=True,
                scaler=scaler,
                teacher_forcing_probability=teacher_forcing_probability,
                selection_rng=rng,
                lambda_quality=args.lambda_quality,
                lambda_ranking=args.lambda_ranking,
            )
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            processed += 1
            final_ious.append(sequence["ious"][-1])
            losses.append(sequence["mean_loss"])
            quality_mae.append(sequence["quality_mae"])
            quality_top1.append(sequence["quality_top1_accuracy"])
            if processed == 1 or processed % args.progress_every == 0 or processed == len(samples):
                print(
                    f"seed={args.seed} phase2 epoch={epoch}/{args.epochs} "
                    f"train={processed}/{len(samples)} size={sample['size_group']} "
                    f"IoU@{args.max_clicks}={final_ious[-1]:.4f} "
                    f"quality_top1={quality_top1[-1]:.3f} "
                    f"grad_norm={float(gradient_norm.detach().cpu()):.4f}",
                    flush=True,
                )
    return {
        "mean_train_iou_at_max_clicks": float(np.mean(final_ious)),
        "mean_train_loss": float(np.mean(losses)),
        "quality_mae": float(np.mean(quality_mae)),
        "quality_top1_accuracy": float(np.mean(quality_top1)),
        "teacher_forcing_probability": teacher_forcing_probability,
    }


def evaluate(
    model: torch.nn.Module,
    quality_head: VIMRFinalMaskQualityHead,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    quality_head.eval()
    amp_enabled = args.amp == "fp16" and args.device.startswith("cuda")
    rows = []
    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            target = target_to_sam_balanced(sample, torch.device(args.device))
            embedding = encode_image(model, sample["path"], torch.device(args.device), amp_enabled)
            sequence = run_multimask_sequence(
                model,
                quality_head,
                embedding,
                target,
                args.max_clicks,
                args.viewport_mode,
                amp_enabled,
                train=False,
            )
            rows.append({"size_group": sample["size_group"], **sequence})
            if index == 1 or index % args.progress_every == 0 or index == len(samples):
                print(
                    f"{prefix} {index}/{len(samples)} size={sample['size_group']} "
                    f"IoU@{args.max_clicks}={sequence['ious'][-1]:.4f} "
                    f"quality_top1={sequence['quality_top1_accuracy']:.3f}",
                    flush=True,
                )
    return rows, summarize_rows(rows, args.max_clicks)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    quality_head: VIMRFinalMaskQualityHead,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "vimr_state_dict": vimr_state_dict(model),
            "quality_head_state_dict": quality_head_state_dict(quality_head),
            "seed": args.seed,
            "epoch": epoch,
            "selection_metric": "mIoU@5",
            "selection_score": metrics["mIoU@5"],
            "validation_metrics": metrics,
            "training_protocol": {
                "multimask_output": True,
                "candidate_count": 3,
                "best_of_3_segmentation_loss": True,
                "quality_regression": True,
                "quality_ranking": True,
                "size_ratios": list(args.size_ratios),
                "small_target_preservation": "adaptive max-pool",
                "warm_start": str(args.init_vimr_checkpoint.resolve().relative_to(ROOT_DIR)),
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_samples, eval_samples, manifest = load_balanced_split(args)
    (args.output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("train_quotas", "validation_quotas")}), flush=True)

    device = torch.device(args.device)
    model = build_model(
        args.base_checkpoint,
        device,
        args.seed,
        lambda_safe=args.lambda_safe,
        lambda_monotonic=args.lambda_monotonic,
        gate_init_probability=args.gate_init_probability,
    )
    warm_start = load_vimr_checkpoint(model, args.init_vimr_checkpoint)
    print(
        f"Warm-started VIMR from seed={warm_start['seed']} epoch={warm_start['epoch']} "
        f"score={warm_start['selection_score']:.6f}",
        flush=True,
    )
    active_vimr = freeze_pretrained_modules(model)
    quality_head = VIMRFinalMaskQualityHead().to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": active_vimr, "lr": args.learning_rate},
            {"params": quality_head.parameters(), "lr": args.quality_learning_rate},
        ],
        weight_decay=1e-4,
    )
    amp_enabled = args.amp == "fp16" and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    active_parameter_count = sum(parameter.numel() for parameter in active_vimr) + sum(
        parameter.numel() for parameter in quality_head.parameters()
    )
    print(f"Phase-2 trainable parameters: {active_parameter_count:,}; AMP={args.amp}", flush=True)

    best_score = -float("inf")
    best_epoch = 0
    history = []
    checkpoint_path = args.output / "best_vimr_multimask_quality.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, quality_head, optimizer, scaler, train_samples, args, epoch)
        _, validation_metrics = evaluate(
            model,
            quality_head,
            eval_samples,
            args,
            f"seed={args.seed} phase2 epoch={epoch} validation",
        )
        record = {"epoch": epoch, **train_metrics, **validation_metrics}
        history.append(record)
        (args.output / "epoch_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        if validation_metrics["mIoU@5"] > best_score:
            best_score = validation_metrics["mIoU@5"]
            best_epoch = epoch
            save_checkpoint(checkpoint_path, model, quality_head, args, epoch, validation_metrics)
            print(f"New phase-2 best epoch={epoch} mIoU@5={best_score:.6f}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["vimr_state_dict"], strict=False)
    quality_head.load_state_dict(checkpoint["quality_head_state_dict"])
    _, final_metrics = evaluate(
        model,
        quality_head,
        eval_samples,
        args,
        f"seed={args.seed} phase2 best-epoch={best_epoch} final",
    )
    result = {
        "protocol": {
            "train_instances": args.train_instances,
            "validation_instances": args.eval_instances,
            "epochs": args.epochs,
            "seed": args.seed,
            "size_ratios": list(args.size_ratios),
            "multimask_candidates": 3,
            "amp": args.amp,
            "elapsed_seconds": time.time() - start,
        },
        "best_epoch": best_epoch,
        "best_score": best_score,
        "active_trainable_parameters": active_parameter_count,
        "epoch_history": history,
        "final_metrics": final_metrics,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
