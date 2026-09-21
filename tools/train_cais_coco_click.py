#!/usr/bin/env python3
"""Train the standalone CAIS network on the strict COCO-Click alignment set."""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficientvit.models.cais import CAISConfig, COCOClickCAISDataset, cais_multitask_loss, load_rgb_image
from efficientvit.sam_model_zoo import create_efficientvit_sam_model

DEFAULT_DATASET = r"data\cais_training_ready"
DEFAULT_CHECKPOINT = r"assets\checkpoints\efficientvit_sam\distilled_model\efficientvit_sam_l0.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--base-checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cais_coco_click_20260919"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--trajectory-samples", type=int, default=16)
    parser.add_argument(
        "--disable-trajectory-input",
        action="store_true",
        help="Ablation: keep click coordinates/types but remove all trajectory information.",
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=("real", "zero", "shuffle"),
        default="real",
        help=(
            "Use aligned trajectories, zero them, or replace them with a deterministic "
            "same-source/same-click-count donor trajectory."
        ),
    )
    parser.add_argument(
        "--prompt-integration",
        choices=("replace", "native_residual"),
        default="replace",
        help="Use replacement CAIS tokens or preserve native SAM points with a zero-init residual.",
    )
    parser.add_argument(
        "--attention-integration",
        choices=("replace", "none", "gated_residual"),
        default="replace",
        help="Replace image features, skip CAIS attention, or apply it through a zero-init residual gate.",
    )
    parser.add_argument("--lambda-iou", type=float, default=1.0)
    parser.add_argument("--lambda-cognitive", type=float, default=0.075)
    parser.add_argument("--boundary-alpha", type=float, default=2.0)
    parser.add_argument("--boundary-width", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9527)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument(
        "--source-sampling",
        choices=("natural", "balanced"),
        default="natural",
        help="Train with the natural source ratio or equal human/synthetic click counts per epoch.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--train-mask-decoder",
        action="store_true",
        help="Also fine-tune the pretrained SAM mask decoder. The default trains CAIS modules only.",
    )
    parser.add_argument(
        "--standard-sam-ablation",
        action="store_true",
        help="Ablation: bypass CAIS modules and fine-tune only the standard EfficientViT-SAM mask decoder.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-image-cache", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run two train samples and two validation samples for integration verification.",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageEmbeddingCache:
    def __init__(self, model: torch.nn.Module, device: torch.device, amp: bool, enabled: bool = True) -> None:
        self.model = model
        self.device = device
        self.amp = amp
        self.enabled = enabled
        self.cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def get(self, image_path: str) -> torch.Tensor:
        cached = self.cache.get(image_path)
        output_dtype = torch.float16 if self.amp else torch.float32
        if cached is not None:
            return cached.to(device=self.device, dtype=output_dtype, non_blocking=True)
        image = load_rgb_image(image_path)
        image_tensor = self.model.transform(image).unsqueeze(0).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            embedding = self.model.image_encoder(image_tensor)
        embedding = embedding.detach()
        if self.enabled:
            self.cache[image_path] = embedding.to(device="cpu", dtype=torch.float16)
        return embedding.to(dtype=output_dtype)


def configure_trainable_parameters(model: torch.nn.Module, train_mask_decoder: bool) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not hasattr(model, "cognitive_prompt_encoder"):
        for parameter in model.mask_decoder.parameters():
            parameter.requires_grad = True
        return [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_modules = [
        model.cognitive_prompt_encoder,
        model.prediction_heads,
    ]
    if model.cais_config.attention_integration != "none":
        trainable_modules.append(model.cognitive_two_way_attention)
    if train_mask_decoder:
        trainable_modules.append(model.mask_decoder)
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    # Residual gates are parameters of the root CAIS model rather than one of
    # the child modules above. They must be trainable, otherwise a zero-init
    # native-residual experiment can never turn either learned branch on.
    for gate_name in ("prompt_residual_gate", "image_residual_gate"):
        gate = getattr(model, gate_name, None)
        if gate is not None:
            gate.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def set_training_mode(model: torch.nn.Module, train_mask_decoder: bool) -> None:
    model.train()
    model.image_encoder.eval()
    model.prompt_encoder.eval()
    if not hasattr(model, "cognitive_prompt_encoder"):
        return
    if not train_mask_decoder:
        model.mask_decoder.eval()


def move_sample(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    keys = (
        "point_coords",
        "point_labels",
        "click_trajectories",
        "trajectory_valid_mask",
        "target",
        "previous_mask",
        "cognitive_target",
        "cognitive_supervision_mask",
    )
    result = dict(sample)
    for key in keys:
        tensor = sample[key]
        if key in {"point_coords", "point_labels", "click_trajectories", "trajectory_valid_mask"}:
            tensor = tensor.unsqueeze(0)
        elif key in {"target", "previous_mask"}:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.reshape(1)
        result[key] = tensor.to(device=device, non_blocking=True)
    return result


def forward_loss(
    model: torch.nn.Module,
    sample: dict[str, Any],
    image_embedding: torch.Tensor,
    config: CAISConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not hasattr(model, "cognitive_prompt_encoder"):
        previous_mask = sample["previous_mask"].to(device=image_embedding.device, dtype=image_embedding.dtype)
        epsilon = torch.finfo(previous_mask.dtype).eps
        mask_prompt = torch.logit(previous_mask.clamp(epsilon, 1.0 - epsilon))
        sparse, dense = model.prompt_encoder(
            points=(sample["point_coords"], sample["point_labels"]),
            boxes=None,
            masks=mask_prompt,
        )
        mask_logits, iou_predictions = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        probability = mask_logits.sigmoid()
        output = {
            "final_mask": probability,
            "iou_predictions": iou_predictions,
            "cognitive_friction": probability.new_zeros(probability.shape[0]),
        }
        losses = cais_multitask_loss(
            probability,
            sample["target"],
            iou_prediction=iou_predictions,
            cognitive_supervision_mask=torch.zeros_like(sample["cognitive_supervision_mask"]),
            boundary_alpha=0.0,
            boundary_width=config.boundary_width,
            lambda_iou=config.lambda_iou,
            lambda_cognitive=0.0,
        )
        return output, losses
    output = model(
        image_embeddings=image_embedding,
        point_coords=sample["point_coords"],
        point_labels=sample["point_labels"],
        click_trajectories=sample["click_trajectories"],
        trajectory_valid_mask=sample["trajectory_valid_mask"],
        previous_mask=sample["previous_mask"],
        multimask_output=False,
    )
    losses = cais_multitask_loss(
        output["final_mask"],
        sample["target"],
        iou_prediction=output["iou_predictions"],
        cognitive_prediction=output["cognitive_friction"],
        cognitive_target=sample["cognitive_target"],
        cognitive_supervision_mask=sample["cognitive_supervision_mask"],
        boundary_alpha=config.boundary_alpha,
        boundary_width=config.boundary_width,
        lambda_iou=config.lambda_iou,
        lambda_cognitive=config.lambda_cognitive,
    )
    return output, losses


def update_metrics(metrics: dict[str, float], output: dict[str, torch.Tensor], losses: dict[str, torch.Tensor]) -> None:
    for key in ("loss", "loss_mask", "loss_bce", "loss_dice", "loss_iou", "loss_cognitive"):
        metrics[key] += float(losses[key].detach())
    iou_target = losses["iou_target"].reshape(-1)
    iou_prediction = output["iou_predictions"].reshape(-1)
    metrics["mask_iou"] += float(iou_target.mean())
    metrics["iou_mae"] += float((iou_prediction - iou_target).abs().mean().detach())
    cognitive_target = losses.get("cognitive_target")
    if cognitive_target is None:
        cognitive_target = output["cognitive_friction"].new_zeros(output["cognitive_friction"].shape)
    supervision_mask = losses.get("cognitive_supervision_mask")
    if supervision_mask is None:
        supervision_mask = torch.ones_like(cognitive_target.reshape(-1), dtype=torch.bool)
    supervision_mask = supervision_mask.reshape(-1).bool()
    if supervision_mask.any():
        error = (
            output["cognitive_friction"].reshape(-1)[supervision_mask] - cognitive_target.reshape(-1)[supervision_mask]
        ).abs()
        metrics["cognitive_mae"] += float(error.mean().detach())
        metrics["cognitive_samples"] += float(supervision_mask.sum())
    metrics["samples"] += 1.0


def finalized_metrics(metrics: dict[str, float]) -> dict[str, float]:
    samples = max(metrics.get("samples", 0.0), 1.0)
    cognitive_samples = max(metrics.get("cognitive_samples", 0.0), 1.0)
    result = {
        key: value / (cognitive_samples if key == "cognitive_mae" else samples)
        for key, value in metrics.items()
        if key not in {"samples", "cognitive_samples"}
    }
    result.setdefault("cognitive_mae", 0.0)
    result["samples"] = int(metrics.get("samples", 0.0))
    result["cognitive_samples"] = int(metrics.get("cognitive_samples", 0.0))
    return result


def sample_indices(length: int, limit: int | None, *, shuffle: bool, seed: int) -> list[int]:
    indices = list(range(length))
    if shuffle:
        generator = random.Random(seed)
        generator.shuffle(indices)
    if limit is not None:
        indices = indices[: min(limit, len(indices))]
    return indices


def shuffled_trajectory_donors(records: list[dict[str, Any]], seed: int) -> dict[int, int]:
    """Create a deterministic derangement within source and prompt-count strata."""
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = (str(record.get("source", "human")), len(record["prompt_history"]))
        groups[key].append(index)
    generator = random.Random(seed)
    donors: dict[int, int] = {}
    for key in sorted(groups):
        indices = groups[key]
        generator.shuffle(indices)
        if len(indices) == 1:
            donors[indices[0]] = indices[0]
            continue
        offset = generator.randrange(1, len(indices))
        rotated = indices[offset:] + indices[:offset]
        donors.update(zip(indices, rotated))
    return donors


def apply_trajectory_mode(
    sample: dict[str, Any],
    mode: str,
    donor_inputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Apply one trajectory control without changing points, masks, or labels."""
    if mode == "real":
        return sample
    result = dict(sample)
    if mode == "zero":
        result["click_trajectories"] = torch.zeros_like(sample["click_trajectories"])
        result["trajectory_valid_mask"] = torch.zeros_like(sample["trajectory_valid_mask"])
        return result
    if mode != "shuffle":
        raise ValueError(f"Unsupported trajectory mode: {mode}")
    if donor_inputs is None:
        raise ValueError("shuffle trajectory mode requires donor inputs")
    if donor_inputs["click_trajectories"].shape != sample["click_trajectories"].shape:
        raise ValueError("Shuffled trajectory donor must match the target prompt-history shape")
    result["click_trajectories"] = donor_inputs["click_trajectories"].clone()
    result["trajectory_valid_mask"] = donor_inputs["trajectory_valid_mask"].clone()
    return result


def source_balanced_indices(
    records: list[dict[str, Any]],
    limit: int | None,
    *,
    seed: int,
) -> tuple[list[int], dict[str, int]]:
    """Sample equal human/synthetic click records without replacement.

    Adjacent human/synthetic pairs keep every even-sized accumulation window
    balanced (the default accumulation size is four). The shuffled synthetic
    subset changes with the epoch seed so unused synthetic records rotate in.
    """
    human = [index for index, record in enumerate(records) if record.get("source", "human") != "synthetic"]
    synthetic = [index for index, record in enumerate(records) if record.get("source", "human") == "synthetic"]
    if not human or not synthetic:
        raise ValueError(
            "--source-sampling balanced requires both human and synthetic train records; "
            f"found human={len(human)}, synthetic={len(synthetic)}"
        )

    generator = random.Random(seed)
    generator.shuffle(human)
    generator.shuffle(synthetic)
    per_source = min(len(human), len(synthetic))
    if limit is not None:
        per_source = min(per_source, max(limit, 0) // 2)
    if per_source == 0:
        raise ValueError("Balanced sampling requires --max-train-samples >= 2")

    indices: list[int] = []
    for human_index, synthetic_index in zip(human[:per_source], synthetic[:per_source]):
        pair = [human_index, synthetic_index]
        generator.shuffle(pair)
        indices.extend(pair)
    return indices, {"human": per_source, "synthetic": per_source}


def run_train_epoch(
    model: torch.nn.Module,
    dataset: COCOClickCAISDataset,
    cache: ImageEmbeddingCache,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: CAISConfig,
    device: torch.device,
    *,
    epoch: int,
    accumulation_steps: int,
    gradient_clip: float,
    log_interval: int,
    max_samples: int | None,
    train_mask_decoder: bool,
    amp: bool,
    seed: int,
    source_sampling: str,
    trajectory_mode: str,
) -> dict[str, float]:
    set_training_mode(model, train_mask_decoder)
    metrics: dict[str, float] = defaultdict(float)
    if source_sampling == "balanced":
        indices, source_counts = source_balanced_indices(dataset.records, max_samples, seed=seed + epoch)
    else:
        indices = sample_indices(len(dataset), max_samples, shuffle=True, seed=seed + epoch)
        source_counts: dict[str, int] = defaultdict(int)
        for index in indices:
            source_counts[str(dataset.records[index].get("source", "human"))] += 1
    print(
        f"train epoch={epoch:02d} sampling={source_sampling} "
        f"human={source_counts.get('human', 0)} synthetic={source_counts.get('synthetic', 0)} "
        f"total={len(indices)} trajectory={trajectory_mode}",
        flush=True,
    )
    trajectory_donors = (
        shuffled_trajectory_donors(dataset.records, seed + epoch) if trajectory_mode == "shuffle" else None
    )
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for step, index in enumerate(indices, start=1):
        raw_sample = dataset[index]
        donor_inputs = (
            dataset.trajectory_inputs(trajectory_donors[index]) if trajectory_donors is not None else None
        )
        sample = move_sample(apply_trajectory_mode(raw_sample, trajectory_mode, donor_inputs), device)
        image_embedding = cache.get(sample["image_path"])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output, losses = forward_loss(model, sample, image_embedding, config)
            scaled_loss = losses["loss"] / accumulation_steps
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}: {losses['loss']}")
        scaler.scale(scaled_loss).backward()
        should_step = step % accumulation_steps == 0 or step == len(indices)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]], gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses["cognitive_target"] = sample["cognitive_target"].detach()
        update_metrics(metrics, output, losses)
        if step == 1 or step % log_interval == 0 or step == len(indices):
            current = finalized_metrics(metrics)
            elapsed = time.perf_counter() - started
            print(
                f"train epoch={epoch:02d} step={step:03d}/{len(indices):03d} "
                f"loss={current['loss']:.4f} iou={current['mask_iou']:.4f} "
                f"cog_mae={current['cognitive_mae']:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    result = finalized_metrics(metrics)
    result["human_samples"] = int(source_counts.get("human", 0))
    result["synthetic_samples"] = int(source_counts.get("synthetic", 0))
    return result


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: COCOClickCAISDataset,
    cache: ImageEmbeddingCache,
    config: CAISConfig,
    device: torch.device,
    *,
    split: str,
    max_samples: int | None,
    amp: bool,
    trajectory_mode: str = "real",
    trajectory_seed: int = 0,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = defaultdict(float)
    indices = sample_indices(len(dataset), max_samples, shuffle=False, seed=0)
    trajectory_donors = (
        shuffled_trajectory_donors(dataset.records, trajectory_seed) if trajectory_mode == "shuffle" else None
    )
    started = time.perf_counter()
    for index in indices:
        raw_sample = dataset[index]
        donor_inputs = (
            dataset.trajectory_inputs(trajectory_donors[index]) if trajectory_donors is not None else None
        )
        sample = move_sample(apply_trajectory_mode(raw_sample, trajectory_mode, donor_inputs), device)
        image_embedding = cache.get(sample["image_path"])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output, losses = forward_loss(model, sample, image_embedding, config)
        losses["cognitive_target"] = sample["cognitive_target"]
        update_metrics(metrics, output, losses)
    result = finalized_metrics(metrics)
    print(
        f"{split} loss={result['loss']:.4f} iou={result['mask_iou']:.4f} "
        f"iou_mae={result['iou_mae']:.4f} cog_mae={result['cognitive_mae']:.4f} "
        f"samples={result['samples']} elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return result


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: CAISConfig,
    dataset: COCOClickCAISDataset,
    args: argparse.Namespace,
    epoch: int,
    best_val_iou: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "architecture": (
            "EfficientViT-SAM-L0-mask-decoder-ablation"
            if args.standard_sam_ablation
            else "CAIS-EfficientViT-SAM-L0"
        ),
        "epoch": epoch,
        "best_val_iou": best_val_iou,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "cais_config": config.to_dict(),
        "cognitive_target": dataset.cognitive_normalizer.to_dict(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "history": history,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.max_train_samples = 2
        args.max_val_samples = 2
        args.max_test_samples = 2
        args.output_dir = args.output_dir / "smoke"
    if args.epochs < 1 or args.accumulation_steps < 1:
        raise ValueError("epochs and accumulation-steps must be positive")
    if args.disable_trajectory_input and args.trajectory_mode != "real":
        raise ValueError("Use either --disable-trajectory-input or --trajectory-mode, not both")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = CAISConfig(
        trajectory_samples=args.trajectory_samples,
        use_trajectory=not args.disable_trajectory_input,
        prompt_integration=args.prompt_integration,
        attention_integration=args.attention_integration,
        boundary_alpha=args.boundary_alpha,
        boundary_width=args.boundary_width,
        lambda_iou=args.lambda_iou,
        lambda_cognitive=args.lambda_cognitive,
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
    )
    if args.standard_sam_ablation:
        if (
            args.disable_trajectory_input
            or args.trajectory_mode != "real"
            or args.prompt_integration != "replace"
            or args.attention_integration != "replace"
            or args.lambda_cognitive != 0.075
            or args.boundary_alpha != 2.0
        ):
            raise ValueError("--standard-sam-ablation cannot be combined with CAIS-only ablation flags")
        print(f"Loading standard EfficientViT-SAM ablation from {args.base_checkpoint}", flush=True)
        model = create_efficientvit_sam_model(
            "efficientvit-sam-l0",
            pretrained=True,
            weight_url=str(args.base_checkpoint),
            image_size=512,
        ).to(device)
    else:
        print(f"Loading independent CAIS model from {args.base_checkpoint}", flush=True)
        model = create_efficientvit_sam_model(
            "efficientvit-sam-cais-l0",
            pretrained=True,
            weight_url=str(args.base_checkpoint),
            cais_config=config.to_dict(),
            image_size=512,
        ).to(device)
    trainable = configure_trainable_parameters(model, args.train_mask_decoder)
    trainable_count = sum(parameter.numel() for parameter in trainable)
    print(f"Trainable parameters: {trainable_count:,}; AMP: {amp}; device: {device}", flush=True)

    train_dataset = COCOClickCAISDataset(
        args.dataset_root,
        "train",
        trajectory_samples=config.trajectory_samples,
    )
    normalizer = train_dataset.cognitive_normalizer
    val_dataset = COCOClickCAISDataset(
        args.dataset_root,
        "val",
        trajectory_samples=config.trajectory_samples,
        cognitive_normalizer=normalizer,
    )
    test_dataset = COCOClickCAISDataset(
        args.dataset_root,
        "test",
        trajectory_samples=config.trajectory_samples,
        cognitive_normalizer=normalizer,
    )
    print(
        f"Strict split clicks: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, init_scale=128.0, growth_interval=2000, enabled=amp)
    cache = ImageEmbeddingCache(model, device, amp=amp, enabled=not args.no_image_cache)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_val_iou = -math.inf

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_iou = float(checkpoint.get("best_val_iou", -math.inf))
        history = list(checkpoint.get("history", []))
        print(f"Resumed at epoch {start_epoch} from {args.resume}", flush=True)

    run_manifest = {
        "architecture": (
            "EfficientViT-SAM-L0 mask-decoder fine-tune ablation"
            if args.standard_sam_ablation
            else "CAIS-EfficientViT-SAM-L0 (independent from VIMR)"
        ),
        "dataset": {
            "root": str(args.dataset_root),
            "train": train_dataset.describe(),
            "val": val_dataset.describe(),
            "test": test_dataset.describe(),
        },
        "config": config.to_dict(),
        "sampling": {
            "strategy": args.source_sampling,
            "unit": "click_record",
            "replacement": False,
            "rotate_seed_each_epoch": True,
            "max_train_samples": args.max_train_samples,
            "trajectory_mode": args.trajectory_mode,
            "shuffle_strata": ["source", "prompt_count"] if args.trajectory_mode == "shuffle" else None,
        },
        "trainable_parameters": trainable_count,
        "base_checkpoint": str(args.base_checkpoint),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_train_epoch(
            model,
            train_dataset,
            cache,
            optimizer,
            scaler,
            config,
            device,
            epoch=epoch,
            accumulation_steps=args.accumulation_steps,
            gradient_clip=args.gradient_clip,
            log_interval=args.log_interval,
            max_samples=args.max_train_samples,
            train_mask_decoder=args.train_mask_decoder,
            amp=amp,
            seed=args.seed,
            source_sampling=args.source_sampling,
            trajectory_mode=args.trajectory_mode,
        )
        val_metrics = evaluate(
            model,
            val_dataset,
            cache,
            config,
            device,
            split="val",
            max_samples=args.max_val_samples,
            amp=amp,
            trajectory_mode=args.trajectory_mode,
            trajectory_seed=args.seed + 100_000,
        )
        scheduler.step()
        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)
        improved = val_metrics["mask_iou"] > best_val_iou
        if improved:
            best_val_iou = val_metrics["mask_iou"]
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            train_dataset,
            args,
            epoch,
            best_val_iou,
            history,
        )
        atomic_torch_save(payload, args.output_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"epoch={epoch:02d} complete best_val_iou={best_val_iou:.4f} " f"cache_images={len(cache.cache)}",
            flush=True,
        )

    best_checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = evaluate(
        model,
        test_dataset,
        cache,
        config,
        device,
        split="test",
        max_samples=args.max_test_samples,
        amp=amp,
        trajectory_mode=args.trajectory_mode,
        trajectory_seed=args.seed + 200_000,
    )
    summary = {
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_val_iou": float(best_checkpoint["best_val_iou"]),
        "test": test_metrics,
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Training finished. Artifacts: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
