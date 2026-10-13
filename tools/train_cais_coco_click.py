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
    parser.add_argument(
        "--decoder-learning-rate",
        type=float,
        help="Mask-decoder learning rate; defaults to --learning-rate.",
    )
    parser.add_argument(
        "--prompt-encoder-learning-rate",
        type=float,
        help="Prompt-encoder learning rate; defaults to --decoder-learning-rate.",
    )
    parser.add_argument(
        "--gate-learning-rate",
        type=float,
        help="Residual-gate learning rate; defaults to --learning-rate.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--trajectory-samples", type=int, default=16)
    parser.add_argument(
        "--trajectory-processing",
        choices=("legacy", "fixed_window_process_v2"),
        default="legacy",
        help=(
            "Use legacy per-fragment stretching or a fixed 2-second clock plus "
            "event-level annotation-process features."
        ),
    )
    parser.add_argument("--trajectory-window-ms", type=float, default=2000.0)
    parser.add_argument("--process-gate-init", type=float, default=0.05)
    parser.add_argument(
        "--disable-trajectory-input",
        action="store_true",
        help="Ablation: keep click coordinates/types but remove all trajectory information.",
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=("real", "zero", "shuffle", "wrong_image"),
        default="real",
        help=(
            "Use aligned trajectories, zero them, or replace them with a deterministic "
            "same-source/same-click-count donor trajectory. wrong_image additionally "
            "requires the donor to come from a different image."
        ),
    )
    parser.add_argument(
        "--normalize-trajectory-input",
        action="store_true",
        help="Standardize trajectory channels with valid train-split statistics.",
    )
    parser.add_argument(
        "--trajectory-encoder",
        choices=("cnn_pool", "tcn_attention"),
        default="cnn_pool",
        help="Encode motion with the legacy CNN/pooling path or a tiny TCN with masked attention pooling.",
    )
    parser.add_argument(
        "--trajectory-token-content",
        choices=("encoded", "constant"),
        default="encoded",
        help="Use encoded motion or an input-independent constant as the decoder trajectory token.",
    )
    parser.add_argument(
        "--prompt-integration",
        choices=("replace", "native", "native_residual", "trajectory_token", "trajectory_dense"),
        default="replace",
        help="Replace prompts, preserve native prompts, add a residual, append trajectory tokens, or add a dense prompt.",
    )
    parser.add_argument("--trajectory-dense-channels", type=int, default=32)
    parser.add_argument("--trajectory-raster-sigma", type=float, default=1.5)
    parser.add_argument("--trajectory-gate-max", type=float, default=1.0)
    parser.add_argument("--trajectory-dropout", type=float, default=0.0)
    parser.add_argument(
        "--interaction-memory",
        choices=("none", "click_mask", "click_mask_trajectory"),
        default="none",
        help="MAIS-style per-round memory using clicks/masks, optionally augmented with trajectories.",
    )
    parser.add_argument("--memory-size", type=int, default=8)
    parser.add_argument("--memory-layers", type=int, default=1)
    parser.add_argument("--memory-heads", type=int, default=8)
    parser.add_argument("--memory-ffn-dim", type=int, default=512)
    parser.add_argument("--memory-mask-channels", type=int, default=32)
    parser.add_argument("--memory-gate-init", type=float, default=0.05)
    parser.add_argument(
        "--attention-integration",
        choices=("replace", "none", "gated_residual"),
        default="replace",
        help="Replace image features, skip CAIS attention, or apply it through a zero-init residual gate.",
    )
    parser.add_argument(
        "--prompt-residual-gate-init",
        type=float,
        default=0.0,
        help="Initial effective gate value for residual or independent trajectory-token integration.",
    )
    parser.add_argument("--lambda-iou", type=float, default=1.0)
    parser.add_argument("--lambda-cognitive", type=float, default=0.075)
    parser.add_argument("--lambda-trajectory-cognitive", type=float, default=0.0)
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
    parser.add_argument(
        "--train-source",
        choices=("all", "human", "synthetic"),
        default="all",
        help="Restrict only the training split to one source; validation and test remain unchanged.",
    )
    parser.add_argument(
        "--train-high-quality-images-only",
        action="store_true",
        help=(
            "Exclude an entire training image when any of its human clicks fails the "
            "target-free trajectory-quality rule. Validation and test remain complete."
        ),
    )
    parser.add_argument(
        "--quality-gate-trajectories",
        action="store_true",
        help=(
            "Keep every image/click/mask sample, but mark trajectories and process features "
            "invalid when the target-free trajectory-quality rule fails. Applied consistently "
            "to train, validation, and test."
        ),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--train-mask-decoder",
        action="store_true",
        help="Also fine-tune the pretrained SAM mask decoder. The default trains CAIS modules only.",
    )
    parser.add_argument(
        "--train-prompt-encoder",
        action="store_true",
        help="Also fine-tune the pretrained SAM prompt encoder; the image encoder remains frozen.",
    )
    parser.add_argument(
        "--decoder-warmup-epochs",
        type=int,
        default=0,
        help="Keep the mask decoder frozen for this many initial epochs.",
    )
    parser.add_argument(
        "--evaluate-trajectory-controls",
        action="store_true",
        help="Evaluate the best checkpoint with real, zero, shuffled, and wrong-image trajectories.",
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


def configure_trainable_parameters(
    model: torch.nn.Module,
    train_mask_decoder: bool,
    train_prompt_encoder: bool = False,
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not hasattr(model, "cognitive_prompt_encoder"):
        for module, enabled in (
            (model.prompt_encoder, train_prompt_encoder),
            (model.mask_decoder, train_mask_decoder),
        ):
            if enabled:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        return [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_modules = [
        model.cognitive_prompt_encoder,
        model.prediction_heads,
    ]
    for module_name in (
        "trajectory_token_adapter",
        "trajectory_dense_prompt_encoder",
        "trajectory_cognitive_head",
        "interaction_memory_attention",
    ):
        if (module := getattr(model, module_name, None)) is not None:
            trainable_modules.append(module)
    if model.cais_config.attention_integration != "none":
        trainable_modules.append(model.cognitive_two_way_attention)
    if train_mask_decoder:
        trainable_modules.append(model.mask_decoder)
    if train_prompt_encoder:
        trainable_modules.append(model.prompt_encoder)
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


def build_optimizer_parameter_groups(
    model: torch.nn.Module,
    trainable: list[torch.nn.Parameter],
    *,
    module_learning_rate: float,
    gate_learning_rate: float,
    decoder_learning_rate: float,
    weight_decay: float,
    prompt_encoder_learning_rate: float | None = None,
) -> list[dict[str, Any]]:
    """Separate new modules, residual gates, and the pretrained decoder."""
    decoder_ids = {id(parameter) for parameter in model.mask_decoder.parameters()}
    prompt_encoder_ids = {id(parameter) for parameter in model.prompt_encoder.parameters()}
    gate_ids = {
        id(gate)
        for name in ("prompt_residual_gate", "image_residual_gate")
        if (gate := getattr(model, name, None)) is not None and gate.requires_grad
    }
    process_gate = getattr(getattr(model, "cognitive_prompt_encoder", None), "process_gate", None)
    if process_gate is not None and process_gate.requires_grad:
        gate_ids.add(id(process_gate))
    trajectory_token_adapter = getattr(model, "trajectory_token_adapter", None)
    if trajectory_token_adapter is not None:
        gate_ids.update(
            id(parameter)
            for parameter in trajectory_token_adapter.gate.parameters()
            if parameter.requires_grad
        )
    trajectory_dense_encoder = getattr(model, "trajectory_dense_prompt_encoder", None)
    if trajectory_dense_encoder is not None:
        gate_ids.update(
            id(parameter)
            for parameter in trajectory_dense_encoder.gate.parameters()
            if parameter.requires_grad
        )
    interaction_memory = getattr(model, "interaction_memory_attention", None)
    if interaction_memory is not None and interaction_memory.gate.requires_grad:
        gate_ids.add(id(interaction_memory.gate))
    grouped = {
        "modules": [
            parameter
            for parameter in trainable
            if id(parameter) not in decoder_ids | prompt_encoder_ids | gate_ids
        ],
        "gates": [parameter for parameter in trainable if id(parameter) in gate_ids],
        "prompt_encoder": [
            parameter for parameter in trainable if id(parameter) in prompt_encoder_ids
        ],
        "decoder": [parameter for parameter in trainable if id(parameter) in decoder_ids],
    }
    learning_rates = {
        "modules": module_learning_rate,
        "gates": gate_learning_rate,
        "prompt_encoder": prompt_encoder_learning_rate or decoder_learning_rate,
        "decoder": decoder_learning_rate,
    }
    return [
        {
            "name": name,
            "params": parameters,
            "lr": learning_rates[name],
            "weight_decay": 0.0 if name == "gates" else weight_decay,
        }
        for name, parameters in grouped.items()
        if parameters
    ]


def set_mask_decoder_trainable(model: torch.nn.Module, enabled: bool) -> None:
    for parameter in model.mask_decoder.parameters():
        parameter.requires_grad = enabled


def set_training_mode(
    model: torch.nn.Module,
    train_mask_decoder: bool,
    train_prompt_encoder: bool = False,
) -> None:
    model.train()
    model.image_encoder.eval()
    if not train_prompt_encoder:
        model.prompt_encoder.eval()
    if not train_mask_decoder:
        model.mask_decoder.eval()
    if not hasattr(model, "cognitive_prompt_encoder"):
        return


def move_sample(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    keys = (
        "point_coords",
        "point_labels",
        "click_trajectories",
        "trajectory_valid_mask",
        "process_features",
        "process_feature_valid",
        "trajectory_coordinate_scale",
        "target",
        "previous_mask",
        "interaction_masks",
        "interaction_mask_valid",
        "cognitive_target",
        "cognitive_supervision_mask",
    )
    result = dict(sample)
    for key in keys:
        tensor = sample[key]
        if key in {
            "point_coords",
            "point_labels",
            "click_trajectories",
            "trajectory_valid_mask",
            "process_features",
            "process_feature_valid",
            "trajectory_coordinate_scale",
        }:
            tensor = tensor.unsqueeze(0)
        elif key in {"target", "previous_mask", "interaction_masks", "interaction_mask_valid"}:
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
        process_features=sample["process_features"],
        process_feature_valid=sample["process_feature_valid"],
        trajectory_coordinate_scale=sample["trajectory_coordinate_scale"],
        previous_mask=sample["previous_mask"],
        interaction_masks=sample["interaction_masks"],
        interaction_mask_valid=sample["interaction_mask_valid"],
        multimask_output=False,
    )
    losses = cais_multitask_loss(
        output["final_mask"],
        sample["target"],
        iou_prediction=output["iou_predictions"],
        cognitive_prediction=output["cognitive_friction"],
        trajectory_cognitive_prediction=output.get("trajectory_cognitive_friction"),
        cognitive_target=sample["cognitive_target"],
        cognitive_supervision_mask=sample["cognitive_supervision_mask"],
        trajectory_cognitive_supervision_mask=(
            sample["trajectory_valid_mask"][:, -1].any(dim=-1)
        ),
        boundary_alpha=config.boundary_alpha,
        boundary_width=config.boundary_width,
        lambda_iou=config.lambda_iou,
        lambda_cognitive=config.lambda_cognitive,
        lambda_trajectory_cognitive=config.lambda_trajectory_cognitive,
    )
    return output, losses


def update_metrics(metrics: dict[str, float], output: dict[str, torch.Tensor], losses: dict[str, torch.Tensor]) -> None:
    for key in (
        "loss",
        "loss_mask",
        "loss_bce",
        "loss_dice",
        "loss_iou",
        "loss_cognitive",
        "loss_trajectory_cognitive",
    ):
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
        trajectory_cognitive = output.get("trajectory_cognitive_friction")
        if trajectory_cognitive is not None:
            trajectory_supervision_mask = losses.get(
                "trajectory_cognitive_supervision_mask", supervision_mask
            ).reshape(-1).bool()
            trajectory_error = (
                trajectory_cognitive.reshape(-1)[trajectory_supervision_mask]
                - cognitive_target.reshape(-1)[trajectory_supervision_mask]
            ).abs()
            if trajectory_error.numel():
                metrics["trajectory_cognitive_mae"] += float(trajectory_error.mean().detach())
                metrics["trajectory_cognitive_samples"] += float(
                    trajectory_supervision_mask.sum()
                )
    trajectory_gate = output.get("trajectory_token_gate")
    if trajectory_gate is None:
        trajectory_gate = output.get("trajectory_dense_gate")
    trajectory_present = output.get("trajectory_present")
    if trajectory_gate is not None and trajectory_present is not None and trajectory_present.any():
        metrics["trajectory_gate_mean"] += float(trajectory_gate[trajectory_present].mean().detach())
        metrics["trajectory_gate_samples"] += 1.0
    memory_gate = output.get("interaction_memory_gate")
    if memory_gate is not None:
        metrics["memory_gate_mean"] += float(memory_gate.detach())
    metrics["samples"] += 1.0


def finalized_metrics(metrics: dict[str, float]) -> dict[str, float]:
    samples = max(metrics.get("samples", 0.0), 1.0)
    cognitive_samples = max(metrics.get("cognitive_samples", 0.0), 1.0)
    trajectory_cognitive_samples = max(metrics.get("trajectory_cognitive_samples", 0.0), 1.0)
    trajectory_gate_samples = max(metrics.get("trajectory_gate_samples", 0.0), 1.0)
    result = {
        key: value
        / (
            cognitive_samples
            if key == "cognitive_mae"
            else trajectory_cognitive_samples
            if key == "trajectory_cognitive_mae"
            else trajectory_gate_samples
            if key == "trajectory_gate_mean"
            else samples
        )
        for key, value in metrics.items()
        if key
        not in {
            "samples",
            "cognitive_samples",
            "trajectory_cognitive_samples",
            "trajectory_gate_samples",
        }
    }
    result.setdefault("cognitive_mae", 0.0)
    result.setdefault("trajectory_cognitive_mae", 0.0)
    result.setdefault("trajectory_gate_mean", 0.0)
    result.setdefault("memory_gate_mean", 0.0)
    result["samples"] = int(metrics.get("samples", 0.0))
    result["cognitive_samples"] = int(metrics.get("cognitive_samples", 0.0))
    result["trajectory_cognitive_samples"] = int(metrics.get("trajectory_cognitive_samples", 0.0))
    result["trajectory_gate_samples"] = int(metrics.get("trajectory_gate_samples", 0.0))
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


def _image_identity(record: dict[str, Any]) -> str:
    """Return a stable image identity for cross-image trajectory controls."""
    return str(record.get("coco_image_id") or record.get("image_ref") or record.get("image_id"))


def wrong_image_trajectory_donors(records: list[dict[str, Any]], seed: int) -> dict[int, int]:
    """Create deterministic same-source/count donors from explicitly different images."""
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = (str(record.get("source", "human")), len(record["prompt_history"]))
        groups[key].append(index)
    generator = random.Random(seed)
    donors: dict[int, int] = {}
    for key in sorted(groups):
        indices = list(groups[key])
        generator.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(f"wrong_image control requires at least two records in stratum {key!r}")
        mapping = None
        for offset in range(1, len(indices)):
            rotated = indices[offset:] + indices[:offset]
            candidate = dict(zip(indices, rotated))
            if all(
                _image_identity(records[target]) != _image_identity(records[donor])
                for target, donor in candidate.items()
            ):
                mapping = candidate
                break
        if mapping is None:
            raise ValueError(f"Could not construct a different-image donor rotation for stratum {key!r}")
        donors.update(mapping)
    return donors


def trajectory_donors(
    records: list[dict[str, Any]], mode: str, seed: int
) -> dict[int, int] | None:
    if mode == "shuffle":
        return shuffled_trajectory_donors(records, seed)
    if mode == "wrong_image":
        return wrong_image_trajectory_donors(records, seed)
    return None


def paired_task_bootstrap_delta(
    real_metrics: dict[str, Any],
    comparison_metrics: dict[str, Any],
    *,
    seed: int,
    iterations: int = 2000,
) -> dict[str, Any]:
    """Bootstrap the task-mean Real-minus-control IoU difference."""
    real = real_metrics["task_level"]["iou_by_task"]
    comparison = comparison_metrics["task_level"]["iou_by_task"]
    task_ids = sorted(set(real) & set(comparison))
    if not task_ids:
        raise ValueError("Paired task bootstrap requires overlapping task IDs")
    deltas = np.asarray([real[task_id] - comparison[task_id] for task_id in task_ids], dtype=np.float64)
    generator = np.random.default_rng(seed)
    bootstrap = deltas[
        generator.integers(0, len(deltas), size=(iterations, len(deltas)))
    ].mean(axis=1)
    return {
        "tasks": len(task_ids),
        "mean_delta": float(deltas.mean()),
        "bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


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
        if "process_features" in sample:
            result["process_features"] = torch.zeros_like(sample["process_features"])
            result["process_feature_valid"] = torch.zeros_like(sample["process_feature_valid"])
        return result
    if mode not in {"shuffle", "wrong_image"}:
        raise ValueError(f"Unsupported trajectory mode: {mode}")
    if donor_inputs is None:
        raise ValueError(f"{mode} trajectory mode requires donor inputs")
    if donor_inputs["click_trajectories"].shape != sample["click_trajectories"].shape:
        raise ValueError("Shuffled trajectory donor must match the target prompt-history shape")
    result["click_trajectories"] = donor_inputs["click_trajectories"].clone()
    result["trajectory_valid_mask"] = donor_inputs["trajectory_valid_mask"].clone()
    if "process_features" in sample:
        result["process_features"] = donor_inputs["process_features"].clone()
        result["process_feature_valid"] = donor_inputs["process_feature_valid"].clone()
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
    train_prompt_encoder: bool,
    amp: bool,
    seed: int,
    source_sampling: str,
    trajectory_mode: str,
) -> dict[str, float]:
    set_training_mode(model, train_mask_decoder, train_prompt_encoder)
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
    donor_mapping = trajectory_donors(dataset.records, trajectory_mode, seed + epoch)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for step, index in enumerate(indices, start=1):
        raw_sample = dataset[index]
        donor_inputs = (
            dataset.trajectory_inputs(donor_mapping[index]) if donor_mapping is not None else None
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
    collect_task_metrics: bool = False,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = defaultdict(float)
    indices = sample_indices(len(dataset), max_samples, shuffle=False, seed=0)
    donor_mapping = trajectory_donors(dataset.records, trajectory_mode, trajectory_seed)
    task_ious: dict[str, list[float]] = defaultdict(list)
    click_index_ious: dict[int, list[float]] = defaultdict(list)
    started = time.perf_counter()
    for index in indices:
        raw_sample = dataset[index]
        donor_inputs = (
            dataset.trajectory_inputs(donor_mapping[index]) if donor_mapping is not None else None
        )
        sample = move_sample(apply_trajectory_mode(raw_sample, trajectory_mode, donor_inputs), device)
        image_embedding = cache.get(sample["image_path"])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output, losses = forward_loss(model, sample, image_embedding, config)
        losses["cognitive_target"] = sample["cognitive_target"]
        update_metrics(metrics, output, losses)
        if collect_task_metrics:
            sample_iou = float(losses["iou_target"].mean())
            task_ious[str(sample["task_run_id"])].append(sample_iou)
            click_index_ious[int(sample["click_index"])].append(sample_iou)
    result = finalized_metrics(metrics)
    if collect_task_metrics:
        task_means = {
            task_id: float(np.mean(values)) for task_id, values in sorted(task_ious.items())
        }
        values = np.asarray(list(task_means.values()), dtype=np.float64)
        generator = np.random.default_rng(trajectory_seed)
        bootstrap = values[
            generator.integers(0, len(values), size=(2000, len(values)))
        ].mean(axis=1)
        result["task_level"] = {
            "tasks": len(task_means),
            "mean_iou": float(values.mean()),
            "bootstrap_95ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "iou_by_task": task_means,
        }
        result["click_index_level"] = {
            str(click_index): {
                "samples": len(click_index_ious[click_index]),
                "mean_iou": float(np.mean(click_index_ious[click_index])),
            }
            for click_index in sorted(click_index_ious)
        }
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
    if args.decoder_warmup_epochs < 0:
        raise ValueError("decoder-warmup-epochs must be non-negative")
    if args.decoder_warmup_epochs and not args.train_mask_decoder:
        raise ValueError("decoder warmup requires --train-mask-decoder")
    if not -1.0 < args.prompt_residual_gate_init < 1.0:
        raise ValueError("prompt-residual-gate-init must be strictly between -1 and 1")
    if args.prompt_integration in {"trajectory_token", "trajectory_dense"} and args.prompt_residual_gate_init <= 0:
        raise ValueError("trajectory token/dense integration requires a gate initialization between 0 and 1")
    if args.trajectory_token_content != "encoded" and args.prompt_integration != "trajectory_token":
        raise ValueError("non-encoded trajectory-token content requires --prompt-integration trajectory_token")
    if args.lambda_trajectory_cognitive < 0:
        raise ValueError("lambda-trajectory-cognitive must be non-negative")
    if args.trajectory_dense_channels < 1 or args.trajectory_raster_sigma <= 0:
        raise ValueError("trajectory dense channels and raster sigma must be positive")
    if not 0 < args.trajectory_gate_max <= 1:
        raise ValueError("trajectory-gate-max must be in (0,1]")
    if args.prompt_integration == "trajectory_dense" and not (
        0 < args.prompt_residual_gate_init < args.trajectory_gate_max
    ):
        raise ValueError("trajectory dense gate initialization must be below trajectory-gate-max")
    if not 0 <= args.trajectory_dropout <= 1:
        raise ValueError("trajectory-dropout must be in [0,1]")
    if args.memory_size < 1 or args.memory_layers < 1 or args.memory_heads < 1:
        raise ValueError("memory size, layers, and heads must be positive")
    if args.memory_ffn_dim < 1 or args.memory_mask_channels < 1:
        raise ValueError("memory FFN dimension and mask channels must be positive")
    if not -1 < args.memory_gate_init < 1:
        raise ValueError("memory-gate-init must be strictly between -1 and 1")
    if not 0 < args.process_gate_init < 1:
        raise ValueError("process-gate-init must be strictly between 0 and 1")
    if args.trajectory_window_ms <= 0:
        raise ValueError("trajectory-window-ms must be positive")
    if args.interaction_memory != "none" and (
        args.prompt_integration != "native" or args.attention_integration != "none"
    ):
        raise ValueError(
            "interaction memory requires --prompt-integration native --attention-integration none"
        )
    for name, value in (
        ("learning-rate", args.learning_rate),
        ("decoder-learning-rate", args.decoder_learning_rate),
        ("prompt-encoder-learning-rate", args.prompt_encoder_learning_rate),
        ("gate-learning-rate", args.gate_learning_rate),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.disable_trajectory_input and args.trajectory_mode != "real":
        raise ValueError("Use either --disable-trajectory-input or --trajectory-mode, not both")
    if args.train_source != "all" and args.source_sampling == "balanced":
        raise ValueError("--source-sampling balanced is incompatible with a single --train-source")
    if args.train_high_quality_images_only and args.train_source != "human":
        raise ValueError("--train-high-quality-images-only requires --train-source human")
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

    train_dataset = COCOClickCAISDataset(
        args.dataset_root,
        "train",
        source_filter=None if args.train_source == "all" else args.train_source,
        trajectory_samples=args.trajectory_samples,
        trajectory_processing=args.trajectory_processing,
        trajectory_window_ms=args.trajectory_window_ms,
        high_quality_images_only=args.train_high_quality_images_only,
        quality_gate_trajectories=args.quality_gate_trajectories,
    )
    normalizer = train_dataset.cognitive_normalizer
    val_dataset = COCOClickCAISDataset(
        args.dataset_root,
        "val",
        trajectory_samples=args.trajectory_samples,
        trajectory_processing=args.trajectory_processing,
        trajectory_window_ms=args.trajectory_window_ms,
        cognitive_normalizer=normalizer,
        quality_gate_trajectories=args.quality_gate_trajectories,
    )
    test_dataset = COCOClickCAISDataset(
        args.dataset_root,
        "test",
        trajectory_samples=args.trajectory_samples,
        trajectory_processing=args.trajectory_processing,
        trajectory_window_ms=args.trajectory_window_ms,
        cognitive_normalizer=normalizer,
        quality_gate_trajectories=args.quality_gate_trajectories,
    )
    trajectory_mean, trajectory_std = (
        train_dataset.trajectory_statistics()
        if args.normalize_trajectory_input
        else ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    )
    process_mean, process_std = train_dataset.process_statistics()
    print(
        f"Strict split clicks: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}",
        flush=True,
    )
    print(
        f"Trajectory normalization: mean={trajectory_mean}, std={trajectory_std}",
        flush=True,
    )
    if process_mean:
        print(
            f"Process normalization: dimensions={len(process_mean)} gate_init={args.process_gate_init}",
            flush=True,
        )

    config = CAISConfig(
        trajectory_samples=args.trajectory_samples,
        trajectory_mean=trajectory_mean,
        trajectory_std=trajectory_std,
        trajectory_encoder=args.trajectory_encoder,
        trajectory_processing=args.trajectory_processing,
        trajectory_window_ms=args.trajectory_window_ms,
        process_feature_dim=len(process_mean),
        process_mean=process_mean,
        process_std=process_std,
        process_gate_init=args.process_gate_init,
        trajectory_token_content=args.trajectory_token_content,
        trajectory_dense_channels=args.trajectory_dense_channels,
        trajectory_raster_sigma=args.trajectory_raster_sigma,
        trajectory_gate_max=args.trajectory_gate_max,
        trajectory_dropout=args.trajectory_dropout,
        use_trajectory=not args.disable_trajectory_input,
        interaction_memory=args.interaction_memory,
        memory_size=args.memory_size,
        memory_layers=args.memory_layers,
        memory_heads=args.memory_heads,
        memory_ffn_dim=args.memory_ffn_dim,
        memory_mask_channels=args.memory_mask_channels,
        memory_gate_init=args.memory_gate_init,
        prompt_integration=args.prompt_integration,
        attention_integration=args.attention_integration,
        prompt_residual_gate_init=args.prompt_residual_gate_init,
        boundary_alpha=args.boundary_alpha,
        boundary_width=args.boundary_width,
        lambda_iou=args.lambda_iou,
        lambda_cognitive=args.lambda_cognitive,
        lambda_trajectory_cognitive=args.lambda_trajectory_cognitive,
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
    )
    if args.standard_sam_ablation:
        if (
            args.disable_trajectory_input
            or args.trajectory_mode != "real"
            or args.normalize_trajectory_input
            or args.trajectory_encoder != "cnn_pool"
            or args.trajectory_token_content != "encoded"
            or args.prompt_integration != "replace"
            or args.attention_integration != "replace"
            or args.prompt_residual_gate_init != 0.0
            or args.lambda_cognitive != 0.075
            or args.lambda_trajectory_cognitive != 0.0
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
    trainable = configure_trainable_parameters(
        model,
        args.train_mask_decoder,
        args.train_prompt_encoder,
    )
    trainable_count = sum(parameter.numel() for parameter in trainable)
    print(f"Trainable parameters: {trainable_count:,}; AMP: {amp}; device: {device}", flush=True)

    optimizer_groups = build_optimizer_parameter_groups(
        model,
        trainable,
        module_learning_rate=args.learning_rate,
        gate_learning_rate=args.gate_learning_rate or args.learning_rate,
        decoder_learning_rate=args.decoder_learning_rate or args.learning_rate,
        weight_decay=args.weight_decay,
        prompt_encoder_learning_rate=(
            args.prompt_encoder_learning_rate
            or args.decoder_learning_rate
            or args.learning_rate
        ),
    )
    optimizer = torch.optim.AdamW(optimizer_groups)
    for group in optimizer.param_groups:
        print(
            f"Optimizer group {group['name']}: params={sum(parameter.numel() for parameter in group['params']):,} "
            f"lr={group['lr']:.2e} weight_decay={group['weight_decay']:.2e}",
            flush=True,
        )
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
            "train_source": args.train_source,
            "unit": "click_record",
            "replacement": False,
            "rotate_seed_each_epoch": True,
            "max_train_samples": args.max_train_samples,
            "trajectory_mode": args.trajectory_mode,
            "shuffle_strata": ["source", "prompt_count"] if args.trajectory_mode == "shuffle" else None,
        },
        "optimization": {
            "decoder_warmup_epochs": args.decoder_warmup_epochs,
            "parameter_groups": [
                {
                    "name": group["name"],
                    "parameters": sum(parameter.numel() for parameter in group["params"]),
                    "initial_learning_rate": group["lr"],
                    "weight_decay": group["weight_decay"],
                }
                for group in optimizer.param_groups
            ],
        },
        "trainable_parameters": trainable_count,
        "base_checkpoint": str(args.base_checkpoint),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        decoder_trainable = args.train_mask_decoder and epoch > args.decoder_warmup_epochs
        set_mask_decoder_trainable(model, decoder_trainable)
        print(
            f"epoch={epoch:02d} mask_decoder={'trainable' if decoder_trainable else 'frozen'}",
            flush=True,
        )
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
            train_mask_decoder=decoder_trainable,
            train_prompt_encoder=args.train_prompt_encoder,
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
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "mask_decoder_trainable": decoder_trainable,
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
    model.eval()
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
        collect_task_metrics=args.evaluate_trajectory_controls,
    )
    trajectory_sensitivity = None
    if args.evaluate_trajectory_controls:
        trajectory_sensitivity = {"val": {}, "test": {}, "paired_task_deltas": {}}
        for split, dataset, limit, seed_offset in (
            ("val", val_dataset, args.max_val_samples, 100_000),
            ("test", test_dataset, args.max_test_samples, 200_000),
        ):
            for trajectory_mode in ("real", "zero", "shuffle", "wrong_image"):
                if split == "test" and trajectory_mode == args.trajectory_mode:
                    metrics = test_metrics
                else:
                    metrics = evaluate(
                        model,
                        dataset,
                        cache,
                        config,
                        device,
                        split=f"{split}/{trajectory_mode}",
                        max_samples=limit,
                        amp=amp,
                        trajectory_mode=trajectory_mode,
                        trajectory_seed=args.seed + seed_offset,
                        collect_task_metrics=True,
                    )
                trajectory_sensitivity[split][trajectory_mode] = metrics
            trajectory_sensitivity["paired_task_deltas"][split] = {
                f"real_minus_{trajectory_mode}": paired_task_bootstrap_delta(
                    trajectory_sensitivity[split]["real"],
                    trajectory_sensitivity[split][trajectory_mode],
                    seed=args.seed + seed_offset + control_index,
                )
                for control_index, trajectory_mode in enumerate(
                    ("zero", "shuffle", "wrong_image"), start=1
                )
            }
    effective_gates = {
        name: float(torch.tanh(gate.detach()).cpu())
        for name in ("prompt_residual_gate", "image_residual_gate")
        if (gate := getattr(model, name, None)) is not None
    }
    if (memory_module := getattr(model, "interaction_memory_attention", None)) is not None:
        effective_gates["interaction_memory_gate"] = float(
            torch.tanh(memory_module.gate.detach()).cpu()
        )
    summary = {
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_val_iou": float(best_checkpoint["best_val_iou"]),
        "test": test_metrics,
        "trajectory_sensitivity": trajectory_sensitivity,
        "effective_residual_gates": effective_gates,
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Training finished. Artifacts: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
