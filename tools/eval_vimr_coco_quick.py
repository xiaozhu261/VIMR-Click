"""Quick, reproducible COCO train2017 holdout evaluation for VIMR-Click.

This is an MVP sanity experiment, not an official COCO val benchmark. It trains
only newly added VIMR modules and compares them with the frozen pretrained SAM
path under independently simulated largest-error corrective clicks.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from efficientvit.models.vimr import VIMRConfig, VIMREfficientViTSam, vimr_interactive_loss
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=ROOT_DIR / "dataset/coco")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt",
    )
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "outputs/vimr_coco_quick")
    parser.add_argument("--train-instances", type=int, default=64)
    parser.add_argument("--eval-instances", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-clicks", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-area-ratio", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_split(
    coco_root: Path,
    train_count: int,
    eval_count: int,
    seed: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotation_path = coco_root / "annotations_trainval2017/annotations/instances_train2017.json"
    image_root = coco_root / "train2017/train2017"
    with annotation_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {record["id"]: record for record in coco["images"]}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        image = images[annotation["image_id"]]
        area_ratio = float(annotation["area"]) / float(image["width"] * image["height"])
        if min_area_ratio <= area_ratio <= max_area_ratio and annotation.get("segmentation"):
            annotations[annotation["image_id"]].append(annotation)

    image_ids = sorted(annotations)
    random.Random(seed).shuffle(image_ids)
    selected = []
    for image_id in image_ids:
        image = images[image_id]
        image_path = image_root / image["file_name"]
        if not image_path.is_file():
            continue
        # One deterministic, well-sized object per image avoids correlated instances.
        annotation = max(annotations[image_id], key=lambda item: item["area"])
        selected.append({"image": image, "annotation": annotation, "path": str(image_path)})
        if len(selected) == train_count + eval_count:
            break
    if len(selected) < train_count + eval_count:
        raise RuntimeError(f"Only found {len(selected)} usable image-instance pairs")
    return selected[:train_count], selected[train_count : train_count + eval_count]


def decode_mask(sample: dict[str, Any]) -> np.ndarray:
    image = sample["image"]
    segmentation = sample["annotation"]["segmentation"]
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, image["height"], image["width"])
        encoded = mask_utils.merge(rles)
    else:
        encoded = segmentation
    return mask_utils.decode(encoded).astype(np.float32)


def target_to_sam(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    height, width = mask.shape
    scale = 256.0 / max(height, width)
    new_height = int(height * scale + 0.5)
    new_width = int(width * scale + 0.5)
    tensor = torch.from_numpy(mask).to(device=device).reshape(1, 1, height, width)
    tensor = F.interpolate(tensor, (new_height, new_width), mode="nearest")
    return F.pad(tensor, (0, 256 - new_width, 0, 256 - new_height))


def encode_image(model: VIMREfficientViTSam, image_path: str, device: torch.device) -> torch.Tensor:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    image_tensor = model.transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.image_encoder(image_tensor)


def initial_click(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    foreground = target[0, 0].detach().cpu().numpy().astype(np.uint8)
    distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    coords = target.new_tensor([[[x * 4.0 + 2.0, y * 4.0 + 2.0]]])
    return coords, torch.ones(1, 1, device=target.device, dtype=torch.long)


def corrective_click(probability: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    error = (target - probability).abs().flatten(1)
    index = error.argmax(dim=1)
    y = torch.div(index, target.shape[-1], rounding_mode="floor")
    x = index % target.shape[-1]
    coords = torch.stack((x.float() * 4.0 + 2.0, y.float() * 4.0 + 2.0), dim=-1).unsqueeze(1)
    labels = target.flatten(1).gather(1, index.unsqueeze(1)).long()
    return coords, labels


def viewport_from_click(
    coords: torch.Tensor, round_index: int, previous_viewport: torch.Tensor | None = None
) -> torch.Tensor:
    viewport = coords.new_zeros(1, 7)
    viewport[:, :2] = coords[:, 0] / 1024.0
    if round_index == 0:
        viewport[:, :2] = 0.5
        return viewport
    viewport[:, 2] = math.log(2.0)
    if previous_viewport is not None:
        viewport[:, 3:5] = viewport[:, :2] - previous_viewport[:, :2]
        viewport[:, 5] = viewport[:, 2] - previous_viewport[:, 2]
    viewport[:, 6] = 1 if round_index == 1 else 3
    return viewport


def viewport_for_mode(
    coords: torch.Tensor,
    round_index: int,
    previous_viewport: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    """Return either the synthetic click-centred viewport or an information-free global control."""
    if mode == "synthetic":
        return viewport_from_click(coords, round_index, previous_viewport)
    if mode == "default":
        viewport = coords.new_zeros(coords.shape[0], 7)
        viewport[:, :2] = 0.5
        return viewport
    raise ValueError(f"Unsupported viewport mode: {mode}")


def probability_to_prompt(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    return torch.logit(mask.clamp(1e-6, 1 - 1e-6))


def baseline_round(
    model: VIMREfficientViTSam,
    embedding: torch.Tensor,
    coords: torch.Tensor,
    labels: torch.Tensor,
    previous: torch.Tensor | None,
) -> torch.Tensor:
    sparse, dense = model.prompt_encoder(
        points=(coords, labels), boxes=None, masks=probability_to_prompt(previous)
    )
    logits, _ = model.mask_decoder(
        image_embeddings=embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return logits.sigmoid()


def run_vimr_sequence(
    model: VIMREfficientViTSam,
    embedding: torch.Tensor,
    target: torch.Tensor,
    max_clicks: int,
    train: bool,
    viewport_mode: str = "synthetic",
) -> tuple[list[torch.Tensor], list[float]]:
    coords, labels = initial_click(target)
    click_events: list[torch.Tensor] = []
    viewports: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    ious: list[float] = []
    previous = None
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
        output = model(
            image_embeddings=embedding,
            point_coords=coords,
            point_labels=labels,
            previous_mask=previous,
            viewport_state=viewport,
            click_history=torch.stack(click_events, dim=1),
            viewport_history=torch.stack(viewports, dim=1).squeeze(2),
            mask_history=torch.stack(masks, dim=1) if masks else None,
            multimask_output=False,
        )
        probability = output["final_mask"]
        if train:
            losses = vimr_interactive_loss(
                probability,
                target,
                previous,
                lambda_safe=model.vimr_config.lambda_safe,
                lambda_monotonic=model.vimr_config.lambda_monotonic,
            )
            (losses["loss"] / max_clicks).backward()
        intersection = ((probability >= 0.5) & (target >= 0.5)).sum().float()
        union = ((probability >= 0.5) | (target >= 0.5)).sum().float()
        ious.append(float((intersection / union.clamp_min(1)).detach().cpu()))
        masks.append(probability.detach())
        previous = probability.detach()
        if round_index + 1 < max_clicks:
            coords, labels = corrective_click(previous, target)
    return masks, ious


def run_baseline_sequence(
    model: VIMREfficientViTSam,
    embedding: torch.Tensor,
    target: torch.Tensor,
    max_clicks: int,
) -> list[float]:
    current_coords, current_labels = initial_click(target)
    all_coords: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    previous = None
    ious = []
    for round_index in range(max_clicks):
        all_coords.append(current_coords)
        all_labels.append(current_labels)
        probability = baseline_round(
            model,
            embedding,
            torch.cat(all_coords, dim=1),
            torch.cat(all_labels, dim=1),
            previous,
        )
        intersection = ((probability >= 0.5) & (target >= 0.5)).sum().float()
        union = ((probability >= 0.5) | (target >= 0.5)).sum().float()
        ious.append(float((intersection / union.clamp_min(1)).cpu()))
        previous = probability
        if round_index + 1 < max_clicks:
            current_coords, current_labels = corrective_click(previous, target)
    return ious


def freeze_pretrained_modules(model: VIMREfficientViTSam) -> list[torch.nn.Parameter]:
    for module in (model.image_encoder, model.prompt_encoder, model.mask_decoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    config = model.vimr_config
    active_prefixes = []
    if config.use_viewport:
        active_prefixes.append("viewport_encoder.")
    if config.use_history:
        active_prefixes.append("history_memory.")
    if config.use_residual:
        active_prefixes.append("residual_analyzer.")
    if config.use_viewport or config.use_history or config.use_residual:
        active_prefixes.append("interaction_fusion.")
    if config.enable_expert_router:
        active_prefixes.append("expert_router.")
    if config.use_safe_update and config.safe_update:
        active_prefixes.append("safe_update.")
    if config.enable_detail_refinement:
        active_prefixes.append("detail_refinement.")
    active_prefixes = tuple(active_prefixes)
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(active_prefixes)
        if parameter.requires_grad:
            trainable.append(parameter)
    return trainable


def train_vimr(
    model: VIMREfficientViTSam,
    samples: list[dict[str, Any]],
    device: torch.device,
    epochs: int,
    max_clicks: int,
    learning_rate: float,
) -> list[float]:
    optimizer = torch.optim.AdamW(freeze_pretrained_modules(model), lr=learning_rate, weight_decay=1e-4)
    losses = []
    for epoch in range(epochs):
        order = list(range(len(samples)))
        random.Random(1000 + epoch).shuffle(order)
        for step, sample_index in enumerate(order, start=1):
            sample = samples[sample_index]
            target = target_to_sam(decode_mask(sample), device)
            embedding = encode_image(model, sample["path"], device)
            optimizer.zero_grad(set_to_none=True)
            _, ious = run_vimr_sequence(model, embedding, target, max_clicks, train=True)
            grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
            optimizer.step()
            losses.append(float(grad_norm.detach().cpu()))
            if step == 1 or step % 10 == 0 or step == len(order):
                print(
                    f"train epoch={epoch + 1}/{epochs} step={step}/{len(order)} "
                    f"last_iou={ious[-1]:.4f} grad_norm={losses[-1]:.4f}",
                    flush=True,
                )
    return losses


def summarize(rows: list[dict[str, Any]], method: str, max_clicks: int) -> dict[str, float]:
    sequences = np.asarray([row[f"{method}_ious"] for row in rows], dtype=np.float64)
    summary = {f"mIoU@{index + 1}": float(sequences[:, index].mean()) for index in range(max_clicks)}
    for threshold in (0.85, 0.90):
        reached = sequences >= threshold
        first = np.where(reached.any(axis=1), reached.argmax(axis=1) + 1, max_clicks + 1)
        summary[f"NoC@{int(threshold * 100)}"] = float(first.mean())
        summary[f"Success@{int(threshold * 100)}/{max_clicks}"] = float(reached.any(axis=1).mean())
    regressions = sequences[:, 1:] < sequences[:, :-1] - 1e-6
    summary["RegressionRate"] = float(regressions.mean())
    return summary


def evaluate(
    model: VIMREfficientViTSam,
    samples: list[dict[str, Any]],
    device: torch.device,
    max_clicks: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            target = target_to_sam(decode_mask(sample), device)
            embedding = encode_image(model, sample["path"], device)
            baseline_ious = run_baseline_sequence(model, embedding, target, max_clicks)
            _, vimr_ious = run_vimr_sequence(model, embedding, target, max_clicks, train=False)
            rows.append(
                {
                    "image_id": sample["image"]["id"],
                    "annotation_id": sample["annotation"]["id"],
                    "category_id": sample["annotation"]["category_id"],
                    "baseline_ious": baseline_ious,
                    "vimr_ious": vimr_ious,
                }
            )
            print(
                f"eval {index}/{len(samples)} baseline={baseline_ious[-1]:.4f} vimr={vimr_ious[-1]:.4f}",
                flush=True,
            )
    return rows, {
        "baseline": summarize(rows, "baseline", max_clicks),
        "vimr": summarize(rows, "vimr", max_clicks),
    }


def save_results(
    output_dir: Path,
    model: VIMREfficientViTSam,
    rows: list[dict[str, Any]],
    summary: dict[str, dict[str, float]],
    protocol: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"protocol": protocol, "metrics": summary}, handle, indent=2)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "baseline", "vimr", "delta_vimr_minus_baseline"))
        for metric in summary["baseline"]:
            baseline = summary["baseline"][metric]
            vimr = summary["vimr"][metric]
            writer.writerow((metric, baseline, vimr, vimr - baseline))
    with (output_dir / "per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_id", "annotation_id", "category_id", "method", "click", "iou"))
        for row in rows:
            for method in ("baseline", "vimr"):
                for click, iou in enumerate(row[f"{method}_ious"], start=1):
                    writer.writerow(
                        (row["image_id"], row["annotation_id"], row["category_id"], method, click, iou)
                    )
    vimr_state = {
        key: value.cpu()
        for key, value in model.state_dict().items()
        if key.startswith((
            "viewport_encoder.",
            "history_memory.",
            "residual_analyzer.",
            "interaction_fusion.",
            "expert_router.",
            "safe_update.",
            "detail_refinement.",
        ))
    }
    torch.save({"vimr_state_dict": vimr_state, "protocol": protocol}, output_dir / "vimr_modules.pt")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_samples, eval_samples = load_split(
        args.coco_root,
        args.train_instances,
        args.eval_instances,
        args.seed,
        args.min_area_ratio,
        args.max_area_ratio,
    )
    config = VIMRConfig(freeze_image_encoder=True, freeze_prompt_encoder=True)
    model = create_efficientvit_sam_model(
        "efficientvit-sam-vimr-l0",
        pretrained=True,
        weight_url=str(args.checkpoint),
        vimr_config=config.to_dict(),
    ).to(device)
    start = time.time()
    grad_norms = train_vimr(
        model,
        train_samples,
        device,
        args.epochs,
        args.max_clicks,
        args.learning_rate,
    )
    rows, summary = evaluate(model, eval_samples, device, args.max_clicks)
    protocol = {
        "dataset": "COCO 2017 train split, deterministic image-disjoint holdout",
        "train_instances": args.train_instances,
        "eval_instances": args.eval_instances,
        "epochs": args.epochs,
        "max_clicks": args.max_clicks,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "elapsed_seconds": time.time() - start,
        "mean_gradient_norm": float(np.mean(grad_norms)),
        "base_modules_frozen": True,
    }
    save_results(args.output, model, rows, summary, protocol)
    print(json.dumps({"protocol": protocol, "metrics": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
