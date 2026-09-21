#!/usr/bin/env python3
"""Evaluate one or more CAIS checkpoints on the same prepared data split."""

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficientvit.models.cais import CAISConfig, COCOClickCAISDataset
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from tools.train_cais_coco_click import (
    ImageEmbeddingCache,
    apply_trajectory_mode,
    evaluate,
    forward_loss,
    move_sample,
    shuffled_trajectory_donors,
)


@torch.no_grad()
def evaluate_subgroups(
    model: torch.nn.Module,
    dataset: COCOClickCAISDataset,
    cache: ImageEmbeddingCache,
    config: CAISConfig,
    device: torch.device,
    amp: bool,
    trajectory_mode: str = "real",
    trajectory_seed: int = 0,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = defaultdict(list)
    model.eval()
    trajectory_donors = (
        shuffled_trajectory_donors(dataset.records, trajectory_seed) if trajectory_mode == "shuffle" else None
    )
    for index in range(len(dataset)):
        raw_sample = dataset[index]
        donor_inputs = (
            dataset.trajectory_inputs(trajectory_donors[index]) if trajectory_donors is not None else None
        )
        sample = move_sample(apply_trajectory_mode(raw_sample, trajectory_mode, donor_inputs), device)
        embedding = cache.get(raw_sample["image_path"])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _, losses = forward_loss(model, sample, embedding, config)
        iou = float(losses["iou_target"].reshape(-1).mean())
        latest_label = int(raw_sample["point_labels"][-1])
        groups["positive" if latest_label == 1 else "negative"].append(iou)
        groups[f"click_{int(raw_sample['click_index'])}"].append(iou)
    return {
        name: {"mask_iou": sum(values) / len(values), "samples": len(values)} for name, values in sorted(groups.items())
    }


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("checkpoint must use non-empty LABEL=PATH")
    return label.strip(), Path(path.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=parse_checkpoint, action="append", required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("assets/checkpoints/efficientvit_sam/distilled_model/efficientvit_sam_l0.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    labels = [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)):
        raise ValueError("Checkpoint labels must be unique")

    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    train_dataset = COCOClickCAISDataset(args.dataset_root, "train")
    normalizer = train_dataset.cognitive_normalizer
    datasets = {
        split: COCOClickCAISDataset(args.dataset_root, split, cognitive_normalizer=normalizer)
        for split in ("val", "test")
    }
    results = {}
    for label, checkpoint_path in args.checkpoint:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = CAISConfig.from_dict(checkpoint["cais_config"])
        checkpoint_arguments = checkpoint.get("arguments", {})
        is_standard_sam = bool(checkpoint_arguments.get("standard_sam_ablation", False))
        trajectory_mode = str(checkpoint_arguments.get("trajectory_mode", "real"))
        trajectory_seed = int(checkpoint_arguments.get("seed", 0))
        if is_standard_sam:
            model = create_efficientvit_sam_model(
                "efficientvit-sam-l0",
                pretrained=True,
                weight_url=str(args.base_checkpoint),
                image_size=512,
            ).to(device)
        else:
            model = create_efficientvit_sam_model(
                "efficientvit-sam-cais-l0",
                pretrained=True,
                weight_url=str(args.base_checkpoint),
                cais_config=config.to_dict(),
                image_size=512,
            ).to(device)
        model.load_state_dict(checkpoint["model"])
        cache = ImageEmbeddingCache(model, device, amp=amp, enabled=True)
        metrics = {}
        for split, dataset in datasets.items():
            metrics[split] = evaluate(
                model,
                dataset,
                cache,
                config,
                device,
                split=f"{label}/{split}",
                max_samples=None,
                amp=amp,
                trajectory_mode=trajectory_mode,
                trajectory_seed=trajectory_seed + (100_000 if split == "val" else 200_000),
            )
        subgroup_metrics = evaluate_subgroups(
            model,
            datasets["test"],
            cache,
            config,
            device,
            amp,
            trajectory_mode=trajectory_mode,
            trajectory_seed=trajectory_seed + 200_000,
        )
        results[label] = {
            "checkpoint": str(checkpoint_path),
            "training_best_epoch": int(checkpoint["epoch"]),
            "training_best_val_iou": float(checkpoint["best_val_iou"]),
            "trajectory_mode": trajectory_mode,
            "evaluation": metrics,
            "test_subgroups": subgroup_metrics,
        }
        del cache, model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "dataset": str(args.dataset_root),
        "split_samples": {split: len(dataset) for split, dataset in datasets.items()},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
