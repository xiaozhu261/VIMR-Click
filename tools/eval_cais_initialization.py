#!/usr/bin/env python3
"""Measure how CAIS integration changes pretrained SAM masks before training."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficientvit.models.cais import CAISConfig, COCOClickCAISDataset
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from tools.train_cais_coco_click import ImageEmbeddingCache, evaluate, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("assets/checkpoints/efficientvit_sam/distilled_model/efficientvit_sam_l0.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9527)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    train_dataset = COCOClickCAISDataset(args.dataset_root, "train")
    datasets = {
        split: COCOClickCAISDataset(
            args.dataset_root,
            split,
            cognitive_normalizer=train_dataset.cognitive_normalizer,
        )
        for split in ("val", "test")
    }
    variants = {
        "standard_sam": None,
        "replacement_cais": CAISConfig(
            freeze_image_encoder=True,
            freeze_prompt_encoder=True,
            boundary_alpha=0.0,
            lambda_cognitive=0.0,
            prompt_integration="replace",
            attention_integration="replace",
        ),
        "zero_gated_native_cais": CAISConfig(
            freeze_image_encoder=True,
            freeze_prompt_encoder=True,
            boundary_alpha=0.0,
            lambda_cognitive=0.0,
            prompt_integration="native_residual",
            attention_integration="gated_residual",
        ),
    }
    results = {}
    for label, config in variants.items():
        set_seed(args.seed)
        if config is None:
            model = create_efficientvit_sam_model(
                "efficientvit-sam-l0",
                pretrained=True,
                weight_url=str(args.base_checkpoint),
                image_size=512,
            ).to(device)
            resolved_config = CAISConfig(boundary_alpha=0.0, lambda_cognitive=0.0)
        else:
            model = create_efficientvit_sam_model(
                "efficientvit-sam-cais-l0",
                pretrained=True,
                weight_url=str(args.base_checkpoint),
                cais_config=config.to_dict(),
                image_size=512,
            ).to(device)
            resolved_config = config
        cache = ImageEmbeddingCache(model, device, amp=amp, enabled=True)
        results[label] = {
            split: evaluate(
                model,
                dataset,
                cache,
                resolved_config,
                device,
                split=f"{label}/{split}",
                max_samples=None,
                amp=amp,
            )
            for split, dataset in datasets.items()
        }
        if config is not None:
            results[label]["gates"] = {
                "prompt": float(torch.tanh(model.prompt_residual_gate).detach().cpu())
                if hasattr(model, "prompt_residual_gate")
                else None,
                "image": float(torch.tanh(model.image_residual_gate).detach().cpu())
                if hasattr(model, "image_residual_gate")
                else None,
            }
        del cache, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "dataset": str(args.dataset_root),
        "seed": args.seed,
        "training_steps": 0,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
