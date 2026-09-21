#!/usr/bin/env python3
"""Evaluate frozen EfficientViT-SAM-L0 on the exact CAIS data protocol."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficientvit.models.cais import COCOClickCAISDataset, hard_iou_score, load_rgb_image
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("assets/checkpoints/efficientvit_sam/distilled_model/efficientvit_sam_l0.pt"),
    )
    parser.add_argument(
        "--cais-summary",
        type=Path,
        default=Path("outputs/cais_coco_click_20260919/training_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cais_coco_click_20260919/baseline_comparison.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser


class EmbeddingCache:
    def __init__(self, model: torch.nn.Module, device: torch.device, amp: bool) -> None:
        self.model = model
        self.device = device
        self.amp = amp
        self.values: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def get(self, image_path: str) -> torch.Tensor:
        if image_path not in self.values:
            image = load_rgb_image(image_path)
            tensor = self.model.transform(image).unsqueeze(0).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                embedding = self.model.image_encoder(tensor)
            self.values[image_path] = embedding.detach().to("cpu", dtype=torch.float16)
        dtype = torch.float16 if self.amp else torch.float32
        return self.values[image_path].to(self.device, dtype=dtype)


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    dataset: COCOClickCAISDataset,
    cache: EmbeddingCache,
    device: torch.device,
    amp: bool,
) -> dict[str, float | int]:
    iou_sum = 0.0
    quality_mae_sum = 0.0
    for index in range(len(dataset)):
        sample = dataset[index]
        embedding = cache.get(sample["image_path"])
        point_coords = sample["point_coords"].unsqueeze(0).to(device)
        point_labels = sample["point_labels"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0).to(device)
        previous = sample["previous_mask"].unsqueeze(0).to(device=device, dtype=embedding.dtype)
        epsilon = torch.finfo(previous.dtype).eps
        mask_prompt = torch.logit(previous.clamp(epsilon, 1.0 - epsilon))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            sparse, dense = model.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=mask_prompt,
            )
            logits, quality = model.mask_decoder(
                image_embeddings=embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            probability = logits.sigmoid()
        actual_iou = hard_iou_score(probability.float(), target.float()).reshape(-1)
        iou_sum += float(actual_iou.mean())
        quality_mae_sum += float((quality.float().reshape(-1) - actual_iou).abs().mean())
    return {
        "mask_iou": iou_sum / len(dataset),
        "iou_prediction_mae": quality_mae_sum / len(dataset),
        "samples": len(dataset),
        "tasks": dataset.describe()["tasks"],
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    model = create_efficientvit_sam_model(
        "efficientvit-sam-l0",
        pretrained=True,
        weight_url=str(args.checkpoint),
        image_size=512,
    ).to(device)
    model.eval()
    cache = EmbeddingCache(model, device, amp)
    datasets = {split: COCOClickCAISDataset(args.dataset_root, split) for split in ("val", "test")}
    baseline = {split: evaluate_split(model, dataset, cache, device, amp) for split, dataset in datasets.items()}
    cais_summary = json.loads(args.cais_summary.read_text(encoding="utf-8"))
    cais = {
        "val": {
            "mask_iou": float(cais_summary["best_val_iou"]),
            "samples": len(datasets["val"]),
            "tasks": datasets["val"].describe()["tasks"],
        },
        "test": {
            "mask_iou": float(cais_summary["test"]["mask_iou"]),
            "iou_prediction_mae": float(cais_summary["test"]["iou_mae"]),
            "cognitive_mae": float(cais_summary["test"]["cognitive_mae"]),
            "samples": int(cais_summary["test"]["samples"]),
            "tasks": datasets["test"].describe()["tasks"],
        },
    }
    payload = {
        "protocol": {
            "dataset": str(args.dataset_root),
            "records": "alignment/modeling_clicks_strict.jsonl",
            "prompt_state": "all recorded click history plus previous-mask prompt",
            "multimask_output": False,
            "mask_resolution": 256,
            "threshold": 0.5,
        },
        "baseline": baseline,
        "cais": cais,
        "delta_cais_minus_baseline": {
            split: {
                "mask_iou_absolute": cais[split]["mask_iou"] - baseline[split]["mask_iou"],
                "mask_iou_relative_percent": 100.0
                * (cais[split]["mask_iou"] - baseline[split]["mask_iou"])
                / baseline[split]["mask_iou"],
            }
            for split in ("val", "test")
        },
        "parameters": {
            "baseline_total": sum(parameter.numel() for parameter in model.parameters()),
            "cais_total": 38_777_426,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
