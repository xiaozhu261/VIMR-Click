#!/usr/bin/env python3
"""Run matched E0-E3 training and paired E4 process shuffling on strict replay."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficientvit.models.cais.coco_click_dataset import PROCESS_FEATURE_NAMES, load_rgb_image
from efficientvit.models.process_sam import ProcessAwareEfficientViTSam, ProcessSAMConfig
from efficientvit.models.process_sam.dataset import (
    COCOClickProcessDataset,
    PROCESS_INPUT_KEYS,
    shuffle_process,
    shuffled_process_donors,
)
from efficientvit.models.process_sam.metrics import binary_iou, boundary_iou, paired_process_delta, replay_metrics
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/cais_synthetic_click2_click3_300_20260921"))
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("assets/checkpoints/efficientvit_sam/distilled_model/efficientvit_sam_l0.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/process_sam_e0_e4"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9527)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--decoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--spatial-gate-init", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-tasks", type=int)
    parser.add_argument("--max-eval-tasks", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="One epoch, two complete five-click tasks per split; integration results only",
    )
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_tasks(dataset, limit=None, *, complete_only=False):
    groups = defaultdict(list)
    for index, record in enumerate(dataset.records):
        groups[record["task_run_id"]].append(index)
    tasks = sorted(groups)
    if complete_only:
        tasks = [
            task for task in tasks if {int(dataset.records[i]["click_index"]) for i in groups[task]} == set(range(1, 6))
        ]
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("No tasks available under the selected replay protocol")
    return [
        index for task in tasks for index in sorted(groups[task], key=lambda i: int(dataset.records[i]["click_index"]))
    ]


class ImageCache:
    def __init__(self, base_model, device, amp):
        self.model, self.device, self.amp = base_model.eval(), device, amp
        self.values = {}

    @torch.no_grad()
    def get(self, image_path):
        if image_path not in self.values:
            image = load_rgb_image(image_path)
            tensor = self.model.transform(image).unsqueeze(0).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                embedding = self.model.image_encoder(tensor)
            # Use identical cached precision for first and later callers.
            self.values[image_path] = embedding.detach().cpu()
        return self.values[image_path].to(self.device)


def model_inputs(sample, device):
    result = {
        key: sample[key].unsqueeze(0).to(device)
        for key in ("point_coords", "point_labels", "previous_mask", *PROCESS_INPUT_KEYS)
    }
    result["coordinate_scale"] = sample["trajectory_coordinate_scale"].unsqueeze(0).to(device)
    if "previous_mask_logits" in sample:
        result.pop("previous_mask")
        if sample["click_index"] > 1:
            result["previous_mask_logits"] = sample["previous_mask_logits"].unsqueeze(0).to(device)
    if sample.get("no_initial_mask", False) and sample["click_index"] == 1:
        result.pop("previous_mask", None)
    for key in ("process_events", "process_event_valid", "process_spatial_valid"):
        if key in sample:
            result[key] = sample[key].unsqueeze(0).to(device)
    return result


def segmentation_loss(output, target):
    logits = output["mask_logits"].float()
    target = target.to(logits)
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1e-6) / (probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6)).mean()
    with torch.no_grad():
        predicted, truth = probability >= 0.5, target >= 0.5
        iou_target = (predicted & truth).sum(dim=(-2, -1)).float() / (predicted | truth).sum(dim=(-2, -1)).clamp_min(1)
    return (
        F.binary_cross_entropy_with_logits(logits, target)
        + dice
        + F.mse_loss(output["iou_predictions"].float(), iou_target)
    )


def train_epoch(
    model, dataset, indices, cache, optimizer, scaler, device, amp, epoch, seed, accumulation, progress_callback=None
):
    model.train()
    order = list(indices)
    random.Random(seed + epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step, index in enumerate(order):
        sample = dataset[index]
        inputs = model_inputs(sample, device)
        embedding = cache.get(sample["image_path"])
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output = model(image_embeddings=embedding, **inputs)
            loss = segmentation_loss(output, sample["target"].unsqueeze(0).to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at sample {sample['event_id']}")
        window_start = step // accumulation * accumulation
        window_size = min(accumulation, len(order) - window_start)
        scaler.scale(loss / window_size).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(
                f"{model.process_config.variant} epoch={epoch} step={step + 1}/{len(order)} loss={np.mean(losses):.4f}",
                flush=True,
            )
            if progress_callback is not None:
                progress_callback(step + 1, len(order), float(np.mean(losses)))
    return float(np.mean(losses))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, dataset, indices, cache, device, amp, *, shuffled=False, seed=0):
    model.eval()
    donors = shuffled_process_donors(dataset.records, seed) if shuffled else None
    rows = []
    for position, index in enumerate(indices):
        sample = dataset[index]
        if donors is not None:
            sample = shuffle_process(sample, dataset.process_inputs(donors[index]))
        inputs = model_inputs(sample, device)
        embedding = cache.get(sample["image_path"])
        if position == 0:
            # Exclude lazy kernel initialization from per-click latency.
            for _ in range(3):
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    model(image_embeddings=embedding, **inputs)
        synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output = model(image_embeddings=embedding, **inputs)
        synchronize(device)
        latency = (time.perf_counter() - started) * 1000
        original_h, original_w = sample["original_size"]
        resized_w, resized_h = sample["trajectory_coordinate_scale"].int().tolist()
        full_logits = model.postprocess_masks(
            output["mask_logits"].float(), (resized_h, resized_w), (original_h, original_w)
        )
        predicted = full_logits[0, 0] > 0
        truth = torch.as_tensor(dataset._decode_ground_truth(dataset.records[index]), device=device)
        rows.append(
            {
                "event_id": sample["event_id"],
                "task_run_id": sample["task_run_id"],
                "click_index": sample["click_index"],
                "iou": binary_iou(predicted, truth),
                "boundary_iou": boundary_iou(predicted, truth),
                "latency_ms": latency,
                "donor_event_id": None if donors is None else dataset.records[donors[index]]["event_id"],
            }
        )
    return replay_metrics(rows), rows


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_progress(output_dir, status, **details):
    write_json(
        output_dir / "progress.json",
        {"status": status, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **details},
    )


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(path, summary):
    lines = [
        "# Process-aware EfficientViT-SAM: E0–E4",
        "",
        f"Result type: {summary['result_type']}",
        "",
        f"Training epochs: {summary['training']['epochs']}; seed: {summary['training']['seed']}.",
        "",
        "| Split | Recorded clicks | Tasks |",
        "| --- | ---: | ---: |",
        *[
            f"| {split} | {coverage['selected_records']} | {coverage['selected_tasks']} |"
            for split, coverage in summary["data_coverage"].items()
        ],
        "",
        "| Experiment | IoU@1 | IoU@3 | IoU@5 | NoC@90 | Boundary IoU | Latency (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["experiments"].items():
        metrics = result["test"]
        values = [metrics[key] for key in ("IoU@1", "IoU@3", "IoU@5", "NoC@90", "Boundary IoU", "latency_ms_mean")]
        lines.append(f"| {name} | " + " | ".join("N/A" if value is None else f"{value:.4f}" for value in values) + " |")
    delta = summary["E3_minus_E4"]
    lines += [
        "",
        f"Paired task-mean IoU delta E3 − E4: {delta['mean_task_iou_delta']:.6f}, bootstrap 95% CI {delta['bootstrap_95ci']}.",
        "",
        "E4 uses E3's exact weights and events; only process inputs are shuffled within source/history-length strata.",
        "All groups replay the recorded clicks and pre-click masks. NoC uses complete five-click prefixes only; failures score 6.",
        "IoU/Boundary IoU are measured at original resolution. Latency includes prompt/process encoding, fusion and decoding with cached image features; excludes IO, image encoding and mask upsampling.",
        "IoU@k over all available events has different task coverage at each k; complete-prefix IoU@k and coverage are in summary.json.",
        f"Test click coverage: {summary['experiments']['E3']['test']['click_coverage']}; complete five-click tasks: {summary['experiments']['E3']['test']['complete_prefix_tasks']}.",
    ]
    if summary["result_type"] == "integration_smoke":
        lines += [
            "",
            "This smoke run checks execution and protocol integrity; it does not establish segmentation improvements.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args=None):
    args = build_parser().parse_args() if args is None else args
    if args.smoke_test:
        args.epochs, args.max_train_tasks, args.max_eval_tasks = 1, 2, 2
    if args.epochs < 1 or args.accumulation_steps < 1:
        raise ValueError("epochs and accumulation_steps must be positive")
    if any(limit is not None and limit < 1 for limit in (args.max_train_tasks, args.max_eval_tasks)):
        raise ValueError("task limits must be positive")
    if args.learning_rate <= 0 or args.decoder_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if (args.output_dir / "run_manifest.json").exists():
        raise FileExistsError("Use a fresh output directory to preserve existing experiment results")
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp = device.type == "cuda" and not args.no_amp
    datasets = {
        split: COCOClickProcessDataset(args.dataset_root, split, source_filter="human")
        for split in ("train", "val", "test")
    }
    selected = {
        split: select_tasks(
            dataset, args.max_train_tasks if split == "train" else args.max_eval_tasks, complete_only=args.smoke_test
        )
        for split, dataset in datasets.items()
    }
    # Every stratum must be shufflable, even when a limited subset is evaluated.
    shuffled_process_donors(datasets["test"].records, args.seed)
    # Normalization is fitted on the actual selected training cohort only.
    features = [datasets["train"].process_inputs(i) for i in selected["train"]]
    matrix = torch.stack([item["process_features"][-1] for item in features])
    matrix = matrix[torch.tensor([bool(item["process_feature_valid"][-1]) for item in features])]
    means, stds = (0.0,) * len(PROCESS_FEATURE_NAMES), (1.0,) * len(PROCESS_FEATURE_NAMES)
    if matrix.shape[0]:
        means = tuple(matrix.mean(dim=0).tolist())
        stds = tuple(matrix.std(dim=0, unbiased=False).clamp_min(1e-6).tolist())
    config_common = dict(process_mean=means, process_std=stds, spatial_gate_init=args.spatial_gate_init)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "clicks": "fixed recorded replay, max 5",
        "previous_mask": "recorded pre_mask, identical for all groups",
        "source": "human only in train/val/test",
        "encoders": "original Image/Prompt encoders frozen",
        "decoder": "same original Mask Decoder, trainable in E0-E3 with identical lr/loss",
        "loss": "BCEWithLogits + Dice + IoU MSE (weight 1 each)",
        "E4": "paired test-time shuffling of E3, same checkpoint; recipient clicks/types/masks fixed",
        "NoC@90": "complete five-click recorded prefixes only; failure 6; not a closed-loop click simulator",
        "Boundary IoU": "original resolution; inner boundary with distance=round(0.02*image_diagonal)",
        "Boundary IoU reference": "https://github.com/bowenc0221/boundary-iou-api",
        "spatial_alignment": "original normalized viewport/path -> resized unpadded image -> SAM padded canvas",
        "normalization": "selected train-only observed process summaries",
    }
    manifest = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "protocol": protocol,
        "base_checkpoint_sha256": file_hash(args.base_checkpoint),
        "dataset_sha256": file_hash(datasets["train"].jsonl_path),
        "process_feature_names": PROCESS_FEATURE_NAMES,
        "selected_event_ids": {
            split: [datasets[split].records[i]["event_id"] for i in indices] for split, indices in selected.items()
        },
        "process_normalization": {"mean": means, "std": stds},
        "data_coverage": {
            split: {
                "selected_records": len(selected[split]),
                "selected_tasks": len({datasets[split].records[i]["task_run_id"] for i in selected[split]}),
                "observed_viewport_records": sum(
                    bool(datasets[split].process_inputs(i)["viewport_valid"][-1]) for i in selected[split]
                ),
            }
            for split in datasets
        },
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    write_progress(args.output_dir, "running", phase="initializing", total_epochs=args.epochs)
    summary = {
        "result_type": "integration_smoke" if args.smoke_test else "matched_replay_experiment",
        "training": {"epochs": args.epochs, "seed": args.seed},
        "data_coverage": manifest["data_coverage"],
        "protocol": protocol,
        "experiments": {},
    }
    cache = None
    for variant in ("E0", "E1", "E2", "E3"):
        set_seed(args.seed)
        base = (
            create_efficientvit_sam_model("efficientvit-sam-l0", weight_url=str(args.base_checkpoint)).to(device).eval()
        )
        model = ProcessAwareEfficientViTSam(base, ProcessSAMConfig(variant=variant, **config_common)).to(device)
        if cache is None:
            cache = ImageCache(model, device, amp)
        else:
            cache.model = model
        folder = args.output_dir / variant
        folder.mkdir()
        params = model.parameter_summary()
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p for name, p in model.named_parameters() if name.startswith("process_") and p.requires_grad
                    ],
                    "lr": args.learning_rate,
                },
                {"params": model.mask_decoder.parameters(), "lr": args.decoder_learning_rate},
            ],
            weight_decay=1e-4,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        best, history = -float("inf"), []
        print(f"{variant} parameters={params} training_samples={len(selected['train'])}", flush=True)
        for epoch in range(1, args.epochs + 1):
            write_progress(
                args.output_dir,
                "running",
                phase="training",
                experiment=variant,
                epoch=epoch,
                total_epochs=args.epochs,
                completed_experiments=list(summary["experiments"]),
            )
            loss = train_epoch(
                model,
                datasets["train"],
                selected["train"],
                cache,
                optimizer,
                scaler,
                device,
                amp,
                epoch,
                args.seed,
                args.accumulation_steps,
            )
            validation, _ = evaluate(model, datasets["val"], selected["val"], cache, device, amp)
            history.append({"epoch": epoch, "train_loss": loss, "val": validation})
            write_json(folder / "history.json", history)
            if validation["mean_iou"] > best:
                best = validation["mean_iou"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": model.process_config.to_dict(),
                        "epoch": epoch,
                        "val_iou": best,
                    },
                    folder / "best.pt",
                )
            print(f"{variant} epoch={epoch} loss={loss:.4f} val_iou={validation['mean_iou']:.4f}", flush=True)
            write_progress(
                args.output_dir,
                "running",
                phase="epoch_completed",
                experiment=variant,
                epoch=epoch,
                total_epochs=args.epochs,
                train_loss=loss,
                val_iou=validation["mean_iou"],
                best_val_iou=best,
                completed_experiments=list(summary["experiments"]),
            )
        write_progress(
            args.output_dir,
            "running",
            phase="testing",
            experiment=variant,
            epoch=args.epochs,
            total_epochs=args.epochs,
            completed_experiments=list(summary["experiments"]),
        )
        checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test, rows = evaluate(model, datasets["test"], selected["test"], cache, device, amp)
        write_json(folder / "test_events.json", rows)
        write_json(folder / "history.json", history)
        summary["experiments"][variant] = {"parameters": params, "best_epoch": checkpoint["epoch"], "test": test}
        write_json(args.output_dir / "partial_summary.json", summary)
        print(f"{variant} test={json.dumps(test)}", flush=True)
        if variant == "E3":
            write_progress(
                args.output_dir,
                "running",
                phase="testing",
                experiment="E4",
                completed_experiments=list(summary["experiments"]),
            )
            shuffled_metrics, shuffled_rows = evaluate(
                model, datasets["test"], selected["test"], cache, device, amp, shuffled=True, seed=args.seed
            )
            e4_folder = args.output_dir / "E4"
            e4_folder.mkdir()
            write_json(e4_folder / "test_events.json", shuffled_rows)
            summary["experiments"]["E4"] = {
                "checkpoint": "../E3/best.pt",
                "parameters": params,
                "test": shuffled_metrics,
            }
            summary["E3_minus_E4"] = paired_process_delta(rows, shuffled_rows, args.seed)
        del checkpoint, optimizer, scaler, model, base
    write_json(args.output_dir / "summary.json", summary)
    write_report(args.output_dir / "report.md", summary)
    write_progress(
        args.output_dir,
        "complete",
        completed_experiments=list(summary["experiments"]),
        total_epochs=args.epochs,
        report="report.md",
        summary="summary.json",
    )
    print(f"Saved five-group results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    existing_run = (arguments.output_dir / "run_manifest.json").exists()
    try:
        main(arguments)
    except Exception as error:
        if not existing_run and arguments.output_dir.is_dir():
            write_progress(
                arguments.output_dir,
                "failed",
                error_type=type(error).__name__,
                error=str(error),
            )
        raise
