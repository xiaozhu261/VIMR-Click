#!/usr/bin/env python3
"""Generate full synthetic processes and train E3 once; all other runs are inference."""

import argparse
from collections import Counter, OrderedDict, defaultdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from efficientvit.models.cais.coco_click_dataset import PROCESS_FEATURE_NAMES, load_rgb_image
from efficientvit.models.process_sam import ProcessAwareEfficientViTSam, ProcessSAMConfig
from efficientvit.models.process_sam.dataset import SyntheticProcessDataset, PROCESS_INPUT_KEYS, shuffled_process_donors
from efficientvit.models.process_sam.metrics import binary_iou, boundary_iou, replay_metrics, paired_process_delta
from efficientvit.models.process_sam.simulation import SimulatedUser, stream_coco_annotations
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from tools.generate_cais_synthetic_clicks import decode_annotation, encode_mask, save_mask
from tools.run_process_sam_experiments import (
    set_seed,
    train_epoch,
    evaluate,
    file_hash,
    write_json,
    write_progress,
    synchronize,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--instances", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=9527)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ROOT / "assets/checkpoints/efficientvit_sam/distilled_model/efficientvit_sam_l0.pt",
    )
    p.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "dataset/coco/annotations_trainval2017/annotations/instances_train2017.json",
    )
    p.add_argument("--image-root", type=Path, default=ROOT / "dataset/coco/train2017/train2017")
    p.add_argument(
        "--human-jsonl",
        type=Path,
        default=ROOT / "outputs/cais_synthetic_click2_click3_300_20260921/alignment/modeling_clicks_strict.jsonl",
    )
    return p


class DiskImageCache:
    def __init__(self, model, folder, device, amp):
        self.model, self.folder, self.device, self.amp = model, folder, device, amp
        self.values = OrderedDict()
        folder.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def get(self, image_path):
        key = Path(image_path).stem
        if key in self.values:
            value = self.values.pop(key)
        else:
            path = self.folder / (key + ".pt")
            if path.exists():
                value = torch.load(path, map_location="cpu", weights_only=True)
            else:
                image = load_rgb_image(str(image_path))
                data = self.model.transform(image).unsqueeze(0).to(self.device)
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                    value = self.model.image_encoder(data).detach().cpu()
                torch.save(value, path)
        self.values[key] = value
        while len(self.values) > 64:
            self.values.popitem(last=False)
        return value.to(self.device)


def select_targets(args):
    raw = [json.loads(l) for l in args.human_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    excluded = {r["coco_image_id"] for r in raw if r.get("source", "human") == "human"}
    names = {r["category_id"]: r["category_name"] for r in raw}
    available = {int(p.stem) for p in args.image_root.glob("*.jpg")}
    generator = random.Random(args.seed)
    pools = defaultdict(list)
    seen = Counter()
    for annotation in stream_coco_annotations(args.annotations):
        image_id = annotation["image_id"]
        if image_id not in available or image_id in excluded or annotation.get("iscrowd") or annotation["area"] < 1024:
            continue
        category = annotation["category_id"]
        seen[category] += 1
        pool = pools[category]
        if len(pool) < 64:
            pool.append(annotation)
        else:
            index = generator.randrange(seen[category])
            if index < 64:
                pool[index] = annotation
    for pool in pools.values():
        generator.shuffle(pool)
    selected, selected_images = [], set()
    while len(selected) < args.instances:
        changed = False
        for category in sorted(pools):
            pool = pools[category]
            while pool and pool[-1]["image_id"] in selected_images:
                pool.pop()
            if not pool:
                continue
            annotation = pool.pop()
            image_id = annotation["image_id"]
            path = args.image_root / f"{image_id:012d}.jpg"
            with Image.open(path) as image:
                width, height = image.size
            gt = decode_annotation(annotation, height, width)
            if not gt.any():
                continue
            selected.append(
                {
                    "image_ref": str(path.resolve()),
                    "coco_image_id": image_id,
                    "annotation_id": annotation["id"],
                    "category_id": category,
                    "category_name": names.get(category, str(category)),
                    "image_width_px": width,
                    "image_height_px": height,
                    "ground_truth_rle": encode_mask(gt),
                }
            )
            selected_images.add(image_id)
            changed = True
            if len(selected) == args.instances:
                break
        if not changed:
            raise ValueError("Not enough unique-image targets")
    indices = list(range(len(selected)))
    generator.shuffle(indices)
    train_n = round(0.7 * len(selected))
    val_n = round(0.15 * len(selected))
    for rank, index in enumerate(indices):
        selected[index]["split"] = "train" if rank < train_n else "val" if rank < train_n + val_n else "test"
        selected[index]["task_run_id"] = f"sim-process-{args.seed}-{selected[index]['annotation_id']}"
        # Each image uses one reproducible user; seeds differ across images.
        selected[index]["user_seed"] = int(args.seed + selected[index]["annotation_id"] * 17)
    return selected


def live_inputs(user, reference, device, previous_logits):
    h, w = reference["image_height_px"], reference["image_width_px"]
    scale = np.array([int(w * 1024 / max(h, w) + 0.5), int(h * 1024 / max(h, w) + 0.5)])
    prompts = user.prompts
    inputs = {key: value.unsqueeze(0).to(device) for key, value in user.process_inputs().items()}
    inputs.update(
        point_coords=torch.tensor(
            [[[p["x_norm"] * scale[0], p["y_norm"] * scale[1]] for p in prompts]], dtype=torch.float32, device=device
        ),
        point_labels=torch.tensor([[p["click_type"] for p in prompts]], device=device),
        coordinate_scale=torch.tensor([scale.tolist()], dtype=torch.float32, device=device),
    )
    if previous_logits is not None:
        inputs["previous_mask_logits"] = previous_logits
    return inputs, (int(scale[1]), int(scale[0]))


@torch.no_grad()
def infer(model, embedding, inputs, amp):
    synchronize(embedding.device)
    started = time.perf_counter()
    with torch.autocast(embedding.device.type, dtype=torch.float16, enabled=amp):
        output = model(image_embeddings=embedding, **inputs)
    synchronize(embedding.device)
    return output, (time.perf_counter() - started) * 1000


def decode(model, output, reference, resized):
    return (
        model.postprocess_masks(
            output["mask_logits"].float(), resized, (reference["image_height_px"], reference["image_width_px"])
        )[0, 0]
        > 0
    )


def result_row(record, prediction, target, latency):
    return {
        "event_id": record["event_id"],
        "task_run_id": record["task_run_id"],
        "coco_image_id": record["coco_image_id"],
        "click_index": record["click_index"],
        "iou": binary_iou(prediction, target),
        "boundary_iou": boundary_iou(prediction, target),
        "latency_ms": latency,
    }


def generate(args, targets, model, cache, device, amp):
    data_root = args.output_dir / "data"
    alignment = data_root / "alignment"
    alignment.mkdir(parents=True)
    baseline = []
    manifest_path = alignment / "modeling_clicks_strict.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for position, reference in enumerate(targets, 1):
            task = reference["task_run_id"]
            gt = mask_utils.decode(reference["ground_truth_rle"]).astype(bool)
            truth = torch.tensor(gt, device=device)
            user = SimulatedUser(reference["user_seed"], *gt.shape)
            folder = data_root / "masks" / task
            folder.mkdir(parents=True)
            previous = np.zeros_like(gt)
            previous_logits = None
            save_mask(folder / "step_0.png", previous)
            np.save(folder / "step_0_logits.npy", np.zeros((1, 256, 256), dtype=np.float32))
            embedding = cache.get(reference["image_ref"])
            for click in range(1, 6):
                record = user.interact(gt, previous, task_run_id=task, event_id=f"{task}:click{click}")
                record.update(reference)
                inputs, resized = live_inputs(user, reference, device, previous_logits)
                output, latency = infer(model, embedding, inputs, amp)
                predicted = decode(model, output, reference, resized)
                row = result_row(record, predicted, truth, latency)
                row["split"] = reference["split"]
                baseline.append(row)
                save_mask(folder / f"step_{click}.png", predicted.cpu().numpy())
                np.save(folder / f"step_{click}_logits.npy", output["mask_logits"][0].float().cpu().numpy())
                record.update(
                    pre_mask_ref=str((folder / f"step_{click-1}.png").relative_to(data_root)),
                    post_mask_ref=str((folder / f"step_{click}.png").relative_to(data_root)),
                    pre_logits_ref=str((folder / f"step_{click-1}_logits.npy").relative_to(data_root)),
                )
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                previous = predicted.cpu().numpy()
                previous_logits = output["mask_logits"].float()
            # The raw task has complete motion/actions; records already contain prefixes.
            write_json(
                folder / "full_process.json",
                {"task_run_id": task, "events": user.events, "clicks": user.prompts, "profile": user.profile},
            )
            if position % 10 == 0 or position == len(targets):
                handle.flush()
                write_progress(
                    args.output_dir,
                    "running",
                    phase="generating_process_data",
                    tasks_done=position,
                    total_tasks=len(targets),
                    generated_clicks=position * 5,
                )
                print(f"Simulation tasks={position}/{len(targets)} clicks={position*5}", flush=True)
    write_json(data_root / "baseline_events.json", baseline)
    write_json(data_root / "targets.json", targets)
    return data_root, baseline


@torch.no_grad()
def closed_loop_test(args, targets, dataset, model, cache, device, amp):
    donors = shuffled_process_donors(dataset.records, args.seed)
    index_by_id = {r["event_id"]: i for i, r in enumerate(dataset.records)}
    results = {"Final": [], "Shuffled": [], "Bypass": []}
    for position, reference in enumerate(targets, 1):
        gt = mask_utils.decode(reference["ground_truth_rle"]).astype(bool)
        truth = torch.tensor(gt, device=device)
        user = SimulatedUser(reference["user_seed"], *gt.shape)
        previous = np.zeros_like(gt)
        previous_logits = None
        embedding = cache.get(reference["image_ref"])
        for click in range(1, 6):
            record = user.interact(
                gt, previous, task_run_id=reference["task_run_id"], event_id=f"{reference['task_run_id']}:click{click}"
            )
            record.update(reference)
            inputs, resized = live_inputs(user, reference, device, previous_logits)
            normal, latency = infer(model, embedding, inputs, amp)
            predicted = decode(model, normal, reference, resized)
            results["Final"].append(result_row(record, predicted, truth, latency))
            # Same recipient image/prompts/pre-logits; intervention does not feed back.
            donor_index = donors[index_by_id[record["event_id"]]]
            donor = dataset.process_inputs(donor_index)
            shuffled = dict(inputs)
            for key in (*PROCESS_INPUT_KEYS, "process_events", "process_event_valid"):
                shuffled[key] = donor[key].unsqueeze(0).to(device)
            changed, elapsed = infer(model, embedding, shuffled, amp)
            results["Shuffled"].append(result_row(record, decode(model, changed, reference, resized), truth, elapsed))
            bypass, elapsed = infer(model, embedding, {**inputs, "use_process": False}, amp)
            results["Bypass"].append(result_row(record, decode(model, bypass, reference, resized), truth, elapsed))
            previous = predicted.cpu().numpy()
            previous_logits = normal["mask_logits"].float()
        if position % 10 == 0 or position == len(targets):
            write_progress(
                args.output_dir, "running", phase="closed_loop_testing", tasks_done=position, total_tasks=len(targets)
            )
            print(f"Final closed-loop test tasks={position}/{len(targets)}", flush=True)
    for name, rows in results.items():
        write_json(args.output_dir / f"test_{name.lower()}_events.json", rows)
    return results


def main(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        # Logs and launch metadata are expected from the background launcher.
        if (args.output_dir / "run_manifest.json").exists():
            raise FileExistsError("Use a fresh run directory")
    if args.instances < 20 or args.epochs < 1:
        raise ValueError("At least 20 targets and one epoch required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    amp = device.type == "cuda"
    torch.set_num_threads(4)
    set_seed(args.seed)
    write_progress(
        args.output_dir, "running", phase="selecting_targets", total_tasks=args.instances, total_epochs=args.epochs
    )
    targets = select_targets(args)
    write_json(
        args.output_dir / "run_manifest.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "trained_variants": ["E3_final"],
            "base_checkpoint_sha256": file_hash(args.base_checkpoint),
            "simulation_policy": "GT error-driven procedural user; noisy focus, cubic cursor paths, variable hover, zoom and pan",
            "limitations": "Oracle-user simulation, not observed human behavior; synthetic gains do not establish human benefit",
            "image_split_policy": "one target per image; human images excluded; grouped 70/15/15",
            "test_protocol": "Final is adaptive closed-loop; Shuffled/Bypass are paired interventions on Final's identical click/pre-logit sequence",
            "initial_mask": "no dense mask on first click; exact previous L0 logits thereafter",
            "event_prefix": "full observed prefix, max 64 ordered events with semantic actions preserved; no future events",
        },
    )
    base = create_efficientvit_sam_model("efficientvit-sam-l0", weight_url=str(args.base_checkpoint)).to(device).eval()
    native = (
        ProcessAwareEfficientViTSam(base, ProcessSAMConfig(variant="E0", train_mask_decoder=False)).to(device).eval()
    )
    cache = DiskImageCache(native, args.output_dir / "image_embeddings", device, amp)
    data_root, baseline = generate(args, targets, native, cache, device, amp)
    baseline_test = [r for r in baseline if r["split"] == "test"]
    write_json(args.output_dir / "frozen_baseline.json", replay_metrics(baseline_test))
    del native, base
    datasets = {s: SyntheticProcessDataset(data_root, s) for s in ["train", "val", "test"]}
    selected = {s: list(range(len(d))) for s, d in datasets.items()}
    matrix = torch.stack([datasets["train"].process_inputs(i)["process_features"][-1] for i in selected["train"]])
    config = ProcessSAMConfig(
        variant="E3",
        use_event_history=True,
        spatial_gate_init=0.0,
        process_mean=tuple(matrix.mean(0).tolist()),
        process_std=tuple(matrix.std(0, unbiased=False).clamp_min(1e-3).tolist()),
    )
    set_seed(args.seed)
    base = create_efficientvit_sam_model("efficientvit-sam-l0", weight_url=str(args.base_checkpoint)).to(device).eval()
    model = ProcessAwareEfficientViTSam(base, config).to(device)
    cache.model = model
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("process_") and p.requires_grad],
                "lr": 1e-4,
            },
            {"params": model.mask_decoder.parameters(), "lr": 1e-5},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    manifest = json.loads((args.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        dataset_sha256=file_hash(datasets["train"].jsonl_path),
        process_config=config.to_dict(),
        parameters=model.parameter_summary(),
        coverage={
            s: {"tasks": len({r["task_run_id"] for r in d.records}), "clicks": len(d)} for s, d in datasets.items()
        },
    )
    write_json(args.output_dir / "run_manifest.json", manifest)
    history, best = [], float("-inf")
    for epoch in range(1, args.epochs + 1):
        write_progress(
            args.output_dir,
            "running",
            phase="training",
            experiment="E3_final",
            epoch=epoch,
            total_epochs=args.epochs,
            training_steps=len(selected["train"]),
        )
        loss = train_epoch(
            model, datasets["train"], selected["train"], cache, optimizer, scaler, device, amp, epoch, args.seed, 4
        )
        metrics, _ = evaluate(model, datasets["val"], selected["val"], cache, device, amp)
        history.append({"epoch": epoch, "train_loss": loss, "val": metrics})
        write_json(args.output_dir / "history.json", history)
        if metrics["mean_iou"] > best:
            best = metrics["mean_iou"]
            torch.save(
                {"model": model.state_dict(), "config": config.to_dict(), "epoch": epoch, "val_iou": best},
                args.output_dir / "best.pt",
            )
        print(
            f"Final E3 epoch={epoch}/{args.epochs} loss={loss:.4f} val_iou={metrics['mean_iou']:.4f} best={best:.4f}",
            flush=True,
        )
        write_progress(
            args.output_dir,
            "running",
            phase="epoch_completed",
            experiment="E3_final",
            epoch=epoch,
            total_epochs=args.epochs,
            best_val_iou=best,
        )
    best_checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    model.eval()
    write_progress(
        args.output_dir, "running", phase="closed_loop_testing", total_tasks=manifest["coverage"]["test"]["tasks"]
    )
    test_targets = [r for r in targets if r["split"] == "test"]
    rows = closed_loop_test(args, test_targets, datasets["test"], model, cache, device, amp)
    summary = {
        "result_type": "oracle_user_synthetic_final_only",
        "training": {
            "epochs": args.epochs,
            "seed": args.seed,
            "trained_variants": ["E3_final"],
            "best_epoch": best_checkpoint["epoch"],
        },
        "coverage": manifest["coverage"],
        "evaluations": {
            "Frozen_L0": replay_metrics(baseline_test),
            **{name: replay_metrics(r) for name, r in rows.items()},
        },
        "Final_minus_Shuffled": paired_process_delta(rows["Final"], rows["Shuffled"], args.seed),
        "Final_minus_Bypass": paired_process_delta(rows["Final"], rows["Bypass"], args.seed),
        "protocol": manifest["test_protocol"],
        "limitations": manifest["limitations"],
        "NoC_protocol": "Final/Frozen_L0: adaptive five-click closed-loop, failures=6; Shuffled/Bypass: paired replay of Final prompts and pre-logits",
        "latency_protocol": "cached image embeddings; prompt/process/fusion/decoder included; image encoding, IO, mask upsampling and simulated-user policy excluded",
    }
    write_json(args.output_dir / "summary.json", summary)
    lines = [
        "Process-aware EfficientViT-SAM synthetic final experiment",
        f"E3 trained once: {args.epochs} epochs; seed {args.seed}; best epoch {best_checkpoint['epoch']}.",
        "Synthetic procedural user uses GT to choose clicks/focus. This does not establish benefit on real human interaction.",
        "Final/Frozen_L0 are adaptive closed-loop. Shuffled/Bypass replay Final's identical prompts and pre-logits.",
        "Variant | IoU@1 | IoU@3 | IoU@5 | NoC@90 | Boundary IoU | Cached latency ms",
    ]
    for name, m in summary["evaluations"].items():
        lines.append(
            name
            + " | "
            + " | ".join(
                f"{m[k]:.4f}" for k in ["IoU@1", "IoU@3", "IoU@5", "NoC@90", "Boundary IoU", "latency_ms_mean"]
            )
        )
    lines.append("Final minus Shuffled: " + json.dumps(summary["Final_minus_Shuffled"]))
    lines.append("Final minus Bypass: " + json.dumps(summary["Final_minus_Bypass"]))
    (args.output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_progress(
        args.output_dir,
        "complete",
        completed_experiments=["E3_final"],
        report="report.txt",
        summary="summary.json",
        total_epochs=args.epochs,
    )
    print("Completed synthetic final-only experiment: " + str(args.output_dir), flush=True)


if __name__ == "__main__":
    arguments = parser().parse_args()
    existing_run = (arguments.output_dir / "run_manifest.json").exists()
    try:
        main(arguments)
    except Exception as error:
        if not existing_run:
            arguments.output_dir.mkdir(parents=True, exist_ok=True)
            write_progress(arguments.output_dir, "failed", error=repr(error))
        raise
