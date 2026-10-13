#!/usr/bin/env python3
"""Matched B0/B1/B2 training, temporal diagnostics, policy shift and human replay."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from efficientvit.models.process_sam import ProcessAwareEfficientViTSam, ProcessSAMConfig
from efficientvit.models.cais.coco_click_dataset import PROCESS_FEATURE_NAMES
from efficientvit.models.process_sam.dataset import SyntheticProcessDataset, viewport_state
from efficientvit.models.process_sam.metrics import binary_iou, boundary_iou, paired_process_delta, replay_metrics
from efficientvit.models.process_sam.simulation import SimulatedUser
from efficientvit.models.process_sam.validation import (
    FullProcessHumanDataset,
    observed_human_prefix,
    temporal_intervention,
)
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from tools.run_process_sam_experiments import (
    evaluate,
    file_hash,
    model_inputs,
    set_seed,
    train_epoch,
    write_json,
    write_progress,
)
from tools.run_process_sam_synthetic_final import DiskImageCache, decode, infer, live_inputs, result_row

POLICIES = {
    "oracle": {},
    "click_focus": {"focus_policy": "click"},
    "click_focus_slow": {"focus_policy": "click", "timing_scale": 1.8},
    "click_focus_sparse": {"focus_policy": "click", "motion_points": 9},
}
INTERVENTIONS = ("history_order", "history_fast", "history_slow", "state_timing_neutral")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, default=ROOT / "outputs/process_sam_synthetic_final_20261010_seed9527")
    p.add_argument("--human-dataset", type=Path, default=ROOT / "outputs/cais_synthetic_click2_click3_300_20260921")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[9527, 9528, 9529])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", action="store_true", help="Keep completed groups and restore epoch training checkpoints")
    return p


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_torch_save(payload, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def random_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def group_metrics(rows, protocol):
    result = replay_metrics(rows)
    result["NoC_protocol"] = protocol
    result["NoC@90_failure_rate"] = (
        result["NoC@90_failures"] / result["complete_prefix_tasks"] if result["complete_prefix_tasks"] else None
    )
    task_scores = defaultdict(list)
    for row in rows:
        task_scores[row["task_run_id"]].append(row["iou"])
    result["mean_task_iou"] = float(np.mean([np.mean(v) for v in task_scores.values()])) if task_scores else None
    z_values = [r["process_max_abs_z"] for r in rows if "process_max_abs_z" in r]
    if z_values:
        result["process_ood_fraction_abs_z_gt10"] = float(np.mean(np.asarray(z_values) > 10))
        result["process_max_abs_z"] = float(max(z_values))
    return result


def select_complete(dataset, limit=None):
    tasks = sorted({r["task_run_id"] for r in dataset.records})
    if limit is not None:
        tasks = tasks[:limit]
    return [i for i, r in enumerate(dataset.records) if r["task_run_id"] in set(tasks)]


def prepare_human(args, synthetic_images):
    source = args.human_dataset / "alignment/modeling_clicks_strict.jsonl"
    all_records = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
    records = [r for r in all_records if r.get("source", "human") == "human"]
    by_task, tasks = defaultdict(list), {}
    raw_hashes = {}
    for record in records:
        by_task[record["task_run_id"]].append(record)
    for name in sorted({r["human_dataset"] for r in records}):
        origin = Path(next(r["image_ref"] for r in records if r["human_dataset"] == name)).parents[2]
        task_file = origin / "interaction_data/interaction_tasks.jsonl"
        raw_hashes[str(task_file)] = file_hash(task_file)
        for line in task_file.open(encoding="utf-8"):
            if line.strip():
                task = json.loads(line)
                if task["task_run_id"] in by_task:
                    tasks[task["task_run_id"]] = task
    target = args.output_dir / "human_data"
    (target / "alignment").mkdir(parents=True)
    transformed, quarantined, details = [], {}, []
    for task_id, group in sorted(by_task.items()):
        try:
            if group[0]["coco_image_id"] in synthetic_images:
                raise ValueError("Human/synthetic image overlap")
            converted = []
            for record in sorted(group, key=lambda r: r["click_index"]):
                updated, audit = observed_human_prefix(record, tasks[task_id], group)
                # Relative references must keep their original dataset root.
                for key in ("image_ref", "pre_mask_ref", "post_mask_ref"):
                    updated[key] = str((args.human_dataset / updated[key]).resolve())
                    if not Path(updated[key]).is_file():
                        raise FileNotFoundError(updated[key])
                converted.append(updated)
                details.append({"event_id": record["event_id"], **audit})
            transformed.extend(converted)
        except (KeyError, ValueError, FileNotFoundError) as error:
            quarantined[task_id] = str(error)
    if not any(r["split"] == "test" for r in transformed):
        raise ValueError("No valid human test prefixes")
    with (target / "alignment/modeling_clicks_strict.jsonl").open("w", encoding="utf-8") as handle:
        for record in transformed:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    audit = {
        "source_hash": file_hash(source),
        "raw_task_hashes": raw_hashes,
        "original_tasks": len(by_task),
        "retained_tasks": len({r["task_run_id"] for r in transformed}),
        "quarantined_tasks": quarantined,
        "records_by_split": dict(Counter(r["split"] for r in transformed)),
        "prefix_checks": details,
        "synthetic_image_overlap": 0,
        "protocol": "Linked current-click task-log marker defines cutoff; locally anchored telemetry; recorded L1 binary masks on later clicks",
        "clock_caveat": "Task and click clocks have variable offsets; task-log cutoff is not claimed to be millisecond-exact release-time alignment",
        "caveat": "Existing strict cohort has final-quality selection and one annotator; no real-user closed-loop claim",
    }
    write_json(target / "audit.json", audit)
    return target, transformed, audit


def make_model(checkpoint, config, device):
    base = create_efficientvit_sam_model("efficientvit-sam-l0", weight_url=str(checkpoint)).to(device).eval()
    return ProcessAwareEfficientViTSam(base, config).to(device)


def load_model(path, base_checkpoint, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(base_checkpoint, ProcessSAMConfig(**saved["config"]), device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, {"best_epoch": saved["epoch"], "val_iou": saved["val_iou"], "checkpoint": str(path.resolve())}


def train_group(args, name, seed, base_checkpoint, source_config, datasets, selected, cache, device, amp):
    folder = args.output_dir / f"seed_{seed}" / name
    if args.resume and (folder / "training.json").is_file():
        finished = read_json(folder / "training.json")
        if finished["epochs"] != args.epochs or finished["seed"] != seed or not (folder / "best.pt").is_file():
            raise ValueError(f"Completed training metadata mismatch: {folder}")
        print(f"Keeping completed training: {name} seed={seed}", flush=True)
        return folder / "best.pt"
    folder.mkdir(parents=True, exist_ok=args.resume)
    config_values = dict(source_config)
    if name == "B0":
        config_values.update(variant="E0", use_event_history=False)
    elif name == "B1":
        config_values.update(history_pooling="mean", spatial_only=True)
    config = ProcessSAMConfig(**config_values)
    set_seed(seed)
    model = make_model(base_checkpoint, config, device)
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
    history, best, best_checkpoint = [], -float("inf"), None
    previous_history = []
    last_path = folder / "last.pt"
    if args.resume and last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["config"] != config.to_dict() or saved["seed"] != seed or saved["epochs"] != args.epochs:
            raise ValueError(f"Training checkpoint protocol mismatch: {last_path}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history, best_checkpoint = saved["history"], saved["best_checkpoint"]
        best = best_checkpoint["val_iou"]
        restore_random_state(saved["random_state"])
        atomic_torch_save(best_checkpoint, folder / "best.pt")
        write_json(folder / "history.json", history)
        print(f"Restored {name} seed={seed} through epoch {saved['epoch']}; optimizer/AMP/RNG included", flush=True)
        del saved
    elif args.resume and (folder / "history.json").is_file():
        # A legacy best.pt contains no optimizer state. Replay the original run,
        # rather than changing its optimizer schedule by restarting from best.pt.
        previous_history = read_json(folder / "history.json")
        archive = folder / ("interrupted_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        if not archive.resolve().is_relative_to(args.output_dir.resolve()):
            raise ValueError("Recovery archive must stay inside the run directory")
        archive.mkdir()
        for filename in ("history.json", "best.pt"):
            if (folder / filename).is_file():
                (folder / filename).rename(archive / filename)
        write_json(
            folder / "recovery.json",
            {
                "method": "same-seed replay from original base; no legacy optimizer state",
                "archive": str(archive.resolve()),
                "previous_history": previous_history,
            },
        )
        print(f"Reconstructing {name} seed={seed} from epoch 1; partial artifacts preserved in {archive}", flush=True)
    for epoch in range(len(history) + 1, args.epochs + 1):

        def progress(step, total, loss):
            write_progress(
                args.output_dir,
                "running",
                phase="training",
                group=name,
                seed=seed,
                epoch=epoch,
                total_epochs=args.epochs,
                step=step,
                total_steps=total,
                loss=loss,
            )

        write_progress(
            args.output_dir,
            "running",
            phase="training",
            group=name,
            seed=seed,
            epoch=epoch,
            total_epochs=args.epochs,
            step=0,
            total_steps=len(selected["train"]),
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
            seed,
            4,
            progress_callback=progress,
        )
        metrics, _ = evaluate(model, datasets["val"], selected["val"], cache, device, amp)
        history.append({"epoch": epoch, "train_loss": loss, "val": metrics})
        write_json(folder / "history.json", history)
        if metrics["mean_iou"] > best:
            best = metrics["mean_iou"]
            best_checkpoint = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "config": config.to_dict(),
                "epoch": epoch,
                "val_iou": best,
            }
            atomic_torch_save(best_checkpoint, folder / "best.pt")
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "config": config.to_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "epochs": args.epochs,
                "seed": seed,
                "history": history,
                "best_checkpoint": best_checkpoint,
                "random_state": random_state(),
            },
            last_path,
        )
        if epoch <= len(previous_history):
            reference = previous_history[epoch - 1]
            recovery = read_json(folder / "recovery.json")
            recovery.setdefault("replayed_epoch_deltas", []).append(
                {
                    "epoch": epoch,
                    "train_loss": loss - reference["train_loss"],
                    "val_mean_iou": metrics["mean_iou"] - reference["val"]["mean_iou"],
                }
            )
            write_json(folder / "recovery.json", recovery)
        print(
            f"{name} seed={seed} epoch={epoch}/{args.epochs} loss={loss:.5f} val_iou={metrics['mean_iou']:.5f}",
            flush=True,
        )
    write_json(
        folder / "training.json",
        {
            "config": config.to_dict(),
            "parameters": model.parameter_summary(),
            "epochs": args.epochs,
            "seed": seed,
            "best_val_iou": best,
        },
    )
    cache.model = None
    del model, optimizer, scaler, best_checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return folder / "best.pt"


@torch.no_grad()
def diagnostics(model, dataset, indices, cache, device, amp, seed):
    conditions = {"normal": [], **{name: [] for name in INTERVENTIONS}}
    for position, index in enumerate(indices):
        sample = dataset[index]
        normal = model_inputs(sample, device)
        embedding = cache.get(sample["image_path"])
        resized_w, resized_h = sample["trajectory_coordinate_scale"].int().tolist()
        original_h, original_w = sample["original_size"]
        truth = torch.tensor(dataset._decode_ground_truth(dataset.records[index]), device=device)
        for name in conditions:
            inputs = (
                normal
                if name == "normal"
                else temporal_intervention(normal, name, seed + position, model.process_config.process_mean)
            )
            if position == 0:
                for _ in range(3):
                    infer(model, embedding, inputs, amp)
            output, elapsed = infer(model, embedding, inputs, amp)
            mask = (
                model.postprocess_masks(
                    output["mask_logits"].float(), (resized_h, resized_w), (original_h, original_w)
                )[0, 0]
                > 0
            )
            conditions[name].append(
                {
                    "event_id": sample["event_id"],
                    "task_run_id": sample["task_run_id"],
                    "click_index": sample["click_index"],
                    "iou": binary_iou(mask, truth),
                    "boundary_iou": boundary_iou(mask, truth),
                    "latency_ms": elapsed,
                }
            )
    return conditions


@torch.no_grad()
def closed_loop(args, model, targets, cache, device, amp, name, seed, policy):
    rows = []
    folder = args.output_dir / f"seed_{seed}" / name
    telemetry_path = folder / f"process_closed_loop_{policy}.jsonl"
    telemetry = telemetry_path.open("w", encoding="utf-8")
    for position, reference in enumerate(targets):
        gt = mask_utils.decode(reference["ground_truth_rle"]).astype(bool)
        truth = torch.tensor(gt, device=device)
        user = SimulatedUser(reference["user_seed"], *gt.shape, **POLICIES[policy])
        previous, previous_logits = np.zeros_like(gt), None
        embedding = cache.get(reference["image_ref"])
        for click in range(1, 6):
            record = user.interact(
                gt, previous, task_run_id=reference["task_run_id"], event_id=f"{reference['task_run_id']}:click{click}"
            )
            record.update(reference)
            inputs, resized = live_inputs(user, reference, device, previous_logits)
            if position == 0 and click == 1:
                for _ in range(3):
                    infer(model, embedding, inputs, amp)
            output, elapsed = infer(model, embedding, inputs, amp)
            prediction = decode(model, output, reference, resized)
            row = result_row(record, prediction, truth, elapsed)
            means = inputs["process_features"].new_tensor(model.process_config.process_mean)
            stds = inputs["process_features"].new_tensor(model.process_config.process_std)
            row["process_max_abs_z"] = float(((inputs["process_features"][0, -1] - means) / stds).abs().max())
            rows.append(row)
            # Save the factual prefix for review; GT remains in target metadata,
            # separate from the tensors that were actually passed to the model.
            telemetry.write(
                json.dumps(
                    {
                        k: record[k]
                        for k in (
                            "event_id",
                            "task_run_id",
                            "click_index",
                            "click_timestamp_ms",
                            "viewport",
                            "prompt_history",
                            "process_events",
                            "preceding_trajectory",
                        )
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            previous = prediction.cpu().numpy()
            previous_logits = output["mask_logits"].float()
        if (position + 1) % 10 == 0:
            write_progress(
                args.output_dir,
                "running",
                phase="closed_loop_evaluation",
                group=name,
                seed=seed,
                policy=policy,
                tasks_done=position + 1,
                total_tasks=len(targets),
            )
    telemetry.close()
    return rows


def human_subgroups(records, rows):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["task_run_id"]].append(record)
    categories = defaultdict(set)
    for task, group in grouped.items():
        events = group[-1]["process_events"]
        first_click_time = next(e["timestamp_ms"] for e in events if e["type"] == "click")
        # Startup auto-fit/preview zoom is not deliberate inter-click behavior.
        events = [e for e in events if e["timestamp_ms"] > first_click_time]
        zoom = any(e["type"] == "zoom" and abs(float(e["viewport"].get("delta_log_zoom", 0))) > 0.01 for e in events)
        pan = any(
            e["type"] == "pan"
            and np.linalg.norm(
                [e["viewport"].get("delta_center_x_norm", 0), e["viewport"].get("delta_center_y_norm", 0)]
            )
            > 0.01
            for e in events
        )
        categories["observed_zoom" if zoom else "no_observed_zoom"].add(task)
        categories["observed_pan" if pan else "no_observed_pan"].add(task)
    return {
        name: group_metrics([r for r in rows if r["task_run_id"] in tasks], "human recorded-mask replay")
        for name, tasks in categories.items()
    }


def save_report(output, summary):
    # Plain-text report: avoid introducing a separate prose artifact workflow.
    lines = [
        "Process SAM supplementary validation",
        "Status: " + summary["status"],
        "B0: decoder-only fine tuning; B1: spatial unordered control; B2: complete temporal E3.",
        "Frozen image/prompt encoders; same original 700/150/150 data and training budgets.",
        "Additional seeds are selected solely by first-seed validation deltas, not test performance.",
    ]
    for seed, result in summary["seeds"].items():
        lines.append(f"\nTraining seed {seed}")
        for name, metrics in result.get("matched_replay", {}).items():
            lines.append(f"{name} replay: " + json.dumps(metrics, ensure_ascii=False))
        for name, policies in result.get("closed_loop", {}).items():
            for policy, metrics in policies.items():
                lines.append(f"{name} closed loop/{policy}: " + json.dumps(metrics, ensure_ascii=False))
        for name, delta in result.get("paired_replay", {}).items():
            lines.append(name + ": " + json.dumps(delta))
        for name, metrics in result.get("human_replay", {}).items():
            lines.append(f"{name} human replay: " + json.dumps(metrics, ensure_ascii=False))
        for name, delta in result.get("paired_human", {}).items():
            lines.append("Human " + name + ": " + json.dumps(delta))
        for name, metrics in result.get("temporal_diagnostics", {}).items():
            lines.append("Temporal " + name + ": " + json.dumps(metrics))
    lines.append("Extra seed decision: " + json.dumps(summary.get("extra_seed_decision"), ensure_ascii=False))
    lines.append("Seed aggregate: " + json.dumps(summary.get("seed_aggregate"), ensure_ascii=False))
    lines.extend(
        [
            "\nLimitations:",
            "Oracle-user training; click-focus policy still uses GT for correction clicks, not viewport size/center.",
            "B1 shares the same recent-window spatial raster (including its coarse temporal support); fine timing/order removed.",
            "Temporal perturbations are sensitivity diagnostics; spatial raster, click and dense-mask inputs stay fixed.",
            "Human transfer is recorded L1 binary-mask replay, not live feedback; strict cohort selection and one annotator limit generalization.",
            "Human task-log click marker defines the prefix cutoff; variable telemetry-clock offsets are audited, not treated as missing data.",
            "Human zoom/pan subgroups use actions AFTER the first click, excluding startup auto-fit.",
            "Latency excludes cached image encoding, IO, original-resolution upsampling and user simulation.",
        ]
    )
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def persist(args, summary):
    write_json(args.output_dir / "summary.json", summary)
    save_report(args.output_dir, summary)


def main(args):
    if args.epochs < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Positive epochs and distinct seeds are required")
    saved_manifest = None
    if args.resume:
        saved_manifest = read_json(args.output_dir / "run_manifest.json")
        for key in ("epochs", "seeds", "smoke_test", "prepare_only", "device"):
            if vars(args)[key] != saved_manifest["args"][key]:
                raise ValueError(f"Resume argument changed: {key}")
        for key in ("source_run", "human_dataset"):
            if vars(args)[key].resolve() != Path(saved_manifest["args"][key]).resolve():
                raise ValueError(f"Resume input changed: {key}")
        for name, expected in saved_manifest["code_hashes"].items():
            if Path(name).name != Path(__file__).name and file_hash(ROOT / name) != expected:
                raise ValueError(f"Model/data/training implementation changed: {name}")
        previous_summary = read_json(args.output_dir / "summary.json")
        if previous_summary["status"] == "complete":
            print("Existing supplementary validation is complete; keeping all results", flush=True)
            return
    elif (args.output_dir / "run_manifest.json").exists():
        raise FileExistsError("Use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((args.source_run / "run_manifest.json").read_text(encoding="utf-8"))
    source_summary = json.loads((args.source_run / "summary.json").read_text(encoding="utf-8"))
    base_checkpoint = Path(source_manifest["args"]["base_checkpoint"])
    data_root = args.source_run / "data"
    if file_hash(base_checkpoint) != source_manifest["base_checkpoint_sha256"]:
        raise ValueError("Base checkpoint changed")
    if file_hash(data_root / "alignment/modeling_clicks_strict.jsonl") != source_manifest["dataset_sha256"]:
        raise ValueError("Original synthetic dataset changed")
    if not args.smoke_test and (
        args.epochs != source_summary["training"]["epochs"] or args.seeds[0] != source_summary["training"]["seed"]
    ):
        raise ValueError("First seed and epoch budget must match reused E3")
    limit = 2 if args.smoke_test else None
    if args.smoke_test:
        args.epochs, args.seeds = 1, args.seeds[:1]
    if not args.resume:
        write_progress(args.output_dir, "running", phase="preparing_data")
    targets = json.loads((data_root / "targets.json").read_text(encoding="utf-8"))
    if args.resume:
        human_root = args.output_dir / "human_data"
        human_audit = read_json(human_root / "audit.json")
        if file_hash(args.human_dataset / "alignment/modeling_clicks_strict.jsonl") != human_audit["source_hash"]:
            raise ValueError("Original human dataset changed")
        for path, expected in human_audit["raw_task_hashes"].items():
            if file_hash(Path(path)) != expected:
                raise ValueError(f"Original human task log changed: {path}")
    else:
        human_root, _, human_audit = prepare_human(args, {r["coco_image_id"] for r in targets})
    test_targets = sorted([r for r in targets if r["split"] == "test"], key=lambda r: r["task_run_id"])
    if limit:
        test_targets = test_targets[:limit]
    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_run": str(args.source_run.resolve()),
        "dataset_sha256": source_manifest["dataset_sha256"],
        "base_checkpoint_sha256": source_manifest["base_checkpoint_sha256"],
        "data_and_user_seeds_fixed": True,
        "training_seeds": args.seeds,
        "extension_gate": "Both paired validation task-mean IoU deltas B2-B0 and B2-B1 > 0; test never selects extension",
        "training": {
            "epochs": args.epochs,
            "process_lr": 1e-4,
            "decoder_lr": 1e-5,
            "weight_decay": 1e-4,
            "accumulation": 4,
            "loss": "BCE + Dice + IoU MSE",
            "selection": "maximum validation event-mean IoU",
        },
        "controls": {
            "B0": "Decoder fine tuning only",
            "B1": "No summaries/time/order; almost parameter-matched mean-pool MLP",
            "B2": "Full E3, first seed reused without retraining",
        },
        "policies": POLICIES,
        "interventions": INTERVENTIONS,
        "human_audit": {k: v for k, v in human_audit.items() if k != "prefix_checks"},
        "spatial_control": "Identical current viewport/click maps and last-two-second trajectory density; B1 lacks explicit timing/order",
        "metrics": "IoU@1/3/5, task-mean IoU, Boundary IoU, NoC@90 failures=6, cached-embedding mean/p95 latency",
        "protocols": "Matched original L0-logit replay; separate per-model adaptive five-click loops; human recorded L1 binary-mask replay",
        "result_type": "integration_smoke" if args.smoke_test else "supplementary_validation",
    }
    if args.resume:
        manifest = saved_manifest
    else:
        write_json(args.output_dir / "run_manifest.json", manifest)
    if args.prepare_only:
        write_progress(args.output_dir, "complete", phase="data_prepared", human_tasks=human_audit["retained_tasks"])
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    amp = device.type == "cuda"
    torch.set_num_threads(4)
    datasets = {s: SyntheticProcessDataset(data_root, s) for s in ("train", "val", "test")}
    selected = {s: select_complete(d, limit) for s, d in datasets.items()}
    human = FullProcessHumanDataset(human_root, "test")
    human_indices = select_complete(human, limit)
    human_features = torch.stack([human.process_inputs(i)["process_features"][-1] for i in human_indices])
    z = (
        (human_features - human_features.new_tensor(source_manifest["process_config"]["process_mean"]))
        / human_features.new_tensor(source_manifest["process_config"]["process_std"])
    ).abs()
    human_shift = {
        "fraction_events_abs_z_gt10": float((z.max(1).values > 10).float().mean()),
        "max_abs_z": float(z.max()),
        "mean_abs_z_by_feature": dict(zip(PROCESS_FEATURE_NAMES, z.mean(0).tolist())),
        "normalization": "Unchanged synthetic TRAIN statistics; constant training fields have a 1e-3 std floor",
    }
    write_json(args.output_dir / "human_data/feature_shift.json", human_shift)
    selected_coverage = {
        s: {"clicks": len(selected[s]), "tasks": len({d.records[i]["task_run_id"] for i in selected[s]})}
        for s, d in datasets.items()
    }
    human_test_coverage = {
        "clicks": len(human_indices),
        "tasks": len({human.records[i]["task_run_id"] for i in human_indices}),
    }
    current_code_hashes = {
        str(path.relative_to(ROOT)): file_hash(path)
        for path in [
            Path(__file__),
            ROOT / "efficientvit/models/process_sam/config.py",
            ROOT / "efficientvit/models/process_sam/modules.py",
            ROOT / "efficientvit/models/process_sam/model.py",
            ROOT / "efficientvit/models/process_sam/simulation.py",
            ROOT / "efficientvit/models/process_sam/validation.py",
            ROOT / "tools/run_process_sam_experiments.py",
        ]
    }
    if args.resume:
        if selected_coverage != manifest["selected_coverage"] or human_test_coverage != manifest["human_test_coverage"]:
            raise ValueError("Resume cohort coverage changed")
        resume_path = args.output_dir / "resume_events.json"
        resumes = read_json(resume_path) if resume_path.is_file() else []
        resumes.append(
            {
                "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
                "previous_status": previous_summary["status"],
                "previous_progress": (
                    read_json(args.output_dir / "pause.json") if (args.output_dir / "pause.json").is_file() else None
                ),
                "code_hashes": current_code_hashes,
                "checkpoint_policy": "Full model, optimizer, AMP, best model and RNG state at each completed epoch",
            }
        )
        write_json(resume_path, resumes)
    else:
        manifest.update(
            selected_coverage=selected_coverage,
            human_test_coverage=human_test_coverage,
            human_feature_shift=human_shift,
            code_hashes=current_code_hashes,
        )
        write_json(args.output_dir / "run_manifest.json", manifest)
    args._active_run = True
    cache = DiskImageCache(None, args.source_run / "image_embeddings", device, amp)
    human_cache = DiskImageCache(None, args.output_dir / "human_image_embeddings", device, amp)
    summary = {
        "status": "running",
        "result_type": manifest["result_type"],
        "seeds": {},
        "protocols": manifest["protocols"],
        "human_coverage": manifest["human_audit"],
        "human_feature_shift": human_shift,
    }
    if args.resume:
        summary = previous_summary
        summary["status"] = "running"
        summary.pop("pause", None)
        summary["resume_history"] = resumes
    persist(args, summary)
    extend = True
    for seed_position, seed in enumerate(args.seeds):
        if seed_position > 0 and not extend:
            summary["extra_seed_decision"] = {"run": False, "reason": "Predeclared validation gate not met"}
            break
        result = summary["seeds"].get(str(seed)) or {
            "checkpoints": {},
            "validation": {},
            "matched_replay": {},
            "paired_replay": {},
            "closed_loop": {},
            "human_replay": {},
        }
        summary["seeds"][str(seed)] = result
        validations, replay_rows, human_rows = {}, {}, {}
        for name in ("B0", "B1", "B2"):
            folder = args.output_dir / f"seed_{seed}" / name
            if args.resume and name in result["human_replay"]:
                validations[name] = read_json(folder / "val_events.json")
                replay_rows[name] = read_json(folder / "test_replay_events.json")
                human_rows[name] = read_json(folder / "human_test_events.json")
                for policy in POLICIES:
                    read_json(folder / f"test_closed_loop_{policy}.json")
                if name == "B2" and "temporal_diagnostics" not in result:
                    raise ValueError("Completed B2 is missing temporal diagnostics")
                print(f"Keeping completed training and evaluation: {name} seed={seed}", flush=True)
                continue
            if name == "B2" and seed_position == 0 and not args.smoke_test:
                path = args.source_run / "best.pt"
                print(f"Reusing existing full E3 checkpoint: {path}", flush=True)
            else:
                path = train_group(
                    args,
                    name,
                    seed,
                    base_checkpoint,
                    source_manifest["process_config"],
                    datasets,
                    selected,
                    cache,
                    device,
                    amp,
                )
            model, metadata = load_model(path, base_checkpoint, device)
            cache.model = human_cache.model = model
            folder = args.output_dir / f"seed_{seed}" / name
            folder.mkdir(parents=True, exist_ok=True)
            result["checkpoints"][name] = metadata
            write_progress(args.output_dir, "running", phase="matched_replay", group=name, seed=seed)
            _, validations[name] = evaluate(model, datasets["val"], selected["val"], cache, device, amp)
            _, replay_rows[name] = evaluate(model, datasets["test"], selected["test"], cache, device, amp)
            result["validation"][name] = group_metrics(validations[name], "fixed original L0-logit replay")
            result["matched_replay"][name] = group_metrics(replay_rows[name], "fixed original L0-logit replay")
            write_json(folder / "val_events.json", validations[name])
            write_json(folder / "test_replay_events.json", replay_rows[name])
            result["closed_loop"][name] = {}
            for policy in POLICIES:
                write_progress(
                    args.output_dir, "running", phase="closed_loop_evaluation", group=name, seed=seed, policy=policy
                )
                rows = closed_loop(args, model, test_targets, cache, device, amp, name, seed, policy)
                write_json(folder / f"test_closed_loop_{policy}.json", rows)
                result["closed_loop"][name][policy] = group_metrics(
                    rows, "adaptive per-model five-click loop; failure=6"
                )
            write_progress(args.output_dir, "running", phase="human_transfer", group=name, seed=seed)
            _, human_rows[name] = evaluate(model, human, human_indices, human_cache, device, amp)
            write_json(folder / "human_test_events.json", human_rows[name])
            result["human_replay"][name] = group_metrics(
                human_rows[name], "recorded L1 binary pre-mask replay; failure=6"
            )
            result["human_replay"][name]["subgroups"] = human_subgroups(
                [human.records[i] for i in human_indices], human_rows[name]
            )
            if name == "B2":
                write_progress(args.output_dir, "running", phase="temporal_diagnostics", seed=seed)
                perturbed = diagnostics(model, datasets["test"], selected["test"], cache, device, amp, seed)
                result["temporal_diagnostics"] = {}
                for condition, rows in perturbed.items():
                    write_json(folder / f"temporal_{condition}_events.json", rows)
                    result["temporal_diagnostics"][condition] = group_metrics(
                        rows, "paired fixed-mask replay; not adaptive NoC"
                    )
                    if condition != "normal":
                        result["temporal_diagnostics"][condition]["normal_minus_changed"] = paired_process_delta(
                            perturbed["normal"], rows, seed
                        )
            persist(args, summary)
            cache.model = human_cache.model = None
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for other in ("B0", "B1"):
            comparison = "B2_minus_" + other
            result["paired_replay"][comparison] = paired_process_delta(replay_rows["B2"], replay_rows[other], seed)
            result.setdefault("paired_validation", {})[comparison] = paired_process_delta(
                validations["B2"], validations[other], seed
            )
            result.setdefault("paired_human", {})[comparison] = paired_process_delta(
                human_rows["B2"], human_rows[other], seed
            )
            result.setdefault("paired_closed_loop", {})[comparison] = {}
            for policy in POLICIES:
                directory = args.output_dir / f"seed_{seed}"
                left = json.loads((directory / "B2" / f"test_closed_loop_{policy}.json").read_text(encoding="utf-8"))
                right = json.loads((directory / other / f"test_closed_loop_{policy}.json").read_text(encoding="utf-8"))
                result["paired_closed_loop"][comparison][policy] = paired_process_delta(left, right, seed)
        if seed_position == 0:
            extend = all(d["mean_task_iou_delta"] > 0 for d in result["paired_validation"].values())
            summary["extra_seed_decision"] = {
                "run": extend,
                "basis": "paired validation only",
                "validation": result["paired_validation"],
            }
        persist(args, summary)
    summary["seed_aggregate"] = {}
    for other in ("B0", "B1"):
        values = [r["paired_replay"]["B2_minus_" + other]["mean_task_iou_delta"] for r in summary["seeds"].values()]
        summary["seed_aggregate"]["B2_minus_" + other] = {
            "seeds": len(values),
            "per_seed_deltas": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "caveat": "Per-seed task bootstrap CIs are in each result; three seeds do not establish broad training robustness",
        }
    summary["status"] = "complete"
    persist(args, summary)
    write_progress(
        args.output_dir, "complete", completed_seeds=list(summary["seeds"]), report="report.txt", summary="summary.json"
    )
    print("Completed supplementary validation: " + str(args.output_dir), flush=True)


if __name__ == "__main__":
    arguments = parser().parse_args()
    existing_run = (arguments.output_dir / "run_manifest.json").exists()
    try:
        main(arguments)
    except Exception as error:
        if (not existing_run or getattr(arguments, "_active_run", False)) and arguments.output_dir.is_dir():
            write_progress(arguments.output_dir, "failed", error_type=type(error).__name__, error=str(error))
        raise
