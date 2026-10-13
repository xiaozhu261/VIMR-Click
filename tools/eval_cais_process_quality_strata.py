#!/usr/bin/env python3
"""Compare Real vs Zero process input by audited trajectory-quality stratum."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficientvit.models.cais import COCOClickCAISDataset, CognitiveTargetNormalizer
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from tools.train_cais_coco_click import ImageEmbeddingCache, apply_trajectory_mode, forward_loss, move_sample


def bootstrap_task_delta(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_task[row["task_run_id"]].append(row["delta"])
    task_deltas = np.asarray([np.mean(values) for values in by_task.values()], dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled = task_deltas[
        generator.integers(0, len(task_deltas), size=(5000, len(task_deltas)))
    ].mean(axis=1)
    return {
        "clicks": len(rows),
        "tasks": len(task_deltas),
        "real_iou": float(np.mean([row["real_iou"] for row in rows])),
        "zero_iou": float(np.mean([row["zero_iou"] for row in rows])),
        "mean_delta": float(task_deltas.mean()),
        "bootstrap_95ci": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/cais_process_trajectory_v2_seed9527_20261008"),
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("outputs/cais_process_trajectory_v2_seed9527_20261008/audit"),
    )
    parser.add_argument("--run-name", default="fixed_window_process_v2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_root = args.experiment_root / args.run_name
    checkpoint = torch.load(run_root / "best.pt", map_location="cpu", weights_only=False)
    run_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    config = checkpoint["cais_config"]
    target_config = checkpoint["cognitive_target"]
    normalizer = CognitiveTargetNormalizer(
        tuple(target_config["lower"]),
        tuple(target_config["upper"]),
    )
    dataset = COCOClickCAISDataset(
        Path(run_manifest["dataset"]["root"]),
        "test",
        trajectory_samples=int(config["trajectory_samples"]),
        trajectory_processing=str(config["trajectory_processing"]),
        trajectory_window_ms=float(config["trajectory_window_ms"]),
        cognitive_normalizer=normalizer,
        quality_gate_trajectories=bool(
            run_manifest["dataset"]["test"].get("quality_gate_trajectories", False)
        ),
    )
    device = torch.device(args.device)
    model = create_efficientvit_sam_model(
        "efficientvit-sam-cais-l0",
        pretrained=False,
        cais_config=config,
        image_size=512,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    amp = device.type == "cuda"
    cache = ImageEmbeddingCache(model, device, amp=amp, enabled=True)
    audit_by_event: dict[str, dict[str, str]] = {}
    with (args.audit_root / "trajectory_audit.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            audit_by_event[row["event_id"]] = row

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            raw = dataset[index]
            embedding = cache.get(raw["image_path"])
            ious = {}
            for mode in ("real", "zero"):
                sample = move_sample(apply_trajectory_mode(raw, mode), device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    _, losses = forward_loss(model, sample, embedding, model.cais_config)
                ious[mode] = float(losses["iou_target"].mean())
            audit = audit_by_event[str(raw["event_id"])]
            rows.append(
                {
                    "event_id": raw["event_id"],
                    "task_run_id": raw["task_run_id"],
                    "click_index": int(raw["click_index"]),
                    "reliable": audit["reliable"].lower() == "true",
                    "real_iou": ious["real"],
                    "zero_iou": ious["zero"],
                    "delta": ious["real"] - ious["zero"],
                }
            )

    groups = {
        "all": rows,
        "reliable": [row for row in rows if row["reliable"]],
        "unreliable": [row for row in rows if not row["reliable"]],
    }
    for click_index in sorted({row["click_index"] for row in rows}):
        groups[f"click_{click_index}"] = [row for row in rows if row["click_index"] == click_index]
    report = {
        name: bootstrap_task_delta(group, seed=9527 + group_index)
        for group_index, (name, group) in enumerate(groups.items())
        if group
    }
    (args.experiment_root / "quality_strata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Process V2 质量分层敏感性",
        "",
        "| 分层 | 点击数 | 任务数 | Real IoU | Zero IoU | 任务级 Real-Zero | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report.items():
        lines.append(
            f"| {name} | {item['clicks']} | {item['tasks']} | {item['real_iou']:.6f} | "
            f"{item['zero_iou']:.6f} | {item['mean_delta']:+.6f} | "
            f"[{item['bootstrap_95ci'][0]:+.6f}, {item['bootstrap_95ci'][1]:+.6f}] |"
        )
    (args.experiment_root / "QUALITY_STRATA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
