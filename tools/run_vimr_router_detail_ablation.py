"""Compare Full VIMR with the G/O/D router and boundary Detail Expert."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from efficientvit.models.vimr import ViewportIntentEncoder

PROTOCOL = ROOT_DIR / "tools/run_vimr_coco_protocol.py"
VARIANTS = (
    ("full_matched", "Full VIMR (matched rerun)", ()),
    ("router", "+G/O/D Router", ("--enable-expert-router",)),
    (
        "router_detail",
        "+G/O/D Router + Detail Expert",
        ("--enable-expert-router", "--enable-detail-refinement"),
    ),
)
METRICS = (
    "mIoU@1",
    "mIoU@2",
    "mIoU@3",
    "mIoU@5",
    "NoC@85",
    "NoC@90",
    "Success@85/5",
    "Success@90/5",
    "RegressionRate",
    "LateRegressionRate@4-5",
    "LateSevereRegressionRate@4-5>0.05",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=ROOT_DIR / "dataset/coco")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt",
    )
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "outputs/vimr_router_detail_ablation")
    parser.add_argument("--train-instances", type=int, default=500)
    parser.add_argument("--eval-instances", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-clicks", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rerun-existing", action="store_true")
    return parser.parse_args()


def protocol_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "train_instances": args.train_instances,
        "validation_instances": args.eval_instances,
        "epochs": args.epochs,
        "max_clicks": args.max_clicks,
        "training_seeds": args.seeds,
        "split_seed": args.split_seed,
        "learning_rate": args.learning_rate,
    }


def validate_protocol(result: dict[str, Any], args: argparse.Namespace, label: str) -> None:
    expected = protocol_signature(args)
    actual = result["protocol"]
    mismatches = {key: (actual.get(key), value) for key, value in expected.items() if actual.get(key) != value}
    if mismatches:
        raise ValueError(f"{label} protocol mismatch: {mismatches}")


def record_from_result(result: dict[str, Any], label: str, source: Path, seeds: list[int]) -> dict[str, Any]:
    return {
        "label": label,
        "source": str(source.resolve()),
        "best_epochs": [result["seeds"][str(seed)]["best_epoch"] for seed in seeds],
        "trainable_parameters": result["seeds"][str(seeds[0])]["trainable_vimr_parameters"],
        "aggregate": result["aggregate"],
    }


def run_variant(
    args: argparse.Namespace,
    directory: str,
    label: str,
    flags: tuple[str, ...],
) -> dict[str, Any]:
    output = args.output / directory
    result_path = output / "results.json"
    if args.rerun_existing or not result_path.is_file():
        command = [
            sys.executable,
            str(PROTOCOL),
            "--coco-root",
            str(args.coco_root),
            "--checkpoint",
            str(args.checkpoint),
            "--output",
            str(output),
            "--train-instances",
            str(args.train_instances),
            "--eval-instances",
            str(args.eval_instances),
            "--epochs",
            str(args.epochs),
            "--max-clicks",
            str(args.max_clicks),
            "--seeds",
            *(str(seed) for seed in args.seeds),
            "--split-seed",
            str(args.split_seed),
            "--learning-rate",
            str(args.learning_rate),
            "--lambda-safe",
            "1.0",
            "--lambda-monotonic",
            "0.5",
            "--gate-init-probability",
            "0.1",
            "--progress-every",
            str(args.progress_every),
            *flags,
        ]
        if args.device:
            command.extend(("--device", args.device))
        print(f"Starting {label}", flush=True)
        subprocess.run(command, cwd=ROOT_DIR, check=True)
    else:
        print(f"Reusing completed {label}: {result_path}", flush=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_protocol(result, args, label)
    return record_from_result(result, label, result_path, args.seeds)


def analyze_router_usage(args: argparse.Namespace) -> dict[str, Any]:
    trajectory_path = ROOT_DIR / "outputs/vimr_viewport_signal_ablation/canonical_interactions.json"
    if not trajectory_path.is_file():
        return {}
    trajectories = json.loads(trajectory_path.read_text(encoding="utf-8"))
    states = torch.tensor(
        [viewport for row in trajectories["validation"] for viewport in row["synthetic_viewports"]],
        dtype=torch.float32,
    )
    usage = {}
    for directory, label, flags in VARIANTS:
        if "--enable-expert-router" not in flags:
            continue
        seed_usage = {}
        for seed in args.seeds:
            checkpoint = torch.load(
                args.output / directory / f"seed_{seed}/best_vimr_modules.pt",
                map_location="cpu",
                weights_only=True,
            )["vimr_state_dict"]
            encoder = ViewportIntentEncoder().eval()
            encoder.load_state_dict(
                {
                    key.removeprefix("viewport_encoder."): value
                    for key, value in checkpoint.items()
                    if key.startswith("viewport_encoder.")
                }
            )
            with torch.inference_mode():
                probabilities = torch.softmax(encoder(states)[1], dim=-1)
            counts = torch.bincount(probabilities.argmax(dim=-1), minlength=3)
            seed_usage[str(seed)] = {
                "mean_probabilities": probabilities.mean(dim=0).tolist(),
                "argmax_fractions": (counts / counts.sum()).tolist(),
            }
        usage[label] = seed_usage
    (args.output / "router_usage.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
    return usage


def save_report(output: Path, records: list[dict[str, Any]], router_usage: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    reference = records[0]
    with (output / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "trainable_parameters", "metric", "mean", "std", "delta_vs_full_vimr"))
        for record in records:
            for metric, values in record["aggregate"].items():
                writer.writerow(
                    (
                        record["label"],
                        record["trainable_parameters"],
                        metric,
                        values["vimr_mean"],
                        values["vimr_std"],
                        values["vimr_mean"] - reference["aggregate"][metric]["vimr_mean"],
                    )
                )
    (output / "ablation_results.json").write_text(json.dumps({"models": records}, indent=2), encoding="utf-8")
    lines = [
        "# Global/Object/Detail Router and Detail Expert ablation",
        "",
        "All variants use the same 500/200 split, 3 epochs, 5-click policy, seeds, and frozen SAM modules.",
        "Safe Update uses lambda_safe=1.0, lambda_monotonic=0.5, gate init=0.1.",
        "",
        "| Model | Trainable params | Best epochs | " + " | ".join(METRICS) + " |",
        "| --- | ---: | --- | " + " | ".join("---:" for _ in METRICS) + " |",
    ]
    for record in records:
        cells = []
        for metric in METRICS:
            value = record["aggregate"][metric]
            cells.append(f"{value['vimr_mean']:.6f} ± {value['vimr_std']:.6f}")
        lines.append(
            f"| {record['label']} | {record['trainable_parameters']:,} | {record['best_epochs']} | "
            + " | ".join(cells)
            + " |"
        )
    if router_usage:
        lines.extend(
            [
                "",
                "## Router usage on fixed synthetic viewport trajectories",
                "",
                "Values are argmax fractions for global/object/detail. Stable semantic routing should not change "
                "drastically between seeds.",
                "",
                "| Model | Seed | Global | Object | Detail |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, seed_usage in router_usage.items():
            for seed, values in seed_usage.items():
                fractions = values["argmax_fractions"]
                lines.append(
                    f"| {label} | {seed} | {fractions[0]:.4f} | {fractions[1]:.4f} | {fractions[2]:.4f} |"
                )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    records = [run_variant(args, *variant) for variant in VARIANTS]
    router_usage = analyze_router_usage(args)
    save_report(args.output, records, router_usage)
    print(f"Router/detail ablation complete: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
