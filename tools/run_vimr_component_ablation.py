"""Run the cumulative Viewport/History/Residual/Safe Update ablation."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT_DIR / "tools/run_vimr_coco_protocol.py"
VARIANTS = (
    (
        "fusion_only",
        "Fusion-only",
        ("--viewport-mode", "default", "--disable-history", "--disable-residual", "--disable-safe-update"),
    ),
    ("viewport", "+Viewport", ("--disable-history", "--disable-residual", "--disable-safe-update")),
    ("history", "+History", ("--disable-residual", "--disable-safe-update")),
    ("residual", "+Residual", ("--disable-safe-update",)),
)
METRICS = (
    "mIoU@1",
    "mIoU@5",
    "NoC@85",
    "NoC@90",
    "Success@85/5",
    "Success@90/5",
    "RegressionRate",
    "MeanRegressionMagnitude",
    "LateSevereRegressionRate@4-5>0.05",
    "LateSevereRegressionRate@4-5>0.10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=ROOT_DIR / "dataset/coco")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt",
    )
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "outputs/vimr_component_ablation")
    parser.add_argument(
        "--full-results",
        type=Path,
        default=ROOT_DIR / "outputs/vimr_safe_update_sweep/S3/results.json",
    )
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


def run_variant(args: argparse.Namespace, directory: str, label: str, flags: tuple[str, ...]) -> dict[str, Any]:
    output = args.output.resolve() / directory
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
            "0",
            "--lambda-monotonic",
            "0",
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
    return {
        "label": label,
        "source": str(result_path.resolve()),
        "best_epochs": [result["seeds"][str(seed)]["best_epoch"] for seed in args.seeds],
        "trainable_parameters": [
            result["seeds"][str(seed)]["trainable_vimr_parameters"] for seed in args.seeds
        ],
        "aggregate": result["aggregate"],
    }


def load_full(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result["protocol"]
    expected = {
        "train_instances": args.train_instances,
        "validation_instances": args.eval_instances,
        "epochs": args.epochs,
        "max_clicks": args.max_clicks,
        "training_seeds": args.seeds,
        "split_seed": args.split_seed,
    }
    mismatches = {key: (protocol.get(key), value) for key, value in expected.items() if protocol.get(key) != value}
    if mismatches:
        raise ValueError(f"Full VIMR reference protocol does not match the ablation: {mismatches}")
    seeds = list(result["seeds"])
    return {
        "label": "Full VIMR",
        "source": str(path.resolve()),
        "best_epochs": [result["seeds"][seed]["best_epoch"] for seed in seeds],
        "trainable_parameters": [result["seeds"][seed]["trainable_vimr_parameters"] for seed in seeds],
        "aggregate": result["aggregate"],
    }


def save_report(output: Path, baseline: dict[str, float], models: list[dict[str, Any]]) -> None:
    records = [
        {
            "label": "Baseline",
            "source": "frozen EfficientViT-SAM baseline from the shared validation runs",
            "best_epochs": [],
            "trainable_parameters": [0],
            "aggregate": {
                metric: {"baseline": value, "vimr_mean": value, "vimr_std": 0.0, "delta_mean": 0.0}
                for metric, value in baseline.items()
            },
        },
        *models,
    ]
    (output / "ablation_results.json").write_text(json.dumps({"models": records}, indent=2), encoding="utf-8")
    with (output / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "trainable_parameters", "metric", "mean", "std", "delta_vs_baseline"))
        for record in records:
            for metric, values in record["aggregate"].items():
                writer.writerow(
                    (
                        record["label"],
                        record["trainable_parameters"][0],
                        metric,
                        values["vimr_mean"],
                        values["vimr_std"],
                        values["vimr_mean"] - baseline[metric],
                    )
                )
    lines = [
        "# VIMR-Click cumulative component ablation",
        "",
        "All trainable variants use the same split, seeds, schedule, and adaptive corrective-click policy.",
        "Fusion-only keeps the same viewport encoder and interaction-fusion capacity as +Viewport but feeds an information-free global viewport at every round.",
        "Full VIMR reuses S3 (lambda_safe=1.0, lambda_monotonic=0.5, gate init=0.1).",
        "",
        "| Model | Trainable params | Best epochs | " + " | ".join(METRICS) + " |",
        "| --- | ---: | --- | " + " | ".join("---:" for _ in METRICS) + " |",
    ]
    for record in records:
        values = [record["aggregate"].get(metric) for metric in METRICS]
        cells = ["n/a" if value is None else f"{value['vimr_mean']:.6f} ± {value['vimr_std']:.6f}" for value in values]
        lines.append(
            f"| {record['label']} | {record['trainable_parameters'][0]:,} | {record['best_epochs']} | "
            + " | ".join(cells)
            + " |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    models = [run_variant(args, *variant) for variant in VARIANTS]
    full = load_full(args.full_results, args)
    models.append(full)
    first = json.loads((args.output / VARIANTS[0][0] / "results.json").read_text(encoding="utf-8"))
    save_report(args.output, first["baseline"], models)
    print(f"Component ablation complete: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
