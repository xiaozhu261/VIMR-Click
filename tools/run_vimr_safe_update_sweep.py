"""Run the S0-S3 Safe Update loss sweep with conservative gate initialization."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from tools.run_vimr_coco_protocol import aggregate_metrics, summarize_protocol


EXPERIMENTS = (
    ("S0", 0.0, 0.0),
    ("S1", 0.2, 0.1),
    ("S2", 0.5, 0.2),
    ("S3", 1.0, 0.5),
)
REPORT_METRICS = (
    "mIoU@1",
    "mIoU@5",
    "NoC@85",
    "NoC@90",
    "Success@85/5",
    "Success@90/5",
    "FinalSuccess@90/5",
    "RegressionRate",
    "MeanRegressionMagnitude",
    "LateRegressionRate@4-5",
    "LateMeanRegressionMagnitude@4-5",
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
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "outputs/vimr_safe_update_sweep")
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=ROOT_DIR / "outputs/vimr_coco_protocol/results.json",
        help="Previous S1 run with the original approximately 0.5 gate initialization.",
    )
    parser.add_argument("--train-instances", type=int, default=500)
    parser.add_argument("--eval-instances", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-clicks", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gate-init-probability", type=float, default=0.1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rerun-existing", action="store_true")
    return parser.parse_args()


def read_method_rows(path: Path, method: str) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != method:
                continue
            annotation_id = int(row["annotation_id"])
            record = grouped.setdefault(
                annotation_id,
                {
                    "image_id": int(row["image_id"]),
                    "annotation_id": annotation_id,
                    "category_id": int(row["category_id"]),
                    "clicks": {},
                },
            )
            record["clicks"][int(row["click"])] = float(row["iou"])
    return [
        {**record, "ious": [record["clicks"][click] for click in sorted(record["clicks"])]}
        for record in grouped.values()
    ]


def load_reference(path: Path, max_clicks: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    seeds = [int(seed) for seed in result["seeds"]]
    baseline = summarize_protocol(read_method_rows(root / f"seed_{seeds[0]}" / "per_instance.csv", "baseline"), max_clicks)
    seed_metrics = {
        seed: summarize_protocol(read_method_rows(root / f"seed_{seed}" / "per_instance.csv", "vimr"), max_clicks)
        for seed in seeds
    }
    return {
        "label": "R0-old-gate",
        "lambda_safe": 0.2,
        "lambda_monotonic": 0.1,
        "gate_init_probability": "approximately 0.5",
        "source": str(path.resolve()),
        "best_epochs": [result["seeds"][str(seed)]["best_epoch"] for seed in seeds],
        "aggregate": aggregate_metrics(baseline, seed_metrics),
    }


def run_experiment(args: argparse.Namespace, label: str, lambda_safe: float, lambda_monotonic: float) -> dict[str, Any]:
    experiment_dir = args.output.resolve() / label
    result_path = experiment_dir / "results.json"
    if args.rerun_existing or not result_path.is_file():
        command = [
            sys.executable,
            str(ROOT_DIR / "tools/run_vimr_coco_protocol.py"),
            "--coco-root",
            str(args.coco_root),
            "--checkpoint",
            str(args.checkpoint),
            "--output",
            str(experiment_dir),
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
            str(lambda_safe),
            "--lambda-monotonic",
            str(lambda_monotonic),
            "--gate-init-probability",
            str(args.gate_init_probability),
            "--progress-every",
            str(args.progress_every),
        ]
        if args.device:
            command.extend(("--device", args.device))
        print(f"Starting {label}: lambda_safe={lambda_safe}, lambda_monotonic={lambda_monotonic}", flush=True)
        subprocess.run(command, cwd=ROOT_DIR, check=True)
    else:
        print(f"Reusing completed {label}: {result_path}", flush=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "lambda_safe": lambda_safe,
        "lambda_monotonic": lambda_monotonic,
        "gate_init_probability": args.gate_init_probability,
        "source": str(result_path.resolve()),
        "best_epochs": [result["seeds"][str(seed)]["best_epoch"] for seed in args.seeds],
        "aggregate": result["aggregate"],
    }


def save_summary(output_dir: Path, experiments: list[dict[str, Any]]) -> None:
    reference = experiments[0]
    with (output_dir / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "experiment",
                "lambda_safe",
                "lambda_monotonic",
                "gate_init_probability",
                "metric",
                "baseline",
                "vimr_mean",
                "vimr_std",
                "delta_vs_baseline",
                "delta_vs_old_gate_reference",
            )
        )
        for experiment in experiments:
            for metric, values in experiment["aggregate"].items():
                reference_value = reference["aggregate"].get(metric, {}).get("vimr_mean")
                writer.writerow(
                    (
                        experiment["label"],
                        experiment["lambda_safe"],
                        experiment["lambda_monotonic"],
                        experiment["gate_init_probability"],
                        metric,
                        values["baseline"],
                        values["vimr_mean"],
                        values["vimr_std"],
                        values["delta_mean"],
                        "" if reference_value is None else values["vimr_mean"] - reference_value,
                    )
                )

    lines = [
        "# VIMR-Click Safe Update sweep",
        "",
        "`R0-old-gate` is the previous S1 loss setting with the original approximately 0.5 gate initialization.",
        "S0-S3 use the conservative gate initialization requested for this sweep.",
        "",
        "| Experiment | lambda_safe | lambda_monotonic | gate init | best epochs | "
        + " | ".join(REPORT_METRICS)
        + " |",
        "| --- | ---: | ---: | ---: | --- | " + " | ".join("---:" for _ in REPORT_METRICS) + " |",
    ]
    for experiment in experiments:
        cells = []
        for metric in REPORT_METRICS:
            values = experiment["aggregate"].get(metric)
            cells.append("n/a" if values is None else f"{values['vimr_mean']:.6f} ± {values['vimr_std']:.6f}")
        lines.append(
            f"| {experiment['label']} | {experiment['lambda_safe']} | {experiment['lambda_monotonic']} | "
            f"{experiment['gate_init_probability']} | {experiment['best_epochs']} | " + " | ".join(cells) + " |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "sweep_results.json").write_text(json.dumps({"experiments": experiments}, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    experiments = []
    reference = load_reference(args.reference_results, args.max_clicks)
    if reference is not None:
        experiments.append(reference)
    for label, lambda_safe, lambda_monotonic in EXPERIMENTS:
        experiments.append(run_experiment(args, label, lambda_safe, lambda_monotonic))
    save_summary(args.output, experiments)
    print(f"Safe Update sweep complete: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
