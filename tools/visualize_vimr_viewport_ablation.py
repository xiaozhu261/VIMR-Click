"""Interactive visual comparison for the controlled viewport-signal ablation.

The GUI replays the locked validation clicks and shows the frozen baseline plus
the four trained viewport conditions.  It can also export the same dashboard to
a PNG without opening a window.
"""

import argparse
import csv
import json
import math
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from tools.eval_vimr_coco_quick import (
    baseline_round,
    decode_mask,
    encode_image,
    freeze_pretrained_modules,
    load_split,
    target_to_sam,
)
from tools.run_vimr_coco_protocol import build_model, load_vimr_checkpoint
from tools.run_vimr_viewport_signal_ablation import mask_iou, trajectory_for_mode

MODES = ("default", "synthetic", "shuffled", "correct")
DISPLAY_NAMES = {
    "baseline": "Frozen baseline",
    "default": "Default viewport",
    "synthetic": "Synthetic viewport",
    "shuffled": "Shuffled viewport",
    "correct": "Correct oracle",
}
RANKINGS = {
    "representative": "Representative (median baseline)",
    "validation": "Validation order",
    "synthetic_wins": "Largest synthetic > shuffled",
    "shuffled_wins": "Largest shuffled > synthetic",
    "oracle_wins": "Largest oracle > synthetic",
    "hardest": "Hardest frozen-baseline cases",
}
PLOT_COLORS = {
    "baseline": "#475569",
    "default": "#64748b",
    "synthetic": "#2563eb",
    "shuffled": "#db2777",
    "correct": "#16a34a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT_DIR / "outputs/vimr_viewport_signal_ablation",
    )
    parser.add_argument("--coco-root", type=Path, default=ROOT_DIR / "dataset/coco")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt",
    )
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--ranking", choices=tuple(RANKINGS), default="representative")
    parser.add_argument("--position", type=int, default=1, help="One-based position within the ranking")
    parser.add_argument("--click", type=int, choices=(1, 2, 3, 4, 5), default=5)
    parser.add_argument("--export", type=Path, default=None, help="Export one dashboard PNG and exit")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def probability_to_original(probability: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
    """Undo the longest-side resize and bottom/right padding used by SAM."""
    scale = 256.0 / max(image_height, image_width)
    resized_height = int(image_height * scale + 0.5)
    resized_width = int(image_width * scale + 0.5)
    cropped = probability[:resized_height, :resized_width]
    return cv2.resize(cropped, (image_width, image_height), interpolation=cv2.INTER_LINEAR)


def prompt_point_to_original(
    point: list[float], image_height: int, image_width: int
) -> tuple[float, float]:
    scale = 1024.0 / max(image_height, image_width)
    return point[0] / scale, point[1] / scale


def viewport_rect_to_original(
    viewport: list[float], image_height: int, image_width: int
) -> tuple[float, float, float, float]:
    """Map a square viewport in SAM's padded frame back to clipped image pixels."""
    scale = 1024.0 / max(image_height, image_width)
    zoom = max(math.exp(viewport[2]), 1e-6)
    half = 0.5 / zoom
    left = (viewport[0] - half) * 1024.0 / scale
    top = (viewport[1] - half) * 1024.0 / scale
    right = (viewport[0] + half) * 1024.0 / scale
    bottom = (viewport[1] + half) * 1024.0 / scale
    left = min(max(left, 0.0), float(image_width))
    top = min(max(top, 0.0), float(image_height))
    right = min(max(right, 0.0), float(image_width))
    bottom = min(max(bottom, 0.0), float(image_height))
    return left, top, max(right - left, 0.0), max(bottom - top, 0.0)


def read_click_curves(path: Path) -> dict[int, list[float]]:
    curves: dict[int, list[float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            annotation_id = int(row["annotation_id"])
            curves.setdefault(annotation_id, []).append(float(row["iou"]))
    return curves


def ranking_indices(
    ranking: str,
    baseline_curves: list[list[float]],
    synthetic_curves: dict[int, list[float]],
    shuffled_curves: dict[int, list[float]],
    oracle_curves: dict[int, list[float]],
    annotation_ids: list[int],
) -> list[int]:
    baseline_final = np.asarray([curve[-1] for curve in baseline_curves])
    synthetic_final = np.asarray([synthetic_curves[item][-1] for item in annotation_ids])
    shuffled_final = np.asarray([shuffled_curves[item][-1] for item in annotation_ids])
    oracle_final = np.asarray([oracle_curves[item][-1] for item in annotation_ids])
    if ranking == "validation":
        return list(range(len(annotation_ids)))
    if ranking == "representative":
        median = float(np.median(baseline_final))
        return np.argsort(np.abs(baseline_final - median)).tolist()
    if ranking == "synthetic_wins":
        return np.argsort(-(synthetic_final - shuffled_final)).tolist()
    if ranking == "shuffled_wins":
        return np.argsort(-(shuffled_final - synthetic_final)).tolist()
    if ranking == "oracle_wins":
        return np.argsort(-(oracle_final - synthetic_final)).tolist()
    if ranking == "hardest":
        return np.argsort(baseline_final).tolist()
    raise ValueError(f"Unknown ranking: {ranking}")


class ViewportAblationExplorer:
    def __init__(self, results_dir: Path, coco_root: Path, checkpoint: Path, device: str) -> None:
        self.results_dir = results_dir.resolve()
        self.device = torch.device(device)
        result_data = json.loads((self.results_dir / "results.json").read_text(encoding="utf-8"))
        self.protocol = result_data["protocol"]
        canonical = json.loads(
            (self.results_dir / "canonical_interactions.json").read_text(encoding="utf-8")
        )
        self.interactions: list[dict[str, Any]] = canonical["validation"]
        _, self.samples = load_split(
            coco_root,
            self.protocol["train_instances"],
            self.protocol["validation_instances"],
            self.protocol["split_seed"],
            0.01,
            0.60,
        )
        if [int(sample["annotation"]["id"]) for sample in self.samples] != [
            item["annotation_id"] for item in self.interactions
        ]:
            raise RuntimeError("COCO validation split does not match the locked interaction manifest")
        shuffle = json.loads(
            (self.results_dir / "shuffled_viewport_manifest.json").read_text(encoding="utf-8")
        )
        self.shuffled_sources: list[int] = shuffle["validation_source_indices"]
        self.model = build_model(
            checkpoint,
            self.device,
            42,
            lambda_safe=0.0,
            lambda_monotonic=0.0,
            use_viewport=True,
            use_history=False,
            use_residual=False,
            use_safe_update=False,
        )
        freeze_pretrained_modules(self.model)
        self.model.eval()
        self.annotation_ids = [item["annotation_id"] for item in self.interactions]
        self.baseline_curves = [item["baseline_ious"] for item in self.interactions]
        self.curves: dict[int, dict[str, dict[int, list[float]]]] = {}
        for seed in self.protocol["training_seeds"]:
            self.curves[seed] = {
                mode: read_click_curves(self.results_dir / mode / f"seed_{seed}" / "per_instance.csv")
                for mode in MODES
            }

    def order(self, ranking: str, seed: int) -> list[int]:
        return ranking_indices(
            ranking,
            self.baseline_curves,
            self.curves[seed]["synthetic"],
            self.curves[seed]["shuffled"],
            self.curves[seed]["correct"],
            self.annotation_ids,
        )

    def resolve_index(self, ranking: str, seed: int, position: int) -> int:
        order = self.order(ranking, seed)
        return order[min(max(position - 1, 0), len(order) - 1)]

    @lru_cache(maxsize=12)
    def infer(self, sample_index: int, seed: int) -> dict[str, Any]:
        sample = self.samples[sample_index]
        interaction = self.interactions[sample_index]
        image = np.asarray(Image.open(sample["path"]).convert("RGB")).copy()
        decoded_target = decode_mask(sample)
        gt = decoded_target.astype(bool)
        target = target_to_sam(decoded_target, self.device)
        embedding = encode_image(self.model, sample["path"], self.device)
        low_res_masks: dict[str, list[np.ndarray]] = {}
        curves: dict[str, list[float]] = {"baseline": interaction["baseline_ious"]}
        trajectories: dict[str, list[list[float]] | None] = {"baseline": None}
        replay_drifts: dict[str, float] = {}

        with torch.inference_mode():
            previous = None
            accumulated_coords = []
            accumulated_labels = []
            baseline_masks = []
            for coords_list, labels_list in zip(
                interaction["click_coords"], interaction["click_labels"]
            ):
                coords = torch.tensor(
                    coords_list, device=self.device, dtype=target.dtype
                ).reshape(1, 1, 2)
                labels = torch.tensor(
                    labels_list, device=self.device, dtype=torch.long
                ).reshape(1, 1)
                accumulated_coords.append(coords)
                accumulated_labels.append(labels)
                probability = baseline_round(
                    self.model,
                    embedding,
                    torch.cat(accumulated_coords, dim=1),
                    torch.cat(accumulated_labels, dim=1),
                    previous,
                )
                baseline_masks.append(probability[0, 0].detach().cpu().numpy())
                previous = probability.detach()
            low_res_masks["baseline"] = baseline_masks

            for mode in MODES:
                load_vimr_checkpoint(
                    self.model,
                    self.results_dir / mode / f"seed_{seed}" / "best_vimr_modules.pt",
                )
                trajectory = trajectory_for_mode(
                    mode,
                    sample_index,
                    self.interactions,
                    self.shuffled_sources,
                    self.protocol["max_clicks"],
                )
                trajectories[mode] = trajectory
                previous = None
                masks = []
                measured_curve = []
                for coords_list, labels_list, viewport_list in zip(
                    interaction["click_coords"], interaction["click_labels"], trajectory
                ):
                    coords = torch.tensor(
                        coords_list, device=self.device, dtype=target.dtype
                    ).reshape(1, 1, 2)
                    labels = torch.tensor(
                        labels_list, device=self.device, dtype=torch.long
                    ).reshape(1, 1)
                    viewport = torch.tensor(
                        viewport_list, device=self.device, dtype=target.dtype
                    ).reshape(1, 7)
                    probability = self.model(
                        image_embeddings=embedding,
                        point_coords=coords,
                        point_labels=labels,
                        previous_mask=previous,
                        viewport_state=viewport,
                        multimask_output=False,
                    )["final_mask"]
                    masks.append(probability[0, 0].detach().cpu().numpy())
                    measured_curve.append(mask_iou(probability, target))
                    previous = probability.detach()
                stored_curve = self.curves[seed][mode][interaction["annotation_id"]]
                max_difference = float(
                    np.max(np.abs(np.asarray(measured_curve) - np.asarray(stored_curve)))
                )
                replay_drifts[mode] = max_difference
                if max_difference > 0.02:
                    warnings.warn(
                        f"Large replay drift for {mode}, seed {seed}, sample {sample_index}: "
                        f"max |delta IoU|={max_difference:.6f}. The displayed mask uses the current "
                        "environment; the experiment CSV remains the authoritative reported metric.",
                        RuntimeWarning,
                    )
                low_res_masks[mode] = masks
                curves[mode] = measured_curve

        height, width = image.shape[:2]
        masks_original = {
            mode: [probability_to_original(mask, height, width) for mask in masks]
            for mode, masks in low_res_masks.items()
        }
        return {
            "sample_index": sample_index,
            "sample": sample,
            "interaction": interaction,
            "image": image,
            "gt": gt,
            "masks": masks_original,
            "curves": curves,
            "trajectories": trajectories,
            "replay_drifts": replay_drifts,
            "max_replay_drift": max(replay_drifts.values(), default=0.0),
            "shuffled_source_annotation_id": self.interactions[
                self.shuffled_sources[sample_index]
            ]["annotation_id"],
        }


def confusion_overlay(image: np.ndarray, gt: np.ndarray, probability: np.ndarray | None) -> np.ndarray:
    output = image.astype(np.float32).copy()
    if probability is None:
        color = np.asarray([34, 197, 94], dtype=np.float32)
        output[gt] = 0.55 * output[gt] + 0.45 * color
        return output.astype(np.uint8)
    prediction = probability >= 0.5
    categories = (
        (prediction & gt, np.asarray([34, 197, 94], dtype=np.float32)),
        (prediction & ~gt, np.asarray([239, 68, 68], dtype=np.float32)),
        (~prediction & gt, np.asarray([245, 158, 11], dtype=np.float32)),
    )
    for mask, color in categories:
        output[mask] = 0.45 * output[mask] + 0.55 * color
    return output.astype(np.uint8)


def draw_clicks(axis: Any, interaction: dict[str, Any], click_index: int, height: int, width: int) -> None:
    for index in range(click_index + 1):
        point = interaction["click_coords"][index][0]
        label = interaction["click_labels"][index][0]
        x, y = prompt_point_to_original(point, height, width)
        current = index == click_index
        axis.scatter(
            [x],
            [y],
            marker="o" if label == 1 else "X",
            s=85 if current else 38,
            c="#22c55e" if label == 1 else "#ef4444",
            edgecolors="white",
            linewidths=1.2,
            zorder=4,
        )


def render_dashboard(result: dict[str, Any], seed: int, click: int) -> Figure:
    click_index = click - 1
    image = result["image"]
    gt = result["gt"]
    height, width = image.shape[:2]
    figure = Figure(figsize=(15.5, 11), dpi=110, constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 0.82))
    panel_specs = (
        ("gt", "Image + GT", 0, 0),
        ("baseline", DISPLAY_NAMES["baseline"], 0, 1),
        ("default", DISPLAY_NAMES["default"], 0, 2),
        ("synthetic", DISPLAY_NAMES["synthetic"], 1, 0),
        ("shuffled", DISPLAY_NAMES["shuffled"], 1, 1),
        ("correct", DISPLAY_NAMES["correct"], 1, 2),
    )
    for mode, name, row, column in panel_specs:
        axis = figure.add_subplot(grid[row, column])
        probability = None if mode == "gt" else result["masks"][mode][click_index]
        axis.imshow(confusion_overlay(image, gt, probability))
        axis.contour(gt.astype(np.uint8), levels=[0.5], colors="white", linewidths=0.65)
        draw_clicks(axis, result["interaction"], click_index, height, width)
        if mode not in {"gt", "baseline"}:
            viewport = result["trajectories"][mode][click_index]
            x, y, viewport_width, viewport_height = viewport_rect_to_original(viewport, height, width)
            axis.add_patch(
                Rectangle(
                    (x, y),
                    viewport_width,
                    viewport_height,
                    fill=False,
                    edgecolor="#38bdf8",
                    linewidth=2.0,
                )
            )
        title = name if mode == "gt" else f"{name}  IoU={result['curves'][mode][click_index]:.3f}"
        axis.set_title(title, fontweight="medium")
        axis.set_axis_off()

    curve_axis = figure.add_subplot(grid[2, :2])
    rounds = np.arange(1, 6)
    for mode in ("baseline", *MODES):
        curve_axis.plot(
            rounds,
            result["curves"][mode],
            marker="o",
            linewidth=2,
            color=PLOT_COLORS[mode],
            label=DISPLAY_NAMES[mode],
        )
    curve_axis.axvline(click, color="#94a3b8", linewidth=1, linestyle="--")
    curve_axis.set_xlim(1, 5)
    curve_axis.set_ylim(0, 1)
    curve_axis.set_xticks(rounds)
    curve_axis.set_xlabel("Click round")
    curve_axis.set_ylabel("IoU")
    curve_axis.set_title("Per-click IoU trajectory", fontweight="medium")
    curve_axis.grid(alpha=0.2)
    curve_axis.legend(ncol=3, fontsize=9, loc="lower right")

    details_axis = figure.add_subplot(grid[2, 2])
    details_axis.set_axis_off()
    interaction = result["interaction"]
    click_label = "positive" if interaction["click_labels"][click_index][0] == 1 else "negative"
    lines = [
        f"Image ID       {interaction['image_id']}",
        f"Annotation ID  {interaction['annotation_id']}",
        f"Seed           {seed}",
        f"Click          {click}/5 ({click_label})",
        "",
        "Viewport at selected click",
    ]
    for mode in MODES:
        state = result["trajectories"][mode][click_index]
        lines.append(
            f"{mode:<9} center=({state[0]:.2f},{state[1]:.2f}) "
            f"zoom={math.exp(state[2]):.2f} action={int(state[6])}"
        )
    lines.extend(("", f"Shuffled source annotation: {result['shuffled_source_annotation_id']}"))
    details_axis.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9.5)
    legend = [
        Patch(facecolor="#22c55e", label="True positive"),
        Patch(facecolor="#ef4444", label="False positive"),
        Patch(facecolor="#f59e0b", label="False negative"),
        Patch(facecolor="#38bdf8", label="Viewport box"),
    ]
    details_axis.legend(handles=legend, loc="lower left", frameon=False, fontsize=9)
    figure.suptitle(
        f"Viewport ablation replay — seed {seed}, click {click}/5",
        fontweight="medium",
    )
    return figure


class ExplorerWindow:
    def __init__(
        self,
        explorer: ViewportAblationExplorer,
        seed: int,
        ranking: str,
        position: int,
        click: int,
    ) -> None:
        import tkinter as tk
        from tkinter import filedialog, ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.filedialog = filedialog
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.explorer = explorer
        self.root = tk.Tk()
        self.root.title("VIMR Viewport Ablation Explorer")
        self.root.geometry("1500x1000")
        self.seed = tk.IntVar(value=seed)
        self.ranking_label = tk.StringVar(value=RANKINGS[ranking])
        self.position = tk.IntVar(value=position)
        self.click = tk.IntVar(value=click)
        self.status = tk.StringVar(value="Ready")
        self.current_result: dict[str, Any] | None = None
        self.canvas = None

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")
        ttk.Label(controls, text="Seed").pack(side="left")
        seed_box = ttk.Combobox(
            controls,
            width=5,
            state="readonly",
            values=self.explorer.protocol["training_seeds"],
            textvariable=self.seed,
        )
        seed_box.pack(side="left", padx=(4, 14))
        ttk.Label(controls, text="Case order").pack(side="left")
        ranking_box = ttk.Combobox(
            controls,
            width=34,
            state="readonly",
            values=list(RANKINGS.values()),
            textvariable=self.ranking_label,
        )
        ranking_box.pack(side="left", padx=(4, 14))
        ttk.Label(controls, text="Position").pack(side="left")
        position_box = ttk.Spinbox(
            controls,
            from_=1,
            to=len(self.explorer.samples),
            width=6,
            textvariable=self.position,
        )
        position_box.pack(side="left", padx=(4, 8))
        ttk.Button(controls, text="Previous", command=lambda: self.move(-1)).pack(side="left")
        ttk.Button(controls, text="Next", command=lambda: self.move(1)).pack(side="left", padx=4)
        ttk.Button(controls, text="Load", command=self.load_case).pack(side="left", padx=(8, 12))
        ttk.Label(controls, text="Click").pack(side="left")
        click_scale = ttk.Scale(
            controls,
            from_=1,
            to=5,
            orient="horizontal",
            length=150,
            command=self.on_click_scale,
        )
        click_scale.set(click)
        click_scale.pack(side="left", padx=(4, 6))
        self.click_label = ttk.Label(controls, text=str(click))
        self.click_label.pack(side="left")
        ttk.Button(controls, text="Save PNG", command=self.save_png).pack(side="right")
        ttk.Label(self.root, textvariable=self.status, padding=(8, 0)).pack(fill="x")
        self.figure_frame = ttk.Frame(self.root)
        self.figure_frame.pack(fill="both", expand=True)
        seed_box.bind("<<ComboboxSelected>>", lambda _event: self.load_case())
        ranking_box.bind("<<ComboboxSelected>>", lambda _event: self.load_case())
        position_box.bind("<Return>", lambda _event: self.load_case())
        self.root.bind("<Left>", lambda _event: self.move(-1))
        self.root.bind("<Right>", lambda _event: self.move(1))
        self.root.after(50, self.load_case)

    def ranking_key(self) -> str:
        reverse = {label: key for key, label in RANKINGS.items()}
        return reverse[self.ranking_label.get()]

    def move(self, delta: int) -> None:
        value = min(max(self.position.get() + delta, 1), len(self.explorer.samples))
        self.position.set(value)
        self.load_case()

    def on_click_scale(self, value: str) -> None:
        click = min(max(int(round(float(value))), 1), 5)
        if click != self.click.get():
            self.click.set(click)
            self.click_label.configure(text=str(click))
            if self.current_result is not None:
                self.redraw()

    def load_case(self) -> None:
        seed = int(self.seed.get())
        position = min(max(int(self.position.get()), 1), len(self.explorer.samples))
        self.position.set(position)
        sample_index = self.explorer.resolve_index(self.ranking_key(), seed, position)
        self.status.set(f"Loading validation sample {sample_index + 1}/200 on {self.explorer.device}...")
        self.root.update_idletasks()
        try:
            self.current_result = self.explorer.infer(sample_index, seed)
            annotation = self.current_result["interaction"]["annotation_id"]
            self.status.set(
                f"Loaded annotation {annotation} · ranking position {position}/200 · "
                f"max replay drift {self.current_result['max_replay_drift']:.4f} · use ←/→ to browse"
            )
            self.redraw()
        except Exception as error:
            self.status.set(f"Error: {error}")
            raise

    def redraw(self) -> None:
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        figure = render_dashboard(self.current_result, int(self.seed.get()), int(self.click.get()))
        self.canvas = self.FigureCanvasTkAgg(figure, master=self.figure_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def save_png(self) -> None:
        if self.current_result is None:
            return
        path = self.filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=(("PNG image", "*.png"),),
            initialfile=f"viewport_comparison_ann_{self.current_result['interaction']['annotation_id']}.png",
        )
        if path:
            figure = render_dashboard(self.current_result, int(self.seed.get()), int(self.click.get()))
            figure.savefig(path, dpi=160)
            self.status.set(f"Saved {path}")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    if args.export is not None:
        matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    explorer = ViewportAblationExplorer(args.results, args.coco_root, args.checkpoint, args.device)
    sample_index = explorer.resolve_index(args.ranking, args.seed, args.position)
    if args.export is not None:
        result = explorer.infer(sample_index, args.seed)
        figure = render_dashboard(result, args.seed, args.click)
        args.export.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.export, dpi=160)
        print(
            f"Saved comparison dashboard: {args.export.resolve()} "
            f"(max replay drift={result['max_replay_drift']:.6f})"
        )
        return
    ExplorerWindow(explorer, args.seed, args.ranking, args.position, args.click).run()


if __name__ == "__main__":
    main()
