import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

COGNITIVE_FEATURE_NAMES = (
    "seeking_time_ms",
    "pre_click_hover_time_ms",
    "hard_brake_count",
    "direction_changes_last_500ms",
    "mean_deceleration_norm_per_s2",
    "trajectory_inefficiency",
)


def _trajectory_inefficiency(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    xy = np.asarray(points, dtype=np.float64)[:, 1:3]
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path_length = float(segment_lengths.sum())
    if path_length <= 1e-8:
        return 0.0
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    return float(np.clip(1.0 - displacement / path_length, 0.0, 1.0))


def cognitive_features(record: dict[str, Any]) -> np.ndarray:
    """Return monotonic hesitation features without using the segmentation target."""
    trajectory = record.get("preceding_trajectory") or {}
    points = trajectory.get("points") or []
    raw = np.asarray(
        [
            max(float(record.get("seeking_time_ms") or 0.0), 0.0),
            max(float(record.get("pre_click_hover_time_ms") or 0.0), 0.0),
            max(float(record.get("hard_brake_count") or 0.0), 0.0),
            max(float(record.get("direction_changes_last_500ms") or 0.0), 0.0),
            max(float(record.get("mean_deceleration_norm_per_s2") or 0.0), 0.0),
            _trajectory_inefficiency(points),
        ],
        dtype=np.float64,
    )
    raw[:5] = np.log1p(raw[:5])
    return raw


@dataclass(frozen=True)
class CognitiveTargetNormalizer:
    """Train-split-only robust scaling used to construct a cognition proxy target."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    feature_names: tuple[str, ...] = COGNITIVE_FEATURE_NAMES

    @classmethod
    def fit(cls, records: Iterable[dict[str, Any]]) -> "CognitiveTargetNormalizer":
        features = np.stack([cognitive_features(record) for record in records], axis=0)
        lower = np.quantile(features, 0.05, axis=0)
        upper = np.quantile(features, 0.95, axis=0)
        upper = np.maximum(upper, lower + 1e-6)
        return cls(tuple(lower.tolist()), tuple(upper.tolist()))

    def transform(self, record: dict[str, Any]) -> float:
        features = cognitive_features(record)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        scaled = np.clip((features - lower) / (upper - lower), 0.0, 1.0)
        return float(scaled.mean())

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "definition": "mean(clip((log/derived feature - train_q05) / (train_q95 - train_q05), 0, 1))",
        }


def _resample_trajectory(
    points: list[list[float]], click_x_norm: float, click_y_norm: float, sample_count: int
) -> np.ndarray:
    """Resample [relative_ms, absolute_x_norm, absolute_y_norm] as [seconds, dx, dy]."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not points:
        return np.zeros((sample_count, 3), dtype=np.float32)
    trajectory = np.asarray(points, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(f"Invalid trajectory shape: {trajectory.shape}")
    order = np.argsort(trajectory[:, 0], kind="stable")
    trajectory = trajectory[order]
    times, unique_indices = np.unique(trajectory[:, 0], return_index=True)
    trajectory = trajectory[unique_indices]
    if len(trajectory) == 1:
        sampled = np.repeat(trajectory, sample_count, axis=0)
    else:
        sample_times = np.linspace(times[0], times[-1], sample_count)
        sampled = np.stack(
            (
                sample_times,
                np.interp(sample_times, times, trajectory[:, 1]),
                np.interp(sample_times, times, trajectory[:, 2]),
            ),
            axis=-1,
        )
    sampled[:, 0] /= 1000.0
    sampled[:, 1] -= float(click_x_norm)
    sampled[:, 2] -= float(click_y_norm)
    return sampled.astype(np.float32)


def _resize_and_pad_mask(mask: np.ndarray, output_size: int = 256) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32)).reshape(1, 1, *mask.shape)
    height, width = mask.shape
    scale = output_size / float(max(height, width))
    resized_height = max(1, int(height * scale + 0.5))
    resized_width = max(1, int(width * scale + 0.5))
    tensor = F.interpolate(tensor, (resized_height, resized_width), mode="nearest")
    tensor = F.pad(tensor, (0, output_size - resized_width, 0, output_size - resized_height))
    return tensor.squeeze(0)


@lru_cache(maxsize=512)
def _load_binary_mask(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


class COCOClickCAISDataset(Dataset):
    """Strict COCO-Click records prepared for the standalone CAIS architecture."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        trajectory_samples: int = 16,
        prompt_input_size: int = 1024,
        mask_output_size: int = 256,
        cognitive_normalizer: Optional[CognitiveTargetNormalizer] = None,
    ) -> None:
        self.root = Path(root)
        self.jsonl_path = self.root / "alignment" / "modeling_clicks_strict.jsonl"
        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"Missing strict alignment file: {self.jsonl_path}")
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            all_records = [json.loads(line) for line in handle if line.strip()]
        self.all_records = all_records
        self.records = [record for record in all_records if record.get("split") == split]
        if not self.records:
            raise ValueError(f"No records found for split={split!r}")
        self.split = split
        self.trajectory_samples = trajectory_samples
        self.prompt_input_size = prompt_input_size
        self.mask_output_size = mask_output_size

        task_records: dict[str, list[dict[str, Any]]] = {}
        for record in all_records:
            task_records.setdefault(record["task_run_id"], []).append(record)
        for records in task_records.values():
            records.sort(key=lambda item: int(item["click_index"]))
        self.task_records = task_records

        if cognitive_normalizer is None:
            train_records = [
                record
                for record in all_records
                if record.get("split") == "train" and record.get("cognitive_supervision", True)
            ]
            if not train_records:
                raise ValueError("No cognitively supervised train records available to fit the normalizer")
            cognitive_normalizer = CognitiveTargetNormalizer.fit(train_records)
        self.cognitive_normalizer = cognitive_normalizer

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _decode_ground_truth(record: dict[str, Any]) -> np.ndarray:
        decoded = mask_utils.decode(record["ground_truth_rle"])
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2)
        return np.asarray(decoded, dtype=np.uint8)

    def _history_records(self, record: dict[str, Any]) -> list[Optional[dict[str, Any]]]:
        """Align retained strict records to the complete prompt history.

        A valid strict record can refer to an earlier click excluded from the
        strict tier. That prompt remains context, while its unavailable motion
        is represented by a false trajectory-valid mask.
        """
        records_by_index = {int(item["click_index"]): item for item in self.task_records[record["task_run_id"]]}
        return [records_by_index.get(index) for index in range(1, len(record["prompt_history"]) + 1)]

    def trajectory_inputs(self, index: int) -> dict[str, torch.Tensor]:
        """Build only trajectory tensors, for controlled trajectory ablations."""
        record = self.records[index]
        trajectories = []
        trajectory_is_valid = []
        for prompt, click_record in zip(record["prompt_history"], self._history_records(record)):
            x_norm = float(prompt["x_norm"])
            y_norm = float(prompt["y_norm"])
            trajectory = (click_record or {}).get("preceding_trajectory") or {}
            points = trajectory.get("points") or []
            trajectories.append(_resample_trajectory(points, x_norm, y_norm, self.trajectory_samples))
            trajectory_is_valid.append(bool(points))
        return {
            "click_trajectories": torch.from_numpy(np.stack(trajectories)),
            "trajectory_valid_mask": torch.tensor(trajectory_is_valid, dtype=torch.bool)
            .unsqueeze(1)
            .expand(-1, self.trajectory_samples)
            .clone(),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self.root / record["image_ref"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        original_height = int(record["image_height_px"])
        original_width = int(record["image_width_px"])
        scale = self.prompt_input_size / float(max(original_height, original_width))
        resized_height = int(original_height * scale + 0.5)
        resized_width = int(original_width * scale + 0.5)

        prompt_history = record["prompt_history"]
        point_coords = []
        point_labels = []
        for prompt in prompt_history:
            x_norm = float(prompt["x_norm"])
            y_norm = float(prompt["y_norm"])
            point_coords.append((x_norm * resized_width, y_norm * resized_height))
            point_labels.append(int(prompt["click_type"]))
        trajectory_inputs = self.trajectory_inputs(index)

        target = self._decode_ground_truth(record)
        if target.shape != (original_height, original_width):
            raise ValueError(
                f"Ground-truth size {target.shape} does not match image size "
                f"{(original_height, original_width)} for {record['event_id']}"
            )
        previous_mask = _load_binary_mask(str(self.root / record["pre_mask_ref"]))
        if previous_mask.shape != target.shape:
            raise ValueError(f"Previous-mask size {previous_mask.shape} does not match target {target.shape}")

        return {
            "event_id": record["event_id"],
            "task_run_id": record["task_run_id"],
            "click_index": int(record["click_index"]),
            "image_path": str(image_path),
            "image_ref": record["image_ref"],
            "point_coords": torch.tensor(point_coords, dtype=torch.float32),
            "point_labels": torch.tensor(point_labels, dtype=torch.long),
            "click_trajectories": trajectory_inputs["click_trajectories"],
            "trajectory_valid_mask": trajectory_inputs["trajectory_valid_mask"],
            "target": _resize_and_pad_mask(target, self.mask_output_size),
            "previous_mask": _resize_and_pad_mask(previous_mask, self.mask_output_size),
            "cognitive_target": torch.tensor(self.cognitive_normalizer.transform(record), dtype=torch.float32),
            "cognitive_supervision_mask": torch.tensor(
                bool(record.get("cognitive_supervision", True)), dtype=torch.bool
            ),
            "source": record.get("source", "human"),
            "original_size": (original_height, original_width),
        }

    def describe(self) -> dict[str, Any]:
        task_ids = {record["task_run_id"] for record in self.records}
        source_counts: dict[str, int] = {}
        for record in self.records:
            source = str(record.get("source", "human"))
            source_counts[source] = source_counts.get(source, 0) + 1
        return {
            "split": self.split,
            "clicks": len(self.records),
            "tasks": len(task_ids),
            "trajectory_samples": self.trajectory_samples,
            "source_clicks": source_counts,
            "cognitively_supervised_clicks": sum(
                bool(record.get("cognitive_supervision", True)) for record in self.records
            ),
            "cognitive_target": self.cognitive_normalizer.to_dict(),
        }


def load_rgb_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))
