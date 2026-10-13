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

PROCESS_FEATURE_NAMES = (
    "has_previous_click",
    "log_active_interval_ms",
    "log_observed_duration_ms",
    "observed_active_coverage",
    "log_raw_point_count",
    "log_unique_xy_count",
    "path_length_norm",
    "displacement_norm",
    "trajectory_inefficiency",
    "previous_click_distance_norm",
    "log_previous_to_click_speed_norm_per_s",
    "log_pre_click_hover_time_ms",
    "log_press_duration_ms",
    "log_peak_speed_norm_per_s",
    "log_mean_deceleration_norm_per_s2",
    "log_hard_brake_count",
    "log_direction_changes_last_500ms",
    "trajectory_end_error_norm",
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


def _resample_trajectory_fixed_window(
    points: list[list[float]],
    click_x_norm: float,
    click_y_norm: float,
    sample_count: int,
    window_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a fixed pre-click clock without extrapolating unobserved motion.

    A quadratic clock allocates more samples near the click, where braking and
    hovering occur. Unlike the legacy sampler, a 68-ms fragment is not stretched
    into a complete 16-point trajectory: only grid points inside the recorded
    interval are marked valid.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    output = np.zeros((sample_count, 3), dtype=np.float32)
    valid = np.zeros(sample_count, dtype=bool)
    if not points:
        return output, valid
    trajectory = np.asarray(points, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(f"Invalid trajectory shape: {trajectory.shape}")
    order = np.argsort(trajectory[:, 0], kind="stable")
    trajectory = trajectory[order]
    _, unique_indices = np.unique(trajectory[:, 0], return_index=True)
    trajectory = trajectory[np.sort(unique_indices)]
    times = trajectory[:, 0]
    phase = np.linspace(1.0, 0.0, sample_count)
    sample_times = -float(window_ms) * np.square(phase)
    valid = (sample_times >= max(float(times[0]), -float(window_ms))) & (
        sample_times <= min(float(times[-1]), 0.0) + 1e-6
    )
    if valid.any():
        output[valid, 0] = (sample_times[valid] / 1000.0).astype(np.float32)
        output[valid, 1] = (
            np.interp(sample_times[valid], times, trajectory[:, 1]) - float(click_x_norm)
        ).astype(np.float32)
        output[valid, 2] = (
            np.interp(sample_times[valid], times, trajectory[:, 2]) - float(click_y_norm)
        ).astype(np.float32)
    # The click endpoint is a real event even when a very short fragment falls
    # between all other fixed clock samples.
    if times[0] <= 0.0 <= times[-1] + 1e-6:
        valid[-1] = True
        output[-1] = np.asarray(
            (0.0, trajectory[-1, 1] - click_x_norm, trajectory[-1, 2] - click_y_norm),
            dtype=np.float32,
        )
    return output, valid


def process_features(
    record: dict[str, Any], previous_record: Optional[dict[str, Any]]
) -> np.ndarray:
    """Summarize annotation-process evidence without inventing missing paths."""
    trajectory = record.get("preceding_trajectory") or {}
    points = trajectory.get("points") or []
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        array = np.zeros((0, 3), dtype=np.float64)
    elif array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Invalid trajectory shape: {array.shape}")
    if len(array):
        array = array[np.argsort(array[:, 0], kind="stable")]
        _, unique_indices = np.unique(array[:, 0], return_index=True)
        array = array[np.sort(unique_indices)]
    xy = array[:, 1:3]
    segments = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.zeros(0)
    path_length = float(segments.sum())
    displacement = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else 0.0
    observed_duration_ms = (
        max(float(array[-1, 0] - array[0, 0]), 0.0)
        if len(array) > 1
        else max(float(trajectory.get("total_duration_ms") or 0.0), 0.0)
    )
    unique_xy_count = len({(round(float(x), 6), round(float(y), 6)) for x, y in xy})
    has_previous = previous_record is not None
    active_interval_ms = 0.0
    previous_click_distance = 0.0
    if previous_record is not None:
        inter_click_ms = max(
            float(record.get("click_timestamp_ms") or 0.0)
            - float(previous_record.get("click_timestamp_ms") or 0.0),
            0.0,
        )
        # Model inference is system waiting time rather than human annotation
        # behavior, so remove it from the usable inter-click interval.
        active_interval_ms = max(
            inter_click_ms - max(float(previous_record.get("inference_latency_ms") or 0.0), 0.0),
            0.0,
        )
        current_xy = np.asarray(
            [float(record.get("click_x_norm") or 0.0), float(record.get("click_y_norm") or 0.0)]
        )
        previous_xy = np.asarray(
            [
                float(previous_record.get("click_x_norm") or 0.0),
                float(previous_record.get("click_y_norm") or 0.0),
            ]
        )
        previous_click_distance = float(np.linalg.norm(current_xy - previous_xy))
    coverage = (
        float(np.clip(observed_duration_ms / active_interval_ms, 0.0, 1.0))
        if active_interval_ms > 1e-6
        else 0.0
    )
    click_speed = (
        previous_click_distance / (active_interval_ms / 1000.0)
        if active_interval_ms > 1e-6
        else 0.0
    )
    end_error = 0.0
    if len(xy):
        end_error = float(
            np.linalg.norm(
                xy[-1]
                - np.asarray(
                    [float(record.get("click_x_norm") or 0.0), float(record.get("click_y_norm") or 0.0)]
                )
            )
        )
    raw = np.asarray(
        [
            float(has_previous),
            np.log1p(active_interval_ms),
            np.log1p(observed_duration_ms),
            coverage,
            np.log1p(len(points)),
            np.log1p(unique_xy_count),
            path_length,
            displacement,
            _trajectory_inefficiency(points),
            previous_click_distance,
            np.log1p(click_speed),
            np.log1p(max(float(record.get("pre_click_hover_time_ms") or 0.0), 0.0)),
            np.log1p(max(float(record.get("press_duration_ms") or 0.0), 0.0)),
            np.log1p(max(float(record.get("peak_speed_norm_per_s") or 0.0), 0.0)),
            np.log1p(max(float(record.get("mean_deceleration_norm_per_s2") or 0.0), 0.0)),
            np.log1p(max(float(record.get("hard_brake_count") or 0.0), 0.0)),
            np.log1p(max(float(record.get("direction_changes_last_500ms") or 0.0), 0.0)),
            end_error,
        ],
        dtype=np.float32,
    )
    return raw


def is_high_quality_trajectory(
    record: dict[str, Any], previous_record: Optional[dict[str, Any]]
) -> bool:
    """Target-free quality rule used for strict image-level train filtering."""
    trajectory = record.get("preceding_trajectory") or {}
    points = np.asarray(trajectory.get("points") or [], dtype=np.float64)
    if points.size == 0:
        return False
    if points.ndim != 2 or points.shape[1] != 3:
        return False
    points = points[np.argsort(points[:, 0], kind="stable")]
    _, unique_indices = np.unique(points[:, 0], return_index=True)
    points = points[np.sort(unique_indices)]
    xy = points[:, 1:3]
    duration_ms = (
        max(float(points[-1, 0] - points[0, 0]), 0.0)
        if len(points) > 1
        else max(float(trajectory.get("total_duration_ms") or 0.0), 0.0)
    )
    path_length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
    unique_xy = len({(round(float(x), 6), round(float(y), 6)) for x, y in xy})
    current_xy = np.asarray(
        [float(record.get("click_x_norm") or 0.0), float(record.get("click_y_norm") or 0.0)]
    )
    end_error = float(np.linalg.norm(xy[-1] - current_xy))
    enough_observation = len(points) >= 5 and duration_ms >= 150.0
    has_motion = unique_xy >= 3 and path_length >= 0.005
    reaches_click = end_error <= 0.01
    if previous_record is None:
        adequate_coverage = duration_ms >= 150.0
    else:
        interval_ms = max(
            float(record.get("click_timestamp_ms") or 0.0)
            - float(previous_record.get("click_timestamp_ms") or 0.0),
            0.0,
        )
        if interval_ms <= 1e-6:
            return False
        previous_xy = np.asarray(
            [
                float(previous_record.get("click_x_norm") or 0.0),
                float(previous_record.get("click_y_norm") or 0.0),
            ]
        )
        click_distance = float(np.linalg.norm(current_xy - previous_xy))
        start_error = float(np.linalg.norm(xy[0] - previous_xy))
        adequate_coverage = duration_ms / interval_ms >= 0.25 and (
            click_distance < 0.03 or start_error <= max(0.05, click_distance * 0.5)
        )
    return bool(enough_observation and has_motion and reaches_click and adequate_coverage)


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
        source_filter: Optional[str] = None,
        trajectory_samples: int = 16,
        trajectory_processing: str = "legacy",
        trajectory_window_ms: float = 2000.0,
        prompt_input_size: int = 1024,
        mask_output_size: int = 256,
        cognitive_normalizer: Optional[CognitiveTargetNormalizer] = None,
        high_quality_images_only: bool = False,
        quality_gate_trajectories: bool = False,
    ) -> None:
        self.root = Path(root)
        self.jsonl_path = self.root / "alignment" / "modeling_clicks_strict.jsonl"
        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"Missing strict alignment file: {self.jsonl_path}")
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            all_records = [json.loads(line) for line in handle if line.strip()]
        self.all_records = all_records
        task_records: dict[str, list[dict[str, Any]]] = {}
        for record in all_records:
            task_records.setdefault(record["task_run_id"], []).append(record)
        for records in task_records.values():
            records.sort(key=lambda item: int(item["click_index"]))
        candidates = [
            record
            for record in all_records
            if record.get("split") == split
            and (source_filter is None or str(record.get("source", "human")) == source_filter)
        ]
        self.high_quality_images_only = bool(high_quality_images_only)
        self.quality_gate_trajectories = bool(quality_gate_trajectories)
        self.excluded_low_quality_images = 0
        if self.high_quality_images_only:
            if split != "train":
                raise ValueError("high_quality_images_only is supported only for the train split")
            if any(str(record.get("source", "human")) != "human" for record in candidates):
                raise ValueError("high_quality_images_only requires a human-only training dataset")
            bad_images = set()
            for record in candidates:
                click_index = int(record["click_index"])
                by_index = {
                    int(item["click_index"]): item for item in task_records[record["task_run_id"]]
                }
                if not is_high_quality_trajectory(record, by_index.get(click_index - 1)):
                    bad_images.add(str(record["image_ref"]))
            candidates = [record for record in candidates if str(record["image_ref"]) not in bad_images]
            self.excluded_low_quality_images = len(bad_images)
        self.records = candidates
        if not self.records:
            raise ValueError(f"No records found for split={split!r}, source_filter={source_filter!r}")
        self.split = split
        self.source_filter = source_filter
        self.trajectory_samples = trajectory_samples
        if trajectory_processing not in {"legacy", "fixed_window_process_v2"}:
            raise ValueError(f"Unsupported trajectory_processing: {trajectory_processing}")
        self.trajectory_processing = trajectory_processing
        self.trajectory_window_ms = float(trajectory_window_ms)
        self.prompt_input_size = prompt_input_size
        self.mask_output_size = mask_output_size

        self.task_records = task_records

        if cognitive_normalizer is None:
            train_records = (
                [record for record in self.records if record.get("cognitive_supervision", True)]
                if self.high_quality_images_only
                else [
                    record
                    for record in all_records
                    if record.get("split") == "train" and record.get("cognitive_supervision", True)
                ]
            )
            if not train_records:
                raise ValueError("No cognitively supervised train records available to fit the normalizer")
            cognitive_normalizer = CognitiveTargetNormalizer.fit(train_records)
        self.cognitive_normalizer = cognitive_normalizer

    def _trajectory_passes_quality_gate(
        self, record: dict[str, Any], previous_record: Optional[dict[str, Any]]
    ) -> bool:
        return not self.quality_gate_trajectories or is_high_quality_trajectory(
            record, previous_record
        )

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
        trajectory_valid_masks = []
        process_summaries = []
        process_is_valid = []
        records_by_index = {
            int(item["click_index"]): item for item in self.task_records[record["task_run_id"]]
        }
        for prompt, click_record in zip(record["prompt_history"], self._history_records(record)):
            x_norm = float(prompt["x_norm"])
            y_norm = float(prompt["y_norm"])
            trajectory = (click_record or {}).get("preceding_trajectory") or {}
            points = trajectory.get("points") or []
            quality_valid = click_record is not None and self._trajectory_passes_quality_gate(
                click_record,
                records_by_index.get(int(click_record["click_index"]) - 1),
            )
            if self.trajectory_processing == "fixed_window_process_v2":
                samples, valid_mask = _resample_trajectory_fixed_window(
                    points,
                    x_norm,
                    y_norm,
                    self.trajectory_samples,
                    self.trajectory_window_ms,
                )
            else:
                samples = _resample_trajectory(points, x_norm, y_norm, self.trajectory_samples)
                valid_mask = np.full(self.trajectory_samples, bool(points), dtype=bool)
            if not quality_valid:
                valid_mask = np.zeros_like(valid_mask, dtype=bool)
            trajectories.append(samples)
            trajectory_valid_masks.append(valid_mask)
            if click_record is None or self.trajectory_processing == "legacy":
                process_summaries.append(np.zeros(len(PROCESS_FEATURE_NAMES), dtype=np.float32))
                process_is_valid.append(False)
            else:
                click_index = int(click_record["click_index"])
                process_summaries.append(
                    process_features(click_record, records_by_index.get(click_index - 1))
                )
                process_is_valid.append(
                    quality_valid and bool(click_record.get("cognitive_supervision", True))
                )
        return {
            "click_trajectories": torch.from_numpy(np.stack(trajectories)),
            "trajectory_valid_mask": torch.from_numpy(np.stack(trajectory_valid_masks)),
            "process_features": torch.from_numpy(np.stack(process_summaries)),
            "process_feature_valid": torch.tensor(process_is_valid, dtype=torch.bool),
        }

    def trajectory_statistics(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return mean/std over valid resampled trajectory points in this split."""
        valid_points = []
        if self.trajectory_processing == "fixed_window_process_v2":
            # Each physical click contributes once. Iterating trajectory_inputs
            # would count early clicks again in every later prompt history.
            for record in self.records:
                records_by_index = {
                    int(item["click_index"]): item
                    for item in self.task_records[record["task_run_id"]]
                }
                click_index = int(record["click_index"])
                if not self._trajectory_passes_quality_gate(
                    record, records_by_index.get(click_index - 1)
                ):
                    continue
                trajectory = record.get("preceding_trajectory") or {}
                samples, valid = _resample_trajectory_fixed_window(
                    trajectory.get("points") or [],
                    float(record.get("click_x_norm") or 0.0),
                    float(record.get("click_y_norm") or 0.0),
                    self.trajectory_samples,
                    self.trajectory_window_ms,
                )
                if valid.any():
                    valid_points.append(torch.from_numpy(samples[valid]))
        else:
            for index in range(len(self.records)):
                inputs = self.trajectory_inputs(index)
                valid = inputs["trajectory_valid_mask"]
                if valid.any():
                    valid_points.append(inputs["click_trajectories"][valid])
        if not valid_points:
            raise ValueError(f"No valid trajectories available in split={self.split!r}")
        points = torch.cat(valid_points, dim=0).float()
        mean = points.mean(dim=0)
        std = points.std(dim=0, unbiased=False).clamp_min(1e-6)
        return tuple(float(value) for value in mean), tuple(float(value) for value in std)

    def process_statistics(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Fit process normalization once per unique train click, never per repeated history slot."""
        if self.trajectory_processing == "legacy":
            return tuple(), tuple()
        features = []
        for record in self.records:
            if not bool(record.get("cognitive_supervision", True)):
                continue
            records_by_index = {
                int(item["click_index"]): item for item in self.task_records[record["task_run_id"]]
            }
            click_index = int(record["click_index"])
            if not self._trajectory_passes_quality_gate(
                record, records_by_index.get(click_index - 1)
            ):
                continue
            features.append(process_features(record, records_by_index.get(click_index - 1)))
        if not features:
            raise ValueError(f"No valid process features available in split={self.split!r}")
        matrix = np.stack(features).astype(np.float64)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.maximum(std, 1e-6)
        return tuple(float(value) for value in mean), tuple(float(value) for value in std)

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

        interaction_masks = []
        interaction_mask_valid = []
        for click_record in self._history_records(record):
            mask_ref = None if click_record is None else click_record.get("pre_mask_ref")
            mask_path = None if not mask_ref else self.root / str(mask_ref)
            if mask_path is None or not mask_path.is_file():
                history_mask = np.zeros_like(target, dtype=bool)
                interaction_mask_valid.append(False)
            else:
                history_mask = _load_binary_mask(str(mask_path))
                if history_mask.shape != target.shape:
                    raise ValueError(
                        f"Interaction-mask size {history_mask.shape} does not match target {target.shape}"
                    )
                interaction_mask_valid.append(True)
            interaction_masks.append(_resize_and_pad_mask(history_mask, self.mask_output_size))

        return {
            "event_id": record["event_id"],
            "task_run_id": record["task_run_id"],
            "click_index": int(record["click_index"]),
            "image_path": str(image_path),
            "image_ref": record["image_ref"],
            "point_coords": torch.tensor(point_coords, dtype=torch.float32),
            "point_labels": torch.tensor(point_labels, dtype=torch.long),
            "trajectory_coordinate_scale": torch.tensor(
                (resized_width, resized_height), dtype=torch.float32
            ),
            "click_trajectories": trajectory_inputs["click_trajectories"],
            "trajectory_valid_mask": trajectory_inputs["trajectory_valid_mask"],
            "process_features": trajectory_inputs["process_features"],
            "process_feature_valid": trajectory_inputs["process_feature_valid"],
            "target": _resize_and_pad_mask(target, self.mask_output_size),
            "previous_mask": _resize_and_pad_mask(previous_mask, self.mask_output_size),
            "interaction_masks": torch.stack(interaction_masks),
            "interaction_mask_valid": torch.tensor(interaction_mask_valid, dtype=torch.bool),
            "cognitive_target": torch.tensor(self.cognitive_normalizer.transform(record), dtype=torch.float32),
            "cognitive_supervision_mask": torch.tensor(
                bool(record.get("cognitive_supervision", True)), dtype=torch.bool
            ),
            "source": record.get("source", "human"),
            "original_size": (original_height, original_width),
        }

    def describe(self) -> dict[str, Any]:
        task_ids = {record["task_run_id"] for record in self.records}
        quality_eligible_clicks = 0
        source_counts: dict[str, int] = {}
        for record in self.records:
            source = str(record.get("source", "human"))
            source_counts[source] = source_counts.get(source, 0) + 1
            records_by_index = {
                int(item["click_index"]): item
                for item in self.task_records[record["task_run_id"]]
            }
            click_index = int(record["click_index"])
            quality_eligible_clicks += int(
                is_high_quality_trajectory(record, records_by_index.get(click_index - 1))
            )
        return {
            "split": self.split,
            "source_filter": self.source_filter,
            "clicks": len(self.records),
            "tasks": len(task_ids),
            "trajectory_samples": self.trajectory_samples,
            "trajectory_processing": self.trajectory_processing,
            "trajectory_window_ms": self.trajectory_window_ms,
            "high_quality_images_only": self.high_quality_images_only,
            "excluded_low_quality_images": self.excluded_low_quality_images,
            "quality_gate_trajectories": self.quality_gate_trajectories,
            "quality_eligible_trajectory_clicks": quality_eligible_clicks,
            "quality_gated_trajectory_clicks": (
                len(self.records) - quality_eligible_clicks
                if self.quality_gate_trajectories
                else 0
            ),
            "process_feature_names": list(PROCESS_FEATURE_NAMES),
            "source_clicks": source_counts,
            "cognitively_supervised_clicks": sum(
                bool(record.get("cognitive_supervision", True)) for record in self.records
            ),
            "cognitive_target": self.cognitive_normalizer.to_dict(),
        }


def load_rgb_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))
