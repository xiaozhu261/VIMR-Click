import math
import random
from collections import defaultdict

import torch

from efficientvit.models.cais.coco_click_dataset import COCOClickCAISDataset


def viewport_state(record: dict) -> tuple[list[float], bool]:
    """Use recorded viewport geometry; never estimate zoom from cursor motion.

    Render dimensions must be in the same logical-pixel units as the viewport.
    Visible width/height are image fractions and may exceed 1 before clipping.
    """
    viewport = record.get("viewport") or {}
    zoom = viewport.get("zoom_scale", record.get("zoom_scale"))
    cx, cy = viewport.get("center_x_norm"), viewport.get("center_y_norm")
    rendered_w = viewport.get("canvas_render_width_logical_px")
    rendered_h = viewport.get("canvas_render_height_logical_px")
    viewport_w, viewport_h = viewport.get("viewport_width_px"), viewport.get("viewport_height_px")
    values = (zoom, cx, cy, rendered_w, rendered_h, viewport_w, viewport_h)
    if any(value is None for value in values):
        return [0.0] * 5, False
    values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in values):
        return [0.0] * 5, False
    zoom, cx, cy, rendered_w, rendered_h, viewport_w, viewport_h = values
    if min(zoom, rendered_w, rendered_h, viewport_w, viewport_h) <= 0:
        return [0.0] * 5, False
    return [zoom, cx, cy, viewport_w / rendered_w, viewport_h / rendered_h], True


class COCOClickProcessDataset(COCOClickCAISDataset):
    """Reuse the fixed strict alignment, clicks, masks and image-grouped splits."""

    def __init__(self, *args, max_clicks=5, **kwargs):
        kwargs.setdefault("trajectory_processing", "fixed_window_process_v2")
        super().__init__(*args, **kwargs)
        self.records = [record for record in self.records if 1 <= len(record["prompt_history"]) <= max_clicks]
        if not self.records:
            raise ValueError("No records within the click budget")

    def process_inputs(self, index: int) -> dict[str, torch.Tensor]:
        # Donors access only behavioral inputs, never masks/targets/IoU labels.
        inputs = self.trajectory_inputs(index)
        states = [viewport_state(record or {}) for record in self._history_records(self.records[index])]
        inputs["viewports"] = torch.tensor([state for state, _ in states], dtype=torch.float32)
        inputs["viewport_valid"] = torch.tensor([valid for _, valid in states], dtype=torch.bool)
        return inputs

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        inputs = self.process_inputs(index)
        sample.update(inputs)
        return sample


def encode_process_events(record: dict, max_events: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode only the observed prefix; keep semantic actions when thinning motion.

    Fields: relative seconds/10, image x/y, log zoom, visible width/height,
    move/zoom/pan/click flags, click label. Time is relative to current click.
    """
    current_time = float(record["click_timestamp_ms"])
    raw = record.get("process_events") or []
    if any(float(e["timestamp_ms"]) > current_time + 1e-6 for e in raw):
        raise ValueError("Process prefix contains future events")
    if any(float(raw[i]["timestamp_ms"]) > float(raw[i + 1]["timestamp_ms"]) for i in range(len(raw) - 1)):
        raise ValueError("Process events must be chronological")
    if len(raw) > max_events:
        semantic = [i for i, e in enumerate(raw) if e["type"] != "move"]
        if len(semantic) >= max_events:
            indices = semantic[-max_events:]
        else:
            import numpy as np

            motion = [i for i, e in enumerate(raw) if e["type"] == "move"]
            chosen = np.linspace(0, len(motion) - 1, max_events - len(semantic)).round().astype(int)
            indices = sorted(semantic + [motion[i] for i in chosen])
        raw = [raw[i] for i in indices]
    values = torch.zeros(max_events, 11)
    valid = torch.zeros(max_events, dtype=torch.bool)
    types = {"move": 6, "zoom": 7, "pan": 8, "click": 9}
    for i, e in enumerate(raw):
        if e["type"] not in types:
            raise ValueError("Unknown process event type")
        viewport = e["viewport"]
        state, good = viewport_state({"viewport": viewport})
        if not good:
            raise ValueError("Synthetic event viewport is incomplete")
        values[i, :6] = torch.tensor(
            [
                (float(e["timestamp_ms"]) - current_time) / 10000,
                e["x_norm"],
                e["y_norm"],
                math.log(state[0]),
                state[3],
                state[4],
            ]
        )
        values[i, types[e["type"]]] = 1
        values[i, 10] = float(e.get("click_type", 0))
        valid[i] = True
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite process event")
    return values, valid


class SyntheticProcessDataset(COCOClickProcessDataset):
    """Explicit synthetic-only data; shared native image/click/mask transforms."""

    def __init__(self, *args, **kwargs):
        from efficientvit.models.cais.coco_click_dataset import CognitiveTargetNormalizer

        kwargs.setdefault("source_filter", "synthetic")
        # This experiment uses segmentation loss, not synthetic cognitive labels.
        kwargs.setdefault("cognitive_normalizer", CognitiveTargetNormalizer((0.0,) * 6, (1.0,) * 6))
        super().__init__(*args, **kwargs)

    def process_inputs(self, index):
        result = super().process_inputs(index)
        result["process_feature_valid"] = torch.tensor(
            [bool(r and r.get("simulated_process_valid")) for r in self._history_records(self.records[index])],
            dtype=torch.bool,
        )
        events, valid = encode_process_events(self.records[index])
        result.update(process_events=events, process_event_valid=valid)
        return result

    def __getitem__(self, index):
        import numpy as np

        result = super().__getitem__(index)
        record = self.records[index]
        result["previous_mask_logits"] = torch.from_numpy(
            np.load(self.root / record["pre_logits_ref"]).astype(np.float32)
        )
        return result


PROCESS_INPUT_KEYS = (
    "process_features",
    "process_feature_valid",
    "viewports",
    "viewport_valid",
    "click_trajectories",
    "trajectory_valid_mask",
)


def shuffled_process_donors(records: list[dict], seed: int) -> dict[int, int]:
    """Deterministic derangement within source and history-length strata."""
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[(str(record.get("source", "human")), len(record["prompt_history"]))].append(index)
    generator = random.Random(seed)
    donors = {}
    for stratum, indices in sorted(groups.items()):
        if len(indices) < 2:
            raise ValueError(f"Cannot shuffle singleton stratum {stratum}; expand evaluation data")
        generator.shuffle(indices)
        # Each task contributes one record per click count. Explicitly check the
        # task identity anyway, instead of silently allowing self-donors.
        mapping = None
        for shift in range(1, len(indices)):
            rotated = indices[shift:] + indices[:shift]
            candidate = dict(zip(indices, rotated))
            if all(records[i]["task_run_id"] != records[j]["task_run_id"] for i, j in candidate.items()):
                mapping = candidate
                break
        if mapping is None:
            raise ValueError(f"No different-task donor permutation for stratum {stratum}")
        donors.update(mapping)
    return donors


def shuffle_process(sample: dict, donor_inputs: dict) -> dict:
    """Keep recipient clicks, labels, image, pre-mask and targets exactly fixed."""
    result = dict(sample)
    for key in PROCESS_INPUT_KEYS:
        if donor_inputs[key].shape != sample[key].shape:
            raise ValueError(f"Process donor shape mismatch for {key}")
        result[key] = donor_inputs[key].clone()
    return result
