"""Controlled temporal interventions and adapters for recorded human prefixes."""

from collections import defaultdict
import copy

import numpy as np
import torch

from .dataset import COCOClickProcessDataset, encode_process_events, viewport_state


def temporal_intervention(inputs, kind, seed=0, process_mean=None):
    """Keep prompts, dense mask, state geometry and spatial raster support fixed.

    These are inference sensitivity diagnostics, not additional training groups.
    Order perturbation stays within move runs, viewport states and recent-window
    strata. Semantic zoom/pan/click events and the current click stay in place.
    """
    result = dict(inputs)
    stream = inputs["process_events"]
    valid = inputs["process_event_valid"].bool()
    recent = valid & (stream[..., 6] > 0.5) & (stream[..., 0] >= -0.2)
    result["process_spatial_valid"] = recent.clone()
    changed = stream.clone()
    if kind == "history_order":
        generator = torch.Generator().manual_seed(seed)
        for batch in range(len(stream)):
            groups = defaultdict(list)
            segment = 0
            for index in range(stream.shape[1]):
                if not valid[batch, index]:
                    continue
                if stream[batch, index, 6] <= 0.5:
                    segment += 1
                    continue
                # View metadata follows the cursor event; keep it identical.
                key = (segment, bool(recent[batch, index]), tuple(stream[batch, index, 3:6].tolist()))
                groups[key].append(index)
            for indices in groups.values():
                if len(indices) < 2:
                    continue
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                donors = [indices[i] for i in permutation]
                changed[batch, indices] = stream[batch, donors]
                changed[batch, indices, 0] = stream[batch, indices, 0]
    elif kind in {"history_fast", "history_slow"}:
        changed[..., 0] *= 0.5 if kind == "history_fast" else 2.0
    elif kind == "state_timing_neutral":
        if process_mean is None:
            raise ValueError("Neutral state timing requires training-only means")
        features = inputs["process_features"].clone()
        columns = [1, 2, 10, 11, 12, 13, 14, 15, 16]
        means = features.new_tensor(process_mean)
        features[..., columns] = means[columns]
        result["process_features"] = features
    else:
        raise ValueError(f"Unknown intervention: {kind}")
    result["process_events"] = changed
    return result


def observed_human_prefix(record, task, task_records):
    """Join factual motion/actions by event ID, cutting at the CURRENT click.

    Task action/full-trajectory clocks are task-relative; click telemetry uses
    the application's monotonic clock. The linked current click anchors the two
    clocks locally. Task-log click markers define the prefix cutoff; the two
    telemetry clocks are not assumed to have one exact global offset. No final
    mask, target preview, outcome or subsequent task-log action is encoded.
    """
    actions = task["actions"]
    linked = {
        a.get("details", {}).get("event_id"): a
        for a in actions
        if a.get("type") == "click" and a.get("details", {}).get("event_id")
    }
    if record["event_id"] not in linked:
        raise ValueError("Missing current click in raw task actions")
    cutoff = float(linked[record["event_id"]]["relative_ms"])
    offset = float(record["click_timestamp_ms"]) - cutoff
    prior_records = [
        r for r in task_records if r["event_id"] in linked and linked[r["event_id"]]["relative_ms"] <= cutoff
    ]
    offsets = [float(r["click_timestamp_ms"]) - float(linked[r["event_id"]]["relative_ms"]) for r in prior_records]

    # Collect both the 100-ms task samples and higher-frequency click snapshots.
    motion = {
        float(p[0]): (float(p[1]), float(p[2])) for p in task["full_trajectory"]["points"] if float(p[0]) <= cutoff
    }
    for previous in prior_records:
        relative_click = float(linked[previous["event_id"]]["relative_ms"])
        for dt, x, y in previous.get("preceding_trajectory", {}).get("points", []):
            time = relative_click + float(dt)
            if 0 <= time <= cutoff:
                motion[time] = (float(x), float(y))
    items = [(t, 1, "move", xy, None) for t, xy in motion.items()]
    for action in actions:
        time = float(action["relative_ms"])
        if time > cutoff:
            continue
        # Contextual viewport snapshots are observed, but system events are not
        # process tokens. They only establish the geometry at earlier motion.
        if action.get("viewport"):
            items.append((time, 0, "view", None, action["viewport"]))
        if action.get("type") in {"zoom", "pan", "click"}:
            items.append((time, 2, action["type"], action.get("details", {}), action.get("viewport")))
    items.sort(key=lambda item: (item[0], item[1]))
    events, current_view = [], None
    skipped_motion = 0
    for time, _, kind, payload, snapshot in items:
        if snapshot is not None and viewport_state({"viewport": snapshot})[1]:
            current_view = snapshot
        if kind == "view":
            continue
        if current_view is None:
            skipped_motion += int(kind == "move")
            continue
        if kind == "move":
            x, y = payload
        elif kind == "click":
            x, y = payload["x_norm"], payload["y_norm"]
        else:
            x, y = current_view["center_x_norm"], current_view["center_y_norm"]
        event = {
            "timestamp_ms": time + offset,
            "type": kind,
            "x_norm": float(x),
            "y_norm": float(y),
            "viewport": copy.deepcopy(current_view),
        }
        if kind == "click":
            event["click_type"] = int(payload.get("prompt_type") == "positive")
        events.append(event)
    if not events or events[-1]["type"] != "click":
        raise ValueError("Observed prefix must end at the linked current click")
    # Avoid addition/subtraction roundoff at a large monotonic timestamp.
    events[-1]["timestamp_ms"] = float(record["click_timestamp_ms"])
    events[-1]["click_type"] = int(record["click_type"])
    # Rebuild the current observed motion interval, with no invented samples.
    earlier_clicks = [
        float(linked[r["event_id"]]["relative_ms"]) for r in prior_records if r["event_id"] != record["event_id"]
    ]
    start = max(earlier_clicks, default=0.0)
    path = [[t - cutoff, x, y] for t, (x, y) in sorted(motion.items()) if start <= t <= cutoff]
    result = dict(record)
    result["process_events"] = events
    result["preceding_trajectory"] = {
        "points": path,
        "total_duration_ms": path[-1][0] - path[0][0] if len(path) > 1 else 0.0,
    }
    result["no_initial_mask"] = True
    encoded, valid = encode_process_events(result)  # validate geometry, finite values and causality
    return result, {
        "motion_before_view_skipped": skipped_motion,
        "clock_offset_range_ms": float(np.ptp(offsets)),
        "full_prefix_events": len(events),
        "encoded_events": int(valid.sum()),
        "encoded_recent_motion": int((valid & (encoded[:, 6] > 0.5) & (encoded[:, 0] >= -0.2)).sum()),
    }


class FullProcessHumanDataset(COCOClickProcessDataset):
    def __init__(self, *args, **kwargs):
        from efficientvit.models.cais.coco_click_dataset import CognitiveTargetNormalizer

        kwargs.setdefault("source_filter", "human")
        kwargs.setdefault("cognitive_normalizer", CognitiveTargetNormalizer((0.0,) * 6, (1.0,) * 6))
        super().__init__(*args, **kwargs)

    def process_inputs(self, index):
        result = super().process_inputs(index)
        events, valid = encode_process_events(self.records[index])
        result.update(process_events=events, process_event_valid=valid)
        return result

    def __getitem__(self, index):
        result = super().__getitem__(index)
        result["no_initial_mask"] = True
        return result
