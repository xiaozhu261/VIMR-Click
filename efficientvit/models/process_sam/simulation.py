"""Procedural users with observed prefixes; GT is used only by the user policy."""

import json
from pathlib import Path

import cv2
import numpy as np


def stream_coco_annotations(path: Path):
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                raise ValueError("COCO annotation array not found")
            buffer += chunk
            index = buffer.find('"annotations"')
            if index >= 0:
                index = buffer.find("[", index)
                if index >= 0:
                    buffer, position = buffer[index + 1 :], 0
                    break
            buffer = buffer[-64:]
        while True:
            while position < len(buffer) and buffer[position] in " \r\n\t,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                chunk = handle.read(1 << 20)
                if not chunk:
                    raise
                buffer, position = buffer[position:] + chunk, 0
                continue
            position = end
            yield value


def viewport(zoom, center, height, width):
    # A square logical viewport, fitted against the image long side at zoom=1.
    rendered_w = 768 * width / max(height, width) * zoom
    rendered_h = 768 * height / max(height, width) * zoom
    return {
        "zoom_scale": float(zoom),
        "center_x_norm": float(center[0]),
        "center_y_norm": float(center[1]),
        "viewport_width_px": 768,
        "viewport_height_px": 768,
        "canvas_render_width_logical_px": float(rendered_w),
        "canvas_render_height_logical_px": float(rendered_h),
        "device_pixel_ratio": 1.0,
    }


class SimulatedUser:
    """Error-driven correction plus imperfect focus, movement, and timing.

    The policy sees target and CURRENT prediction as a simulated annotator.
    Outputs to the model contain coordinates/actions only, never GT pixels,
    error masks, measured IoU, or behavior after the current click.
    """

    def __init__(self, seed, height, width, *, focus_policy="oracle", timing_scale=1.0, motion_points=18):
        if focus_policy not in {"oracle", "click"} or timing_scale <= 0 or motion_points < 3:
            raise ValueError("Invalid simulated-user policy")
        self.generator = np.random.default_rng(seed)
        self.behavior_generator = np.random.default_rng(seed + 104729)
        self.focus_policy = focus_policy
        self.timing_scale = timing_scale
        self.motion_points = motion_points
        self.height, self.width = height, width
        self.profile = int(self.generator.integers(3))
        self.clock = 0.0
        self.cursor = self.generator.uniform(0.1, 0.9, size=2)
        self.zoom = 1.0
        self.center = np.array([0.5, 0.5])
        self.events = []
        self.prompts = []
        self.records = []

    def _event(self, kind, xy, click_type=0):
        event = {
            "timestamp_ms": float(self.clock),
            "type": kind,
            "x_norm": float(xy[0]),
            "y_norm": float(xy[1]),
            "viewport": viewport(self.zoom, self.center, self.height, self.width),
        }
        if kind == "click":
            event["click_type"] = int(click_type)
        self.events.append(event)

    def interact(self, target, prediction, *, task_run_id, event_id):
        from .dataset import viewport_state

        candidates = []
        for label, error in [(1, target & ~prediction), (0, prediction & ~target)]:
            number, labels, stats, _ = cv2.connectedComponentsWithStats(error.astype(np.uint8), connectivity=8)
            if number > 1:
                index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                component = labels == index
                candidates.append((int(stats[index, cv2.CC_STAT_AREA]), label, component, stats[index]))
        if not candidates:
            component = target.copy()
            ys, xs = np.where(component)
            stats = np.array([xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, len(xs)])
            candidates = [(len(xs), 1, component, stats)]
        _, label, component, stats = max(candidates, key=lambda item: item[0])
        distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        # Users need not click the exact error-center argmax.
        choices = np.argwhere(distance >= max(0.65 * float(distance.max()), 0.5))
        y, x = choices[int(self.generator.integers(len(choices)))]
        click = np.array([(float(x) + 0.5) / self.width, (float(y) + 0.5) / self.height])
        focus = np.array([(stats[0] + stats[2] / 2) / self.width, (stats[1] + stats[3] / 2) / self.height])
        focus = np.clip(focus + self.generator.normal(0, 0.035, 2), 0, 1)
        extent = max(stats[2] / self.width, stats[3] / self.height)
        desired_zoom = float(np.clip(0.9 / max(extent, 0.2) * self.generator.uniform(0.85, 1.15), 1, 4.5))
        # Include a non-focus minority so process is an imperfect cue.
        if self.generator.random() < 0.2:
            desired_zoom = 1.0
            focus = np.array([0.5, 0.5])
        elif self.focus_policy == "click":
            # No GT region center/extent is used for these viewport decisions.
            focus = np.clip(click + self.behavior_generator.normal(0, 0.06, 2), 0, 1)
            desired_zoom = float(self.behavior_generator.uniform(1, 4.5))
        self.clock += float(self.generator.uniform(80, 160))
        self.zoom = desired_zoom
        self._event("zoom", self.center)
        self.clock += float(self.generator.uniform(60, 140))
        self.center = focus
        self._event("pan", self.center)
        duration = float(self.generator.uniform(550, 1400) * (0.75, 1.0, 1.4)[self.profile])
        hover = float(self.generator.uniform(80, 260) * (1.0, 1.2, 1.6)[self.profile])
        duration *= self.timing_scale
        hover *= self.timing_scale
        start_time = self.clock
        start = self.cursor.copy()
        control = np.clip((start + click) / 2 + self.generator.normal(0, 0.045, 2), 0, 1)
        trajectory = []
        for fraction in np.linspace(0, 1, self.motion_points):
            phase = float(fraction * fraction * (3 - 2 * fraction))
            xy = (1 - phase) ** 2 * start + 2 * (1 - phase) * phase * control + phase**2 * click
            self.clock = start_time + duration * float(fraction)
            self._event("move", xy)
            trajectory.append([self.clock, float(xy[0]), float(xy[1])])
        self.clock += hover
        self._event("move", click)
        trajectory.append([self.clock, *click.tolist()])
        press = float(self.generator.uniform(50, 130))
        self.clock += press
        self._event("click", click, label)
        trajectory.append([self.clock, *click.tolist()])
        relative = [[time - self.clock, px, py] for time, px, py in trajectory]
        self.prompts.append({"x_norm": float(click[0]), "y_norm": float(click[1]), "click_type": int(label)})
        speeds = np.linalg.norm(np.diff(np.array(trajectory)[:, 1:], axis=0), axis=1) / np.maximum(
            np.diff(np.array(trajectory)[:, 0]) / 1000, 1e-6
        )
        record = {
            "task_run_id": task_run_id,
            "event_id": event_id,
            "click_index": len(self.prompts),
            "click_type": int(label),
            "click_timestamp_ms": float(self.clock),
            "click_x_norm": float(click[0]),
            "click_y_norm": float(click[1]),
            "press_duration_ms": press,
            "pre_click_hover_time_ms": hover,
            "peak_speed_norm_per_s": float(speeds.max()),
            "mean_deceleration_norm_per_s2": float(
                np.maximum(-np.diff(speeds), 0).mean() * 1000 / (duration / (self.motion_points - 1))
            ),
            "hard_brake_count": int((np.diff(speeds) < -0.1).sum()),
            "direction_changes_last_500ms": 0,
            "viewport": viewport(self.zoom, self.center, self.height, self.width),
            "preceding_trajectory": {"total_duration_ms": float(duration + hover + press), "points": relative},
            "prompt_history": [dict(p) for p in self.prompts],
            "process_events": [dict(e) for e in self.events],
            "simulated_process_valid": True,
            "inference_latency_ms": 0.0,
            "cognitive_supervision": False,
            "source": "synthetic",
            "simulation_profile": self.profile,
        }
        viewport_state(record)
        self.cursor = click
        self.records.append(record)
        return record

    def process_inputs(self):
        import torch
        from efficientvit.models.cais.coco_click_dataset import process_features, _resample_trajectory_fixed_window
        from .dataset import encode_process_events, viewport_state

        summaries, paths, valids, views = [], [], [], []
        for i, r in enumerate(self.records):
            summaries.append(process_features(r, self.records[i - 1] if i else None))
            path, valid = _resample_trajectory_fixed_window(
                r["preceding_trajectory"]["points"], r["click_x_norm"], r["click_y_norm"], 16, 2000
            )
            paths.append(path)
            valids.append(valid)
            views.append(viewport_state(r)[0])
        events, event_valid = encode_process_events(self.records[-1])
        return {
            "process_features": torch.tensor(np.stack(summaries)),
            "process_feature_valid": torch.ones(len(summaries), dtype=torch.bool),
            "viewports": torch.tensor(views),
            "viewport_valid": torch.ones(len(views), dtype=torch.bool),
            "click_trajectories": torch.tensor(np.stack(paths)),
            "trajectory_valid_mask": torch.tensor(np.stack(valids)),
            "process_events": events,
            "process_event_valid": event_valid,
        }
