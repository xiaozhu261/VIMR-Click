import copy
import unittest

import numpy as np
import torch

from efficientvit.models.process_sam import InteractionProcessEncoder, ProcessSAMConfig
from efficientvit.models.process_sam.dataset import encode_process_events
from efficientvit.models.process_sam.simulation import SimulatedUser, viewport
from efficientvit.models.process_sam.validation import observed_human_prefix, temporal_intervention
from test_process_sam import encode, inputs


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def stream_inputs(self):
        target = np.zeros((60, 80), dtype=bool)
        target[15:50, 20:65] = True
        user = SimulatedUser(17, *target.shape)
        for step in range(3):
            user.interact(target, np.zeros_like(target), task_run_id="t", event_id=str(step))
        data = inputs(count=3)
        data.update({key: value.unsqueeze(0) for key, value in user.process_inputs().items()})
        return data

    def test_temporal_diagnostics_preserve_spatial_maps_and_current_state(self):
        encoder = InteractionProcessEncoder(ProcessSAMConfig(use_event_history=True)).eval()
        data = self.stream_inputs()
        reference = encode(encoder, data)
        for condition in ("history_order", "history_fast", "history_slow"):
            changed = temporal_intervention(data, condition, seed=19)
            actual = encode(encoder, changed)
            for key in ("point_coords", "point_labels", "viewports", "process_features"):
                self.assertTrue(torch.equal(data[key], changed[key]))
            self.assertTrue(torch.equal(reference.state_token, actual.state_token))
            self.assertTrue(torch.allclose(reference.spatial_maps, actual.spatial_maps, atol=1e-6), condition)
            self.assertFalse(torch.allclose(reference.history_token, actual.history_token), condition)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            temporal_intervention(data, "wrong")

    def test_spatial_control_is_order_and_fine_timing_invariant(self):
        config = ProcessSAMConfig(use_event_history=True, history_pooling="mean", spatial_only=True)
        encoder = InteractionProcessEncoder(config).eval()
        data = self.stream_inputs()
        reference = encode(encoder, data)
        for condition in ("history_order", "history_fast", "history_slow"):
            changed = temporal_intervention(data, condition, seed=19)
            changed["process_features"] = torch.randn_like(data["process_features"]) * 100
            actual = encode(encoder, changed)
            self.assertTrue(torch.equal(reference.state_token, actual.state_token))
            self.assertTrue(torch.allclose(reference.history_token, actual.history_token, atol=1e-6))
            self.assertTrue(torch.allclose(reference.spatial_maps, actual.spatial_maps, atol=1e-6))
        gru = InteractionProcessEncoder(ProcessSAMConfig(use_event_history=True))
        difference = sum(p.numel() for p in gru.parameters()) - sum(p.numel() for p in encoder.parameters())
        self.assertEqual(difference, 256)

    def test_human_prefix_rejects_future_and_uses_observed_view(self):
        view = viewport(2, [0.5, 0.5], 60, 80)
        record = {
            "event_id": "c",
            "click_type": 1,
            "click_timestamp_ms": 10500,
            "preceding_trajectory": {"points": [[-100, 0.4, 0.4], [0, 0.5, 0.5]]},
        }
        task = {
            "full_trajectory": {"points": [[50, 0.2, 0.3], [300, 0.4, 0.4], [700, 0.9, 0.9]]},
            "actions": [
                {"relative_ms": 100, "type": "zoom", "viewport": view},
                {
                    "relative_ms": 500,
                    "type": "click",
                    "viewport": view,
                    "details": {"event_id": "c", "prompt_type": "positive", "x_norm": 0.5, "y_norm": 0.5},
                },
                {"relative_ms": 600, "type": "pan", "viewport": viewport(4, [0.9, 0.9], 60, 80)},
            ],
        }
        result, audit = observed_human_prefix(record, task, [record])
        self.assertEqual(audit["motion_before_view_skipped"], 1)
        self.assertEqual(result["process_events"][-1]["timestamp_ms"], 10500)
        self.assertTrue(all(e["timestamp_ms"] <= 10500 for e in result["process_events"]))
        self.assertFalse(any(e["type"] == "pan" for e in result["process_events"]))
        self.assertTrue(all(e["viewport"]["zoom_scale"] == 2 for e in result["process_events"]))
        values, valid = encode_process_events(result)
        self.assertEqual(int(valid.sum()), len(result["process_events"]))
        self.assertTrue((values[..., 0] <= 0).all())
        bad = copy.deepcopy(task)
        bad["actions"][1]["details"]["event_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "Missing current"):
            observed_human_prefix(record, bad, [record])

    def test_click_focus_does_not_reveal_target_extent_given_the_same_click(self):
        # A translation-symmetric singleton error fixes the click; viewport
        # decisions must use that click and an independent behavior RNG only.
        first = np.zeros((5, 5), dtype=bool)
        first[2, 2] = True
        second = np.zeros_like(first)
        second[1:4, 1:4] = True
        for seed in range(5):
            a, b = SimulatedUser(seed, 5, 5, focus_policy="click"), SimulatedUser(seed, 5, 5, focus_policy="click")
            ra = a.interact(first, np.zeros_like(first), task_run_id="t", event_id="c")
            rb = b.interact(second, np.zeros_like(second), task_run_id="t", event_id="c")
            self.assertEqual(ra["click_x_norm"], rb["click_x_norm"])
            self.assertEqual(ra["click_y_norm"], rb["click_y_norm"])
            self.assertEqual(ra["viewport"], rb["viewport"])


if __name__ == "__main__":
    unittest.main()
