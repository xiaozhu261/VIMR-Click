import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

from efficientvit.models.process_sam import (
    InteractionProcessEncoder,
    ProcessAwareEfficientViTSam,
    ProcessAwareSamPredictor,
    ProcessSAMConfig,
)
from efficientvit.models.process_sam.dataset import (
    PROCESS_INPUT_KEYS,
    shuffle_process,
    shuffled_process_donors,
    viewport_state,
    encode_process_events,
)
from efficientvit.models.process_sam.metrics import boundary_iou, paired_process_delta, replay_metrics
from efficientvit.models.efficientvit.sam import EfficientViTSam


class CountingImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 256, 1)
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return self.projection(F.adaptive_avg_pool2d(image, (64, 64)))


def make_base_model():
    return EfficientViTSam(
        image_encoder=CountingImageEncoder(),
        prompt_encoder=PromptEncoder(
            embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=16
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=1, embedding_dim=256, mlp_dim=512, num_heads=8),
            transformer_dim=256,
            iou_head_depth=2,
            iou_head_hidden_dim=128,
        ),
        image_size=(1024, 64),
    )


def inputs(count=3, batch=1):
    return {
        "image_embeddings": torch.randn(batch, 256, 64, 64),
        "point_coords": torch.tensor([[[320.0, 480.0], [400.0, 500.0], [550.0, 460.0], [650.0, 480.0], [700.0, 500.0]]])
        .expand(batch, -1, -1)[:, :count]
        .clone(),
        "point_labels": torch.ones(batch, count, dtype=torch.long),
        "process_features": torch.randn(batch, count, 18),
        "process_feature_valid": torch.ones(batch, count, dtype=torch.bool),
        "viewports": torch.tensor([1.5, 0.6, 0.5, 0.4, 0.5]).expand(batch, count, -1).clone(),
        "viewport_valid": torch.ones(batch, count, dtype=torch.bool),
        "click_trajectories": torch.randn(batch, count, 8, 3) * 0.03,
        "trajectory_valid_mask": torch.ones(batch, count, 8, dtype=torch.bool),
        "coordinate_scale": torch.tensor([[1024.0, 768.0]]).expand(batch, -1).clone(),
    }


def encode(encoder, data):
    return encoder(**{key: value for key, value in data.items() if key != "image_embeddings"})


class ProcessSAMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(27)
        torch.set_num_threads(2)

    def test_original_modules_and_e0_mask_are_preserved(self):
        base = make_base_model().eval()
        model = ProcessAwareEfficientViTSam(base, ProcessSAMConfig(variant="E0")).eval()
        self.assertIs(model.image_encoder, base.image_encoder)
        self.assertIs(model.prompt_encoder, base.prompt_encoder)
        self.assertIs(model.mask_decoder, base.mask_decoder)
        data = inputs(count=2)
        data["previous_mask"] = torch.rand(1, 1, 256, 256)
        with torch.no_grad():
            probability = data["previous_mask"]
            eps = torch.finfo(probability.dtype).eps
            sparse, dense = base.prompt_encoder(
                points=(data["point_coords"], data["point_labels"]),
                boxes=None,
                masks=torch.logit(probability.clamp(eps, 1 - eps)),
            )
            expected, _ = base.mask_decoder(
                image_embeddings=data["image_embeddings"],
                image_pe=base.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            result = model(**data)
        self.assertTrue(torch.equal(expected, result["mask_logits"]))
        self.assertIsNone(result["state_token"])

    def test_three_outputs_have_expected_shapes(self):
        encoder = InteractionProcessEncoder(ProcessSAMConfig()).eval()
        result = encode(encoder, inputs(count=5, batch=2))
        self.assertEqual(tuple(result.state_token.shape), (2, 256))
        self.assertEqual(tuple(result.history_token.shape), (2, 256))
        self.assertEqual(tuple(result.spatial_features.shape), (2, 256, 64, 64))
        self.assertEqual(tuple(result.spatial_maps.shape), (2, 4, 64, 64))

    def test_simulated_process_is_deterministic_and_prefix_only(self):
        from efficientvit.models.process_sam.simulation import SimulatedUser

        truth = np.zeros((48, 64), dtype=bool)
        truth[8:39, 12:52] = True
        users = [SimulatedUser(29, 48, 64) for _ in range(2)]
        prefixes = []
        for user in users:
            for i in range(3):
                record = user.interact(truth, np.zeros_like(truth), task_run_id="a", event_id=f"e{i}")
                self.assertTrue(
                    all(e["timestamp_ms"] <= record["click_timestamp_ms"] for e in record["process_events"])
                )
                x = int(record["click_x_norm"] * 64)
                y = int(record["click_y_norm"] * 48)
                self.assertTrue(truth[y, x])
            prefixes.append(user.records)
        self.assertEqual(prefixes[0], prefixes[1])
        self.assertLess(len(prefixes[0][0]["process_events"]), len(prefixes[0][-1]["process_events"]))
        events, valid = encode_process_events(prefixes[0][-1])
        self.assertEqual(tuple(events.shape), (64, 11))
        self.assertEqual(int(valid.sum()), 64)
        self.assertEqual(int(events[:, 7].sum()), 3)
        self.assertEqual(int(events[:, 8].sum()), 3)
        self.assertEqual(int(events[:, 9].sum()), 3)
        broken = dict(prefixes[0][-1])
        broken["process_events"] = [dict(e) for e in broken["process_events"]]
        broken["process_events"][-1]["timestamp_ms"] += 1
        with self.assertRaisesRegex(ValueError, "future"):
            encode_process_events(broken)

    def test_event_history_changes_history_and_masks_padding(self):
        encoder = InteractionProcessEncoder(ProcessSAMConfig(use_event_history=True)).eval()
        first = inputs(count=1)
        first["process_events"] = torch.zeros(1, 4, 11)
        first["process_events"][0, :, 0] = torch.tensor([-0.3, -0.2, -0.1, 0])
        first["process_events"][0, :, 1:3] = 0.5
        first["process_events"][0, :, 4:6] = 1
        first["process_events"][0, :, 6] = 1
        first["process_event_valid"] = torch.ones(1, 4, dtype=torch.bool)
        second = dict(first)
        second["process_events"] = first["process_events"].clone()
        second["process_events"][0, 0, 7] = 1
        a, b = encode(encoder, first), encode(encoder, second)
        self.assertTrue(torch.equal(a.state_token, b.state_token))
        self.assertFalse(torch.allclose(a.history_token, b.history_token))
        padded = dict(first)
        padded["process_events"] = torch.cat((first["process_events"], torch.full((1, 1, 11), float("nan"))), 1)
        padded["process_event_valid"] = torch.cat(
            (first["process_event_valid"], torch.zeros(1, 1, dtype=torch.bool)), 1
        )
        c = encode(encoder, padded)
        self.assertTrue(torch.allclose(a.history_token, c.history_token))
        self.assertTrue(torch.allclose(a.spatial_maps, c.spatial_maps))
        first["process_events"][0, -1, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "follow"):
            encode(encoder, first)

    def test_native_logit_mask_prompt_is_preserved(self):
        base = make_base_model().eval()
        model = ProcessAwareEfficientViTSam(base, ProcessSAMConfig(variant="E0")).eval()
        data = inputs(count=1)
        previous = torch.randn(1, 1, 256, 256) * 5
        with torch.no_grad():
            sparse, dense = base.prompt_encoder(
                points=(data["point_coords"], data["point_labels"]), boxes=None, masks=previous
            )
            expected, _ = base.mask_decoder(
                image_embeddings=data["image_embeddings"],
                image_pe=base.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            actual = model(**data, previous_mask_logits=previous)
        self.assertTrue(torch.equal(expected, actual["mask_logits"]))

    def test_gru_uses_history_while_current_state_is_fixed(self):
        encoder = InteractionProcessEncoder(ProcessSAMConfig(variant="E2")).eval()
        first = inputs()
        second = dict(first)
        second["process_features"] = first["process_features"].clone()
        second["process_features"][:, 0] += 4
        a, b = encode(encoder, first), encode(encoder, second)
        self.assertTrue(torch.equal(a.state_token, b.state_token))
        self.assertFalse(torch.allclose(a.history_token, b.history_token))

    def test_masked_history_is_invariant_to_padding_and_nan_payloads(self):
        encoder = InteractionProcessEncoder(ProcessSAMConfig()).eval()
        data = inputs(count=2)
        padded = dict(data)
        for key in ("point_coords", "process_features", "viewports", "click_trajectories"):
            padded[key] = torch.cat((data[key], torch.full_like(data[key][:, :1], float("nan"))), dim=1)
        padded["point_labels"] = torch.cat((data["point_labels"], torch.tensor([[-1]])), dim=1)
        for key in ("process_feature_valid", "viewport_valid", "trajectory_valid_mask"):
            padded[key] = torch.cat((data[key], torch.zeros_like(data[key][:, :1])), dim=1)
        a, b = encode(encoder, data), encode(encoder, padded)
        for key in ("state_token", "history_token", "spatial_features", "spatial_maps"):
            self.assertTrue(torch.allclose(getattr(a, key), getattr(b, key), atol=1e-6), key)

    def test_zero_spatial_gate_preserves_image_and_bypass_preserves_mask(self):
        model = ProcessAwareEfficientViTSam(make_base_model(), ProcessSAMConfig(spatial_gate_init=0)).eval()
        data = inputs()
        with torch.no_grad():
            output = model(**data)
            bypass = model(**data, use_process=False)
        self.assertTrue(torch.equal(output["decoder_image_embeddings"], data["image_embeddings"]))
        self.assertEqual(float(output["spatial_gate"]), 0)
        self.assertEqual(tuple(output["process_tokens"].shape), (1, 2, 256))
        self.assertFalse(torch.allclose(output["mask_logits"], bypass["mask_logits"]))

    def test_training_gradients_reach_process_branches_only_with_frozen_base(self):
        model = ProcessAwareEfficientViTSam(make_base_model(), ProcessSAMConfig(train_mask_decoder=False)).train()
        output = model(**inputs())
        output["mask_logits"].square().mean().backward()
        for name in ("state_mlp", "history_gru", "spatial_encoder"):
            self.assertTrue(
                any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in getattr(model.process_encoder, name).parameters()
                ),
                name,
            )
        self.assertIsNotNone(model.process_fusion.alpha.grad)
        self.assertGreater(float(model.process_fusion.alpha.grad.abs()), 0)
        for name in ("image_encoder", "prompt_encoder", "mask_decoder"):
            self.assertFalse(
                any(p.requires_grad or p.grad is not None for p in getattr(model, name).parameters()), name
            )
        self.assertFalse(model.image_encoder.training)
        self.assertFalse(model.prompt_encoder.training)

    def test_spatial_maps_align_with_unpadded_image(self):
        encoder = InteractionProcessEncoder(ProcessSAMConfig()).eval()
        data = inputs(count=1)
        data["point_coords"][0, 0] = torch.tensor([800.0, 400.0])
        data["viewports"][0, 0] = torch.tensor([2.0, 0.75, 0.5, 0.25, 0.25])
        data["trajectory_valid_mask"].zero_()
        result = encode(encoder, data)
        self.assertEqual(float(result.spatial_maps[:, :, 48:].abs().sum()), 0)
        self.assertEqual(float(result.spatial_maps[:, 3].sum()), 0)
        peak = int(result.spatial_maps[0, 1].flatten().argmax())
        self.assertIn(peak % 64, (49, 50))
        self.assertEqual(float(result.spatial_maps[0, 0, 24, 48]), 1)
        self.assertEqual(float(result.spatial_maps[0, 0, 24, 10]), 0)

    def test_shuffling_preserves_prompts_masks_targets_and_is_non_mutating(self):
        data = inputs()
        sample = {key: value[0] for key, value in data.items()}
        sample["target"] = torch.rand(1, 256, 256)
        sample["previous_mask"] = torch.rand(1, 256, 256)
        donor = {key: sample[key].clone() for key in PROCESS_INPUT_KEYS}
        donor["process_features"] += 7
        donor["viewports"] *= 2
        result = shuffle_process(sample, donor)
        for key in ("point_coords", "point_labels", "image_embeddings", "target", "previous_mask"):
            self.assertIs(result[key], sample[key])
        self.assertFalse(torch.equal(result["process_features"], sample["process_features"]))
        result["process_features"].zero_()
        self.assertGreater(float(donor["process_features"].abs().sum()), 0)

    def test_donors_are_stratified_derangements_and_singletons_fail(self):
        records = [
            {"task_run_id": f"t{i}", "source": source, "prompt_history": [{}] * count}
            for source in ("human", "synthetic")
            for count in (1, 2)
            for i in range(4)
        ]
        donors = shuffled_process_donors(records, seed=9)
        self.assertEqual(donors, shuffled_process_donors(records, seed=9))
        self.assertEqual(set(donors.values()), set(range(len(records))))
        for target, donor in donors.items():
            self.assertNotEqual(records[target]["task_run_id"], records[donor]["task_run_id"])
            self.assertEqual(records[target]["source"], records[donor]["source"])
            self.assertEqual(len(records[target]["prompt_history"]), len(records[donor]["prompt_history"]))
        with self.assertRaises(ValueError):
            shuffled_process_donors(records[:1], seed=9)

    def test_metrics_do_not_fabricate_missing_clicks_and_use_complete_prefixes(self):
        rows = [
            {
                "event_id": f"e{i}",
                "task_run_id": "a",
                "click_index": i,
                "iou": 0.91 if i >= 3 else 0.7,
                "boundary_iou": 0.5,
                "latency_ms": 10.0,
            }
            for i in range(1, 6)
        ]
        rows.append(
            {
                "event_id": "b1",
                "task_run_id": "b",
                "click_index": 1,
                "iou": 0.1,
                "boundary_iou": 0.1,
                "latency_ms": 10.0,
            }
        )
        metrics = replay_metrics(rows)
        self.assertEqual(metrics["NoC@90"], 3)
        self.assertEqual(metrics["complete_prefix_tasks"], 1)
        self.assertEqual(metrics["click_coverage"]["1"], 2)
        self.assertIsNone(replay_metrics(rows[-1:])["IoU@5"])
        self.assertIsNone(replay_metrics(rows[-1:])["NoC@90"])
        self.assertAlmostEqual(paired_process_delta(rows, rows, seed=7)["mean_task_iou_delta"], 0)

    def test_boundary_iou_includes_the_image_border(self):
        full = torch.ones(20, 20, dtype=torch.bool)
        empty = torch.zeros_like(full)
        self.assertEqual(boundary_iou(full, full), 1)
        self.assertEqual(boundary_iou(full, empty), 0)

    def test_no_valid_interaction_adds_no_decoder_token(self):
        model = ProcessAwareEfficientViTSam(make_base_model(), ProcessSAMConfig()).eval()
        data = inputs(count=2)
        data["point_labels"].fill_(-1)
        with torch.no_grad():
            process = model(**data)
            native = model(**data, use_process=False)
        self.assertTrue(torch.equal(process["mask_logits"], native["mask_logits"]))

    def test_viewport_reads_recorded_geometry_and_masks_missing_values(self):
        state, valid = viewport_state(
            {
                "viewport": {
                    "zoom_scale": 2.0,
                    "center_x_norm": 0.7,
                    "center_y_norm": 0.5,
                    "viewport_width_px": 400,
                    "viewport_height_px": 300,
                    "canvas_render_width_logical_px": 800,
                    "canvas_render_height_logical_px": 600,
                }
            }
        )
        self.assertTrue(valid)
        self.assertEqual(state, [2.0, 0.7, 0.5, 0.5, 0.5])
        self.assertFalse(viewport_state({})[1])

    def test_stateful_predictor_caches_image_and_reset_clears_process(self):
        model = ProcessAwareEfficientViTSam(make_base_model(), ProcessSAMConfig()).eval()
        predictor = ProcessAwareSamPredictor(model)
        predictor.set_image(np.zeros((60, 80, 3), dtype=np.uint8))
        result = predictor.predict([40, 30, 1], viewport=[2, 0.5, 0.5, 0.5, 0.5])
        self.assertEqual(result["masks"].shape, (1, 60, 80))
        predictor.predict([50, 35, 0])
        self.assertEqual(model.image_encoder.calls, 1)
        self.assertEqual(len(predictor.clicks), 2)
        predictor.reset_interaction()
        self.assertEqual(len(predictor.clicks), 0)
        self.assertIsNone(predictor.previous_mask)


if __name__ == "__main__":
    unittest.main()
