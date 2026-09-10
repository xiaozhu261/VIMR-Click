import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

from efficientvit.models.efficientvit.sam import EfficientViTSam
from efficientvit.models.vimr import (
    ClickHistoryMemory,
    ResidualGatedSafeUpdate,
    VIMRConfig,
    VIMREfficientViTSam,
    VIMREfficientViTSamPredictor,
    vimr_interactive_loss,
)
from efficientvit.samcore.trainer.vimr_trainer import VIMRTrainer
from tools.eval_vimr_coco_quick import freeze_pretrained_modules


class CountingImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 256, 1)
        self.calls = 0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.projection(F.adaptive_avg_pool2d(image, (64, 64)))


def make_base_model() -> EfficientViTSam:
    return EfficientViTSam(
        image_encoder=CountingImageEncoder(),
        prompt_encoder=PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
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


def direct_inputs(batch_size: int, history: int, previous: bool = True) -> dict[str, torch.Tensor]:
    data = {
        "image_embeddings": torch.randn(batch_size, 256, 64, 64),
        "point_coords": torch.tensor([[[512.0, 512.0]]] * batch_size),
        "point_labels": torch.ones(batch_size, 1, dtype=torch.long),
        "click_history": torch.rand(batch_size, history, 3),
        "viewport_history": torch.zeros(batch_size, history, 7),
    }
    data["click_history"][..., 2] = 1
    data["viewport_history"][..., :2] = 0.5
    if previous:
        data["previous_mask"] = torch.rand(batch_size, 1, 256, 256)
        data["mask_history"] = torch.rand(batch_size, max(history - 1, 1), 1, 256, 256)
    return data


class VIMRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(7)
        cls.model = VIMREfficientViTSam(make_base_model(), VIMRConfig())
        cls.model.eval()

    def test_01_baseline_checkpoint_and_forward(self) -> None:
        baseline = make_base_model().eval()
        clone = make_base_model().eval()
        clone.load_state_dict(baseline.state_dict())
        record = {
            "image": torch.randn(3, 64, 64),
            "point_coords": torch.tensor([[[512.0, 512.0]]]),
            "point_labels": torch.ones(1, 1, dtype=torch.long),
        }
        masks, scores = clone([record], multimask_output=False)
        self.assertEqual(tuple(masks.shape), (1, 1, 1, 256, 256))
        self.assertEqual(tuple(scores.shape), (1, 1, 1))

    def test_02_forward_b1_t1_shapes(self) -> None:
        with torch.no_grad():
            output = self.model(**direct_inputs(1, 1))
        self.assertEqual(tuple(output["final_mask"].shape), (1, 1, 256, 256))
        self.assertEqual(tuple(output["viewport_token"].shape), (1, 256))
        self.assertEqual(tuple(output["history_summary"].shape), (1, 256))
        self.assertEqual(tuple(output["residual_token"].shape), (1, 256))

    def test_03_forward_b2_t8(self) -> None:
        with torch.no_grad():
            output = self.model(**direct_inputs(2, 8))
        self.assertEqual(tuple(output["final_mask"].shape), (2, 1, 256, 256))
        self.assertEqual(tuple(output["history_tokens"].shape), (2, 8, 256))

    def test_04_no_previous_mask_uses_draft(self) -> None:
        with torch.no_grad():
            output = self.model(**direct_inputs(1, 1, previous=False))
        self.assertTrue(torch.equal(output["final_mask"], output["draft_mask"]))
        self.assertTrue(torch.equal(output["gate_map"], torch.ones_like(output["gate_map"])))

    def test_05_missing_viewport_uses_global_default(self) -> None:
        data = direct_inputs(1, 1, previous=False)
        with torch.no_grad():
            default_output = self.model(**data)
            data["viewport_state"] = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
            explicit_output = self.model(**data)
        self.assertTrue(torch.allclose(default_output["viewport_token"], explicit_output["viewport_token"]))

    def test_06_history_truncates_and_supports_padding(self) -> None:
        memory = ClickHistoryMemory(max_history=16).eval()
        clicks = torch.rand(2, 20, 3)
        clicks[..., 2] = 1
        valid = torch.ones(2, 20, dtype=torch.bool)
        valid[1, -2:] = False
        with torch.no_grad():
            tokens, summary, truncated_valid = memory(clicks, torch.randn(2, 20, 256), valid_mask=valid)
        self.assertEqual(tuple(tokens.shape), (2, 16, 256))
        self.assertEqual(tuple(summary.shape), (2, 256))
        self.assertEqual(tuple(truncated_valid.shape), (2, 16))

    def test_07_backward_reaches_vimr_modules(self) -> None:
        model = VIMREfficientViTSam(make_base_model(), VIMRConfig()).train()
        output = model(**direct_inputs(1, 2))
        losses = vimr_interactive_loss(output["final_mask"], torch.rand(1, 1, 256, 256), output["draft_mask"].detach())
        losses["loss"].backward()
        for name in ("viewport_encoder", "history_memory", "residual_analyzer", "interaction_fusion", "safe_update"):
            module = getattr(model, name)
            self.assertTrue(any(parameter.grad is not None for parameter in module.parameters()), name)

    def test_08_frozen_encoder_has_no_gradient(self) -> None:
        model = VIMREfficientViTSam(make_base_model(), VIMRConfig(freeze_image_encoder=True)).train()
        data = direct_inputs(1, 1)
        data.pop("image_embeddings")
        data["images"] = torch.randn(1, 3, 64, 64)
        model(**data)["final_mask"].mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.image_encoder.parameters()))

    def test_09_original_checkpoint_keys_remain_compatible(self) -> None:
        baseline = make_base_model()
        vimr = VIMREfficientViTSam(make_base_model(), VIMRConfig())
        incompatible = vimr.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        prefixes = ("viewport_encoder", "history_memory", "residual_analyzer", "interaction_fusion", "safe_update")
        prefixes = (*prefixes, "expert_router", "detail_refinement")
        self.assertTrue(all(key.startswith(prefixes) for key in incompatible.missing_keys))

    def test_10_three_round_predictor_encodes_image_once(self) -> None:
        model = VIMREfficientViTSam(make_base_model(), VIMRConfig()).eval()
        predictor = VIMREfficientViTSamPredictor(model)
        predictor.set_image_batch(torch.randn(1, 3, 64, 64))
        for index in range(3):
            predictor.predict(np.array([[20.0 + index, 20.0, 1.0]], dtype=np.float32))
        self.assertEqual(model.image_encoder.calls, 1)
        self.assertEqual(len(predictor.click_history), 3)

    def test_11_module_parameter_budgets(self) -> None:
        summary = self.model.parameter_summary()
        self.assertLess(summary["viewport_encoder"], 500_000)
        self.assertLess(summary["residual_analyzer"], 1_000_000)
        self.assertLess(summary["residual_gate"], 1_000_000)

    def test_12_training_adapter_backward(self) -> None:
        model = VIMREfficientViTSam(make_base_model(), VIMRConfig()).train()
        trainer = VIMRTrainer.__new__(VIMRTrainer)
        trainer.model = model
        trainer.amp = "fp32"
        trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
        result = trainer.run_step(
            {
                "image": torch.randn(1, 3, 64, 64),
                "masks": (torch.rand(1, 1, 64, 64) > 0.5).float(),
                "points": torch.tensor([[[512.0, 512.0]]]),
                "bboxs": torch.zeros(1, 1, 4),
            }
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertTrue(any(parameter.grad is not None for parameter in model.safe_update.parameters()))

    def test_13_conservative_gate_initialization(self) -> None:
        module = ResidualGatedSafeUpdate(initial_gate_probability=0.1).eval()
        previous = torch.rand(2, 1, 16, 16)
        draft = torch.rand(2, 1, 16, 16)
        zeros = torch.zeros(2, 1, 16, 16)
        with torch.no_grad():
            _, gate, _ = module(previous, draft, None, zeros, zeros, zeros, None)
        self.assertTrue(torch.allclose(gate, torch.full_like(gate, 0.1), atol=1e-6))

    def test_14_ablation_freezes_disabled_modules(self) -> None:
        model = VIMREfficientViTSam(
            make_base_model(),
            VIMRConfig(use_viewport=True, use_history=True, use_residual=False, use_safe_update=False),
        )
        trainable = freeze_pretrained_modules(model)
        expected = sum(parameter.numel() for parameter in model.viewport_encoder.parameters())
        expected += sum(parameter.numel() for parameter in model.history_memory.parameters())
        expected += sum(parameter.numel() for parameter in model.interaction_fusion.parameters())
        self.assertEqual(sum(parameter.numel() for parameter in trainable), expected)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.residual_analyzer.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.safe_update.parameters()))

    def test_15_router_and_detail_expert_are_active_and_shape_safe(self) -> None:
        model = VIMREfficientViTSam(
            make_base_model(),
            VIMRConfig(enable_expert_router=True, enable_detail_refinement=True),
        ).train()
        output = model(**direct_inputs(2, 2))
        self.assertEqual(tuple(output["router_probabilities"].shape), (2, 3))
        self.assertTrue(torch.allclose(output["router_probabilities"].sum(dim=-1), torch.ones(2), atol=1e-6))
        self.assertEqual(tuple(output["expert_tokens"].shape), (2, 3, 256))
        self.assertEqual(tuple(output["detail_delta_map"].shape), (2, 1, 256, 256))
        output["final_mask"].mean().backward()
        for name in ("expert_router", "detail_refinement"):
            self.assertTrue(any(parameter.grad is not None for parameter in getattr(model, name).parameters()), name)

    def test_16_new_modules_initialize_as_identity(self) -> None:
        model = VIMREfficientViTSam(
            make_base_model(),
            VIMRConfig(enable_expert_router=True, enable_detail_refinement=True),
        ).eval()
        data = direct_inputs(1, 2)
        with torch.no_grad():
            output = model(**data)
        self.assertTrue(torch.count_nonzero(output["expert_tokens"]) == 0)
        self.assertTrue(torch.count_nonzero(output["detail_delta_map"]) == 0)


if __name__ == "__main__":
    unittest.main()
