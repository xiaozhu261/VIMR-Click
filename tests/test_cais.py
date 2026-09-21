import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

from efficientvit.models.cais import (
    CAISConfig,
    CAISEfficientViTSam,
    CAISEfficientViTSamPredictor,
    cais_multitask_loss,
)
from efficientvit.models.efficientvit.sam import EfficientViTSam
from efficientvit.models.vimr import VIMRConfig, VIMREfficientViTSam
from efficientvit.sam_model_zoo import REGISTERED_CAIS_MODEL, REGISTERED_VIMR_MODEL
from efficientvit.samcore.trainer.cais_trainer import CAISTrainer
from tools.train_cais_coco_click import (
    apply_trajectory_mode,
    configure_trainable_parameters,
    shuffled_trajectory_donors,
)


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


def direct_inputs(batch_size: int = 1, click_count: int = 1) -> dict[str, torch.Tensor]:
    return {
        "image_embeddings": torch.randn(batch_size, 256, 64, 64),
        "point_coords": torch.full((batch_size, click_count, 2), 512.0),
        "point_labels": torch.ones(batch_size, click_count, dtype=torch.long),
        "click_trajectories": torch.randn(batch_size, click_count, 8, 3),
    }


class CAISTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(11)

    def test_01_cais_and_vimr_are_disjoint_model_types(self) -> None:
        cais = CAISEfficientViTSam(make_base_model(), CAISConfig(cognitive_layers=1))
        vimr = VIMREfficientViTSam(make_base_model(), VIMRConfig())
        self.assertNotIsInstance(cais, VIMREfficientViTSam)
        self.assertFalse(any("viewport" in name or "history_memory" in name for name, _ in cais.named_modules()))
        self.assertFalse(any("cognitive" in name or "trajectory" in name for name, _ in vimr.named_modules()))

    def test_02_forward_returns_mask_quality_and_cognition(self) -> None:
        model = CAISEfficientViTSam(make_base_model(), CAISConfig(cognitive_layers=1)).eval()
        with torch.no_grad():
            output = model(**direct_inputs(batch_size=2, click_count=2))
        self.assertEqual(tuple(output["final_mask"].shape), (2, 1, 256, 256))
        self.assertEqual(tuple(output["iou_predictions"].shape), (2, 1))
        self.assertEqual(tuple(output["cognitive_friction"].shape), (2,))
        self.assertEqual(tuple(output["cognitive_prompt_tokens"].shape), (2, 2, 256))
        self.assertTrue(torch.all((0 <= output["iou_predictions"]) & (output["iou_predictions"] <= 1)))
        self.assertTrue(torch.all((0 <= output["cognitive_friction"]) & (output["cognitive_friction"] <= 1)))

    def test_03_multitask_backward_reaches_only_cais_modules(self) -> None:
        model = CAISEfficientViTSam(make_base_model(), CAISConfig(cognitive_layers=1)).train()
        output = model(**direct_inputs())
        losses = cais_multitask_loss(
            output["final_mask"],
            torch.rand(1, 1, 256, 256),
            iou_prediction=output["iou_predictions"],
            cognitive_prediction=output["cognitive_friction"],
            cognitive_target=torch.tensor([0.5]),
        )
        losses["loss"].backward()
        for name in ("cognitive_prompt_encoder", "cognitive_two_way_attention", "prediction_heads"):
            self.assertTrue(any(parameter.grad is not None for parameter in getattr(model, name).parameters()), name)

    def test_03b_trajectory_ablation_ignores_motion_values(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(), CAISConfig(cognitive_layers=1, use_trajectory=False)
        ).eval()
        first = direct_inputs()
        second = dict(first)
        second["click_trajectories"] = torch.randn_like(first["click_trajectories"]) * 100.0
        with torch.no_grad():
            first_output = model(**first)
            second_output = model(**second)
        self.assertTrue(torch.allclose(first_output["final_mask"], second_output["final_mask"]))
        self.assertTrue(
            torch.allclose(first_output["cognitive_friction"], second_output["cognitive_friction"])
        )

    def test_03c_zero_gated_native_integration_preserves_sam_mask(self) -> None:
        baseline = make_base_model().eval()
        cais_base = make_base_model()
        cais_base.load_state_dict(baseline.state_dict())
        cais = CAISEfficientViTSam(
            cais_base,
            CAISConfig(
                cognitive_layers=1,
                prompt_integration="native_residual",
                attention_integration="gated_residual",
            ),
        ).eval()
        inputs = direct_inputs()
        inputs["previous_mask"] = torch.rand(1, 1, 256, 256)
        with torch.no_grad():
            epsilon = torch.finfo(inputs["previous_mask"].dtype).eps
            mask_prompt = torch.logit(inputs["previous_mask"].clamp(epsilon, 1.0 - epsilon))
            sparse, dense = baseline.prompt_encoder(
                points=(inputs["point_coords"], inputs["point_labels"]), boxes=None, masks=mask_prompt
            )
            baseline_logits, _ = baseline.mask_decoder(
                image_embeddings=inputs["image_embeddings"],
                image_pe=baseline.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            output = cais(**inputs)
        self.assertEqual(float(output["prompt_residual_gate"]), 0.0)
        self.assertEqual(float(output["image_residual_gate"]), 0.0)
        self.assertTrue(torch.allclose(output["mask_logits"], baseline_logits))

    def test_03d_native_residual_gates_are_trainable(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                prompt_integration="native_residual",
                attention_integration="gated_residual",
            ),
        )
        configure_trainable_parameters(model, train_mask_decoder=False)
        self.assertTrue(model.prompt_residual_gate.requires_grad)
        self.assertTrue(model.image_residual_gate.requires_grad)

    def test_03e_trajectory_controls_are_stratified_and_non_mutating(self) -> None:
        records = [
            {"source": "human", "prompt_history": [{}]},
            {"source": "human", "prompt_history": [{}]},
            {"source": "synthetic", "prompt_history": [{}, {}]},
            {"source": "synthetic", "prompt_history": [{}, {}]},
        ]
        donors = shuffled_trajectory_donors(records, seed=17)
        self.assertEqual(donors, shuffled_trajectory_donors(records, seed=17))
        for target, donor in donors.items():
            self.assertNotEqual(target, donor)
            self.assertEqual(records[target]["source"], records[donor]["source"])
            self.assertEqual(len(records[target]["prompt_history"]), len(records[donor]["prompt_history"]))

        sample = {
            "click_trajectories": torch.ones(1, 8, 3),
            "trajectory_valid_mask": torch.ones(1, 8, dtype=torch.bool),
        }
        zero = apply_trajectory_mode(sample, "zero")
        self.assertFalse(zero["click_trajectories"].any())
        self.assertFalse(zero["trajectory_valid_mask"].any())
        self.assertTrue(sample["click_trajectories"].all())

        donor = {
            "click_trajectories": torch.full((1, 8, 3), 2.0),
            "trajectory_valid_mask": torch.ones(1, 8, dtype=torch.bool),
        }
        shuffled = apply_trajectory_mode(sample, "shuffle", donor)
        self.assertTrue(torch.equal(shuffled["click_trajectories"], donor["click_trajectories"]))
        self.assertIsNot(shuffled["click_trajectories"], donor["click_trajectories"])

    def test_04_high_friction_relaxes_boundary_penalty(self) -> None:
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 4:12, 4:12] = 1
        prediction = torch.full_like(target, 0.5)
        low = cais_multitask_loss(prediction, target, cognitive_target=torch.tensor([0.0]))
        high = cais_multitask_loss(prediction, target, cognitive_target=torch.tensor([1.0]))
        self.assertGreater(low["loss_bce"], high["loss_bce"])
        self.assertGreater(low["boundary_weight"].max(), high["boundary_weight"].max())

    def test_05_predictor_caches_image_and_accumulates_trajectories(self) -> None:
        model = CAISEfficientViTSam(make_base_model(), CAISConfig(cognitive_layers=1)).eval()
        predictor = CAISEfficientViTSamPredictor(model)
        predictor.set_image_batch(torch.randn(1, 3, 64, 64))
        trajectory = np.zeros((1, 6, 3), dtype=np.float32)
        for index in range(2):
            result = predictor.predict(
                np.array([[20.0 + index, 20.0, float(index == 0)]], dtype=np.float32),
                click_trajectories=trajectory,
            )
        self.assertEqual(model.image_encoder.calls, 1)
        self.assertEqual(len(predictor.prompt_coords), 2)
        self.assertEqual(len(predictor.trajectory_history), 2)
        self.assertEqual(result["cognitive_vector"].shape, (256,))

    def test_05b_synthetic_rows_do_not_supervise_cognition(self) -> None:
        prediction = torch.full((2, 1, 8, 8), 0.5)
        losses = cais_multitask_loss(
            prediction,
            torch.zeros_like(prediction),
            cognitive_prediction=torch.tensor([0.9, 0.2]),
            cognitive_target=torch.tensor([0.1, 0.8]),
            cognitive_supervision_mask=torch.tensor([False, True]),
        )
        expected = F.smooth_l1_loss(torch.tensor([0.2]), torch.tensor([0.8]))
        self.assertTrue(torch.allclose(losses["loss_cognitive"], expected))
        self.assertEqual(losses["cognitive_supervision_mask"].tolist(), [False, True])

    def test_06_base_checkpoint_has_only_cais_missing_keys(self) -> None:
        baseline = make_base_model()
        cais = CAISEfficientViTSam(make_base_model(), CAISConfig(cognitive_layers=1))
        incompatible = cais.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        prefixes = ("cognitive_prompt_encoder", "cognitive_two_way_attention", "prediction_heads")
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(key.startswith(prefixes) for key in incompatible.missing_keys))

    def test_07_model_registries_and_trainers_are_separate(self) -> None:
        self.assertFalse(set(REGISTERED_CAIS_MODEL).intersection(REGISTERED_VIMR_MODEL))
        model = CAISEfficientViTSam(make_base_model(), CAISConfig(cognitive_layers=1)).train()
        trainer = CAISTrainer.__new__(CAISTrainer)
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
        self.assertTrue(any(parameter.grad is not None for parameter in model.prediction_heads.parameters()))


if __name__ == "__main__":
    unittest.main()
