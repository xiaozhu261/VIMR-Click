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
from efficientvit.models.cais.coco_click_dataset import (
    _resample_trajectory_fixed_window,
    is_high_quality_trajectory,
)
from efficientvit.models.efficientvit.sam import EfficientViTSam
from efficientvit.models.vimr import VIMRConfig, VIMREfficientViTSam
from efficientvit.sam_model_zoo import REGISTERED_CAIS_MODEL, REGISTERED_VIMR_MODEL
from efficientvit.samcore.trainer.cais_trainer import CAISTrainer
from tools.train_cais_coco_click import (
    apply_trajectory_mode,
    build_optimizer_parameter_groups,
    configure_trainable_parameters,
    shuffled_trajectory_donors,
    wrong_image_trajectory_donors,
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
            "process_features": torch.ones(1, 3),
            "process_feature_valid": torch.ones(1, dtype=torch.bool),
        }
        zero = apply_trajectory_mode(sample, "zero")
        self.assertFalse(zero["click_trajectories"].any())
        self.assertFalse(zero["trajectory_valid_mask"].any())
        self.assertFalse(zero["process_features"].any())
        self.assertFalse(zero["process_feature_valid"].any())
        self.assertTrue(sample["click_trajectories"].all())

        donor = {
            "click_trajectories": torch.full((1, 8, 3), 2.0),
            "trajectory_valid_mask": torch.ones(1, 8, dtype=torch.bool),
            "process_features": torch.full((1, 3), 2.0),
            "process_feature_valid": torch.ones(1, dtype=torch.bool),
        }
        shuffled = apply_trajectory_mode(sample, "shuffle", donor)
        self.assertTrue(torch.equal(shuffled["click_trajectories"], donor["click_trajectories"]))
        self.assertIsNot(shuffled["click_trajectories"], donor["click_trajectories"])
        self.assertTrue(torch.equal(shuffled["process_features"], donor["process_features"]))

        wrong_image_records = [
            {"source": "human", "prompt_history": [{}], "image_ref": "a.jpg"},
            {"source": "human", "prompt_history": [{}], "image_ref": "b.jpg"},
            {"source": "human", "prompt_history": [{}], "image_ref": "c.jpg"},
        ]
        wrong_image = wrong_image_trajectory_donors(wrong_image_records, seed=17)
        for target, donor_index in wrong_image.items():
            self.assertNotEqual(
                wrong_image_records[target]["image_ref"],
                wrong_image_records[donor_index]["image_ref"],
            )
        controlled = apply_trajectory_mode(sample, "wrong_image", donor)
        self.assertTrue(torch.equal(controlled["click_trajectories"], donor["click_trajectories"]))

    def test_03f_missing_trajectory_has_exactly_zero_motion_vector(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_mean=(-0.35, 0.01, 0.04),
                trajectory_std=(0.36, 0.10, 0.10),
            ),
        )
        encoder = model.cognitive_prompt_encoder
        with torch.no_grad():
            encoder.trajectory_projection.bias.fill_(7.0)
            _, trajectory_vector, _, trajectory_present = encoder(
                torch.tensor([[[512.0, 512.0]]]),
                torch.ones(1, 1, dtype=torch.long),
                torch.randn(1, 1, 8, 3),
                trajectory_valid_mask=torch.zeros(1, 1, 8, dtype=torch.bool),
            )
        self.assertTrue(torch.equal(trajectory_vector, torch.zeros_like(trajectory_vector)))
        self.assertFalse(trajectory_present.any())

    def test_03fa_fixed_clock_does_not_expand_short_fragment(self) -> None:
        points = [
            [-68.0, 0.62, 0.70],
            [-66.0, 0.62, 0.70],
            [0.0, 0.62, 0.70],
        ]
        samples, valid = _resample_trajectory_fixed_window(points, 0.62, 0.70, 16, 2000.0)
        self.assertEqual(samples.shape, (16, 3))
        self.assertGreaterEqual(int(valid.sum()), 1)
        self.assertLess(int(valid.sum()), 16)
        self.assertTrue(valid[-1])

    def test_03fb_process_summary_is_masked_and_trains_its_gate(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                process_feature_dim=3,
                process_mean=(0.0, 0.0, 0.0),
                process_std=(1.0, 1.0, 1.0),
                process_gate_init=0.05,
            ),
        )
        encoder = model.cognitive_prompt_encoder
        args = (
            torch.tensor([[[512.0, 512.0]]]),
            torch.ones(1, 1, dtype=torch.long),
            torch.zeros(1, 1, 8, 3),
            torch.zeros(1, 1, 8, dtype=torch.bool),
        )
        _, vector, _, present = encoder(
            *args,
            process_features=torch.ones(1, 1, 3),
            process_feature_valid=torch.ones(1, 1, dtype=torch.bool),
        )
        self.assertTrue(present.all())
        vector.sum().backward()
        self.assertIsNotNone(encoder.process_gate.grad)
        with torch.no_grad():
            _, missing_vector, _, missing_present = encoder(
                *args,
                process_features=torch.ones(1, 1, 3),
                process_feature_valid=torch.zeros(1, 1, dtype=torch.bool),
            )
        self.assertFalse(missing_present.any())
        self.assertTrue(torch.equal(missing_vector, torch.zeros_like(missing_vector)))

    def test_03fc_high_quality_rule_rejects_truncated_fragment(self) -> None:
        complete = {
            "click_x_norm": 0.50,
            "click_y_norm": 0.50,
            "click_timestamp_ms": 1000.0,
            "preceding_trajectory": {
                "total_duration_ms": 400.0,
                "points": [
                    [-400.0, 0.30, 0.30],
                    [-300.0, 0.35, 0.35],
                    [-200.0, 0.40, 0.40],
                    [-100.0, 0.45, 0.45],
                    [0.0, 0.50, 0.50],
                ],
            },
        }
        truncated = dict(complete)
        truncated["preceding_trajectory"] = {
            "total_duration_ms": 68.0,
            "points": [
                [-68.0, 0.50, 0.50],
                [-66.0, 0.50, 0.50],
                [0.0, 0.50, 0.50],
            ],
        }
        self.assertTrue(is_high_quality_trajectory(complete, None))
        self.assertFalse(is_high_quality_trajectory(truncated, None))

    def test_03fd_trajectory_cognitive_loss_ignores_gated_click(self) -> None:
        probability = torch.full((1, 1, 8, 8), 0.5)
        losses = cais_multitask_loss(
            probability,
            torch.ones_like(probability),
            trajectory_cognitive_prediction=torch.tensor([0.9], requires_grad=True),
            cognitive_target=torch.tensor([0.1]),
            cognitive_supervision_mask=torch.tensor([True]),
            trajectory_cognitive_supervision_mask=torch.tensor([False]),
            lambda_trajectory_cognitive=0.05,
        )
        self.assertEqual(float(losses["loss_trajectory_cognitive"]), 0.0)
        self.assertFalse(losses["trajectory_cognitive_supervision_mask"].any())

    def test_03g_nonzero_prompt_gate_exposes_trajectory_gradient(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                prompt_integration="native_residual",
                attention_integration="none",
                prompt_residual_gate_init=0.05,
            ),
        ).train()
        trainable = configure_trainable_parameters(model, train_mask_decoder=True)
        groups = build_optimizer_parameter_groups(
            model,
            trainable,
            module_learning_rate=1e-4,
            gate_learning_rate=1e-3,
            decoder_learning_rate=1e-5,
            weight_decay=1e-4,
        )
        self.assertEqual(
            {group["name"]: group["lr"] for group in groups},
            {"modules": 1e-4, "gates": 1e-3, "decoder": 1e-5},
        )
        output = model(**direct_inputs())
        output["mask_logits"].mean().backward()
        self.assertAlmostEqual(float(torch.tanh(model.prompt_residual_gate).detach()), 0.05, places=6)
        trajectory_grads = [
            parameter.grad
            for parameter in model.cognitive_prompt_encoder.trajectory_cnn.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(trajectory_grads)
        self.assertTrue(any(torch.count_nonzero(gradient).item() for gradient in trajectory_grads))

    def test_03ga_prompt_encoder_has_separate_pretrained_learning_rate(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                prompt_integration="native",
                attention_integration="none",
                interaction_memory="click_mask_trajectory",
            ),
        )
        trainable = configure_trainable_parameters(
            model,
            train_mask_decoder=True,
            train_prompt_encoder=True,
        )
        groups = build_optimizer_parameter_groups(
            model,
            trainable,
            module_learning_rate=8e-4,
            gate_learning_rate=8e-4,
            decoder_learning_rate=8e-5,
            prompt_encoder_learning_rate=8e-5,
            weight_decay=1e-4,
        )
        learning_rates = {group["name"]: group["lr"] for group in groups}
        self.assertEqual(learning_rates["modules"], 8e-4)
        self.assertEqual(learning_rates["gates"], 8e-4)
        self.assertEqual(learning_rates["prompt_encoder"], 8e-5)
        self.assertEqual(learning_rates["decoder"], 8e-5)
        self.assertFalse(any(parameter.requires_grad for parameter in model.image_encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.prompt_encoder.parameters()))

    def test_03h_independent_trajectory_tokens_preserve_native_points(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_encoder="tcn_attention",
                prompt_integration="trajectory_token",
                attention_integration="none",
                prompt_residual_gate_init=0.05,
                lambda_trajectory_cognitive=0.05,
            ),
        ).eval()
        inputs = direct_inputs(click_count=2)
        native_sparse, _ = model.prompt_encoder(
            points=(inputs["point_coords"], inputs["point_labels"]), boxes=None, masks=None
        )
        captured: dict[str, torch.Tensor] = {}

        def capture_sparse(_module, _args, kwargs) -> None:
            captured["sparse"] = kwargs["sparse_prompt_embeddings"].detach()

        hook = model.mask_decoder.register_forward_pre_hook(capture_sparse, with_kwargs=True)
        try:
            with torch.no_grad():
                output = model(**inputs)
        finally:
            hook.remove()
        self.assertEqual(captured["sparse"].shape[1], native_sparse.shape[1] + 2)
        self.assertTrue(torch.allclose(captured["sparse"][:, : native_sparse.shape[1]], native_sparse))
        self.assertTrue(torch.allclose(output["trajectory_token_gate"], torch.full((1, 2), 0.05)))
        self.assertEqual(tuple(output["trajectory_cognitive_friction"].shape), (1,))

    def test_03i_invalid_trajectory_appends_no_token(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_encoder="tcn_attention",
                prompt_integration="trajectory_token",
                attention_integration="none",
                prompt_residual_gate_init=0.05,
            ),
        ).eval()
        inputs = direct_inputs()
        inputs["trajectory_valid_mask"] = torch.zeros(1, 1, 8, dtype=torch.bool)
        native_sparse, _ = model.prompt_encoder(
            points=(inputs["point_coords"], inputs["point_labels"]), boxes=None, masks=None
        )
        captured: dict[str, torch.Tensor] = {}

        def capture_sparse(_module, _args, kwargs) -> None:
            captured["sparse"] = kwargs["sparse_prompt_embeddings"].detach()

        hook = model.mask_decoder.register_forward_pre_hook(capture_sparse, with_kwargs=True)
        try:
            with torch.no_grad():
                output = model(**inputs)
        finally:
            hook.remove()
        self.assertEqual(captured["sparse"].shape[1], native_sparse.shape[1])
        self.assertFalse(output["trajectory_token_gate"].any())

    def test_03j_trajectory_only_cognition_supervises_tcn(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_encoder="tcn_attention",
                prompt_integration="trajectory_token",
                attention_integration="none",
                prompt_residual_gate_init=0.05,
                lambda_trajectory_cognitive=0.05,
            ),
        ).train()
        output = model(**direct_inputs())
        losses = cais_multitask_loss(
            output["final_mask"],
            torch.rand(1, 1, 256, 256),
            trajectory_cognitive_prediction=output["trajectory_cognitive_friction"],
            cognitive_target=torch.tensor([0.8]),
            lambda_iou=0.0,
            lambda_cognitive=0.0,
            lambda_trajectory_cognitive=0.05,
        )
        losses["loss"].backward()
        self.assertGreater(float(losses["loss_trajectory_cognitive"].detach()), 0.0)
        self.assertTrue(
            any(
                parameter.grad is not None and torch.count_nonzero(parameter.grad).item()
                for parameter in model.cognitive_prompt_encoder.trajectory_cnn.parameters()
            )
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.trajectory_cognitive_head.parameters())
        )

    def test_03k_constant_trajectory_token_hides_motion_from_mask_decoder(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_encoder="tcn_attention",
                trajectory_token_content="constant",
                prompt_integration="trajectory_token",
                attention_integration="none",
                prompt_residual_gate_init=0.05,
                lambda_trajectory_cognitive=0.05,
            ),
        ).eval()
        first = direct_inputs()
        second = dict(first)
        second["click_trajectories"] = torch.randn_like(first["click_trajectories"]) * 100.0
        with torch.no_grad():
            first_output = model(**first)
            second_output = model(**second)
        self.assertTrue(torch.allclose(first_output["final_mask"], second_output["final_mask"]))
        self.assertFalse(
            torch.allclose(first_output["trajectory_vectors"], second_output["trajectory_vectors"])
        )
        self.assertTrue(
            torch.equal(first_output["trajectory_token_gate"], second_output["trajectory_token_gate"])
        )

    def test_03l_dense_trajectory_prompt_preserves_native_points_and_uses_motion(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_encoder="tcn_attention",
                prompt_integration="trajectory_dense",
                attention_integration="none",
                prompt_residual_gate_init=0.05,
                lambda_trajectory_cognitive=0.05,
            ),
        ).eval()
        inputs = direct_inputs()
        inputs["click_trajectories"] = torch.tensor(
            [[[[ -1.0, -0.30, 0.0], [-0.5, -0.15, 0.0], [0.0, 0.0, 0.0]]]]
        )
        inputs["trajectory_valid_mask"] = torch.ones(1, 1, 3, dtype=torch.bool)
        inputs["trajectory_coordinate_scale"] = torch.tensor([[1024.0, 1024.0]])
        native_sparse, _ = model.prompt_encoder(
            points=(inputs["point_coords"], inputs["point_labels"]), boxes=None, masks=None
        )
        captured: dict[str, torch.Tensor] = {}

        def capture_prompts(_module, _args, kwargs) -> None:
            captured["sparse"] = kwargs["sparse_prompt_embeddings"].detach()

        hook = model.mask_decoder.register_forward_pre_hook(capture_prompts, with_kwargs=True)
        try:
            with torch.no_grad():
                output = model(**inputs)
        finally:
            hook.remove()
        self.assertTrue(torch.allclose(captured["sparse"], native_sparse))
        self.assertGreater(torch.count_nonzero(output["trajectory_dense_prompt"]).item(), 0)
        self.assertGreater(torch.count_nonzero(output["trajectory_raster"]).item(), 0)
        self.assertTrue(
            torch.allclose(output["trajectory_dense_gate"], torch.full((1, 1), 0.05), atol=0.005)
        )

        shifted = dict(inputs)
        shifted["click_trajectories"] = inputs["click_trajectories"].clone()
        shifted["click_trajectories"][..., 2] += 0.25
        with torch.no_grad():
            shifted_output = model(**shifted)
        self.assertFalse(torch.allclose(output["trajectory_raster"], shifted_output["trajectory_raster"]))

    def test_03m_missing_trajectory_has_zero_dense_prompt(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_encoder="tcn_attention",
                prompt_integration="trajectory_dense",
                attention_integration="none",
                prompt_residual_gate_init=0.05,
            ),
        ).eval()
        inputs = direct_inputs()
        inputs["trajectory_valid_mask"] = torch.zeros(1, 1, 8, dtype=torch.bool)
        with torch.no_grad():
            output = model(**inputs)
        self.assertTrue(torch.equal(output["trajectory_dense_prompt"], torch.zeros_like(output["trajectory_dense_prompt"])))
        self.assertTrue(torch.equal(output["trajectory_raster"], torch.zeros_like(output["trajectory_raster"])))
        self.assertFalse(output["trajectory_dense_gate"].any())

    def test_03n_dense_mask_loss_reaches_tcn_and_spatial_encoder(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_encoder="tcn_attention",
                prompt_integration="trajectory_dense",
                attention_integration="none",
                prompt_residual_gate_init=0.05,
            ),
        ).train()
        configure_trainable_parameters(model, train_mask_decoder=True)
        output = model(**direct_inputs())
        output["mask_logits"].mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.count_nonzero(parameter.grad).item()
                for parameter in model.cognitive_prompt_encoder.trajectory_cnn.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None and torch.count_nonzero(parameter.grad).item()
                for parameter in model.trajectory_dense_prompt_encoder.encoder.parameters()
            )
        )

    def test_03o_dense_gate_is_capped_and_dropout_can_disable_prompt(self) -> None:
        capped = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_encoder="tcn_attention",
                prompt_integration="trajectory_dense",
                attention_integration="none",
                prompt_residual_gate_init=0.05,
                trajectory_gate_max=0.25,
            ),
        ).eval()
        with torch.no_grad():
            capped.trajectory_dense_prompt_encoder.gate.weight.zero_()
            capped.trajectory_dense_prompt_encoder.gate.bias.fill_(100.0)
            capped_output = capped(**direct_inputs())
        self.assertLessEqual(float(capped_output["trajectory_dense_gate"].max()), 0.25)
        self.assertGreater(float(capped_output["trajectory_dense_gate"].min()), 0.249)

        dropped = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_encoder="tcn_attention",
                prompt_integration="trajectory_dense",
                attention_integration="none",
                prompt_residual_gate_init=0.05,
                trajectory_gate_max=0.25,
                trajectory_dropout=1.0,
            ),
        ).train()
        dropped_output = dropped(**direct_inputs())
        self.assertFalse(dropped_output["trajectory_dense_gate"].any())
        self.assertFalse(dropped_output["trajectory_dense_prompt"].any())

    def test_03p_interaction_memory_preserves_native_prompt_path(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_encoder="tcn_attention",
                prompt_integration="native",
                attention_integration="none",
                interaction_memory="click_mask_trajectory",
                memory_layers=1,
                memory_size=4,
                memory_gate_init=0.05,
            ),
        ).eval()
        inputs = direct_inputs(click_count=2)
        inputs["interaction_masks"] = torch.rand(1, 2, 1, 256, 256)
        inputs["interaction_mask_valid"] = torch.ones(1, 2, dtype=torch.bool)
        native_sparse, _ = model.prompt_encoder(
            points=(inputs["point_coords"], inputs["point_labels"]), boxes=None, masks=None
        )
        captured: dict[str, torch.Tensor] = {}

        def capture_sparse(_module, _args, kwargs) -> None:
            captured["sparse"] = kwargs["sparse_prompt_embeddings"].detach()

        hook = model.mask_decoder.register_forward_pre_hook(capture_sparse, with_kwargs=True)
        try:
            with torch.no_grad():
                output = model(**inputs)
        finally:
            hook.remove()
        self.assertTrue(torch.allclose(captured["sparse"], native_sparse))
        self.assertEqual(tuple(output["interaction_memory_tokens"].shape), (1, 2, 256))
        self.assertAlmostEqual(float(output["interaction_memory_gate"]), 0.05, places=6)

    def test_03q_click_mask_memory_is_trajectory_invariant(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_encoder="tcn_attention",
                prompt_integration="native",
                attention_integration="none",
                interaction_memory="click_mask",
            ),
        ).eval()
        first = direct_inputs(click_count=2)
        first["interaction_masks"] = torch.rand(1, 2, 1, 256, 256)
        first["interaction_mask_valid"] = torch.ones(1, 2, dtype=torch.bool)
        second = dict(first)
        second["click_trajectories"] = torch.randn_like(first["click_trajectories"]) * 100.0
        with torch.no_grad():
            first_output = model(**first)
            second_output = model(**second)
        self.assertTrue(torch.allclose(first_output["final_mask"], second_output["final_mask"]))

    def test_03r_trajectory_memory_changes_mask_and_receives_gradient(self) -> None:
        model = CAISEfficientViTSam(
            make_base_model(),
            CAISConfig(
                cognitive_layers=1,
                trajectory_encoder="tcn_attention",
                prompt_integration="native",
                attention_integration="none",
                interaction_memory="click_mask_trajectory",
            ),
        ).train()
        configure_trainable_parameters(model, train_mask_decoder=True)
        inputs = direct_inputs(click_count=2)
        inputs["interaction_masks"] = torch.rand(1, 2, 1, 256, 256)
        inputs["interaction_mask_valid"] = torch.ones(1, 2, dtype=torch.bool)
        output = model(**inputs)
        output["mask_logits"].mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.count_nonzero(parameter.grad).item()
                for parameter in model.cognitive_prompt_encoder.trajectory_cnn.parameters()
            )
        )
        self.assertIsNotNone(model.interaction_memory_attention.trajectory_projection.weight.grad)
        groups = build_optimizer_parameter_groups(
            model,
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            module_learning_rate=1e-4,
            gate_learning_rate=1e-3,
            decoder_learning_rate=1e-5,
            weight_decay=1e-4,
        )
        self.assertIn(id(model.interaction_memory_attention.gate), {id(p) for g in groups if g["name"] == "gates" for p in g["params"]})

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
