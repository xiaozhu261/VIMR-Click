from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientvit.models.efficientvit.sam import EfficientViTSam

from .config import CAISConfig
from .modules import CAISMultiTaskHeads, CognitivePromptEncoder, CognitiveTwoWayAttention


@dataclass
class CAISInteractionState:
    point_coords: Optional[torch.Tensor] = None
    point_labels: Optional[torch.Tensor] = None
    click_trajectories: Optional[torch.Tensor] = None
    trajectory_valid_mask: Optional[torch.Tensor] = None
    previous_mask: Optional[torch.Tensor] = None


class CAISEfficientViTSam(EfficientViTSam):
    """Standalone CAIS architecture sharing only EfficientViT-SAM base modules."""

    def __init__(self, base_model: EfficientViTSam, config: CAISConfig | dict[str, Any] | None = None) -> None:
        super().__init__(
            image_encoder=base_model.image_encoder,
            prompt_encoder=base_model.prompt_encoder,
            mask_decoder=base_model.mask_decoder,
            image_size=base_model.image_size,
        )
        self.cais_config = config if isinstance(config, CAISConfig) else CAISConfig.from_dict(config)
        cfg = self.cais_config
        self.cognitive_prompt_encoder = CognitivePromptEncoder(
            d_model=cfg.d_model,
            trajectory_channels=cfg.trajectory_channels,
        )
        self.cognitive_two_way_attention = CognitiveTwoWayAttention(
            d_model=cfg.d_model,
            num_heads=cfg.cognitive_heads,
            num_layers=cfg.cognitive_layers,
            ffn_dim=cfg.cognitive_ffn_dim,
        )
        self.prediction_heads = CAISMultiTaskHeads(d_model=cfg.d_model)
        if cfg.prompt_integration not in {"replace", "native_residual"}:
            raise ValueError(f"Unsupported prompt integration: {cfg.prompt_integration}")
        if cfg.attention_integration not in {"replace", "none", "gated_residual"}:
            raise ValueError(f"Unsupported attention integration: {cfg.attention_integration}")
        if cfg.prompt_integration == "native_residual":
            self.prompt_residual_gate = nn.Parameter(torch.zeros(()))
        if cfg.attention_integration == "gated_residual":
            self.image_residual_gate = nn.Parameter(torch.zeros(()))
        self.set_freeze_policy(cfg.freeze_image_encoder, cfg.freeze_prompt_encoder)

    def set_freeze_policy(self, freeze_image_encoder: bool = True, freeze_prompt_encoder: bool = False) -> None:
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = not freeze_image_encoder
        for parameter in self.prompt_encoder.parameters():
            parameter.requires_grad = not freeze_prompt_encoder

    def parameter_summary(self) -> dict[str, int]:
        modules: dict[str, nn.Module] = {
            "image_encoder": self.image_encoder,
            "prompt_encoder": self.prompt_encoder,
            "mask_decoder": self.mask_decoder,
            "cognitive_prompt_encoder": self.cognitive_prompt_encoder,
            "cognitive_two_way_attention": self.cognitive_two_way_attention,
            "prediction_heads": self.prediction_heads,
        }
        summary = {
            name: sum(parameter.numel() for parameter in module.parameters()) for name, module in modules.items()
        }
        summary["total"] = sum(parameter.numel() for parameter in self.parameters())
        summary["trainable"] = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return summary

    @staticmethod
    def _resize_probability(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        # Preserve the native SAM mask-prompt path exactly when the previous
        # low-resolution mask already has the decoder's expected resolution.
        # A redundant interpolation can perturb FP16 probabilities enough to
        # change the prompt embedding and defeat zero-gate identity.
        if mask.shape[-2:] != size:
            mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
        return mask.clamp(0, 1)

    def _sam_prompt(
        self,
        batch_size: int,
        points: Optional[tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        prompt_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse, dense = self.prompt_encoder(points=points, boxes=boxes, masks=prompt_mask)
        if sparse.shape[0] == 1 and batch_size > 1:
            sparse = sparse.expand(batch_size, -1, -1)
        if dense.shape[0] == 1 and batch_size > 1:
            dense = dense.expand(batch_size, -1, -1, -1)
        return sparse, dense

    def forward(
        self,
        batched_input: Optional[list[dict[str, Any]]] = None,
        multimask_output: bool = False,
        *,
        image_embeddings: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        click_trajectories: Optional[torch.Tensor] = None,
        trajectory_valid_mask: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        previous_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """Run CAIS with cached image features and trajectory-aware point prompts."""
        if batched_input is not None:
            return super().forward(batched_input, multimask_output)
        if image_embeddings is None:
            if images is None:
                raise ValueError("Either image_embeddings or preprocessed images must be provided")
            image_embeddings = self.image_encoder(images)
        batch_size, channels = image_embeddings.shape[:2]
        if channels != self.cais_config.d_model:
            raise ValueError(f"Expected {self.cais_config.d_model} image channels, got {channels}")
        if point_coords is None or point_labels is None:
            raise ValueError("point_coords and point_labels are required for CAIS direct inference")
        device, dtype = image_embeddings.device, image_embeddings.dtype
        # SAM's pretrained positional prompt encoder normally receives the
        # dataset's FP32 coordinates even when image features/decoder run in
        # autocast FP16. Keep that native dtype for the SAM branch; separately
        # cast coordinates for the learned CAIS encoder.
        native_point_coords = point_coords.to(device=device)
        point_coords = native_point_coords.to(dtype=dtype)
        point_labels = point_labels.to(device=device)
        if point_labels.shape != point_coords.shape[:2]:
            raise ValueError("point labels must match point coordinates")

        if click_trajectories is None:
            click_trajectories = image_embeddings.new_zeros(batch_size, point_coords.shape[1], 1, 3)
            trajectory_valid_mask = torch.zeros(batch_size, point_coords.shape[1], 1, device=device, dtype=torch.bool)
        else:
            click_trajectories = click_trajectories.to(device=device, dtype=dtype)
            if trajectory_valid_mask is not None:
                trajectory_valid_mask = trajectory_valid_mask.to(device=device, dtype=torch.bool)
        if not self.cais_config.use_trajectory:
            click_trajectories = torch.zeros_like(click_trajectories)
            trajectory_valid_mask = torch.zeros(
                click_trajectories.shape[:-1], device=device, dtype=torch.bool
            )

        prompt_tokens, trajectory_vectors, prompt_valid_mask = self.cognitive_prompt_encoder(
            point_coords,
            point_labels,
            click_trajectories,
            trajectory_valid_mask,
            input_image_size=self.prompt_encoder.input_image_size,
        )
        if self.cais_config.attention_integration == "none":
            decoder_image_embeddings = image_embeddings
            refined_prompt_tokens = prompt_tokens
        else:
            refined_image_embeddings, refined_prompt_tokens = self.cognitive_two_way_attention(
                image_embeddings, prompt_tokens, prompt_valid_mask
            )
            if self.cais_config.attention_integration == "gated_residual":
                image_gate = torch.tanh(self.image_residual_gate).to(dtype=dtype)
                decoder_image_embeddings = image_embeddings + image_gate * (
                    refined_image_embeddings - image_embeddings
                )
            else:
                decoder_image_embeddings = refined_image_embeddings

        prompt_mask = None
        if previous_mask is not None:
            previous_low_res = self._resize_probability(previous_mask.to(device=device, dtype=dtype), (256, 256))
            eps = torch.finfo(previous_low_res.dtype).eps
            prompt_mask = torch.logit(previous_low_res.clamp(eps, 1 - eps))
        if boxes is not None:
            boxes = boxes.to(device=device, dtype=dtype)
        if self.cais_config.prompt_integration == "native_residual":
            sparse_embeddings, dense_embeddings = self._sam_prompt(
                batch_size,
                (native_point_coords, point_labels),
                boxes,
                prompt_mask,
            )
            residual_tokens = (
                trajectory_vectors
                if self.cais_config.attention_integration == "none"
                else refined_prompt_tokens
            )
            prompt_gate = torch.tanh(self.prompt_residual_gate).to(dtype=dtype)
            point_count = point_coords.shape[1]
            sparse_embeddings = sparse_embeddings.clone()
            sparse_embeddings[:, :point_count] = (
                sparse_embeddings[:, :point_count] + prompt_gate * residual_tokens
            )
        else:
            sparse_embeddings, dense_embeddings = self._sam_prompt(batch_size, None, boxes, prompt_mask)
            sparse_embeddings = torch.cat((sparse_embeddings, refined_prompt_tokens), dim=1)

        mask_outputs = []
        iou_outputs = []
        image_pe = self.prompt_encoder.get_dense_pe()
        for batch_index in range(batch_size):
            sample_logits, sample_iou = self.mask_decoder(
                image_embeddings=decoder_image_embeddings[batch_index : batch_index + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings[batch_index : batch_index + 1],
                dense_prompt_embeddings=dense_embeddings[batch_index : batch_index + 1],
                multimask_output=multimask_output,
            )
            mask_outputs.append(sample_logits)
            iou_outputs.append(sample_iou)
        mask_logits = torch.cat(mask_outputs, dim=0)
        sam_iou_predictions = torch.cat(iou_outputs, dim=0)
        mask_probability = mask_logits.sigmoid()
        iou_predictions, cognitive_friction, cognitive_vector = self.prediction_heads(
            decoder_image_embeddings,
            refined_prompt_tokens,
            prompt_valid_mask,
            sam_iou_predictions,
        )
        return {
            "image_embeddings": image_embeddings,
            "decoder_image_embeddings": decoder_image_embeddings,
            "mask_logits": mask_logits,
            "final_mask": mask_probability,
            "iou_predictions": iou_predictions,
            "sam_iou_predictions": sam_iou_predictions,
            "cognitive_friction": cognitive_friction,
            "cognitive_vector": cognitive_vector,
            "cognitive_prompt_tokens": refined_prompt_tokens,
            "trajectory_vectors": trajectory_vectors,
            "prompt_valid_mask": prompt_valid_mask,
            "prompt_residual_gate": (
                torch.tanh(self.prompt_residual_gate)
                if hasattr(self, "prompt_residual_gate")
                else mask_probability.new_ones(())
            ),
            "image_residual_gate": (
                torch.tanh(self.image_residual_gate)
                if hasattr(self, "image_residual_gate")
                else mask_probability.new_ones(())
            ),
        }


def build_cais_efficientvit_sam(
    base_model: EfficientViTSam, config: CAISConfig | dict[str, Any] | None = None
) -> CAISEfficientViTSam:
    return CAISEfficientViTSam(base_model, config)
