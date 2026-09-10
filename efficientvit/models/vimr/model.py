from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientvit.models.efficientvit.sam import EfficientViTSam

from .config import VIMRConfig
from .modules import (
    ClickHistoryMemory,
    GlobalObjectDetailRouter,
    MaskResidualAnalyzer,
    ResidualGatedSafeUpdate,
    ViewportConditionedInteractionFusion,
    ViewportIntentEncoder,
    click_spatial_maps,
)


@dataclass
class VIMRInteractionState:
    """Tensor state consumed by one VIMR-Click interaction."""

    point_coords: Optional[torch.Tensor] = None
    point_labels: Optional[torch.Tensor] = None
    click_history: Optional[torch.Tensor] = None
    viewport_state: Optional[torch.Tensor] = None
    viewport_history: Optional[torch.Tensor] = None
    previous_mask: Optional[torch.Tensor] = None
    mask_history: Optional[torch.Tensor] = None
    history_valid_mask: Optional[torch.Tensor] = None


class DetailRefinement(nn.Module):
    """Boundary-aware detail expert operating in SAM's 256-pixel mask space."""

    def __init__(self, image_channels: int = 256, image_detail_channels: int = 8, strength: float = 1.0) -> None:
        super().__init__()
        self.strength = strength
        self.image_projection = nn.Conv2d(image_channels, image_detail_channels, 1)
        input_channels = image_detail_channels + 6
        output = nn.Conv2d(24, 1, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.refiner = nn.Sequential(
            nn.Conv2d(input_channels, 24, 1),
            nn.GELU(),
            nn.Conv2d(24, 24, 3, padding=1, groups=24),
            nn.GELU(),
            nn.Conv2d(24, 24, 1),
            nn.GELU(),
            output,
        )

    def _refine_single(
        self,
        mask: torch.Tensor,
        image_detail: torch.Tensor,
        viewport_map: torch.Tensor,
        uncertainty: Optional[torch.Tensor],
        positive_click_map: torch.Tensor,
        negative_click_map: torch.Tensor,
        detail_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = mask.shape[-2:]
        viewport = F.interpolate(viewport_map, size, mode="bilinear", align_corners=False)
        positive = F.interpolate(positive_click_map, size, mode="bilinear", align_corners=False)
        negative = F.interpolate(negative_click_map, size, mode="bilinear", align_corners=False)
        if uncertainty is None:
            uncertainty = 1.0 - (2.0 * mask - 1.0).abs()
        else:
            uncertainty = F.interpolate(uncertainty, size, mode="bilinear", align_corners=False)
        dilation = F.max_pool2d(mask, 3, stride=1, padding=1)
        erosion = -F.max_pool2d(-mask, 3, stride=1, padding=1)
        boundary = dilation - erosion
        fields = torch.cat((image_detail, mask, uncertainty, boundary, viewport, positive, negative), dim=1)
        focus = (uncertainty + boundary).clamp(0, 1) * (0.25 + 0.75 * viewport)
        correction = self.refiner(fields) * focus * detail_weight[:, None, None, None] * self.strength
        eps = torch.finfo(mask.dtype).eps
        refined = torch.sigmoid(torch.logit(mask.clamp(eps, 1 - eps)) + correction)
        return refined, correction

    def forward(
        self,
        mask: torch.Tensor,
        image_embedding: torch.Tensor,
        viewport_map: torch.Tensor,
        uncertainty: Optional[torch.Tensor],
        positive_click_map: torch.Tensor,
        negative_click_map: torch.Tensor,
        detail_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_detail = self.image_projection(image_embedding)
        image_detail = F.interpolate(image_detail, mask.shape[-2:], mode="bilinear", align_corners=False)
        if detail_weight is None:
            detail_weight = mask.new_ones(mask.shape[0])
        if mask.shape[1] == 1:
            return self._refine_single(
                mask, image_detail, viewport_map, uncertainty, positive_click_map, negative_click_map, detail_weight
            )
        refined, corrections = [], []
        for index in range(mask.shape[1]):
            item, correction = self._refine_single(
                mask[:, index : index + 1],
                image_detail,
                viewport_map,
                uncertainty,
                positive_click_map,
                negative_click_map,
                detail_weight,
            )
            refined.append(item)
            corrections.append(correction)
        return torch.cat(refined, dim=1), torch.cat(corrections, dim=1)


class VIMREfficientViTSam(EfficientViTSam):
    """EfficientViT-SAM with ablatable interaction memory and safe mask updates.

    The inherited ``image_encoder``, ``prompt_encoder`` and ``mask_decoder`` names
    are intentionally unchanged so original EfficientViT-SAM checkpoints load
    without key remapping.
    """

    def __init__(self, base_model: EfficientViTSam, config: VIMRConfig | dict[str, Any] | None = None) -> None:
        super().__init__(
            image_encoder=base_model.image_encoder,
            prompt_encoder=base_model.prompt_encoder,
            mask_decoder=base_model.mask_decoder,
            image_size=base_model.image_size,
        )
        self.vimr_config = config if isinstance(config, VIMRConfig) else VIMRConfig.from_dict(config)
        cfg = self.vimr_config
        self.viewport_encoder = ViewportIntentEncoder(
            d_model=cfg.d_model,
            fourier_levels=cfg.viewport_fourier_levels,
            action_embedding_dim=cfg.action_embedding_dim,
        )
        self.history_memory = ClickHistoryMemory(
            d_model=cfg.d_model,
            num_heads=cfg.memory_heads,
            num_layers=cfg.memory_layers,
            ffn_dim=cfg.memory_ffn_dim,
            max_history=cfg.max_history,
            fourier_levels=cfg.viewport_fourier_levels,
        )
        self.residual_analyzer = MaskResidualAnalyzer(d_model=cfg.d_model)
        self.interaction_fusion = ViewportConditionedInteractionFusion(
            d_model=cfg.d_model, num_heads=cfg.memory_heads, num_layers=2
        )
        self.expert_router = GlobalObjectDetailRouter(
            d_model=cfg.d_model, temperature=cfg.router_temperature
        )
        self.safe_update = ResidualGatedSafeUpdate(
            use_residual_feature=True,
            initial_gate_probability=cfg.gate_init_probability,
        )
        self.detail_refinement = DetailRefinement(
            image_channels=cfg.d_model, strength=cfg.detail_refinement_strength
        )
        self.set_freeze_policy(cfg.freeze_image_encoder, cfg.freeze_prompt_encoder)

    def set_freeze_policy(self, freeze_image_encoder: bool = True, freeze_prompt_encoder: bool = False) -> None:
        """Configure the MVP fine-tuning policy without changing module train/eval modes."""
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = not freeze_image_encoder
        for parameter in self.prompt_encoder.parameters():
            parameter.requires_grad = not freeze_prompt_encoder

    def parameter_summary(self) -> dict[str, int]:
        modules = {
            "image_encoder": self.image_encoder,
            "prompt_encoder": self.prompt_encoder,
            "mask_decoder": self.mask_decoder,
            "viewport_encoder": self.viewport_encoder,
            "history_memory": self.history_memory,
            "residual_analyzer": self.residual_analyzer,
            "interaction_fusion": self.interaction_fusion,
            "expert_router": self.expert_router,
            "residual_gate": self.safe_update,
            "detail_refinement": self.detail_refinement,
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
        return F.interpolate(mask, size=size, mode="bilinear", align_corners=False).clamp(0, 1)

    @staticmethod
    def _history_statistics(mask_history: Optional[torch.Tensor], steps: int, reference: torch.Tensor) -> torch.Tensor:
        """Compute area, mean confidence, uncertainty, and temporal change as ``[B,T,4]``."""
        batch_size = reference.shape[0]
        output = reference.new_zeros(batch_size, steps, 4)
        if mask_history is None or steps == 0:
            return output
        if mask_history.ndim == 4:
            mask_history = mask_history.unsqueeze(2)
        masks = mask_history[:, -steps:].clamp(0, 1)
        available = masks.shape[1]
        probability = masks
        uncertainty = 1.0 - (2.0 * probability - 1.0).abs()
        confidence = torch.maximum(probability, 1.0 - probability)
        change = torch.zeros_like(probability)
        if available > 1:
            change[:, 1:] = (probability[:, 1:] - probability[:, :-1]).abs()
        stats = torch.stack(
            (
                probability.mean(dim=(2, 3, 4)),
                confidence.mean(dim=(2, 3, 4)),
                uncertainty.mean(dim=(2, 3, 4)),
                change.mean(dim=(2, 3, 4)),
            ),
            dim=-1,
        )
        output[:, -available:] = stats
        return output

    def _encode_history(
        self,
        click_history: torch.Tensor,
        viewport_history: Optional[torch.Tensor],
        mask_history: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, steps = click_history.shape[:2]
        if viewport_history is None:
            viewport_history = ViewportIntentEncoder.default_state(
                batch_size * steps, click_history.device, click_history.dtype
            ).reshape(batch_size, steps, 7)
        else:
            viewport_history = viewport_history[:, -steps:]
        viewport_tokens, _ = self.viewport_encoder(viewport_history.reshape(-1, 7))
        viewport_tokens = viewport_tokens.reshape(batch_size, steps, -1)
        if not self.vimr_config.use_viewport:
            viewport_tokens = torch.zeros_like(viewport_tokens)
        statistics = self._history_statistics(mask_history, steps, click_history)
        statistics_valid = torch.zeros(batch_size, steps, device=click_history.device, dtype=torch.bool)
        if mask_history is not None:
            available = min(mask_history.shape[1], steps)
            if available > 0:
                statistics_valid[:, -available:] = True
        return self.history_memory(
            click_history,
            viewport_tokens,
            statistics,
            valid_mask,
            statistics_valid_mask=statistics_valid,
        )

    def forward(
        self,
        batched_input: Optional[list[dict[str, Any]]] = None,
        multimask_output: bool = False,
        *,
        image_embeddings: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        previous_mask: Optional[torch.Tensor] = None,
        viewport_state: Optional[torch.Tensor] = None,
        click_history: Optional[torch.Tensor] = None,
        viewport_history: Optional[torch.Tensor] = None,
        mask_history: Optional[torch.Tensor] = None,
        history_valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """Run one interaction using cached embeddings or preprocessed images.

        Direct API shapes: image embeddings ``[B,256,64,64]``, point coordinates
        ``[B,N,2]`` in SAM's 1024-pixel prompt frame, labels ``[B,N]``, previous
        mask ``[B,1,H,W]``, viewport ``[B,7]``, and histories ``[B,T,...]``.
        Passing legacy ``batched_input`` delegates to the untouched baseline path.
        """
        if batched_input is not None:
            return super().forward(batched_input, multimask_output)
        if image_embeddings is None:
            if images is None:
                raise ValueError("Either image_embeddings or preprocessed images must be provided")
            image_embeddings = self.image_encoder(images)
        batch_size, channels, embed_h, embed_w = image_embeddings.shape
        if channels != self.vimr_config.d_model:
            raise ValueError(f"Expected {self.vimr_config.d_model} image channels, got {channels}")
        dtype, device = image_embeddings.dtype, image_embeddings.device
        if viewport_state is None:
            viewport_state = ViewportIntentEncoder.default_state(batch_size, device, dtype)
        else:
            viewport_state = viewport_state.to(device=device, dtype=dtype)

        points = None if point_coords is None else (point_coords, point_labels)
        prompt_mask = None
        previous_low_res = None
        if previous_mask is not None:
            previous_low_res = self._resize_probability(previous_mask.to(device=device, dtype=dtype), (256, 256))
            eps = torch.finfo(previous_low_res.dtype).eps
            prompt_mask = torch.logit(previous_low_res.clamp(eps, 1 - eps))
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=boxes, masks=prompt_mask)

        viewport_token = image_embeddings.new_zeros(batch_size, self.vimr_config.d_model)
        router_logits = image_embeddings.new_zeros(batch_size, 3)
        viewport_map = image_embeddings.new_ones(batch_size, 1, 256, 256)
        if self.vimr_config.use_viewport or self.vimr_config.use_history:
            viewport_token, router_logits = self.viewport_encoder(viewport_state)
            viewport_map = self.viewport_encoder.spatial_map(viewport_state, (256, 256))

        history_tokens = image_embeddings.new_zeros(batch_size, 0, self.vimr_config.d_model)
        history_summary = image_embeddings.new_zeros(batch_size, self.vimr_config.d_model)
        history_mask = torch.zeros(batch_size, 0, device=device, dtype=torch.bool)
        if self.vimr_config.use_history and click_history is not None and click_history.shape[1] > 0:
            history_tokens, history_summary, history_mask = self._encode_history(
                click_history.to(device=device, dtype=dtype),
                None if viewport_history is None else viewport_history.to(device=device, dtype=dtype),
                None if mask_history is None else mask_history.to(device=device, dtype=dtype),
                None if history_valid_mask is None else history_valid_mask.to(device=device),
            )

        residual_feature_map = image_embeddings.new_zeros(batch_size, self.vimr_config.d_model, embed_h, embed_w)
        residual_token = image_embeddings.new_zeros(batch_size, self.vimr_config.d_model)
        uncertainty = None
        residual_statistics = image_embeddings.new_zeros(batch_size, 5)
        if self.vimr_config.use_residual and previous_low_res is not None:
            older_mask = None
            if mask_history is not None and mask_history.shape[1] > 1:
                older_mask = self._resize_probability(
                    mask_history[:, -2].to(device=device, dtype=dtype), (embed_h, embed_w)
                )
            residual_feature_map, residual_token, uncertainty, residual_statistics = self.residual_analyzer(
                self._resize_probability(previous_low_res, (embed_h, embed_w)), older_mask
            )

        router_probabilities = torch.softmax(router_logits, dim=-1)
        expert_tokens = image_embeddings.new_zeros(batch_size, 3, self.vimr_config.d_model)
        if self.vimr_config.enabled and self.vimr_config.use_viewport and self.vimr_config.enable_expert_router:
            viewport_token, router_probabilities, expert_tokens = self.expert_router(
                image_embeddings, viewport_token, viewport_map, router_logits
            )

        if point_coords is not None and point_labels is not None:
            point_count = point_coords.shape[1]
            current_click_token = sparse_embeddings[:, max(point_count - 1, 0)]
            positive_map, negative_map = click_spatial_maps(
                point_coords.to(dtype=dtype), point_labels, (256, 256), self.prompt_encoder.input_image_size
            )
        else:
            current_click_token = image_embeddings.new_zeros(batch_size, self.vimr_config.d_model)
            positive_map = negative_map = image_embeddings.new_zeros(batch_size, 1, 256, 256)

        extra_tokens = []
        if self.vimr_config.enabled and self.vimr_config.use_viewport:
            extra_tokens.append(viewport_token)
        if self.vimr_config.enabled and self.vimr_config.use_history and click_history is not None:
            extra_tokens.append(history_summary)
        if self.vimr_config.enabled and self.vimr_config.use_residual and previous_low_res is not None:
            extra_tokens.append(residual_token)
        if extra_tokens:
            fusion_input = torch.stack((current_click_token, *extra_tokens), dim=1)
            conditioned = self.interaction_fusion(image_embeddings, fusion_input)[:, 1:]
            sparse_embeddings = torch.cat((sparse_embeddings, conditioned), dim=1)

        # SAM's decoder repeats a single image embedding across prompt batches;
        # decode per sample to support true image batches without B-by-B expansion.
        draft_outputs = []
        iou_outputs = []
        image_pe = self.prompt_encoder.get_dense_pe()
        for batch_index in range(batch_size):
            sample_logits, sample_iou = self.mask_decoder(
                image_embeddings=image_embeddings[batch_index : batch_index + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings[batch_index : batch_index + 1],
                dense_prompt_embeddings=dense_embeddings[batch_index : batch_index + 1],
                multimask_output=multimask_output,
            )
            draft_outputs.append(sample_logits)
            iou_outputs.append(sample_iou)
        draft_logits = torch.cat(draft_outputs, dim=0)
        iou_predictions = torch.cat(iou_outputs, dim=0)
        draft_mask = draft_logits.sigmoid()
        if self.vimr_config.enabled and self.vimr_config.use_safe_update and self.vimr_config.safe_update:
            final_mask, gate_map, delta_map = self.safe_update(
                previous_low_res,
                draft_mask,
                uncertainty,
                viewport_map,
                positive_map,
                negative_map,
                residual_feature_map if self.vimr_config.use_residual else None,
            )
        else:
            final_mask = draft_mask
            gate_map = torch.ones_like(draft_mask)
            delta_map = draft_mask if previous_low_res is None else draft_mask - previous_low_res
        detail_delta_map = torch.zeros_like(final_mask)
        if self.vimr_config.enabled and self.vimr_config.enable_detail_refinement:
            detail_weight = router_probabilities[:, 2] if self.vimr_config.enable_expert_router else None
            final_mask, detail_delta_map = self.detail_refinement(
                final_mask,
                image_embeddings,
                viewport_map,
                uncertainty,
                positive_map,
                negative_map,
                detail_weight,
            )
        eps = torch.finfo(final_mask.dtype).eps
        final_logits = torch.logit(final_mask.clamp(eps, 1 - eps))
        return {
            "image_embeddings": image_embeddings,
            "mask_logits": final_logits,
            "final_mask": final_mask,
            "draft_logits": draft_logits,
            "draft_mask": draft_mask,
            "iou_predictions": iou_predictions,
            "gate_map": gate_map,
            "delta_map": delta_map,
            "viewport_map": viewport_map,
            "viewport_token": viewport_token,
            "router_logits": router_logits,
            "router_probabilities": router_probabilities,
            "expert_tokens": expert_tokens,
            "detail_delta_map": detail_delta_map,
            "history_tokens": history_tokens,
            "history_summary": history_summary,
            "history_valid_mask": history_mask,
            "residual_feature_map": residual_feature_map,
            "residual_token": residual_token,
            "residual_statistics": residual_statistics,
        }


def build_vimr_efficientvit_sam(
    base_model: EfficientViTSam, config: VIMRConfig | dict[str, Any] | None = None
) -> VIMREfficientViTSam:
    return VIMREfficientViTSam(base_model, config)
