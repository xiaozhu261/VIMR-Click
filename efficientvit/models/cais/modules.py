import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CognitivePromptEncoder(nn.Module):
    """Fuse click position, click type, and pre-click motion into one token per click."""

    def __init__(
        self,
        d_model: int = 256,
        trajectory_channels: int = 64,
        max_position_frequency: float = 64.0,
    ) -> None:
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by four for 2-D sine/cosine encoding")
        frequencies = torch.linspace(1.0, max_position_frequency, d_model // 4) * (2.0 * math.pi)
        self.register_buffer("position_frequencies", frequencies, persistent=False)
        self.type_embedding = nn.Embedding(3, d_model)
        self.trajectory_cnn = nn.Sequential(
            nn.Conv1d(3, trajectory_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(trajectory_channels, d_model, 3, padding=1),
            nn.GELU(),
        )
        self.trajectory_projection = nn.Linear(2 * d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def _position_encoding(self, normalized_coords: torch.Tensor) -> torch.Tensor:
        angles = normalized_coords.unsqueeze(-1) * self.position_frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=-2)

    @staticmethod
    def _align_trajectories(
        trajectories: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        click_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if trajectories.ndim == 3:
            trajectories = trajectories.unsqueeze(1)
            if valid_mask is not None and valid_mask.ndim == 2:
                valid_mask = valid_mask.unsqueeze(1)
        if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
            raise ValueError(
                "click_trajectories must have shape [B,N,T,3] or [B,T,3], " f"got {tuple(trajectories.shape)}"
            )
        if trajectories.shape[1] not in (1, click_count):
            raise ValueError(
                f"trajectory click dimension must be 1 or match {click_count} clicks, " f"got {trajectories.shape[1]}"
            )
        if valid_mask is None:
            valid_mask = torch.ones(trajectories.shape[:-1], device=trajectories.device, dtype=torch.bool)
        elif valid_mask.shape != trajectories.shape[:-1]:
            raise ValueError(
                f"trajectory_valid_mask must have shape {tuple(trajectories.shape[:-1])}, "
                f"got {tuple(valid_mask.shape)}"
            )
        else:
            valid_mask = valid_mask.bool()
        if trajectories.shape[1] == click_count:
            return trajectories, valid_mask

        aligned = trajectories.new_zeros(trajectories.shape[0], click_count, trajectories.shape[2], 3)
        aligned_mask = torch.zeros(
            trajectories.shape[0], click_count, trajectories.shape[2], device=trajectories.device, dtype=torch.bool
        )
        aligned[:, -1] = trajectories[:, 0]
        aligned_mask[:, -1] = valid_mask[:, 0]
        return aligned, aligned_mask

    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        click_trajectories: torch.Tensor,
        trajectory_valid_mask: Optional[torch.Tensor] = None,
        input_image_size: tuple[int, int] = (1024, 1024),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if point_coords.ndim != 3 or point_coords.shape[-1] != 2:
            raise ValueError(f"point_coords must have shape [B,N,2], got {tuple(point_coords.shape)}")
        if point_labels.shape != point_coords.shape[:2]:
            raise ValueError(
                f"point_labels must have shape {tuple(point_coords.shape[:2])}, got {tuple(point_labels.shape)}"
            )
        trajectories, trajectory_valid_mask = self._align_trajectories(
            click_trajectories, trajectory_valid_mask, point_coords.shape[1]
        )
        if trajectories.shape[0] != point_coords.shape[0]:
            raise ValueError("point and trajectory batch dimensions must match")

        normalized = point_coords.clone()
        normalized[..., 0] = normalized[..., 0] / float(input_image_size[1])
        normalized[..., 1] = normalized[..., 1] / float(input_image_size[0])
        position = self._position_encoding(normalized)
        click_valid = point_labels >= 0
        type_indices = torch.where(click_valid, point_labels.clamp(0, 1), torch.full_like(point_labels, 2)).long()
        type_embedding = self.type_embedding(type_indices)

        batch_size, click_count, sample_count, _ = trajectories.shape
        flat_trajectory = trajectories.reshape(batch_size * click_count, sample_count, 3)
        flat_valid = trajectory_valid_mask.reshape(batch_size * click_count, sample_count)
        encoded = self.trajectory_cnn(
            (flat_trajectory * flat_valid.unsqueeze(-1).to(flat_trajectory.dtype)).transpose(1, 2)
        ).transpose(1, 2)
        weights = flat_valid.unsqueeze(-1).to(encoded.dtype)
        mean_pool = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        max_pool = encoded.masked_fill(~flat_valid.unsqueeze(-1), -torch.inf).amax(dim=1)
        max_pool = torch.where(flat_valid.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool))
        trajectory_vector = self.trajectory_projection(torch.cat((mean_pool, max_pool), dim=-1))
        trajectory_vector = trajectory_vector.reshape(batch_size, click_count, -1)

        prompt_tokens = self.output_norm(position + type_embedding + trajectory_vector)
        prompt_tokens = prompt_tokens * click_valid.unsqueeze(-1).to(prompt_tokens.dtype)
        trajectory_vector = trajectory_vector * click_valid.unsqueeze(-1).to(trajectory_vector.dtype)
        return prompt_tokens, trajectory_vector, click_valid


class CognitiveTwoWayAttentionLayer(nn.Module):
    """One prompt/image bidirectional refinement layer."""

    def __init__(self, d_model: int = 256, num_heads: int = 8, ffn_dim: int = 1024) -> None:
        super().__init__()
        self.prompt_self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.prompt_to_image = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.image_to_prompt = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.prompt_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.image_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.prompt_ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))
        self.image_ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))

    def forward(
        self,
        image_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_valid = prompt_valid_mask.clone()
        all_padding = ~safe_valid.any(dim=1)
        if all_padding.any():
            safe_valid[all_padding, 0] = True
        padding_mask = ~safe_valid

        attended, _ = self.prompt_self_attention(
            prompt_tokens, prompt_tokens, prompt_tokens, key_padding_mask=padding_mask, need_weights=False
        )
        prompt_tokens = self.prompt_norms[0](prompt_tokens + attended)
        attended, _ = self.prompt_to_image(prompt_tokens, image_tokens, image_tokens, need_weights=False)
        prompt_tokens = self.prompt_norms[1](prompt_tokens + attended)
        prompt_tokens = self.prompt_norms[2](prompt_tokens + self.prompt_ffn(prompt_tokens))

        attended, _ = self.image_to_prompt(
            image_tokens, prompt_tokens, prompt_tokens, key_padding_mask=padding_mask, need_weights=False
        )
        image_tokens = self.image_norms[0](image_tokens + attended)
        image_tokens = self.image_norms[1](image_tokens + self.image_ffn(image_tokens))
        prompt_tokens = prompt_tokens * prompt_valid_mask.unsqueeze(-1).to(prompt_tokens.dtype)
        return image_tokens, prompt_tokens


class CognitiveTwoWayAttention(nn.Module):
    """Alternately refine CAIS prompt tokens and cached EfficientViT features."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CognitiveTwoWayAttentionLayer(d_model, num_heads, ffn_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        image_embedding: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tokens = image_embedding.flatten(2).transpose(1, 2)
        for layer in self.layers:
            image_tokens, prompt_tokens = layer(image_tokens, prompt_tokens, prompt_valid_mask)
        refined_image = image_tokens.transpose(1, 2).reshape_as(image_embedding)
        return refined_image, prompt_tokens


class CAISMultiTaskHeads(nn.Module):
    """Predict mask IoU and cognitive friction from CAIS prompt/image context."""

    def __init__(self, d_model: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.context_projection = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.iou_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cognitive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        image_embedding: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_valid_mask: torch.Tensor,
        base_iou_predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = prompt_valid_mask.unsqueeze(-1).to(prompt_tokens.dtype)
        prompt_context = (prompt_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        image_context = F.adaptive_avg_pool2d(image_embedding, 1).flatten(1)
        context = self.context_projection(torch.cat((image_context, prompt_context), dim=-1))
        candidate_count = base_iou_predictions.shape[1]
        per_candidate_context = context.unsqueeze(1).expand(-1, candidate_count, -1)
        iou_features = torch.cat((per_candidate_context, base_iou_predictions.unsqueeze(-1)), dim=-1)
        predicted_iou = torch.sigmoid(self.iou_head(iou_features).squeeze(-1))
        cognitive_friction = torch.sigmoid(self.cognitive_head(context).squeeze(-1))
        return predicted_iou, cognitive_friction, prompt_context
