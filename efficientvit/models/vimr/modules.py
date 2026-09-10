import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_encode(values: torch.Tensor, levels: int) -> torch.Tensor:
    """Encode the last dimension with dyadic sine/cosine features."""
    frequencies = torch.pow(2.0, torch.arange(levels, device=values.device, dtype=values.dtype)) * math.pi
    angles = values.unsqueeze(-1) * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=-2)


class ViewportIntentEncoder(nn.Module):
    """Encode a seven-value viewport state into a token and router logits."""

    def __init__(self, d_model: int = 256, fourier_levels: int = 6, action_embedding_dim: int = 32) -> None:
        super().__init__()
        self.d_model = d_model
        self.fourier_levels = fourier_levels
        self.action_embedding = nn.Embedding(4, action_embedding_dim)
        input_dim = 6 * 2 * fourier_levels + action_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.router = nn.Linear(d_model, 3)

    @staticmethod
    def default_state(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        state = torch.zeros(batch_size, 7, device=device, dtype=dtype)
        state[:, :2] = 0.5
        return state

    def forward(self, viewport_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return viewport token ``[B,D]`` and global/object/detail logits ``[B,3]``."""
        if viewport_state.ndim != 2 or viewport_state.shape[-1] != 7:
            raise ValueError(f"viewport_state must have shape [B,7], got {tuple(viewport_state.shape)}")
        continuous = viewport_state[:, :6]
        action = viewport_state[:, 6].long().clamp(0, 3)
        token = self.mlp(
            torch.cat((fourier_encode(continuous, self.fourier_levels), self.action_embedding(action)), -1)
        )
        return token, self.router(token)

    def spatial_map(self, viewport_state: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Create a differentiable Gaussian viewport ROI map with shape ``[B,1,H,W]``."""
        height, width = size
        y = torch.linspace(0, 1, height, device=viewport_state.device, dtype=viewport_state.dtype)
        x = torch.linspace(0, 1, width, device=viewport_state.device, dtype=viewport_state.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        cx = viewport_state[:, 0, None, None]
        cy = viewport_state[:, 1, None, None]
        zoom = viewport_state[:, 2, None, None].exp().clamp(0.25, 16.0)
        sigma = (0.5 / zoom).clamp(0.03, 1.0)
        roi = torch.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))
        return roi.unsqueeze(1)


class GlobalObjectDetailRouter(nn.Module):
    """Route the viewport token through global, object, and detail image experts.

    The three experts see, respectively, a global image summary, a viewport-weighted
    object summary, and viewport-weighted high-frequency image features.  Their
    residual outputs are mixed by the viewport encoder's three router logits.
    """

    def __init__(self, d_model: int = 256, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("router temperature must be positive")
        self.temperature = temperature
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(3)
            ]
        )
        # Start as an exact identity so enabling routing is safe for old checkpoints.
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    @staticmethod
    def _weighted_pool(feature: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight = F.interpolate(weight, feature.shape[-2:], mode="bilinear", align_corners=False).clamp_min(0)
        return (feature * weight).sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1e-6)

    def forward(
        self,
        image_embedding: torch.Tensor,
        viewport_token: torch.Tensor,
        viewport_map: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return routed token ``[B,D]``, probabilities ``[B,3]``, and expert tokens ``[B,3,D]``."""
        global_context = F.adaptive_avg_pool2d(image_embedding, 1).flatten(1)
        object_context = self._weighted_pool(image_embedding, viewport_map)
        local_mean = F.avg_pool2d(image_embedding, 3, stride=1, padding=1)
        detail_context = self._weighted_pool((image_embedding - local_mean).abs(), viewport_map.square())
        contexts = (global_context, object_context, detail_context)
        expert_deltas = torch.stack(
            [expert(torch.cat((viewport_token, context), dim=-1)) for expert, context in zip(self.experts, contexts)],
            dim=1,
        )
        probabilities = torch.softmax(router_logits / self.temperature, dim=-1)
        routed_token = viewport_token + (probabilities.unsqueeze(-1) * expert_deltas).sum(dim=1)
        return routed_token, probabilities, expert_deltas


class ClickHistoryMemory(nn.Module):
    """Causal Transformer memory over click, viewport, and mask-statistic tokens."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        max_history: int = 16,
        fourier_levels: int = 6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_history = max_history
        self.fourier_levels = fourier_levels
        self.label_embedding = nn.Embedding(2, 32)
        input_dim = 2 * 2 * fourier_levels + 32 + d_model + 4 + 2
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_history, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

    def forward(
        self,
        click_history: torch.Tensor,
        viewport_embeddings: torch.Tensor,
        mask_statistics: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        statistics_valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return causal tokens, last-valid summary, and the truncated valid mask.

        Shapes are clicks ``[B,T,3]``, viewport embeddings ``[B,T,D]``, statistics
        ``[B,T,4]``, tokens ``[B,min(T,max_history),D]``.
        """
        if click_history.ndim != 3 or click_history.shape[-1] != 3:
            raise ValueError(f"click_history must have shape [B,T,3], got {tuple(click_history.shape)}")
        click_history = click_history[:, -self.max_history :]
        viewport_embeddings = viewport_embeddings[:, -self.max_history :]
        batch_size, steps, _ = click_history.shape
        if valid_mask is None:
            valid_mask = click_history[..., 2] >= 0
        else:
            valid_mask = valid_mask[:, -self.max_history :].bool()
        if mask_statistics is None:
            mask_statistics = click_history.new_zeros(batch_size, steps, 4)
            statistics_valid_mask = torch.zeros(batch_size, steps, device=click_history.device, dtype=torch.bool)
        else:
            mask_statistics = mask_statistics[:, -self.max_history :]
            if statistics_valid_mask is None:
                statistics_valid_mask = torch.ones(
                    batch_size, steps, device=click_history.device, dtype=torch.bool
                )
            else:
                statistics_valid_mask = statistics_valid_mask[:, -self.max_history :].bool()
        labels = click_history[..., 2].long().clamp(0, 1)
        features = torch.cat(
            (
                fourier_encode(click_history[..., :2], self.fourier_levels),
                self.label_embedding(labels),
                viewport_embeddings,
                mask_statistics,
                statistics_valid_mask.to(click_history.dtype).unsqueeze(-1),
                valid_mask.to(click_history.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        tokens = self.input_projection(features) + self.position_embedding[:, :steps]
        causal_mask = torch.triu(torch.ones(steps, steps, device=tokens.device, dtype=torch.bool), diagonal=1)
        safe_valid_mask = valid_mask.clone()
        all_padding = ~safe_valid_mask.any(dim=1)
        if all_padding.any():
            safe_valid_mask[all_padding, 0] = True
        tokens = self.transformer(tokens, mask=causal_mask, src_key_padding_mask=~safe_valid_mask)
        tokens = tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)
        positions = torch.arange(steps, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        last_index = positions.masked_fill(~valid_mask, -1).amax(dim=1).clamp_min(0)
        summary = tokens[torch.arange(batch_size, device=tokens.device), last_index]
        summary = summary * valid_mask.any(dim=1, keepdim=True).to(summary.dtype)
        return tokens, summary, valid_mask


class MaskResidualAnalyzer(nn.Module):
    """Analyze prior mask uncertainty and temporal residuals with a light CNN."""

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, d_model, 3, padding=1),
        )

    def forward(
        self, previous_mask: torch.Tensor, older_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return feature map ``[B,D,H,W]``, token ``[B,D]``, uncertainty, and five statistics."""
        probability = previous_mask.clamp(0, 1)
        uncertainty = 1.0 - (2.0 * probability - 1.0).abs()
        residual = torch.zeros_like(probability) if older_mask is None else probability - older_mask.clamp(0, 1)
        positive = residual.relu()
        negative = (-residual).relu()
        feature_map = self.cnn(torch.cat((probability, uncertainty, positive, negative), dim=1))
        token = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        statistics = torch.stack(
            (
                probability.mean(dim=(1, 2, 3)),
                uncertainty.mean(dim=(1, 2, 3)),
                residual.abs().mean(dim=(1, 2, 3)),
                positive.mean(dim=(1, 2, 3)),
                negative.mean(dim=(1, 2, 3)),
            ),
            dim=-1,
        )
        return feature_map, token, uncertainty, statistics


class ViewportConditionedInteractionFusion(nn.Module):
    """Condition compact interaction tokens on cached image features by cross-attention."""

    def __init__(self, d_model: int = 256, num_heads: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads, dropout=0.0, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, image_embedding: torch.Tensor, interaction_tokens: torch.Tensor) -> torch.Tensor:
        """Return image-conditioned interaction tokens ``[B,K,D]``."""
        image_tokens = image_embedding.flatten(2).transpose(1, 2)  # [B,HW,D]
        output = interaction_tokens
        for attention, norm in zip(self.layers, self.norms):
            attended, _ = attention(output, image_tokens, image_tokens, need_weights=False)
            output = norm(output + attended)
        return output


def click_spatial_maps(
    point_coords: Optional[torch.Tensor],
    point_labels: Optional[torch.Tensor],
    size: tuple[int, int],
    input_image_size: tuple[int, int] = (1024, 1024),
    sigma: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize positive and negative click Gaussians as ``[B,1,H,W]`` maps."""
    if point_coords is None or point_labels is None:
        raise ValueError("point_coords and point_labels are required to infer batch size")
    batch_size = point_coords.shape[0]
    height, width = size
    dtype, device = point_coords.dtype, point_coords.device
    y = torch.linspace(0, 1, height, device=device, dtype=dtype)
    x = torch.linspace(0, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    normalized = point_coords.clone()
    normalized[..., 0] = normalized[..., 0] / input_image_size[1]
    normalized[..., 1] = normalized[..., 1] / input_image_size[0]
    distance = (xx - normalized[..., 0, None, None]) ** 2 + (yy - normalized[..., 1, None, None]) ** 2
    maps = torch.exp(-0.5 * distance / (sigma**2))
    positive = (maps * (point_labels == 1)[..., None, None]).amax(dim=1, keepdim=True)
    negative = (maps * (point_labels == 0)[..., None, None]).amax(dim=1, keepdim=True)
    return positive.reshape(batch_size, 1, height, width), negative.reshape(batch_size, 1, height, width)


class ResidualGatedSafeUpdate(nn.Module):
    """Blend a draft mask with the previous mask through a learned spatial gate."""

    def __init__(self, use_residual_feature: bool = True, initial_gate_probability: float = 0.1) -> None:
        super().__init__()
        if not 0.0 < initial_gate_probability < 1.0:
            raise ValueError("initial_gate_probability must be between 0 and 1")
        self.use_residual_feature = use_residual_feature
        input_channels = 8 if use_residual_feature else 7
        output_layer = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, math.log(initial_gate_probability / (1.0 - initial_gate_probability)))
        self.gate = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            output_layer,
            nn.Sigmoid(),
        )

    def forward(
        self,
        previous_mask: Optional[torch.Tensor],
        draft_mask: torch.Tensor,
        uncertainty: Optional[torch.Tensor],
        viewport_map: torch.Tensor,
        positive_click_map: torch.Tensor,
        negative_click_map: torch.Tensor,
        residual_feature_map: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return final probability, gate, and draft-minus-previous delta maps."""
        if previous_mask is None:
            return draft_mask, torch.ones_like(draft_mask), draft_mask
        previous = F.interpolate(previous_mask, draft_mask.shape[-2:], mode="bilinear", align_corners=False)
        if previous.shape[1] != draft_mask.shape[1]:
            previous = previous.expand(-1, draft_mask.shape[1], -1, -1)
        uncertainty = 1.0 - (2.0 * previous - 1.0).abs() if uncertainty is None else F.interpolate(
            uncertainty, draft_mask.shape[-2:], mode="bilinear", align_corners=False
        )
        spatial = [viewport_map, positive_click_map, negative_click_map]
        spatial = [F.interpolate(item, draft_mask.shape[-2:], mode="bilinear", align_corners=False) for item in spatial]
        batch_size, candidates = draft_mask.shape[:2]
        delta = draft_mask - previous
        gate_input = torch.cat((previous, draft_mask, delta.abs(), uncertainty, *spatial), dim=1)
        if self.use_residual_feature:
            if residual_feature_map is None:
                residual_map = torch.zeros_like(draft_mask[:, :1])
            else:
                residual_map = F.interpolate(
                    residual_feature_map.mean(dim=1, keepdim=True),
                    draft_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            gate_input = torch.cat((gate_input, residual_map), dim=1)
        if candidates != 1:
            # Apply the same single-candidate gate network independently to each SAM candidate.
            per_candidate = []
            for index in range(candidates):
                fields = [
                    previous[:, index : index + 1],
                    draft_mask[:, index : index + 1],
                    delta[:, index : index + 1].abs(),
                    uncertainty[:, :1],
                    *spatial,
                ]
                if self.use_residual_feature:
                    fields.append(residual_map)
                per_candidate.append(self.gate(torch.cat(fields, dim=1)))
            gate = torch.cat(per_candidate, dim=1)
        else:
            gate = self.gate(gate_input)
        final = (previous + gate * delta).clamp(0, 1)
        return final, gate, delta
