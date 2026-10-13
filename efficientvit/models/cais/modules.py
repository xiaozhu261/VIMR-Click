import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualBlock(nn.Module):
    """Small dilated residual block for short pre-click trajectories."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
        )
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.layers(features))


class CognitivePromptEncoder(nn.Module):
    """Fuse click position, click type, and pre-click motion into one token per click."""

    def __init__(
        self,
        d_model: int = 256,
        trajectory_channels: int = 64,
        max_position_frequency: float = 64.0,
        trajectory_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        trajectory_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        trajectory_encoder: str = "cnn_pool",
        process_feature_dim: int = 0,
        process_mean: tuple[float, ...] = (),
        process_std: tuple[float, ...] = (),
        process_gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by four for 2-D sine/cosine encoding")
        frequencies = torch.linspace(1.0, max_position_frequency, d_model // 4) * (2.0 * math.pi)
        self.register_buffer("position_frequencies", frequencies, persistent=False)
        if len(trajectory_mean) != 3 or len(trajectory_std) != 3:
            raise ValueError("trajectory_mean and trajectory_std must each contain three values")
        if any(value <= 0 for value in trajectory_std):
            raise ValueError("trajectory_std values must be positive")
        self.register_buffer(
            "trajectory_mean", torch.tensor(trajectory_mean, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "trajectory_std", torch.tensor(trajectory_std, dtype=torch.float32), persistent=False
        )
        if trajectory_encoder not in {"cnn_pool", "tcn_attention"}:
            raise ValueError(f"Unsupported trajectory encoder: {trajectory_encoder}")
        self.trajectory_encoder = trajectory_encoder
        self.process_feature_dim = int(process_feature_dim)
        if self.process_feature_dim < 0:
            raise ValueError("process_feature_dim must be non-negative")
        if self.process_feature_dim:
            if len(process_mean) != self.process_feature_dim or len(process_std) != self.process_feature_dim:
                raise ValueError("process_mean/process_std must match process_feature_dim")
            if any(value <= 0 for value in process_std):
                raise ValueError("process_std values must be positive")
            if not 0.0 < process_gate_init < 1.0:
                raise ValueError("process_gate_init must be strictly between 0 and 1")
            self.register_buffer(
                "process_mean", torch.tensor(process_mean, dtype=torch.float32), persistent=False
            )
            self.register_buffer(
                "process_std", torch.tensor(process_std, dtype=torch.float32), persistent=False
            )
            self.process_projection = nn.Sequential(
                nn.LayerNorm(self.process_feature_dim),
                nn.Linear(self.process_feature_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.process_gate = nn.Parameter(torch.atanh(torch.tensor(float(process_gate_init))))
        self.type_embedding = nn.Embedding(3, d_model)
        if trajectory_encoder == "tcn_attention":
            # Base channels plus first differences in x/y, speed, and speed change.
            self.trajectory_cnn = nn.Sequential(
                nn.Conv1d(7, trajectory_channels, 1),
                nn.GELU(),
                TemporalResidualBlock(trajectory_channels, dilation=1),
                TemporalResidualBlock(trajectory_channels, dilation=2),
                nn.Conv1d(trajectory_channels, d_model, 1),
                nn.GELU(),
            )
            self.trajectory_attention = nn.Linear(d_model, 1)
            self.trajectory_projection = nn.Linear(d_model, d_model)
        else:
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
        process_features: Optional[torch.Tensor] = None,
        process_feature_valid: Optional[torch.Tensor] = None,
        input_image_size: tuple[int, int] = (1024, 1024),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        trajectory_mean = self.trajectory_mean.to(device=flat_trajectory.device, dtype=flat_trajectory.dtype)
        trajectory_std = self.trajectory_std.to(device=flat_trajectory.device, dtype=flat_trajectory.dtype)
        flat_trajectory = (flat_trajectory - trajectory_mean) / trajectory_std
        if self.trajectory_encoder == "tcn_attention":
            delta_xy = torch.diff(flat_trajectory[..., 1:3], dim=1, prepend=flat_trajectory[:, :1, 1:3])
            previous_valid = torch.cat(
                (torch.zeros_like(flat_valid[:, :1]), flat_valid[:, :-1]), dim=1
            )
            adjacent_valid = flat_valid & previous_valid
            delta_xy = delta_xy * adjacent_valid.unsqueeze(-1).to(delta_xy.dtype)
            speed = torch.linalg.vector_norm(delta_xy, dim=-1, keepdim=True)
            speed_change = torch.diff(speed, dim=1, prepend=speed[:, :1])
            previous_adjacent_valid = torch.cat(
                (torch.zeros_like(adjacent_valid[:, :1]), adjacent_valid[:, :-1]), dim=1
            )
            speed_change = speed_change * (adjacent_valid & previous_adjacent_valid).unsqueeze(-1).to(
                speed_change.dtype
            )
            trajectory_features = torch.cat((flat_trajectory, delta_xy, speed, speed_change), dim=-1)
        else:
            trajectory_features = flat_trajectory
        encoded = self.trajectory_cnn(
            (trajectory_features * flat_valid.unsqueeze(-1).to(flat_trajectory.dtype)).transpose(1, 2)
        ).transpose(1, 2)
        weights = flat_valid.unsqueeze(-1).to(encoded.dtype)
        trajectory_present = flat_valid.any(dim=1, keepdim=True)
        if self.trajectory_encoder == "tcn_attention":
            safe_valid = flat_valid.clone()
            safe_valid[~trajectory_present.squeeze(1), 0] = True
            attention_logits = self.trajectory_attention(encoded).squeeze(-1)
            attention_weights = torch.softmax(
                attention_logits.masked_fill(~safe_valid, -torch.inf), dim=1
            ).unsqueeze(-1)
            attention_weights = attention_weights * weights
            pooled = (encoded * attention_weights).sum(dim=1)
            trajectory_vector = self.trajectory_projection(pooled)
        else:
            mean_pool = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            max_pool = encoded.masked_fill(~flat_valid.unsqueeze(-1), -torch.inf).amax(dim=1)
            max_pool = torch.where(trajectory_present, max_pool, torch.zeros_like(max_pool))
            trajectory_vector = self.trajectory_projection(torch.cat((mean_pool, max_pool), dim=-1))
        # Projection bias must not turn a missing trajectory into a generic
        # learned prompt adapter. Missing motion is represented exactly by 0.
        trajectory_vector = trajectory_vector * trajectory_present.to(trajectory_vector.dtype)
        trajectory_vector = trajectory_vector.reshape(batch_size, click_count, -1)
        trajectory_present = trajectory_present.reshape(batch_size, click_count)

        if self.process_feature_dim:
            if process_features is None:
                process_features = trajectories.new_zeros(
                    batch_size, click_count, self.process_feature_dim
                )
                process_feature_valid = torch.zeros(
                    batch_size, click_count, device=trajectories.device, dtype=torch.bool
                )
            else:
                if process_features.ndim == 2:
                    process_features = process_features.unsqueeze(1)
                    if process_feature_valid is not None and process_feature_valid.ndim == 1:
                        process_feature_valid = process_feature_valid.unsqueeze(1)
                if process_features.shape[0] != batch_size or process_features.shape[-1] != self.process_feature_dim:
                    raise ValueError(
                        "process_features must have shape [B,N,F] with "
                        f"F={self.process_feature_dim}, got {tuple(process_features.shape)}"
                    )
                if process_features.shape[1] == 1 and click_count > 1:
                    aligned_process = process_features.new_zeros(
                        batch_size, click_count, self.process_feature_dim
                    )
                    aligned_process[:, -1] = process_features[:, 0]
                    process_features = aligned_process
                    if process_feature_valid is not None:
                        aligned_valid = torch.zeros(
                            batch_size, click_count, device=trajectories.device, dtype=torch.bool
                        )
                        aligned_valid[:, -1] = process_feature_valid[:, 0]
                        process_feature_valid = aligned_valid
                elif process_features.shape[1] != click_count:
                    raise ValueError("process feature click dimension must be 1 or match point count")
                process_features = process_features.to(
                    device=trajectories.device, dtype=trajectories.dtype
                )
                if process_feature_valid is None:
                    process_feature_valid = torch.ones(
                        batch_size, click_count, device=trajectories.device, dtype=torch.bool
                    )
                elif process_feature_valid.shape != (batch_size, click_count):
                    raise ValueError(
                        f"process_feature_valid must have shape {(batch_size, click_count)}"
                    )
                else:
                    process_feature_valid = process_feature_valid.to(
                        device=trajectories.device, dtype=torch.bool
                    )
            normalized_process = (
                process_features
                - self.process_mean.to(device=trajectories.device, dtype=trajectories.dtype)
            ) / self.process_std.to(device=trajectories.device, dtype=trajectories.dtype)
            process_vector = self.process_projection(normalized_process)
            process_vector = process_vector * process_feature_valid.unsqueeze(-1).to(
                process_vector.dtype
            )
            trajectory_vector = trajectory_vector + torch.tanh(self.process_gate).to(
                trajectory_vector.dtype
            ) * process_vector
            trajectory_present = trajectory_present | process_feature_valid

        prompt_tokens = self.output_norm(position + type_embedding + trajectory_vector)
        prompt_tokens = prompt_tokens * click_valid.unsqueeze(-1).to(prompt_tokens.dtype)
        trajectory_vector = trajectory_vector * click_valid.unsqueeze(-1).to(trajectory_vector.dtype)
        trajectory_present = trajectory_present & click_valid
        return prompt_tokens, trajectory_vector, click_valid, trajectory_present


class TrajectoryTokenAdapter(nn.Module):
    """Build separately gated sparse tokens without modifying native SAM points."""

    def __init__(self, d_model: int = 256, bottleneck: int = 64, gate_init: float = 0.05) -> None:
        super().__init__()
        if bottleneck < 1:
            raise ValueError("trajectory token bottleneck must be positive")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("trajectory token gate_init must be strictly between 0 and 1")
        self.adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )
        self.gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(gate_init / (1.0 - gate_init)))

    def forward(
        self, trajectory_vectors: torch.Tensor, trajectory_present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate(trajectory_vectors)).squeeze(-1)
        gate = gate * trajectory_present.to(gate.dtype)
        tokens = self.adapter(trajectory_vectors) * gate.unsqueeze(-1)
        tokens = tokens * trajectory_present.unsqueeze(-1).to(tokens.dtype)
        return tokens, gate


class TrajectoryDensePromptEncoder(nn.Module):
    """Rasterize motion into a spatial prompt aligned with SAM image features."""

    def __init__(
        self,
        d_model: int = 256,
        hidden_channels: int = 32,
        sigma: float = 1.5,
        gate_init: float = 0.05,
        gate_max: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("trajectory dense-prompt channels must be positive")
        if sigma <= 0:
            raise ValueError("trajectory raster sigma must be positive")
        if not 0.0 < gate_max <= 1.0:
            raise ValueError("trajectory dense-prompt gate_max must be in (0,1]")
        if not 0.0 < gate_init < gate_max:
            raise ValueError("trajectory dense-prompt gate_init must be strictly between 0 and gate_max")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("trajectory dropout must be in [0,1]")
        self.sigma = float(sigma)
        self.gate_max = float(gate_max)
        self.dropout = float(dropout)
        # Bias-free convolutions make missing motion an exact zero residual,
        # rather than an input-independent learned dense prompt.
        self.encoder = nn.Sequential(
            nn.Conv2d(6, hidden_channels, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, d_model, 3, padding=1, bias=False),
        )
        self.gate = nn.Linear(d_model, 1)
        # A tiny non-zero projection lets mask supervision reach the temporal
        # encoder on step one while keeping the effective gate near gate_init.
        nn.init.normal_(self.gate.weight, std=1e-3)
        initial_ratio = gate_init / gate_max
        nn.init.constant_(self.gate.bias, math.log(initial_ratio / (1.0 - initial_ratio)))

    def _rasterize(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        trajectories: torch.Tensor,
        valid_mask: torch.Tensor,
        coordinate_scale: torch.Tensor,
        gate: torch.Tensor,
        output_size: tuple[int, int],
        input_image_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size = trajectories.shape[0]
        output_height, output_width = output_size
        input_height, input_width = input_image_size
        dtype, device = trajectories.dtype, trajectories.device

        scale_x = coordinate_scale[:, None, None, 0]
        scale_y = coordinate_scale[:, None, None, 1]
        trajectory_x = point_coords[..., 0].unsqueeze(-1) + trajectories[..., 1] * scale_x
        trajectory_y = point_coords[..., 1].unsqueeze(-1) + trajectories[..., 2] * scale_y
        trajectory_x = trajectory_x * ((output_width - 1) / float(input_width))
        trajectory_y = trajectory_y * ((output_height - 1) / float(input_height))

        grid_x = torch.arange(output_width, device=device, dtype=dtype).view(1, 1, 1, 1, output_width)
        grid_y = torch.arange(output_height, device=device, dtype=dtype).view(1, 1, 1, output_height, 1)
        distance_squared = (
            (grid_x - trajectory_x[..., None, None]).square()
            + (grid_y - trajectory_y[..., None, None]).square()
        )
        valid = valid_mask & (point_labels >= 0).unsqueeze(-1)
        kernel = torch.exp(-0.5 * distance_squared / (self.sigma**2))
        kernel = kernel * valid[..., None, None].to(dtype)
        kernel = kernel * gate[..., None, None, None]

        positive = (point_labels == 1)[..., None, None, None].to(dtype)
        negative = (point_labels == 0)[..., None, None, None].to(dtype)
        positive_map = (kernel * positive).amax(dim=(1, 2))
        negative_map = (kernel * negative).amax(dim=(1, 2))

        times = trajectories[..., 0]
        large = torch.finfo(dtype).max
        time_min = times.masked_fill(~valid, large).amin(dim=-1, keepdim=True)
        time_max = times.masked_fill(~valid, -large).amax(dim=-1, keepdim=True)
        time_min = torch.where(valid.any(dim=-1, keepdim=True), time_min, torch.zeros_like(time_min))
        time_max = torch.where(valid.any(dim=-1, keepdim=True), time_max, torch.zeros_like(time_max))
        recency = ((times - time_min) / (time_max - time_min).clamp_min(1e-6)).clamp(0, 1)
        recency_map = (kernel * recency[..., None, None]).amax(dim=(1, 2))

        positions = torch.stack((trajectory_x, trajectory_y), dim=-1)
        delta = torch.diff(positions, dim=2, prepend=positions[:, :, :1])
        adjacent_valid = valid & torch.cat((valid[:, :, :1], valid[:, :, :-1]), dim=2)
        speed = torch.linalg.vector_norm(delta, dim=-1)
        max_speed = speed.masked_fill(~adjacent_valid, 0).amax(dim=-1, keepdim=True).clamp_min(1e-6)
        slowness = (1.0 - speed / max_speed).clamp(0, 1) * adjacent_valid.to(dtype)
        slowness_map = (kernel * slowness[..., None, None]).amax(dim=(1, 2))

        direction = delta / speed.clamp_min(1e-6).unsqueeze(-1)
        direction = direction * adjacent_valid.unsqueeze(-1).to(dtype)
        density = kernel.sum(dim=(1, 2)).clamp_min(1e-6)
        direction_x = (kernel * direction[..., 0, None, None]).sum(dim=(1, 2)) / density
        direction_y = (kernel * direction[..., 1, None, None]).sum(dim=(1, 2)) / density
        return torch.stack(
            (positive_map, negative_map, recency_map, slowness_map, direction_x, direction_y), dim=1
        ).reshape(batch_size, 6, output_height, output_width)

    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        click_trajectories: torch.Tensor,
        trajectory_valid_mask: Optional[torch.Tensor],
        trajectory_vectors: torch.Tensor,
        trajectory_present: torch.Tensor,
        coordinate_scale: torch.Tensor,
        output_size: tuple[int, int],
        input_image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectories, valid_mask = CognitivePromptEncoder._align_trajectories(
            click_trajectories, trajectory_valid_mask, point_coords.shape[1]
        )
        normalized_vectors = F.layer_norm(trajectory_vectors, (trajectory_vectors.shape[-1],))
        gate = self.gate_max * torch.sigmoid(self.gate(normalized_vectors)).squeeze(-1)
        gate = gate * trajectory_present.to(gate.dtype)
        if self.training and self.dropout > 0:
            keep = torch.rand_like(gate) >= self.dropout
            gate = gate * keep.to(gate.dtype)
        raster = self._rasterize(
            point_coords,
            point_labels,
            trajectories,
            valid_mask,
            coordinate_scale,
            gate,
            output_size,
            input_image_size,
        )
        dense_prompt = self.encoder(raster)
        batch_present = trajectory_present.any(dim=1, keepdim=True).to(dense_prompt.dtype)
        dense_prompt = dense_prompt * batch_present[:, :, None, None]
        return dense_prompt, gate, raster


class TrajectoryCognitiveHead(nn.Module):
    """Predict cognition from motion alone so supervision cannot bypass trajectories."""

    def __init__(self, d_model: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trajectory_vectors: torch.Tensor, trajectory_present: torch.Tensor) -> torch.Tensor:
        weights = trajectory_present.unsqueeze(-1).to(trajectory_vectors.dtype)
        context = (trajectory_vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return torch.sigmoid(self.head(context).squeeze(-1))


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


class InteractionMemoryAttention(nn.Module):
    """Condition image features on per-round click, mask, and optional motion memories.

    The native SAM click embeddings remain untouched. Each memory slot represents
    the state seen by the user before one click, rather than treating the whole
    prompt history as one unordered set.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        ffn_dim: int = 512,
        mask_channels: int = 32,
        max_memories: int = 8,
        gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        if max_memories < 1:
            raise ValueError("max_memories must be positive")
        if not -1.0 < gate_init < 1.0:
            raise ValueError("memory gate_init must be strictly between -1 and 1")
        self.max_memories = int(max_memories)
        self.click_projection = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        # Bias-free projection keeps a missing/zero trajectory exactly neutral.
        self.trajectory_projection = nn.Linear(d_model, d_model, bias=False)
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, mask_channels, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mask_channels, d_model, 1, bias=False),
        )
        self.round_embedding = nn.Embedding(max_memories, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sparse_memory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.image_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.dense_score = nn.Linear(d_model, 1)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.gate = nn.Parameter(torch.atanh(torch.tensor(float(gate_init))))

    @staticmethod
    def _normalize_masks(
        masks: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        memory_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if masks is None:
            masks = torch.zeros(batch_size, memory_count, 1, 1, 1, device=device, dtype=dtype)
            valid_mask = torch.zeros(batch_size, memory_count, device=device, dtype=torch.bool)
        else:
            masks = masks.to(device=device, dtype=dtype)
            if masks.ndim == 4:
                masks = masks.unsqueeze(2)
            if masks.ndim != 5 or masks.shape[:2] != (batch_size, memory_count):
                raise ValueError(
                    "interaction_masks must have shape [B,N,1,H,W] or [B,N,H,W], "
                    f"got {tuple(masks.shape)}"
                )
            if masks.shape[2] != 1:
                masks = masks[:, :, :1]
            if valid_mask is None:
                valid_mask = torch.ones(batch_size, memory_count, device=device, dtype=torch.bool)
            else:
                valid_mask = valid_mask.to(device=device, dtype=torch.bool)
                if valid_mask.shape != (batch_size, memory_count):
                    raise ValueError(
                        f"interaction_mask_valid must have shape {(batch_size, memory_count)}, "
                        f"got {tuple(valid_mask.shape)}"
                    )
        return masks, valid_mask

    def forward(
        self,
        image_embedding: torch.Tensor,
        click_embeddings: torch.Tensor,
        click_valid_mask: torch.Tensor,
        trajectory_vectors: torch.Tensor,
        interaction_masks: Optional[torch.Tensor],
        interaction_mask_valid: Optional[torch.Tensor],
        *,
        include_trajectory: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = image_embedding.shape
        if click_embeddings.shape != trajectory_vectors.shape:
            raise ValueError("click and trajectory memory tensors must have matching shapes")
        if click_embeddings.shape[:2] != click_valid_mask.shape:
            raise ValueError("click_valid_mask must match memory click dimensions")

        memory_count = click_embeddings.shape[1]
        interaction_masks, interaction_mask_valid = self._normalize_masks(
            interaction_masks,
            interaction_mask_valid,
            batch_size=batch_size,
            memory_count=memory_count,
            device=image_embedding.device,
            dtype=image_embedding.dtype,
        )
        start = max(memory_count - self.max_memories, 0)
        click_embeddings = click_embeddings[:, start:]
        click_valid_mask = click_valid_mask[:, start:].bool()
        trajectory_vectors = trajectory_vectors[:, start:]
        interaction_masks = interaction_masks[:, start:]
        interaction_mask_valid = interaction_mask_valid[:, start:]
        memory_count = click_embeddings.shape[1]

        flat_masks = interaction_masks.reshape(-1, 1, *interaction_masks.shape[-2:])
        flat_masks = F.interpolate(flat_masks, size=(height, width), mode="bilinear", align_corners=False)
        dense_memories = self.mask_encoder(flat_masks).reshape(
            batch_size, memory_count, channels, height, width
        )
        dense_memories = dense_memories * interaction_mask_valid[:, :, None, None, None].to(
            dense_memories.dtype
        )
        dense_tokens = dense_memories.mean(dim=(-2, -1))

        round_indices = torch.arange(memory_count, device=image_embedding.device)
        memory_tokens = self.click_projection(click_embeddings) + dense_tokens
        memory_tokens = memory_tokens + self.round_embedding(round_indices).unsqueeze(0)
        if include_trajectory:
            memory_tokens = memory_tokens + self.trajectory_projection(trajectory_vectors)
        memory_tokens = memory_tokens * click_valid_mask.unsqueeze(-1).to(memory_tokens.dtype)

        safe_valid = click_valid_mask.clone()
        no_valid = ~safe_valid.any(dim=1)
        if no_valid.any():
            safe_valid[no_valid, -1] = True
        padding_mask = ~safe_valid
        memory_tokens = self.sparse_memory_encoder(memory_tokens, src_key_padding_mask=padding_mask)
        memory_tokens = memory_tokens * click_valid_mask.unsqueeze(-1).to(memory_tokens.dtype)

        image_tokens = image_embedding.flatten(2).transpose(1, 2)
        sparse_context, _ = self.cross_attention(
            self.image_norm(image_tokens),
            self.memory_norm(memory_tokens),
            self.memory_norm(memory_tokens),
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        dense_logits = self.dense_score(dense_tokens).squeeze(-1)
        dense_valid = interaction_mask_valid & click_valid_mask
        safe_dense_valid = dense_valid.clone()
        no_dense = ~safe_dense_valid.any(dim=1)
        if no_dense.any():
            safe_dense_valid[no_dense, -1] = True
        dense_weights = torch.softmax(dense_logits.masked_fill(~safe_dense_valid, -torch.inf), dim=1)
        dense_weights = dense_weights * dense_valid.to(dense_weights.dtype)
        dense_weights = dense_weights / dense_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        dense_context = (dense_memories * dense_weights[:, :, None, None, None]).sum(dim=1)
        dense_context = dense_context.flatten(2).transpose(1, 2)

        residual = self.output_projection(sparse_context + dense_context)
        effective_gate = torch.tanh(self.gate).to(dtype=image_embedding.dtype)
        refined = image_tokens + effective_gate * residual
        refined = refined.transpose(1, 2).reshape_as(image_embedding)
        return refined, memory_tokens, effective_gate


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
