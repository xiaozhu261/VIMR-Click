from dataclasses import dataclass

import torch
from torch import nn

from .config import ProcessSAMConfig


@dataclass
class ProcessEncoding:
    state_token: torch.Tensor
    history_token: torch.Tensor | None
    spatial_features: torch.Tensor | None
    spatial_maps: torch.Tensor | None
    image_valid_map: torch.Tensor | None
    valid: torch.Tensor


class InteractionProcessEncoder(nn.Module):
    """Encode current state, ordered history, and spatial focus independently.

    Viewports are (zoom, center_x, center_y, width, height) in ORIGINAL image
    fractions. Trajectories are relative (time, dx, dy), also in original image
    fractions. coordinate_scale maps those fractions into the padded SAM frame.
    Neither image features nor prompt embeddings are inputs to this encoder.
    """

    def __init__(self, config: ProcessSAMConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("process_mean", torch.tensor(config.process_mean or (0.0,) * config.process_dim))
        self.register_buffer("process_std", torch.tensor(config.process_std or (1.0,) * config.process_dim))
        # Coordinates/type, process summaries + validity, viewport + validity.
        self.state_mlp = nn.Sequential(
            nn.Linear(3 + config.process_dim + 1 + 5 + 1, config.state_hidden),
            nn.GELU(),
            nn.Linear(config.state_hidden, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        if config.use_history:
            if config.history_pooling == "gru":
                self.history_gru = nn.GRUCell(config.d_model, config.history_hidden)
            else:
                # Almost the same parameter budget as GRUCell (256 fewer).
                self.history_pool = nn.Sequential(
                    nn.Linear(config.d_model, 3 * config.history_hidden),
                    nn.GELU(),
                    nn.Linear(3 * config.history_hidden, config.history_hidden),
                )
            self.history_projection = nn.Linear(config.history_hidden, config.d_model)
            if config.use_event_history:
                self.event_mlp = nn.Sequential(
                    nn.Linear(config.event_dim, config.state_hidden),
                    nn.GELU(),
                    nn.Linear(config.state_hidden, config.d_model),
                    nn.LayerNorm(config.d_model),
                )
        if config.use_spatial:
            self.spatial_encoder = nn.Sequential(
                nn.Conv2d(4, config.spatial_hidden, 3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(config.spatial_hidden, config.d_model, 1, bias=False),
            )

    @staticmethod
    def _last(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
        last = indices.masked_fill(~valid, -1).amax(dim=1)
        result = values[torch.arange(values.shape[0], device=values.device), last.clamp_min(0)]
        return result * valid.any(dim=1).reshape(-1, *([1] * (result.ndim - 1))).to(result.dtype)

    def _spatial_maps(
        self,
        coords,
        labels,
        valid,
        trajectories,
        trajectory_valid,
        viewports,
        viewport_valid,
        scale,
        canvas_size,
        embedding_size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count = valid.shape
        height, width = embedding_size
        canvas_h, canvas_w = canvas_size
        gy, gx = torch.meshgrid(
            (torch.arange(height, device=coords.device, dtype=coords.dtype) + 0.5) / height,
            (torch.arange(width, device=coords.device, dtype=coords.dtype) + 0.5) / width,
            indexing="ij",
        )
        image_valid = (gx[None] < scale[:, 0, None, None] / canvas_w) & (gy[None] < scale[:, 1, None, None] / canvas_h)
        normalized_coords = coords / coords.new_tensor((canvas_w, canvas_h))
        dx = (gx[None, None] - normalized_coords[:, :, 0, None, None]) * width
        dy = (gy[None, None] - normalized_coords[:, :, 1, None, None]) * height
        heat = torch.exp(-(dx.square() + dy.square()) / (2 * self.config.raster_sigma**2))
        positive = (heat * (valid & (labels == 1))[:, :, None, None]).amax(dim=1)
        negative = (heat * (valid & (labels == 0))[:, :, None, None]).amax(dim=1)

        # The current observed viewport, clipped by the real image's SAM extent.
        current_viewport = self._last(viewports, valid)
        current_viewport_valid = self._last(viewport_valid.unsqueeze(-1), valid).squeeze(-1).bool()
        image_x = gx[None] * canvas_w / scale[:, 0, None, None]
        image_y = gy[None] * canvas_h / scale[:, 1, None, None]
        viewport = ((image_x - current_viewport[:, 1, None, None]).abs() <= current_viewport[:, 3, None, None] / 2) & (
            (image_y - current_viewport[:, 2, None, None]).abs() <= current_viewport[:, 4, None, None] / 2
        )
        viewport = viewport & current_viewport_valid[:, None, None]

        # Relative paths are anchored to this task's clicks, including in E4.
        path_xy = coords[:, :, None, :] + trajectories[..., 1:] * scale[:, None, None, :]
        path_xy = path_xy / coords.new_tensor((canvas_w, canvas_h))
        tx = (gx[None, None, None] - path_xy[..., 0, None, None]) * width
        ty = (gy[None, None, None] - path_xy[..., 1, None, None]) * height
        path_heat = torch.exp(-(tx.square() + ty.square()) / (2 * self.config.raster_sigma**2))
        path_valid = trajectory_valid & valid[:, :, None]
        trajectory_map = (path_heat * path_valid[..., None, None]).sum(dim=(1, 2))
        trajectory_map = trajectory_map / path_valid.sum(dim=(1, 2)).clamp_min(1)[:, None, None]
        maps = torch.stack((viewport.to(coords.dtype), positive, negative, trajectory_map), dim=1)
        return maps * image_valid[:, None], image_valid[:, None].to(coords.dtype)

    def forward(
        self,
        point_coords,
        point_labels,
        *,
        process_features=None,
        process_feature_valid=None,
        viewports=None,
        viewport_valid=None,
        click_trajectories=None,
        trajectory_valid_mask=None,
        process_events=None,
        process_event_valid=None,
        process_spatial_valid=None,
        coordinate_scale=None,
        canvas_size=(1024, 1024),
        embedding_size=(64, 64),
    ) -> ProcessEncoding:
        batch, count, _ = point_coords.shape
        if not 1 <= count <= self.config.max_clicks or point_labels.shape != (batch, count):
            raise ValueError(f"Expected 1..{self.config.max_clicks} clicks with matching labels")
        if not ((point_labels == -1) | (point_labels == 0) | (point_labels == 1)).all():
            raise ValueError("Point labels must be -1 (padding), 0, or 1")
        valid = point_labels >= 0
        coords = torch.where(valid[..., None], point_coords, torch.zeros_like(point_coords))
        scale = (
            coords.new_tensor((canvas_size[1], canvas_size[0])).expand(batch, -1)
            if coordinate_scale is None
            else coordinate_scale.to(coords)
        )
        if scale.shape != (batch, 2) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("coordinate_scale must be positive finite [B,2] (resized_width,resized_height)")
        if process_features is None:
            process_features = coords.new_zeros(batch, count, self.config.process_dim)
            process_feature_valid = torch.zeros_like(valid)
        if process_features.shape != (batch, count, self.config.process_dim):
            raise ValueError("process_features must have shape [B,N,process_dim]")
        if process_feature_valid is not None and process_feature_valid.shape != (batch, count):
            raise ValueError("process_feature_valid must have shape [B,N]")
        process_valid = (
            valid if process_feature_valid is None else process_feature_valid.to(valid.device).bool() & valid
        )
        process = (process_features.to(coords) - self.process_mean) / self.process_std
        process = torch.where(process_valid[..., None], process, torch.zeros_like(process))
        if self.config.spatial_only:
            process = torch.zeros_like(process)
            process_valid = torch.zeros_like(process_valid)
        if viewports is None:
            viewports = coords.new_zeros(batch, count, 5)
            viewport_valid = torch.zeros_like(valid)
        if viewports.shape != (batch, count, 5):
            raise ValueError("viewports must have shape [B,N,5] (zoom,cx,cy,width,height)")
        if viewport_valid is not None and viewport_valid.shape != (batch, count):
            raise ValueError("viewport_valid must have shape [B,N]")
        view_valid = valid if viewport_valid is None else viewport_valid.to(valid.device).bool() & valid
        views = torch.where(view_valid[..., None], viewports.to(coords), torch.zeros_like(viewports.to(coords)))
        if (view_valid & (views[..., 0] <= 0)).any():
            raise ValueError("Observed zoom must be positive")
        if (view_valid[..., None] & (views[..., 3:] <= 0)).any():
            raise ValueError("Observed viewport dimensions must be positive")
        view_state = torch.cat((views[..., :1].clamp_min(1e-6).log(), views[..., 1:]), dim=-1)
        view_state = torch.where(view_valid[..., None], view_state, torch.zeros_like(view_state))
        event_states = torch.cat(
            (
                coords / scale[:, None],
                point_labels.clamp_min(0).to(coords)[..., None],
                process,
                process_valid.to(coords)[..., None],
                view_state,
                view_valid.to(coords)[..., None],
            ),
            dim=-1,
        )
        if not torch.isfinite(event_states).all():
            raise ValueError("Observed interaction features must be finite")
        events = self.state_mlp(event_states) * valid[..., None]
        state_token = self._last(events, valid)
        stream = stream_valid = None
        if self.config.use_event_history and process_events is not None:
            stream = process_events.to(coords)
            if stream.ndim != 3 or stream.shape[0] != batch or stream.shape[-1] != self.config.event_dim:
                raise ValueError("process_events must be [B,L,11]")
            if not 1 <= stream.shape[1] <= self.config.max_process_events:
                raise ValueError("Process event budget exceeded")
            stream_valid = (
                torch.ones(stream.shape[:2], device=coords.device, dtype=torch.bool)
                if process_event_valid is None
                else process_event_valid.to(coords.device).bool()
            )
            if stream_valid.shape != stream.shape[:2]:
                raise ValueError("Process event validity must be [B,L]")
            stream_valid = stream_valid & valid.any(dim=1, keepdim=True)
            stream = torch.where(stream_valid[..., None], stream, 0.0)
            if not torch.isfinite(stream).all() or (stream[..., 0] > 1e-6).any():
                raise ValueError("Process events must be finite and cannot follow the current click")
        history_token = None
        if self.config.use_history:
            hidden = events.new_zeros(batch, self.config.history_hidden)
            history_stream = stream
            if stream is not None and self.config.spatial_only:
                history_stream = stream.clone()
                history_stream[..., 0] = 0
            history_events = events if stream is None else self.event_mlp(history_stream)
            history_valid = valid if stream is None else stream_valid
            if self.config.history_pooling == "mean":
                pooled = (history_events * history_valid[..., None]).sum(1)
                pooled = pooled / history_valid.sum(1).clamp_min(1)[:, None]
                hidden = self.history_pool(pooled)
            else:
                for index in range(history_events.shape[1]):
                    updated = self.history_gru(history_events[:, index], hidden)
                    hidden = torch.where(history_valid[:, index, None], updated, hidden)
            history_token = self.history_projection(hidden) * valid.any(dim=1, keepdim=True)
        maps = spatial = image_valid = None
        if self.config.use_spatial:
            if click_trajectories is None:
                click_trajectories = coords.new_zeros(batch, count, 1, 3)
                trajectory_valid_mask = torch.zeros(batch, count, 1, device=coords.device, dtype=torch.bool)
            if (
                click_trajectories.ndim != 4
                or click_trajectories.shape[:2] != (batch, count)
                or click_trajectories.shape[-1] != 3
            ):
                raise ValueError("click_trajectories must have shape [B,N,T,3]")
            path_valid = (
                torch.ones(click_trajectories.shape[:-1], device=coords.device, dtype=torch.bool)
                if trajectory_valid_mask is None
                else trajectory_valid_mask.to(coords.device).bool()
            )
            if path_valid.shape != click_trajectories.shape[:-1]:
                raise ValueError("trajectory_valid_mask must match [B,N,T]")
            path_valid = path_valid & valid[..., None]
            paths = torch.where(path_valid[..., None], click_trajectories.to(coords), 0.0)
            if not torch.isfinite(paths).all():
                raise ValueError("Observed trajectories must be finite")
            maps, image_valid = self._spatial_maps(
                coords,
                point_labels,
                valid,
                paths,
                path_valid,
                views,
                view_valid,
                scale,
                canvas_size,
                embedding_size,
            )
            if stream is not None:
                # Use only observed cursor positions in the last two seconds;
                # zoom/pan centers are not cursor observations.
                h, w = embedding_size
                gy, gx = torch.meshgrid(
                    (torch.arange(h, device=coords.device, dtype=coords.dtype) + 0.5) / h,
                    (torch.arange(w, device=coords.device, dtype=coords.dtype) + 0.5) / w,
                    indexing="ij",
                )
                xy = stream[..., 1:3] * scale[:, None] / coords.new_tensor((canvas_size[1], canvas_size[0]))
                dx = (gx[None, None] - xy[..., 0, None, None]) * w
                dy = (gy[None, None] - xy[..., 1, None, None]) * h
                observed = stream_valid & (stream[..., 6] > 0.5) & (stream[..., 0] >= -0.2)
                if process_spatial_valid is not None:
                    if process_spatial_valid.shape != stream_valid.shape:
                        raise ValueError("Spatial event mask must match [B,L]")
                    observed = process_spatial_valid.to(coords.device).bool() & stream_valid & (stream[..., 6] > 0.5)
                heat = torch.exp(-(dx.square() + dy.square()) / (2 * self.config.raster_sigma**2))
                density = (heat * observed[..., None, None]).sum(dim=1) / observed.sum(dim=1).clamp_min(1)[
                    :, None, None
                ]
                maps = torch.cat((maps[:, :3], density[:, None] * image_valid), dim=1)
            spatial = self.spatial_encoder(maps) * image_valid
        return ProcessEncoding(state_token, history_token, spatial, maps, image_valid, valid.any(dim=1))


class ProcessAwareFusion(nn.Module):
    """A: append process tokens. B: also gate local visual features."""

    def __init__(self, config: ProcessSAMConfig) -> None:
        super().__init__()
        self.config = config
        if config.use_spatial:
            self.attention_head = nn.Sequential(
                nn.Conv2d(config.d_model, config.spatial_hidden, 1),
                nn.GELU(),
                nn.Conv2d(config.spatial_hidden, 1, 1),
                nn.Tanh(),
            )
            self.alpha = nn.Parameter(torch.atanh(torch.tensor(float(config.spatial_gate_init))))

    def forward(self, image_embeddings, sparse_embeddings, encoding: ProcessEncoding):
        process_tokens = [encoding.state_token]
        if encoding.history_token is not None:
            process_tokens.append(encoding.history_token)
        tokens = torch.stack(process_tokens, dim=1)
        attention = gate = None
        fused_image = image_embeddings
        if self.config.use_spatial:
            context = encoding.state_token
            if encoding.history_token is not None:
                context = context + encoding.history_token
            attention = self.attention_head(encoding.spatial_features + context[:, :, None, None])
            attention = attention * encoding.image_valid_map * encoding.valid[:, None, None, None]
            gate = self.alpha.tanh().to(image_embeddings.dtype)
            fused_image = image_embeddings * (1 + gate * attention)
        # SAM has no sparse padding mask. Append per sample; no valid
        # interaction adds NO token (zero tokens still affect attention).
        fused_sparse = [
            torch.cat((sparse, tokens[index : index + 1].to(sparse.dtype)), dim=1) if encoding.valid[index] else sparse
            for index, sparse in enumerate(sparse_embeddings)
        ]
        return fused_image, fused_sparse, tokens, attention, gate
