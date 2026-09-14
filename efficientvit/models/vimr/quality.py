"""Quality calibration for VIMR multimask outputs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class VIMRFinalMaskQualityHead(nn.Module):
    """Predict IoU after VIMR safe-update rather than before it.

    SAM's original IoU head scores the draft decoder masks. VIMR changes those
    masks afterwards, so this head combines draft scores, final-mask statistics
    and interaction context to calibrate the candidates actually returned.
    """

    def __init__(self, context_dim: int = 256, context_hidden: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim * 4, context_hidden),
            nn.GELU(),
            nn.Linear(context_hidden, hidden_dim),
            nn.GELU(),
        )
        self.candidate_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _candidate_mean(value: torch.Tensor, candidate_count: int) -> torch.Tensor:
        if value.shape[1] == 1 and candidate_count > 1:
            value = value.expand(-1, candidate_count, -1, -1)
        return value.mean(dim=(-2, -1))

    def forward(
        self,
        output: dict[str, torch.Tensor],
        round_index: int,
        max_clicks: int,
    ) -> torch.Tensor:
        final_mask = output["final_mask"].clamp(1e-6, 1 - 1e-6)
        draft_mask = output["draft_mask"].clamp(1e-6, 1 - 1e-6)
        candidate_count = final_mask.shape[1]
        base_quality = output["iou_predictions"]
        area = final_mask.mean(dim=(-2, -1))
        confidence = torch.maximum(final_mask, 1.0 - final_mask).mean(dim=(-2, -1))
        uncertainty = (1.0 - (2.0 * final_mask - 1.0).abs()).mean(dim=(-2, -1))
        gate = self._candidate_mean(output["gate_map"], candidate_count)
        delta = self._candidate_mean(output["delta_map"].abs(), candidate_count)
        draft_change = (final_mask - draft_mask).abs().mean(dim=(-2, -1))
        round_fraction = final_mask.new_full(
            (final_mask.shape[0], candidate_count),
            float(round_index + 1) / float(max_clicks),
        )
        candidate_features = torch.stack(
            (
                base_quality,
                base_quality.sigmoid(),
                area,
                confidence,
                uncertainty,
                gate,
                delta,
                draft_change,
                round_fraction,
            ),
            dim=-1,
        )

        image_context = output["image_embeddings"].mean(dim=(-2, -1))
        context = torch.cat(
            (
                image_context,
                output["viewport_token"],
                output["history_summary"],
                output["residual_token"],
            ),
            dim=-1,
        )
        context = self.context_projection(context).unsqueeze(1).expand(-1, candidate_count, -1)
        return torch.sigmoid(self.candidate_predictor(torch.cat((context, candidate_features), dim=-1)).squeeze(-1))


def hard_candidate_iou(candidates: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return hard IoU targets with shape ``[B,K]``."""
    target = F.interpolate(target.float(), candidates.shape[-2:], mode="nearest") >= 0.5
    predicted = candidates >= 0.5
    intersection = (predicted & target).sum(dim=(-2, -1)).float()
    union = (predicted | target).sum(dim=(-2, -1)).float()
    return intersection / union.clamp_min(1.0)


def quality_calibration_loss(
    predicted_quality: torch.Tensor,
    target_quality: torch.Tensor,
    ranking_margin: float = 0.05,
    target_tolerance: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Smooth-L1 calibration plus pairwise candidate-ranking supervision."""
    target_quality = target_quality.detach()
    regression = F.smooth_l1_loss(predicted_quality, target_quality)
    ranking_terms = []
    candidate_count = predicted_quality.shape[1]
    for left in range(candidate_count):
        for right in range(left + 1, candidate_count):
            target_delta = target_quality[:, left] - target_quality[:, right]
            valid = target_delta.abs() > target_tolerance
            if valid.any():
                predicted_delta = predicted_quality[:, left] - predicted_quality[:, right]
                direction = target_delta.sign()
                ranking_terms.append(F.relu(ranking_margin - direction * predicted_delta)[valid].mean())
    ranking = (
        torch.stack(ranking_terms).mean()
        if ranking_terms
        else predicted_quality.new_zeros(())
    )
    return {"loss_quality": regression, "loss_ranking": ranking}


def quality_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return calibration MAE and top-candidate agreement for logging."""
    return {
        "quality_mae": float((predicted - target).abs().mean().detach().cpu()),
        "quality_top1_accuracy": float(
            (predicted.argmax(dim=1) == target.argmax(dim=1)).float().mean().detach().cpu()
        ),
    }


def quality_head_state_dict(head: VIMRFinalMaskQualityHead) -> dict[str, Any]:
    return {key: value.detach().cpu() for key, value in head.state_dict().items()}
