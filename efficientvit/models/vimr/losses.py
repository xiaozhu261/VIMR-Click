from typing import Optional

import torch
import torch.nn.functional as F


def soft_dice_score(probability: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * intersection + eps) / (denominator + eps)


def vimr_interactive_loss(
    final_mask: torch.Tensor,
    target: torch.Tensor,
    previous_mask: Optional[torch.Tensor] = None,
    lambda_safe: float = 0.2,
    lambda_monotonic: float = 0.1,
    correct_threshold: float = 0.1,
    monotonic_margin: float = 0.0,
) -> dict[str, torch.Tensor]:
    """BCE+Dice segmentation loss with safe-update and monotonic refinement terms."""
    target = F.interpolate(target.float(), final_mask.shape[-2:], mode="nearest")
    probability = final_mask.clamp(1e-6, 1 - 1e-6)
    bce = F.binary_cross_entropy(probability, target)
    dice = 1.0 - soft_dice_score(probability, target).mean()
    zero = probability.new_zeros(())
    safe = zero
    monotonic = zero
    if previous_mask is not None:
        previous = F.interpolate(previous_mask.detach(), final_mask.shape[-2:], mode="bilinear", align_corners=False)
        correct_previous = ((previous - target).abs() < correct_threshold).to(probability.dtype)
        safe = (correct_previous * (probability - previous).abs()).sum() / correct_previous.sum().clamp_min(1.0)
        quality_current = soft_dice_score(probability, target)
        quality_previous = soft_dice_score(previous, target)
        monotonic = F.relu(quality_previous - quality_current + monotonic_margin).mean()
    total = bce + dice + lambda_safe * safe + lambda_monotonic * monotonic
    return {
        "loss": total,
        "loss_bce": bce,
        "loss_dice": dice,
        "loss_safe": safe,
        "loss_monotonic": monotonic,
    }
