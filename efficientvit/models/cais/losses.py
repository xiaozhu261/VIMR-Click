from typing import Optional

import torch
import torch.nn.functional as F


def soft_dice_score(probability: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * intersection + eps) / (denominator + eps)


def hard_iou_score(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted = probability >= 0.5
    truth = target >= 0.5
    if truth.shape[1] == 1 and predicted.shape[1] > 1:
        truth = truth.expand(-1, predicted.shape[1], -1, -1)
    intersection = (predicted & truth).sum(dim=(-2, -1)).float()
    union = (predicted | truth).sum(dim=(-2, -1)).float()
    return intersection / union.clamp_min(1.0)


def mask_boundary_band(target: torch.Tensor, width: int = 2) -> torch.Tensor:
    if width < 1:
        raise ValueError("boundary width must be at least one pixel")
    kernel_size = 2 * width + 1
    dilation = F.max_pool2d(target, kernel_size, stride=1, padding=width)
    erosion = -F.max_pool2d(-target, kernel_size, stride=1, padding=width)
    return (dilation - erosion).clamp(0, 1)


def cais_multitask_loss(
    final_mask: torch.Tensor,
    target: torch.Tensor,
    *,
    iou_prediction: Optional[torch.Tensor] = None,
    cognitive_prediction: Optional[torch.Tensor] = None,
    trajectory_cognitive_prediction: Optional[torch.Tensor] = None,
    iou_target: Optional[torch.Tensor] = None,
    cognitive_target: Optional[torch.Tensor] = None,
    cognitive_supervision_mask: Optional[torch.Tensor] = None,
    trajectory_cognitive_supervision_mask: Optional[torch.Tensor] = None,
    boundary_alpha: float = 2.0,
    boundary_width: int = 2,
    lambda_iou: float = 1.0,
    lambda_cognitive: float = 0.075,
    lambda_trajectory_cognitive: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Boundary-adaptive mask loss with independent IoU and cognition supervision."""
    if lambda_iou < 0 or lambda_cognitive < 0 or lambda_trajectory_cognitive < 0:
        raise ValueError("auxiliary loss weights must be non-negative")
    target = F.interpolate(target.float(), final_mask.shape[-2:], mode="nearest")
    if target.shape[1] == 1 and final_mask.shape[1] > 1:
        target = target.expand(-1, final_mask.shape[1], -1, -1)
    # Keep the probability-form loss numerically stable under CUDA autocast.
    probability = final_mask.float().clamp(1e-6, 1 - 1e-6)
    target = target.float()
    boundary = mask_boundary_band(target, boundary_width)

    supervision_mask = None
    if cognitive_supervision_mask is not None:
        supervision_mask = cognitive_supervision_mask.to(device=probability.device, dtype=torch.bool)
        supervision_mask = supervision_mask.reshape(supervision_mask.shape[0], -1).any(dim=1)
    if cognitive_target is not None:
        friction = cognitive_target.to(device=probability.device, dtype=probability.dtype)
        friction = friction.reshape(friction.shape[0], -1).mean(dim=1)
        if supervision_mask is not None and cognitive_prediction is not None:
            predicted_friction = cognitive_prediction.detach().to(device=probability.device, dtype=probability.dtype)
            predicted_friction = predicted_friction.reshape(predicted_friction.shape[0], -1).mean(dim=1)
            friction = torch.where(supervision_mask, friction, predicted_friction)
    elif cognitive_prediction is not None:
        friction = cognitive_prediction.detach().to(device=probability.device, dtype=probability.dtype)
    else:
        friction = probability.new_zeros(probability.shape[0])
    friction = friction.reshape(friction.shape[0], -1).mean(dim=1).clamp(0, 1).reshape(-1, 1, 1, 1)
    boundary_weight = 1.0 + boundary_alpha * (1.0 - friction) * boundary
    bce_map = -(target * probability.log() + (1.0 - target) * torch.log1p(-probability))
    bce = (boundary_weight * bce_map).mean()
    dice = 1.0 - soft_dice_score(probability, target).mean()
    mask_loss = bce + dice

    zero = probability.new_zeros(())
    resolved_iou_target = hard_iou_score(probability.detach(), target)
    if iou_target is not None:
        resolved_iou_target = iou_target.to(device=probability.device, dtype=probability.dtype)
    iou_loss = zero if iou_prediction is None else F.mse_loss(iou_prediction, resolved_iou_target.detach())

    cognitive_loss = zero
    if cognitive_prediction is not None and cognitive_target is not None:
        prediction = cognitive_prediction.reshape(cognitive_prediction.shape[0], -1).mean(dim=1)
        cognition = cognitive_target.to(device=prediction.device, dtype=prediction.dtype)
        cognition = cognition.reshape(cognition.shape[0], -1).mean(dim=1).clamp(0, 1)
        per_sample_loss = F.smooth_l1_loss(prediction, cognition, reduction="none")
        if supervision_mask is None:
            cognitive_loss = per_sample_loss.mean()
        elif supervision_mask.any():
            cognitive_loss = per_sample_loss[supervision_mask].mean()

    trajectory_cognitive_loss = zero
    trajectory_supervision_mask = supervision_mask
    if trajectory_cognitive_supervision_mask is not None:
        trajectory_supervision_mask = trajectory_cognitive_supervision_mask.to(
            device=probability.device, dtype=torch.bool
        )
        trajectory_supervision_mask = trajectory_supervision_mask.reshape(
            trajectory_supervision_mask.shape[0], -1
        ).any(dim=1)
        if supervision_mask is not None:
            trajectory_supervision_mask = trajectory_supervision_mask & supervision_mask
    if trajectory_cognitive_prediction is not None and cognitive_target is not None:
        prediction = trajectory_cognitive_prediction.reshape(
            trajectory_cognitive_prediction.shape[0], -1
        ).mean(dim=1)
        cognition = cognitive_target.to(device=prediction.device, dtype=prediction.dtype)
        cognition = cognition.reshape(cognition.shape[0], -1).mean(dim=1).clamp(0, 1)
        per_sample_loss = F.smooth_l1_loss(prediction, cognition, reduction="none")
        if trajectory_supervision_mask is None:
            trajectory_cognitive_loss = per_sample_loss.mean()
        elif trajectory_supervision_mask.any():
            trajectory_cognitive_loss = per_sample_loss[trajectory_supervision_mask].mean()

    total = (
        mask_loss
        + lambda_iou * iou_loss
        + lambda_cognitive * cognitive_loss
        + lambda_trajectory_cognitive * trajectory_cognitive_loss
    )
    return {
        "loss": total,
        "loss_mask": mask_loss,
        "loss_bce": bce,
        "loss_dice": dice,
        "loss_iou": iou_loss,
        "loss_cognitive": cognitive_loss,
        "loss_trajectory_cognitive": trajectory_cognitive_loss,
        "boundary_band": boundary.detach(),
        "boundary_weight": boundary_weight.detach(),
        "iou_target": resolved_iou_target.detach(),
        "cognitive_supervision_mask": (
            supervision_mask.detach()
            if supervision_mask is not None
            else torch.ones(probability.shape[0], device=probability.device, dtype=torch.bool)
        ),
        "trajectory_cognitive_supervision_mask": (
            trajectory_supervision_mask.detach()
            if trajectory_supervision_mask is not None
            else torch.ones(probability.shape[0], device=probability.device, dtype=torch.bool)
        ),
    }
