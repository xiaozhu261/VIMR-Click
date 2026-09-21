from typing import Any

import torch
import torch.nn.functional as F

from efficientvit.models.cais import cais_multitask_loss
from efficientvit.samcore.trainer.sam_trainer import SAMTrainer

__all__ = ["CAISTrainer"]


class CAISTrainer(SAMTrainer):
    """Two-round trainer for the standalone CAIS architecture."""

    @staticmethod
    def _corrective_click(probability: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        resized_target = F.interpolate(target, probability.shape[-2:], mode="nearest")
        error = (resized_target - probability).abs().flatten(1)
        flat_index = error.argmax(dim=1)
        width = probability.shape[-1]
        y = torch.div(flat_index, width, rounding_mode="floor")
        x = flat_index % width
        coords = torch.stack(
            (
                (x.float() + 0.5) * (1024.0 / width),
                (y.float() + 0.5) * (1024.0 / probability.shape[-2]),
            ),
            dim=-1,
        ).unsqueeze(1)
        labels = resized_target.flatten(1).gather(1, flat_index.unsqueeze(1)).long()
        return coords, labels

    @staticmethod
    def _synthetic_approach_trajectory(point_coords: torch.Tensor, samples: int) -> torch.Tensor:
        progress = torch.linspace(0, 1, samples, device=point_coords.device, dtype=point_coords.dtype)
        path = torch.stack(
            (
                -2.0 * (1.0 - progress),
                -0.05 * (1.0 - progress),
                0.03 * (1.0 - progress),
            ),
            dim=-1,
        )
        return path.reshape(1, 1, samples, 3).expand(point_coords.shape[0], point_coords.shape[1], -1, -1)

    def run_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        images = feed_dict["image"]
        target = feed_dict["masks"][:, :1]
        first_coords = feed_dict["points"][:, :1]
        if images.shape[-1] == 512:
            first_coords = first_coords * 2
        labels = torch.ones(first_coords.shape[:2], device=images.device, dtype=torch.long)
        batch_size = images.shape[0]
        cfg = self.model.module.cais_config if hasattr(self.model, "module") else self.model.cais_config

        first_trajectory = feed_dict.get("click_trajectories")
        first_trajectory_mask = feed_dict.get("trajectory_valid_mask")
        if first_trajectory is None:
            first_trajectory = self._synthetic_approach_trajectory(first_coords, cfg.trajectory_samples)
        else:
            first_trajectory = first_trajectory.to(device=images.device, dtype=images.dtype)
            if first_trajectory.ndim == 3:
                first_trajectory = first_trajectory.unsqueeze(1)
            first_trajectory = first_trajectory[:, :1]
            if first_trajectory_mask is not None:
                first_trajectory_mask = first_trajectory_mask.to(device=images.device, dtype=torch.bool)
                if first_trajectory_mask.ndim == 2:
                    first_trajectory_mask = first_trajectory_mask.unsqueeze(1)
                first_trajectory_mask = first_trajectory_mask[:, :1]
        cognitive_target = feed_dict.get("cognitive_friction")
        if cognitive_target is not None:
            cognitive_target = cognitive_target.to(device=images.device, dtype=images.dtype)
            cognitive_target = cognitive_target.reshape(batch_size, -1)[:, 0]
        cognitive_supervision_mask = feed_dict.get("cognitive_supervision_mask")
        if cognitive_supervision_mask is not None:
            cognitive_supervision_mask = cognitive_supervision_mask.to(device=images.device, dtype=torch.bool)
            cognitive_supervision_mask = cognitive_supervision_mask.reshape(batch_size, -1)[:, 0]

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            first = self.model(
                images=images,
                point_coords=first_coords,
                point_labels=labels,
                click_trajectories=first_trajectory,
                trajectory_valid_mask=first_trajectory_mask,
                multimask_output=False,
            )
            previous = first["final_mask"].detach()
            second_coords, second_labels = self._corrective_click(previous, target)
            second_trajectory = self._synthetic_approach_trajectory(second_coords, first_trajectory.shape[2])
            combined_trajectory_mask = None
            if first_trajectory_mask is not None:
                second_mask = torch.ones(second_trajectory.shape[:-1], device=images.device, dtype=torch.bool)
                combined_trajectory_mask = torch.cat((first_trajectory_mask, second_mask), dim=1)
            second = self.model(
                image_embeddings=first["image_embeddings"],
                point_coords=torch.cat((first_coords, second_coords), dim=1),
                point_labels=torch.cat((labels, second_labels), dim=1),
                click_trajectories=torch.cat((first_trajectory, second_trajectory), dim=1),
                trajectory_valid_mask=combined_trajectory_mask,
                previous_mask=previous,
                multimask_output=False,
            )
            losses = cais_multitask_loss(
                second["final_mask"],
                target,
                iou_prediction=second["iou_predictions"],
                cognitive_prediction=second["cognitive_friction"],
                cognitive_target=cognitive_target,
                cognitive_supervision_mask=cognitive_supervision_mask,
                boundary_alpha=cfg.boundary_alpha,
                boundary_width=cfg.boundary_width,
                lambda_iou=cfg.lambda_iou,
                lambda_cognitive=cfg.lambda_cognitive,
            )

        self.scaler.scale(losses["loss"]).backward()
        return {**losses, "output": second["final_mask"]}
