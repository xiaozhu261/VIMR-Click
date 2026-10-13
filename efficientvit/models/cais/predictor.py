from typing import Any, Optional

import numpy as np
import torch

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

from .model import CAISEfficientViTSam


class CAISEfficientViTSamPredictor:
    """Stateful CAIS predictor that caches image features and accumulates clicks."""

    def __init__(self, model: CAISEfficientViTSam) -> None:
        self.model = model
        self.base_predictor = EfficientViTSamPredictor(model)
        self.reset_interaction()

    @property
    def is_image_set(self) -> bool:
        return self.base_predictor.is_image_set

    @property
    def features(self) -> Optional[torch.Tensor]:
        return self.base_predictor.features

    @torch.inference_mode()
    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        self.base_predictor.set_image(image, image_format)
        self.reset_interaction()

    @torch.inference_mode()
    def set_image_batch(self, image: torch.Tensor) -> None:
        self.base_predictor.set_image_batch(image)
        self.reset_interaction()

    def reset_interaction(self) -> None:
        self.prompt_coords: list[torch.Tensor] = []
        self.prompt_labels: list[torch.Tensor] = []
        self.trajectory_history: list[Optional[torch.Tensor]] = []
        self.trajectory_mask_history: list[Optional[torch.Tensor]] = []
        self.interaction_mask_history: list[Optional[torch.Tensor]] = []
        self.previous_mask: Optional[torch.Tensor] = None

    @staticmethod
    def _normalize_trajectory_input(
        click_count: int,
        trajectories: Optional[np.ndarray | torch.Tensor],
        valid_mask: Optional[np.ndarray | torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if trajectories is None:
            return None, None
        trajectory_tensor = torch.as_tensor(trajectories, device=device, dtype=dtype)
        if trajectory_tensor.ndim == 2:
            trajectory_tensor = trajectory_tensor.unsqueeze(0)
        if trajectory_tensor.ndim != 3 or trajectory_tensor.shape[0] != click_count or trajectory_tensor.shape[-1] != 3:
            raise ValueError(
                "click_trajectories must have shape [N,T,3] matching the new clicks, "
                f"got {tuple(trajectory_tensor.shape)}"
            )
        if valid_mask is None:
            mask_tensor = torch.ones(trajectory_tensor.shape[:-1], device=device, dtype=torch.bool)
        else:
            mask_tensor = torch.as_tensor(valid_mask, device=device, dtype=torch.bool)
            if mask_tensor.ndim == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            if mask_tensor.shape != trajectory_tensor.shape[:-1]:
                raise ValueError(
                    f"trajectory_valid_mask must have shape {tuple(trajectory_tensor.shape[:-1])}, "
                    f"got {tuple(mask_tensor.shape)}"
                )
        return trajectory_tensor, mask_tensor

    def _trajectory_batch(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        available = [trajectory for trajectory in self.trajectory_history if trajectory is not None]
        if not available:
            return None, None
        max_samples = max(trajectory.shape[0] for trajectory in available)
        trajectories = torch.zeros(1, len(self.trajectory_history), max_samples, 3, device=device, dtype=dtype)
        valid_mask = torch.zeros(1, len(self.trajectory_history), max_samples, device=device, dtype=torch.bool)
        for index, (trajectory, mask) in enumerate(zip(self.trajectory_history, self.trajectory_mask_history)):
            if trajectory is not None and mask is not None:
                trajectories[0, index, : trajectory.shape[0]] = trajectory
                valid_mask[0, index, : trajectory.shape[0]] = mask
        return trajectories, valid_mask

    @torch.inference_mode()
    def predict(
        self,
        clicks: np.ndarray | torch.Tensor,
        multimask_output: bool = False,
        return_logits: bool = False,
        *,
        click_trajectories: Optional[np.ndarray | torch.Tensor] = None,
        trajectory_valid_mask: Optional[np.ndarray | torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Append new ``(x,y,label)`` clicks and their relative ``(t,x,y)`` trajectories."""
        if not self.is_image_set or self.features is None:
            raise RuntimeError("An image must be set before CAIS prediction")
        device, dtype = self.features.device, self.features.dtype
        clicks_tensor = torch.as_tensor(clicks, device=device, dtype=dtype)
        if clicks_tensor.ndim == 1:
            clicks_tensor = clicks_tensor.unsqueeze(0)
        if clicks_tensor.ndim != 2 or clicks_tensor.shape[-1] != 3:
            raise ValueError(f"clicks must have shape [N,3], got {tuple(clicks_tensor.shape)}")

        transformed_coords = self.base_predictor.apply_coords_torch(clicks_tensor[:, :2]).to(device=device, dtype=dtype)
        new_labels = clicks_tensor[:, 2].long()
        self.prompt_coords.append(transformed_coords)
        self.prompt_labels.append(new_labels)
        trajectories, valid_mask = self._normalize_trajectory_input(
            len(clicks_tensor), click_trajectories, trajectory_valid_mask, device, dtype
        )
        for index in range(len(clicks_tensor)):
            self.trajectory_history.append(None if trajectories is None else trajectories[index])
            self.trajectory_mask_history.append(None if valid_mask is None else valid_mask[index])
            self.interaction_mask_history.append(
                None if self.previous_mask is None else self.previous_mask[0].detach().clone()
            )
        trajectory_batch, trajectory_mask_batch = self._trajectory_batch(device, dtype)
        interaction_masks = torch.zeros(
            1,
            len(self.interaction_mask_history),
            1,
            256,
            256,
            device=device,
            dtype=dtype,
        )
        interaction_mask_valid = torch.zeros(
            1, len(self.interaction_mask_history), device=device, dtype=torch.bool
        )
        for index, mask in enumerate(self.interaction_mask_history):
            if mask is not None:
                interaction_masks[0, index] = mask.to(device=device, dtype=dtype)
                interaction_mask_valid[0, index] = True

        output = self.model(
            image_embeddings=self.features,
            point_coords=torch.cat(self.prompt_coords, dim=0).unsqueeze(0),
            point_labels=torch.cat(self.prompt_labels, dim=0).unsqueeze(0),
            click_trajectories=trajectory_batch,
            trajectory_valid_mask=trajectory_mask_batch,
            trajectory_coordinate_scale=self.features.new_tensor(
                (self.base_predictor.input_size[1], self.base_predictor.input_size[0])
            ).unsqueeze(0),
            previous_mask=self.previous_mask,
            interaction_masks=interaction_masks,
            interaction_mask_valid=interaction_mask_valid,
            multimask_output=multimask_output,
        )
        candidate = (
            output["iou_predictions"].argmax(dim=1)
            if multimask_output
            else torch.zeros(1, dtype=torch.long, device=device)
        )
        batch_indices = torch.arange(output["final_mask"].shape[0], device=device)
        selected_low_res = output["final_mask"][batch_indices, candidate].unsqueeze(1)
        full_resolution = self.model.postprocess_masks(
            selected_low_res,
            self.base_predictor.input_size,
            self.base_predictor.original_size,
        )
        self.previous_mask = selected_low_res
        masks = torch.logit(full_resolution.clamp(1e-6, 1 - 1e-6)) if return_logits else full_resolution > 0.5
        return {
            "masks": masks[0].cpu().numpy(),
            "iou_predictions": output["iou_predictions"][0].cpu().numpy(),
            "sam_iou_predictions": output["sam_iou_predictions"][0].cpu().numpy(),
            "cognitive_friction": float(output["cognitive_friction"][0].cpu()),
            "cognitive_vector": output["cognitive_vector"][0].cpu().numpy(),
            "low_res_masks": selected_low_res[0].cpu().numpy(),
        }
