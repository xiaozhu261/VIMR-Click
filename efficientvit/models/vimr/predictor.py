from typing import Any, Optional

import numpy as np
import torch

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

from .model import VIMREfficientViTSam
from .modules import ViewportIntentEncoder


class VIMREfficientViTSamPredictor:
    """Stateful multi-round predictor that reuses one cached image embedding."""

    def __init__(self, model: VIMREfficientViTSam) -> None:
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
        """Encode an image once and reset target-specific interaction state."""
        self.base_predictor.set_image(image, image_format)
        self.reset_interaction()

    @torch.inference_mode()
    def set_image_batch(self, image: torch.Tensor) -> None:
        self.base_predictor.set_image_batch(image)
        self.reset_interaction()

    def reset_interaction(self) -> None:
        """Clear clicks, viewports, and prior masks without discarding image features."""
        self.click_history: list[torch.Tensor] = []
        self.viewport_history: list[torch.Tensor] = []
        self.mask_history: list[torch.Tensor] = []
        self.previous_mask: Optional[torch.Tensor] = None

    def _default_viewport(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return ViewportIntentEncoder.default_state(1, device, dtype)[0]

    @torch.inference_mode()
    def predict(
        self,
        clicks: np.ndarray | torch.Tensor,
        viewport_state: Optional[np.ndarray | torch.Tensor] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> dict[str, Any]:
        """Run one click round and automatically update interaction history.

        ``clicks`` has shape ``[N,3]`` with original-image ``(x,y,label)`` rows.
        The current SAM prompt receives all rows supplied in this call; one history
        event is recorded using the final row.
        """
        if not self.is_image_set or self.features is None:
            raise RuntimeError("An image must be set before VIMR prediction")
        device, dtype = self.features.device, self.features.dtype
        clicks_tensor = torch.as_tensor(clicks, device=device, dtype=dtype)
        if clicks_tensor.ndim == 1:
            clicks_tensor = clicks_tensor.unsqueeze(0)
        if clicks_tensor.ndim != 2 or clicks_tensor.shape[-1] != 3:
            raise ValueError(f"clicks must have shape [N,3], got {tuple(clicks_tensor.shape)}")

        original_coords = clicks_tensor[:, :2]
        transformed_coords = self.base_predictor.apply_coords_torch(original_coords).to(device=device, dtype=dtype)
        point_coords = transformed_coords.unsqueeze(0)
        point_labels = clicks_tensor[:, 2].long().unsqueeze(0)
        original_height, original_width = self.base_predictor.original_size
        normalized_event = clicks_tensor[-1].clone()
        normalized_event[0] /= original_width
        normalized_event[1] /= original_height
        current_viewport = (
            self._default_viewport(dtype, device)
            if viewport_state is None
            else torch.as_tensor(viewport_state, device=device, dtype=dtype).reshape(7)
        )
        click_history = torch.stack((*self.click_history, normalized_event), dim=0).unsqueeze(0)
        viewport_history = torch.stack((*self.viewport_history, current_viewport), dim=0).unsqueeze(0)
        mask_history = torch.stack(self.mask_history, dim=1) if self.mask_history else None

        output = self.model(
            image_embeddings=self.features,
            point_coords=point_coords,
            point_labels=point_labels,
            previous_mask=self.previous_mask,
            viewport_state=current_viewport.unsqueeze(0),
            click_history=click_history,
            viewport_history=viewport_history,
            mask_history=mask_history,
            multimask_output=multimask_output,
        )
        candidate = (
            output["iou_predictions"].argmax(dim=1)
            if multimask_output
            else torch.zeros(1, dtype=torch.long, device=device)
        )
        batch_indices = torch.arange(output["final_mask"].shape[0], device=device)
        selected_low_res = output["final_mask"][batch_indices, candidate].unsqueeze(1)
        selected_gate = output["gate_map"][batch_indices, candidate].unsqueeze(1)
        selected_delta = output["delta_map"][batch_indices, candidate].unsqueeze(1)
        full_resolution = self.model.postprocess_masks(
            selected_low_res,
            self.base_predictor.input_size,
            self.base_predictor.original_size,
        )
        self.click_history.append(normalized_event)
        self.viewport_history.append(current_viewport)
        self.mask_history.append(selected_low_res)
        self.previous_mask = selected_low_res

        masks = torch.logit(full_resolution.clamp(1e-6, 1 - 1e-6)) if return_logits else full_resolution > 0.5
        return {
            "masks": masks[0].cpu().numpy(),
            "iou_predictions": output["iou_predictions"][0].cpu().numpy(),
            "low_res_masks": selected_low_res[0].cpu().numpy(),
            "draft_mask": output["draft_mask"][0].cpu().numpy(),
            "gate_map": selected_gate[0].cpu().numpy(),
            "delta_map": selected_delta[0].cpu().numpy(),
            "viewport_map": output["viewport_map"][0].cpu().numpy(),
        }
