from typing import Any

import torch
import torch.nn.functional as F

from efficientvit.models.vimr import vimr_interactive_loss
from efficientvit.samcore.trainer.sam_trainer import SAMTrainer

__all__ = ["VIMRTrainer"]


class VIMRTrainer(SAMTrainer):
    """Two-round interaction trainer for the VIMR-Click MVP.

    It reuses the existing SA-1B data provider, selects one object per image,
    encodes the image once, then synthesizes a corrective second click from the
    largest current error.
    """

    @staticmethod
    def _corrective_click(probability: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        resized_target = F.interpolate(target, probability.shape[-2:], mode="nearest")
        error = (resized_target - probability).abs().flatten(1)
        flat_index = error.argmax(dim=1)
        width = probability.shape[-1]
        y = torch.div(flat_index, width, rounding_mode="floor")
        x = flat_index % width
        scale_x = 1024.0 / width
        scale_y = 1024.0 / probability.shape[-2]
        coords = torch.stack(((x.float() + 0.5) * scale_x, (y.float() + 0.5) * scale_y), dim=-1).unsqueeze(1)
        labels = resized_target.flatten(1).gather(1, flat_index.unsqueeze(1)).long()
        return coords, labels

    def run_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        images = feed_dict["image"]
        target = feed_dict["masks"][:, :1]
        first_coords = feed_dict["points"][:, :1]
        if images.shape[-1] == 512:
            first_coords = first_coords * 2
        labels = torch.ones(first_coords.shape[:2], device=images.device, dtype=torch.long)
        batch_size = images.shape[0]
        first_history = torch.cat((first_coords / 1024.0, labels.unsqueeze(-1).float()), dim=-1)

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            first = self.model(
                images=images,
                point_coords=first_coords,
                point_labels=labels,
                click_history=first_history,
                multimask_output=False,
            )
            previous = first["final_mask"].detach()
            second_coords, second_labels = self._corrective_click(previous, target)
            second_event = torch.cat((second_coords / 1024.0, second_labels.unsqueeze(-1).float()), dim=-1)
            click_history = torch.cat((first_history, second_event), dim=1)
            viewport = torch.zeros(batch_size, 7, device=images.device, dtype=images.dtype)
            viewport[:, :2] = second_coords[:, 0] / 1024.0
            viewport[:, 2] = torch.log(torch.tensor(2.0, device=images.device, dtype=images.dtype))
            viewport[:, 6] = 1
            default_viewport = torch.zeros_like(viewport)
            default_viewport[:, :2] = 0.5
            viewport_history = torch.stack((default_viewport, viewport), dim=1)
            second = self.model(
                image_embeddings=first["image_embeddings"],
                point_coords=second_coords,
                point_labels=second_labels,
                previous_mask=previous,
                viewport_state=viewport,
                click_history=click_history,
                viewport_history=viewport_history,
                mask_history=previous.unsqueeze(1),
                multimask_output=False,
            )
            cfg = self.model.module.vimr_config if hasattr(self.model, "module") else self.model.vimr_config
            losses = vimr_interactive_loss(
                second["final_mask"],
                target,
                previous,
                lambda_safe=cfg.lambda_safe,
                lambda_monotonic=cfg.lambda_monotonic,
                monotonic_margin=cfg.monotonic_margin,
            )

        self.scaler.scale(losses["loss"]).backward()
        return {**losses, "output": second["final_mask"]}
