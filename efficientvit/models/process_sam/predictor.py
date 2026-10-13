import numpy as np
import torch

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

from .model import ProcessAwareEfficientViTSam


class ProcessAwareSamPredictor:
    """Cache the image once and append one click with its observed telemetry."""

    def __init__(self, model: ProcessAwareEfficientViTSam):
        self.model = model
        self.base_predictor = EfficientViTSamPredictor(model)
        self.reset_interaction()

    def reset_interaction(self):
        self.clicks = []
        self.process_history = []
        self.viewport_history = []
        self.trajectory_history = []
        self.previous_mask = None

    @torch.inference_mode()
    def set_image(self, image: np.ndarray, image_format="RGB"):
        self.base_predictor.set_image(image, image_format)
        self.reset_interaction()

    @torch.inference_mode()
    def predict(self, click, *, process_features=None, viewport=None, trajectory=None, multimask_output=False):
        """click=(x_px,y_px,label); viewport=(zoom,cx,cy,w,h) in image fractions.

        trajectory=[T,3] contains relative (t,dx,dy); process_features follows
        PROCESS_FEATURE_NAMES from the strict dataset. Missing telemetry is masked.
        """
        if not self.base_predictor.is_image_set:
            raise RuntimeError("Set an image before prediction")
        if len(self.clicks) >= self.model.process_config.max_clicks:
            raise ValueError("Click budget exhausted; reset_interaction to start a new object")
        features = self.base_predictor.features
        device = features.device
        new_click = torch.as_tensor(click, dtype=torch.float32, device=device)
        if new_click.shape != (3,) or not torch.isfinite(new_click).all() or new_click[2] not in (0, 1):
            raise ValueError("Provide exactly one finite (x,y,label) click, label 0 or 1")
        process = (
            None if process_features is None else torch.as_tensor(process_features, dtype=torch.float32, device=device)
        )
        view = None if viewport is None else torch.as_tensor(viewport, dtype=torch.float32, device=device)
        path = None if trajectory is None else torch.as_tensor(trajectory, dtype=torch.float32, device=device)
        if process is not None and (
            process.shape != (self.model.process_config.process_dim,) or not torch.isfinite(process).all()
        ):
            raise ValueError("process_features must be a finite [process_dim] vector")
        if view is not None and (
            view.shape != (5,) or not torch.isfinite(view).all() or view[0] <= 0 or (view[3:] <= 0).any()
        ):
            raise ValueError("viewport must be finite (zoom,cx,cy,w,h), with positive zoom/size")
        if path is not None and (
            path.ndim != 2 or path.shape[-1] != 3 or len(path) == 0 or not torch.isfinite(path).all()
        ):
            raise ValueError("trajectory must be a finite nonempty [T,3] array")
        # Build a tentative history; commit only when inference succeeds.
        clicks = self.clicks + [new_click]
        processes = self.process_history + [process]
        views = self.viewport_history + [view]
        paths = self.trajectory_history + [path]
        count = len(clicks)
        max_samples = max((len(value) for value in paths if value is not None), default=1)
        trajectories = torch.zeros(1, count, max_samples, 3, device=device)
        trajectory_valid = torch.zeros(1, count, max_samples, device=device, dtype=torch.bool)
        for index, value in enumerate(paths):
            if value is not None:
                trajectories[0, index, : len(value)] = value
                trajectory_valid[0, index, : len(value)] = True
        raw_clicks = torch.stack(clicks)
        output = self.model(
            image_embeddings=features,
            point_coords=self.base_predictor.apply_coords_torch(raw_clicks[:, :2]).unsqueeze(0),
            point_labels=raw_clicks[:, 2].long().unsqueeze(0),
            process_features=torch.stack(
                [
                    torch.zeros(self.model.process_config.process_dim, device=device) if value is None else value
                    for value in processes
                ]
            ).unsqueeze(0),
            process_feature_valid=torch.tensor([[value is not None for value in processes]], device=device),
            viewports=torch.stack(
                [torch.zeros(5, device=device) if value is None else value for value in views]
            ).unsqueeze(0),
            viewport_valid=torch.tensor([[value is not None for value in views]], device=device),
            click_trajectories=trajectories,
            trajectory_valid_mask=trajectory_valid,
            coordinate_scale=features.new_tensor(
                (self.base_predictor.input_size[1], self.base_predictor.input_size[0])
            ).unsqueeze(0),
            previous_mask=self.previous_mask,
            multimask_output=multimask_output,
        )
        candidate = int(output["iou_predictions"][0].argmax()) if multimask_output else 0
        low_logits = output["mask_logits"][:, candidate : candidate + 1]
        full_logits = self.model.postprocess_masks(
            low_logits, self.base_predictor.input_size, self.base_predictor.original_size
        )
        self.clicks, self.process_history, self.viewport_history, self.trajectory_history = (
            clicks,
            processes,
            views,
            paths,
        )
        self.previous_mask = low_logits.sigmoid()
        return {
            "masks": (full_logits[0] > 0).cpu().numpy(),
            "low_res_masks": self.previous_mask[0].cpu().numpy(),
            "iou_predictions": output["iou_predictions"][0].cpu().numpy(),
            "spatial_attention": (
                None if output["spatial_attention"] is None else output["spatial_attention"][0].cpu().numpy()
            ),
            "spatial_gate": None if output["spatial_gate"] is None else float(output["spatial_gate"]),
        }
