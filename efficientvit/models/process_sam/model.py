import torch
from torch.nn import functional as F

from efficientvit.models.efficientvit.sam import EfficientViTSam

from .config import ProcessSAMConfig
from .modules import InteractionProcessEncoder, ProcessAwareFusion


class ProcessAwareEfficientViTSam(EfficientViTSam):
    """Three parallel encoders with the original SAM mask decoder."""

    def __init__(self, base_model: EfficientViTSam, config: ProcessSAMConfig | None = None) -> None:
        super().__init__(
            base_model.image_encoder, base_model.prompt_encoder, base_model.mask_decoder, base_model.image_size
        )
        self.process_config = config or ProcessSAMConfig()
        if self.process_config.d_model != self.prompt_encoder.embed_dim:
            raise ValueError("Process dimensions must match the native SAM prompt dimensions")
        if self.process_config.variant != "E0":
            self.process_encoder = InteractionProcessEncoder(self.process_config)
            self.process_fusion = ProcessAwareFusion(self.process_config)
        for module in (self.image_encoder, self.prompt_encoder):
            module.requires_grad_(False)
        self.mask_decoder.requires_grad_(self.process_config.train_mask_decoder)

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_encoder.eval()
        self.prompt_encoder.eval()
        if not self.process_config.train_mask_decoder:
            self.mask_decoder.eval()
        return self

    def parameter_summary(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "process": sum(p.numel() for name, p in self.named_parameters() if name.startswith("process_")),
        }

    def forward(
        self,
        *,
        image_embeddings=None,
        images=None,
        point_coords,
        point_labels,
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
        previous_mask=None,
        previous_mask_logits=None,
        boxes=None,
        multimask_output=False,
        use_process=True,
    ) -> dict[str, torch.Tensor | None]:
        if image_embeddings is None:
            if images is None:
                raise ValueError("Provide cached image_embeddings or preprocessed images")
            with torch.no_grad():
                image_embeddings = self.image_encoder(images)
        batch = image_embeddings.shape[0]
        coords = point_coords.to(device=image_embeddings.device)  # retain native FP32 positions
        labels = point_labels.to(device=image_embeddings.device)
        if coords.ndim != 3 or coords.shape[0] != batch or coords.shape[-1] != 2 or labels.shape != coords.shape[:2]:
            raise ValueError("Point tensors must be [B,N,2] and [B,N] matching image batch")
        if not 1 <= coords.shape[1] <= self.process_config.max_clicks:
            raise ValueError(f"This protocol supports 1..{self.process_config.max_clicks} clicks")
        mask_prompt = None
        if previous_mask_logits is not None:
            if previous_mask is not None:
                raise ValueError("Supply probability or logit mask prompts, not both")
            mask_prompt = previous_mask_logits.to(device=image_embeddings.device, dtype=torch.float32)
            if mask_prompt.shape != (batch, 1, 256, 256) or not torch.isfinite(mask_prompt).all():
                raise ValueError("Previous logits must be finite [B,1,256,256]")
        elif previous_mask is not None:
            probability = previous_mask.to(image_embeddings)
            if probability.ndim == 3:
                probability = probability.unsqueeze(1)
            if probability.shape[-2:] != (256, 256):
                probability = F.interpolate(probability, (256, 256), mode="bilinear", align_corners=False)
            eps = torch.finfo(probability.dtype).eps
            mask_prompt = torch.logit(probability.clamp(eps, 1 - eps))
        # Decode native prompts per sample: SAM's point padding token is a real
        # prompt token, so remove batch padding before the prompt encoder.
        sparse_list, dense_list = [], []
        for index in range(batch):
            valid = labels[index] >= 0
            sparse, dense = self.prompt_encoder(
                points=(coords[index : index + 1, valid], labels[index : index + 1, valid]),
                boxes=None if boxes is None else boxes[index : index + 1].to(coords.device),
                masks=None if mask_prompt is None else mask_prompt[index : index + 1],
            )
            sparse_list.append(sparse)
            dense_list.append(dense)
        encoding = attention = gate = process_tokens = None
        decoder_image = image_embeddings
        if self.process_config.variant != "E0" and use_process:
            encoding = self.process_encoder(
                coords,
                labels,
                process_features=process_features,
                process_feature_valid=process_feature_valid,
                viewports=viewports,
                viewport_valid=viewport_valid,
                click_trajectories=click_trajectories,
                trajectory_valid_mask=trajectory_valid_mask,
                process_events=process_events,
                process_event_valid=process_event_valid,
                process_spatial_valid=process_spatial_valid,
                coordinate_scale=coordinate_scale,
                canvas_size=self.prompt_encoder.input_image_size,
                embedding_size=image_embeddings.shape[-2:],
            )
            decoder_image, sparse_list, process_tokens, attention, gate = self.process_fusion(
                image_embeddings, sparse_list, encoding
            )
        logits, qualities = [], []
        for index in range(batch):
            sparse = sparse_list[index]
            sample_logits, sample_quality = self.mask_decoder(
                image_embeddings=decoder_image[index : index + 1],
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense_list[index],
                multimask_output=multimask_output,
            )
            logits.append(sample_logits)
            qualities.append(sample_quality)
        mask_logits = torch.cat(logits)
        return {
            "mask_logits": mask_logits,
            "final_mask": mask_logits.sigmoid(),
            "iou_predictions": torch.cat(qualities),
            "decoder_image_embeddings": decoder_image,
            "state_token": None if encoding is None else encoding.state_token,
            "history_token": None if encoding is None else encoding.history_token,
            "spatial_features": None if encoding is None else encoding.spatial_features,
            "spatial_maps": None if encoding is None else encoding.spatial_maps,
            "process_tokens": process_tokens,
            "spatial_attention": attention,
            "spatial_gate": gate,
        }
