from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CAISConfig:
    """Configuration for the standalone cognitive interactive segmentation model."""

    d_model: int = 256
    cognitive_layers: int = 2
    cognitive_heads: int = 8
    cognitive_ffn_dim: int = 1024
    trajectory_channels: int = 64
    trajectory_samples: int = 16
    trajectory_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    trajectory_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    trajectory_encoder: str = "cnn_pool"
    trajectory_processing: str = "legacy"
    trajectory_window_ms: float = 2000.0
    process_feature_dim: int = 0
    process_mean: tuple[float, ...] = ()
    process_std: tuple[float, ...] = ()
    process_gate_init: float = 0.05
    trajectory_token_bottleneck: int = 64
    trajectory_token_content: str = "encoded"
    trajectory_dense_channels: int = 32
    trajectory_raster_sigma: float = 1.5
    trajectory_gate_max: float = 1.0
    trajectory_dropout: float = 0.0
    use_trajectory: bool = True
    interaction_memory: str = "none"
    memory_size: int = 8
    memory_layers: int = 1
    memory_heads: int = 8
    memory_ffn_dim: int = 512
    memory_mask_channels: int = 32
    memory_gate_init: float = 0.05
    prompt_integration: str = "replace"
    attention_integration: str = "replace"
    prompt_residual_gate_init: float = 0.0
    boundary_alpha: float = 2.0
    boundary_width: int = 2
    lambda_iou: float = 1.0
    lambda_cognitive: float = 0.075
    lambda_trajectory_cognitive: float = 0.0
    freeze_image_encoder: bool = True
    freeze_prompt_encoder: bool = False

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "CAISConfig":
        if config is None:
            return cls()
        normalized = {key.lower(): value for key, value in config.items()}
        for key in ("trajectory_mean", "trajectory_std", "process_mean", "process_std"):
            if key in normalized:
                normalized[key] = tuple(float(value) for value in normalized[key])
        valid = cls.__dataclass_fields__
        return cls(**{key: value for key, value in normalized.items() if key in valid})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
