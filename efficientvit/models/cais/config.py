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
    use_trajectory: bool = True
    prompt_integration: str = "replace"
    attention_integration: str = "replace"
    boundary_alpha: float = 2.0
    boundary_width: int = 2
    lambda_iou: float = 1.0
    lambda_cognitive: float = 0.075
    freeze_image_encoder: bool = True
    freeze_prompt_encoder: bool = False

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "CAISConfig":
        if config is None:
            return cls()
        normalized = {key.lower(): value for key, value in config.items()}
        valid = cls.__dataclass_fields__
        return cls(**{key: value for key, value in normalized.items() if key in valid})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
