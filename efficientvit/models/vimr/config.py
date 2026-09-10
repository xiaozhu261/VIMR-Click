from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class VIMRConfig:
    """Configuration for the VIMR-Click MVP modules."""

    enabled: bool = True
    use_viewport: bool = True
    use_history: bool = True
    use_residual: bool = True
    use_safe_update: bool = True
    safe_update: bool = True
    d_model: int = 256
    memory_layers: int = 4
    memory_heads: int = 8
    memory_ffn_dim: int = 1024
    max_history: int = 16
    viewport_fourier_levels: int = 6
    action_embedding_dim: int = 32
    lambda_safe: float = 0.2
    lambda_monotonic: float = 0.1
    monotonic_margin: float = 0.0
    gate_init_probability: float = 0.1
    freeze_image_encoder: bool = True
    freeze_prompt_encoder: bool = False
    enable_expert_router: bool = False
    enable_detail_refinement: bool = False
    router_temperature: float = 1.0
    detail_refinement_strength: float = 1.0

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "VIMRConfig":
        if config is None:
            return cls()
        normalized = {key.lower(): value for key, value in config.items()}
        valid = cls.__dataclass_fields__
        return cls(**{key: value for key, value in normalized.items() if key in valid})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
