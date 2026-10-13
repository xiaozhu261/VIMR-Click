from dataclasses import asdict, dataclass


@dataclass
class ProcessSAMConfig:
    """A/B process fusion over the unchanged EfficientViT-SAM interfaces."""

    variant: str = "E3"
    d_model: int = 256
    state_hidden: int = 128
    history_hidden: int = 128
    spatial_hidden: int = 32
    process_dim: int = 18
    process_mean: tuple[float, ...] = ()
    process_std: tuple[float, ...] = ()
    max_clicks: int = 5
    raster_sigma: float = 1.5
    spatial_gate_init: float = 0.05
    train_mask_decoder: bool = True
    use_event_history: bool = False
    event_dim: int = 11
    max_process_events: int = 64
    history_pooling: str = "gru"
    spatial_only: bool = False

    def __post_init__(self) -> None:
        if self.variant not in {"E0", "E1", "E2", "E3"}:
            raise ValueError("variant must be E0, E1, E2, or E3; E4 evaluates E3 with shuffled process")
        if self.max_clicks < 1 or self.raster_sigma <= 0:
            raise ValueError("max_clicks and raster_sigma must be positive")
        if self.event_dim != 11 or self.max_process_events < 1:
            raise ValueError("Process events require 11 fields and a positive event budget")
        if self.use_event_history and not self.use_history:
            raise ValueError("Event history requires E2 or E3")
        if self.history_pooling not in {"gru", "mean"}:
            raise ValueError("history_pooling must be gru or mean")
        if self.spatial_only and (not self.use_event_history or self.history_pooling != "mean"):
            raise ValueError("Spatial-only control requires unordered event pooling")
        if not -1 < self.spatial_gate_init < 1:
            raise ValueError("spatial_gate_init must be strictly between -1 and 1")
        if bool(self.process_mean) != bool(self.process_std):
            raise ValueError("process_mean and process_std must be supplied together")
        if self.process_mean and (
            len(self.process_mean) != self.process_dim or len(self.process_std) != self.process_dim
        ):
            raise ValueError("process normalization must match process_dim")
        if any(value <= 0 for value in self.process_std):
            raise ValueError("process_std must be positive")

    @property
    def use_history(self) -> bool:
        return self.variant in {"E2", "E3"}

    @property
    def use_spatial(self) -> bool:
        return self.variant == "E3"

    def to_dict(self) -> dict:
        return asdict(self)
