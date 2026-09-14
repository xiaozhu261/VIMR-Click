"""Stable AMP entry point for balanced VIMR multimask phase-2 training."""

from __future__ import annotations

import torch

import tools.run_vimr_coco_multimask_balanced as phase2
from tools.run_vimr_coco_multimask_balanced_v2 import fp32_interactive_loss


def main() -> None:
    phase2.vimr_interactive_loss = fp32_interactive_loss
    original_grad_scaler = torch.amp.GradScaler

    def conservative_grad_scaler(device: str, enabled: bool):
        return original_grad_scaler(
            device,
            enabled=enabled,
            init_scale=1024.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )

    phase2.torch.amp.GradScaler = conservative_grad_scaler
    phase2.main()


if __name__ == "__main__":
    main()
