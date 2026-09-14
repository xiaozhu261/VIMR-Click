"""AMP-safe entry point for balanced VIMR multimask phase-2 training."""

from __future__ import annotations

import torch

import tools.run_vimr_coco_multimask_balanced as phase2
from efficientvit.models.vimr.losses import vimr_interactive_loss as probability_loss


def fp32_interactive_loss(
    final_mask: torch.Tensor,
    target: torch.Tensor,
    previous_mask: torch.Tensor | None = None,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Keep the probability-domain BCE and Dice terms outside CUDA autocast."""
    with torch.amp.autocast("cuda", enabled=False):
        return probability_loss(
            final_mask.float(),
            target.float(),
            None if previous_mask is None else previous_mask.float(),
            **kwargs,
        )


def main() -> None:
    phase2.vimr_interactive_loss = fp32_interactive_loss
    phase2.main()


if __name__ == "__main__":
    main()
