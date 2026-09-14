"""Full COCO evaluation of seed-42 VIMR with predicted-IoU multimask selection."""

from __future__ import annotations

import json

import numpy as np
import torch

import tools.eval_coco_click_vimr_full as single_eval
from applications.efficientvit_sam.sam_eval_utils import Clicker
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from tools.eval_vimr_coco_quick import viewport_for_mode


PROFILED_BASELINE_MULTIMASK_MACS_G = 36.63825664
PROFILED_VIMR_MULTIMASK_MACS_G = 50.19106048
PUBLISHED_BASELINE_MACS_G = 35.0
NORMALIZED_VIMR_MULTIMASK_MACS_G = (
    PUBLISHED_BASELINE_MACS_G
    + PROFILED_VIMR_MULTIMASK_MACS_G
    - PROFILED_BASELINE_MULTIMASK_MACS_G
)


def run_vimr_interaction_multimask(
    model: torch.nn.Module,
    predictor: EfficientViTSamPredictor,
    target: np.ndarray,
    clicks: list[int],
    viewport_mode: str,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Select one of three VIMR masks using predicted rather than oracle IoU."""
    clicker = Clicker(gt_mask=target)
    prediction = np.zeros_like(target, dtype=bool)
    click_events: list[torch.Tensor] = []
    viewports: list[torch.Tensor] = []
    mask_history: list[torch.Tensor] = []
    previous: torch.Tensor | None = None
    masks_at_click: dict[int, np.ndarray] = {}
    scores_at_click: dict[int, float] = {}
    device = predictor.features.device
    dtype = predictor.features.dtype

    for round_index in range(clicks[-1]):
        clicker.make_next_click(prediction)
        click = clicker.clicks_list[-1]
        original_coords = torch.tensor(
            [[click.coords[1], click.coords[0]]], device=device, dtype=dtype
        )
        coords = predictor.apply_coords_torch(original_coords).to(device=device, dtype=dtype).unsqueeze(0)
        labels = torch.tensor([[int(click.is_positive)]], device=device, dtype=torch.long)
        viewport = viewport_for_mode(
            coords,
            round_index,
            viewports[-1] if viewports else None,
            viewport_mode,
        )
        event = torch.cat((coords[:, 0] / 1024.0, labels[:, 0:1].to(dtype=dtype)), dim=-1)
        click_events.append(event)
        viewports.append(viewport)
        output = model(
            image_embeddings=predictor.features,
            point_coords=coords,
            point_labels=labels,
            previous_mask=previous,
            viewport_state=viewport,
            click_history=torch.stack(click_events, dim=1),
            viewport_history=torch.stack(viewports, dim=1).squeeze(2),
            mask_history=torch.stack(mask_history, dim=1) if mask_history else None,
            multimask_output=True,
        )
        candidate = int(output["iou_predictions"][0].argmax().item())
        probability = output["final_mask"][:, candidate : candidate + 1]
        full_resolution = model.postprocess_masks(
            probability,
            predictor.input_size,
            predictor.original_size,
        )
        prediction = full_resolution[0, 0].detach().cpu().numpy() >= 0.5
        click_number = round_index + 1
        if click_number in clicks:
            masks_at_click[click_number] = prediction.copy()
            scores_at_click[click_number] = float(
                output["iou_predictions"][0, candidate].detach().cpu()
            )
        mask_history.append(probability.detach())
        previous = probability.detach()

    return masks_at_click, scores_at_click


original_save_outputs = single_eval.save_outputs


def save_multimask_outputs(output_dir, payload) -> None:
    payload["protocol"].update(
        {
            "multimask_output": True,
            "candidate_count": 3,
            "mask_selection": (
                "highest model-predicted IoU among three VIMR candidates; "
                "ground-truth IoU is never used for selection"
            ),
            "checkpoint_training_mask_mode": "single-mask",
        }
    )
    payload["vimr_complexity"].update(
        {
            "MACs(G)": NORMALIZED_VIMR_MULTIMASK_MACS_G,
            "ProfiledBaselineMultimaskMACs(G)": PROFILED_BASELINE_MULTIMASK_MACS_G,
            "ProfiledVIMRMultimaskMACs(G)": PROFILED_VIMR_MULTIMASK_MACS_G,
            "macs_protocol": (
                "PyTorch 2.6 FlopCounterMode, MAC=FLOP/2; 512x512 encoder plus the "
                "maximum-history click-5 three-candidate path. Table MACs add the measured "
                "VIMR overhead to the published 35G baseline."
            ),
        }
    )
    for row in payload["table"]:
        if row["method"] == "VIMR seed 42":
            row["method"] = "VIMR seed 42 multimask"
            row["MACs(G)"] = NORMALIZED_VIMR_MULTIMASK_MACS_G
    original_save_outputs(output_dir, payload)
    print(
        json.dumps(
            {
                "multimask_selection": "predicted IoU argmax",
                "normalized_MACs_G": NORMALIZED_VIMR_MULTIMASK_MACS_G,
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    single_eval.run_vimr_interaction = run_vimr_interaction_multimask
    single_eval.save_outputs = save_multimask_outputs
    single_eval.main()


if __name__ == "__main__":
    main()
