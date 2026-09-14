"""Full COCO evaluation for a phase-2 VIMR checkpoint with a calibrated quality head."""

from __future__ import annotations

import json

import numpy as np
import torch

import tools.eval_coco_click_vimr_full as base_eval
from applications.efficientvit_sam.sam_eval_utils import Clicker
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.models.vimr.quality import VIMRFinalMaskQualityHead
from tools.eval_vimr_coco_quick import viewport_for_mode


PROFILED_BASELINE_MULTIMASK_MACS_G = 36.63825664
PROFILED_VIMR_MULTIMASK_MACS_G = 50.19106048
PUBLISHED_BASELINE_MACS_G = 35.0
QUALITY_HEAD_PARAMETERS = 148417
NORMALIZED_MACS_G = (
    PUBLISHED_BASELINE_MACS_G
    + PROFILED_VIMR_MULTIMASK_MACS_G
    - PROFILED_BASELINE_MULTIMASK_MACS_G
)

quality_head: VIMRFinalMaskQualityHead | None = None
original_load_checkpoint = base_eval.load_vimr_checkpoint
original_save_outputs = base_eval.save_outputs


def load_phase2_checkpoint(model, path):
    global quality_head
    checkpoint = original_load_checkpoint(model, path)
    if "quality_head_state_dict" not in checkpoint:
        raise KeyError(f"{path} does not contain quality_head_state_dict")
    quality_head = VIMRFinalMaskQualityHead().to(next(model.parameters()).device)
    quality_head.load_state_dict(checkpoint["quality_head_state_dict"])
    quality_head.eval()
    return checkpoint


def run_quality_selected_interaction(
    model: torch.nn.Module,
    predictor: EfficientViTSamPredictor,
    target: np.ndarray,
    clicks: list[int],
    viewport_mode: str,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    if quality_head is None:
        raise RuntimeError("The phase-2 quality head was not loaded")
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
        calibrated_quality = quality_head(output, round_index, clicks[-1])
        candidate = int(calibrated_quality[0].argmax().item())
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
            scores_at_click[click_number] = float(calibrated_quality[0, candidate].detach().cpu())
        mask_history.append(probability.detach())
        previous = probability.detach()
    return masks_at_click, scores_at_click


def save_quality_outputs(output_dir, payload) -> None:
    checkpoint = torch.load(payload["protocol"]["checkpoint"], map_location="cpu", weights_only=True)
    payload["protocol"].update(
        {
            "multimask_output": True,
            "candidate_count": 3,
            "mask_selection": "highest calibrated final-mask IoU prediction; no GT selection",
            "checkpoint_training_protocol": checkpoint.get("training_protocol", {}),
        }
    )
    payload["vimr_complexity"].update(
        {
            "Params(M)": payload["vimr_complexity"]["Params(M)"] + QUALITY_HEAD_PARAMETERS / 1e6,
            "QualityHeadParams(M)": QUALITY_HEAD_PARAMETERS / 1e6,
            "MACs(G)": NORMALIZED_MACS_G,
            "ProfiledBaselineMultimaskMACs(G)": PROFILED_BASELINE_MULTIMASK_MACS_G,
            "ProfiledVIMRMultimaskMACs(G)": PROFILED_VIMR_MULTIMASK_MACS_G,
            "macs_protocol": (
                "PyTorch 2.6 FlopCounterMode, MAC=FLOP/2; quality-head MACs are negligible "
                "at table precision; VIMR overhead is normalized to the published 35G baseline."
            ),
        }
    )
    for row in payload["table"]:
        if row["method"] == "VIMR seed 42":
            row["method"] = "VIMR phase2 calibrated multimask"
            row["Params(M)"] += QUALITY_HEAD_PARAMETERS / 1e6
            row["MACs(G)"] = NORMALIZED_MACS_G
    original_save_outputs(output_dir, payload)
    print(
        json.dumps(
            {
                "selection": "calibrated final-mask IoU argmax",
                "quality_head_parameters": QUALITY_HEAD_PARAMETERS,
                "normalized_MACs_G": NORMALIZED_MACS_G,
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    base_eval.load_vimr_checkpoint = load_phase2_checkpoint
    base_eval.run_vimr_interaction = run_quality_selected_interaction
    base_eval.save_outputs = save_quality_outputs
    base_eval.main()


if __name__ == "__main__":
    main()
