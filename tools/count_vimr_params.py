import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from efficientvit.models.vimr import VIMREfficientViTSam
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Count EfficientViT-SAM and VIMR parameters")
    parser.add_argument("--model", default="efficientvit-sam-vimr-l0")
    args = parser.parse_args()
    model = create_efficientvit_sam_model(args.model, pretrained=False)
    if not isinstance(model, VIMREfficientViTSam):
        raise ValueError(f"{args.model} is not a VIMR model")
    summary = model.parameter_summary()
    labels = {
        "image_encoder": "EfficientViT encoder",
        "prompt_encoder": "Prompt encoder",
        "mask_decoder": "Mask decoder",
        "viewport_encoder": "Viewport encoder",
        "history_memory": "History memory",
        "residual_analyzer": "Residual analyzer",
        "interaction_fusion": "Interaction fusion",
        "expert_router": "G/O/D expert router",
        "residual_gate": "Residual gate",
        "detail_refinement": "Detail expert",
        "total": "Total params",
        "trainable": "Trainable params",
    }
    for key, label in labels.items():
        print(f"{label:24s} {summary[key]:12,d}")
    added = sum(
        summary[key]
        for key in (
            "viewport_encoder",
            "history_memory",
            "residual_analyzer",
            "interaction_fusion",
            "expert_router",
            "residual_gate",
            "detail_refinement",
        )
    )
    print(f"{'VIMR added params':24s} {added:12,d}")


if __name__ == "__main__":
    main()
