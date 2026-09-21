import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from efficientvit.models.cais import CAISEfficientViTSam
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Count EfficientViT-SAM and CAIS parameters")
    parser.add_argument("--model", default="efficientvit-sam-cais-l0")
    args = parser.parse_args()
    model = create_efficientvit_sam_model(args.model, pretrained=False)
    if not isinstance(model, CAISEfficientViTSam):
        raise ValueError(f"{args.model} is not a CAIS model")
    summary = model.parameter_summary()
    labels = {
        "image_encoder": "EfficientViT encoder",
        "prompt_encoder": "Prompt encoder",
        "mask_decoder": "Mask decoder",
        "cognitive_prompt_encoder": "Cognitive prompt encoder",
        "cognitive_two_way_attention": "Cognitive two-way decoder",
        "prediction_heads": "IoU/cognitive heads",
        "total": "Total params",
        "trainable": "Trainable params",
    }
    for key, label in labels.items():
        print(f"{label:28s} {summary[key]:12,d}")
    added = sum(summary[key] for key in ("cognitive_prompt_encoder", "cognitive_two_way_attention", "prediction_heads"))
    print(f"{'CAIS added params':28s} {added:12,d}")


if __name__ == "__main__":
    main()
