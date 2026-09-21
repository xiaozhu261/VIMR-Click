import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.models.cais import CAISEfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def save_map(path: Path, value: np.ndarray) -> None:
    array = (np.squeeze(value).clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def demo_trajectory(samples: int = 16) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    return np.stack(
        (-2.0 * (1.0 - progress), -0.05 * (1.0 - progress), 0.03 * (1.0 - progress)),
        axis=-1,
    )[None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone CAIS interactive demo")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="efficientvit-sam-cais-l0")
    parser.add_argument("--output", type=Path, default=Path("outputs/cais_demo"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model = create_efficientvit_sam_model(
            args.model,
            pretrained=False,
            cais_config=checkpoint.get("cais_config"),
        )
        model.load_state_dict(checkpoint["model"])
        print(
            f"Loaded trained CAIS checkpoint from epoch {checkpoint.get('epoch', 'unknown')} "
            f"(best validation IoU={checkpoint.get('best_val_iou', 'unknown')})"
        )
    else:
        model = create_efficientvit_sam_model(args.model, pretrained=True, weight_url=str(args.checkpoint))
    predictor = CAISEfficientViTSamPredictor(model.to(args.device).eval())
    predictor.set_image(image)
    height, width = image.shape[:2]
    clicks = (
        np.array([[0.50 * width, 0.50 * height, 1]], dtype=np.float32),
        np.array([[0.65 * width, 0.50 * height, 0]], dtype=np.float32),
        np.array([[0.60 * width, 0.60 * height, 1]], dtype=np.float32),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    for index, click in enumerate(clicks, start=1):
        result = predictor.predict(click, click_trajectories=demo_trajectory())
        save_map(args.output / f"mask_{index}.png", result["masks"])
        print(
            f"round={index} iou_score={float(result['iou_predictions'][0]):.6f} "
            f"cognitive_friction={result['cognitive_friction']:.6f}"
        )


if __name__ == "__main__":
    main()
