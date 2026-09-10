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

from efficientvit.models.vimr import VIMREfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def save_map(path: Path, value: np.ndarray) -> None:
    array = np.squeeze(value)
    array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-round VIMR-Click MVP demo")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="efficientvit-sam-vimr-l0")
    parser.add_argument("--output", type=Path, default=Path("outputs/vimr_demo"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    model = create_efficientvit_sam_model(args.model, pretrained=True, weight_url=str(args.checkpoint))
    model = model.to(args.device).eval()
    predictor = VIMREfficientViTSamPredictor(model)
    predictor.set_image(image)
    height, width = image.shape[:2]
    interactions = [
        (np.array([[0.50 * width, 0.50 * height, 1]], dtype=np.float32), [0.5, 0.5, 0, 0, 0, 0, 0]),
        (np.array([[0.65 * width, 0.50 * height, 0]], dtype=np.float32), [0.5, 0.5, np.log(2), 0, 0, np.log(2), 1]),
        (np.array([[0.60 * width, 0.60 * height, 1]], dtype=np.float32), [0.6, 0.6, np.log(2), 0.1, 0.1, 0, 3]),
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    previous = None
    for index, (click, viewport) in enumerate(interactions, start=1):
        result = predictor.predict(click, np.asarray(viewport, dtype=np.float32))
        mask = result["masks"].astype(np.uint8)
        save_map(args.output / f"mask_{index}.png", mask)
        if index > 1:
            save_map(args.output / f"gate_{index}.png", result["gate_map"])
        change = 0.0 if previous is None else float(np.not_equal(mask, previous).mean())
        print(
            f"round={index} mask_change_ratio={change:.6f} "
            f"gate_mean={float(result['gate_map'].mean()):.6f} prediction_area={float(mask.mean()):.6f}"
        )
        previous = mask


if __name__ == "__main__":
    main()
