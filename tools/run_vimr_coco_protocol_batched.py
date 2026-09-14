"""Run the VIMR COCO protocol with batched frozen image encoding.

The research protocol, optimizer steps, interactive click sequence, losses,
validation, and artifact format are inherited from run_vimr_coco_protocol.
Only the frozen image encoder is evaluated in batches to reduce GPU idle gaps.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import tools.run_vimr_coco_protocol as protocol
from tools.eval_vimr_coco_quick import (
    decode_mask,
    run_baseline_sequence,
    run_vimr_sequence,
    target_to_sam,
)


def extract_encoder_batch_size(argv: list[str]) -> int:
    batch_size = 16
    for index, argument in enumerate(list(argv)):
        if argument == "--encoder-batch-size":
            if index + 1 >= len(argv):
                raise ValueError("--encoder-batch-size requires an integer")
            batch_size = int(argv[index + 1])
            del argv[index : index + 2]
            break
        if argument.startswith("--encoder-batch-size="):
            batch_size = int(argument.split("=", 1)[1])
            argv.remove(argument)
            break
    if batch_size < 1:
        raise ValueError("--encoder-batch-size must be positive")
    return batch_size


ENCODER_BATCH_SIZE = extract_encoder_batch_size(sys.argv)


def encode_image_batch(model, samples: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    tensors = []
    for sample in samples:
        image = np.asarray(Image.open(sample["path"]).convert("RGB"))
        tensors.append(model.transform(image))
    images = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        return model.image_encoder(images)


def evaluate_method(
    model,
    samples: list[dict[str, Any]],
    device: torch.device,
    max_clicks: int,
    method: str,
    progress_every: int,
    prefix: str,
    viewport_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if method not in {"baseline", "vimr"}:
        raise ValueError(f"Unsupported method: {method}")
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch_start in range(0, len(samples), ENCODER_BATCH_SIZE):
            batch_samples = samples[batch_start : batch_start + ENCODER_BATCH_SIZE]
            embeddings = encode_image_batch(model, batch_samples, device)
            for batch_offset, (sample, embedding) in enumerate(zip(batch_samples, embeddings)):
                index = batch_start + batch_offset + 1
                target = target_to_sam(decode_mask(sample), device)
                embedding = embedding.unsqueeze(0)
                if method == "baseline":
                    ious = run_baseline_sequence(model, embedding, target, max_clicks)
                else:
                    _, ious = run_vimr_sequence(
                        model,
                        embedding,
                        target,
                        max_clicks,
                        train=False,
                        viewport_mode=viewport_mode,
                    )
                rows.append(
                    {
                        "image_id": sample["image"]["id"],
                        "annotation_id": sample["annotation"]["id"],
                        "category_id": sample["annotation"]["category_id"],
                        "ious": ious,
                    }
                )
                if index == 1 or index % progress_every == 0 or index == len(samples):
                    print(f"{prefix} {index}/{len(samples)} IoU@{max_clicks}={ious[-1]:.4f}", flush=True)
    return rows, protocol.summarize_protocol(rows, max_clicks)


def train_one_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    samples: list[dict[str, Any]],
    device: torch.device,
    epoch: int,
    seed: int,
    max_clicks: int,
    progress_every: int,
    viewport_mode: str,
) -> dict[str, float]:
    protocol.set_vimr_train_mode(model)
    order = list(range(len(samples)))
    random.Random(seed * 10_000 + epoch).shuffle(order)
    grad_norms = []
    final_ious = []
    for batch_start in range(0, len(order), ENCODER_BATCH_SIZE):
        batch_indices = order[batch_start : batch_start + ENCODER_BATCH_SIZE]
        batch_samples = [samples[index] for index in batch_indices]
        embeddings = encode_image_batch(model, batch_samples, device)
        for batch_offset, (sample, embedding) in enumerate(zip(batch_samples, embeddings)):
            step = batch_start + batch_offset + 1
            target = target_to_sam(decode_mask(sample), device)
            optimizer.zero_grad(set_to_none=True)
            _, ious = run_vimr_sequence(
                model,
                embedding.unsqueeze(0),
                target,
                max_clicks,
                train=True,
                viewport_mode=viewport_mode,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
            optimizer.step()
            grad_norms.append(float(grad_norm.detach().cpu()))
            final_ious.append(ious[-1])
            if step == 1 or step % progress_every == 0 or step == len(order):
                print(
                    f"seed={seed} epoch={epoch} train={step}/{len(order)} "
                    f"IoU@{max_clicks}={ious[-1]:.4f} grad_norm={grad_norms[-1]:.4f}",
                    flush=True,
                )
    return {
        "mean_train_iou_at_max_clicks": float(np.mean(final_ious)),
        "mean_gradient_norm": float(np.mean(grad_norms)),
    }


def main() -> None:
    print(
        f"Using frozen image-encoder batch size {ENCODER_BATCH_SIZE}; "
        "interactive optimizer steps remain per instance",
        flush=True,
    )
    protocol.evaluate_method = evaluate_method
    protocol.train_one_epoch = train_one_epoch
    protocol.main()


if __name__ == "__main__":
    main()
