# VIMR-Click COCO 2017 quick comparison

This is a small-sample sanity experiment, not an official COCO val benchmark.
The local dataset contains train2017 images but no val2017 images, so seed 42
was used to create image-disjoint train and held-out subsets from train2017.

## Protocol

- Official EfficientViT-SAM-L0 checkpoint, SHA256 prefix `C4F994B01A16D48`.
- 64 training instances from distinct images; 32 held-out instances from other images.
- One eligible instance per image, area ratio in `[0.01, 0.60]`.
- VIMR-only training for one epoch at learning rate `1e-4`; all EfficientViT/SAM modules frozen.
- Three clicks: an initial positive distance-center click followed by each method's own largest-error clicks.
- Baseline accumulates SAM point prompts and previous-mask input; VIMR uses the current SAM point plus explicit history.
- `NoC` is capped at four when the target is not reached within three clicks. Lower is better.

## Results

| Metric | Baseline | VIMR | VIMR - Baseline |
| --- | ---: | ---: | ---: |
| mIoU@1 | 0.5028 | 0.7284 | +0.2256 |
| mIoU@2 | 0.6659 | 0.7373 | +0.0714 |
| mIoU@3 | 0.5948 | 0.7530 | +0.1582 |
| Median IoU@1 | 0.4525 | 0.8081 | +0.3556 |
| Median IoU@2 | 0.7155 | 0.8379 | +0.1224 |
| Median IoU@3 | 0.7178 | 0.8500 | +0.1321 |
| NoC@85 | 2.7188 | 2.5938 | -0.1250 |
| Success@85 within 3 | 59.38% | 50.00% | -9.38 pp |
| NoC@90 | 3.1250 | 3.0312 | -0.0938 |
| Success@90 within 3 | 40.62% | 37.50% | -3.12 pp |
| Regression rate | 40.62% | 45.31% | +4.69 pp |
| Per-instance win rate @3 | — | 53.12% | — |

Bootstrap 95% confidence intervals for paired mean IoU deltas were
`[+0.1247,+0.3305]`, `[-0.0281,+0.1748]`, and `[+0.0156,+0.3028]` at clicks
one through three. With only 32 held-out instances, these intervals should be
treated as diagnostic rather than publication evidence.

## Interpretation

VIMR improves mean and median IoU, especially at clicks one and three. It does
not yet improve threshold success or regression rate. The distribution is more
concentrated near IoU 0.8–0.85, while the baseline is more bimodal. This points
to useful learned conditioning but an insufficiently calibrated safe-update
gate after only 64 training examples.
