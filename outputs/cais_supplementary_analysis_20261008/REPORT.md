# CAIS supplementary candidate-space analysis

## Candidate-budget / radius ablation (GT upper bound)

| Candidates | Maximum radius | All mean gain | Gated mean gain | Gated gain > 0.01 |
|---:|---:|---:|---:|---:|
| 1 | Original | +0.000000 | +0.000000 | 0.00% |
| 9 | 0.625% | +0.016333 | +0.052936 | 58.06% |
| 17 | 1.25% | +0.027071 | +0.071704 | 64.52% |
| 25 | 2.5% | +0.030341 | +0.079114 | 74.19% |
| 33 | 5.0% | +0.041297 | +0.097817 | 77.42% |

## Trajectory-direction mechanism check

- Valid corrections: 138
- Real mean |cosine|: 0.5976
- Shuffled mean |cosine|: 0.6415
- Real - Shuffle: -0.0439, 95% CI [-0.0979, +0.0107]
- Best correction opposite to approach direction: 47.83%

## SAM quality-head ranking diagnostic

- Mean within-click quality/actual-IoU correlation: 0.4238
- Quality Top-1 equals oracle point: 4.23%
- Quality Top-3 contains oracle point: 16.20%
- Mean oracle candidate rank by SAM quality: 18.18

## Subgroups by original IoU

| Subgroup | N | Mean grid gain | Gain > 0.01 |
|---|---:|---:|---:|
| iou_lt_0.6 | 7 | +0.372140 | 85.71% |
| iou_0.6_to_0.8 | 34 | +0.069249 | 79.41% |
| iou_ge_0.8 | 101 | +0.008957 | 21.78% |
