# CAIS oracle local click-refinement experiment

## Protocol

Frozen EfficientViT-SAM-L0; all prompt history, click type, previous mask, image features, and decoder settings are held fixed. Only the current click coordinate is replaced by the nearest point with a safe interior margin in the corresponding GT error component.

- Boundary threshold: 3.0 px
- Safe interior margin: 5.0 px
- Maximum correction: 5.0% of image long side
- Human clicks audited: 1590
- Hard clicks: 188
- Eligible local corrections: 173

## Frozen-SAM results

| Metric | Value |
|---|---:|
| Eligible clicks | 173 |
| Original mean IoU | 0.837568 |
| Oracle mean IoU | 0.856459 |
| Mean IoU delta | +0.018891 |
| Median IoU delta | +0.000368 |
| 5% trimmed mean delta | +0.006739 |
| Win rate | 61.27% |
| Harm rate | 32.95% |
| Delta > 0.01 | 19.08% |
| Delta < -0.01 | 5.20% |
| Mean correction distance | 7.16 px |
| Task-bootstrap 95% CI | [+0.009215, +0.030690] |

## Predeclared decision

- Mean IoU delta >= 0.02: `False`
- Win rate >= 60%: `True`
- Bootstrap CI lower bound > 0: `True`
- Proceed to a learned trajectory click refiner: `False`

## Subgroups

| Group | N | Mean delta | 95% task-bootstrap CI | Win rate |
|---|---:|---:|---:|---:|
| positive | 142 | +0.023330 | [+0.011982, +0.037612] | 64.08% |
| negative | 31 | -0.001445 | [-0.008501, +0.004250] | 48.39% |
| inside_boundary | 99 | +0.016853 | [+0.006803, +0.028921] | 57.58% |
| outside_error | 74 | +0.021617 | [+0.005017, +0.044743] | 66.22% |

The overall mean should be interpreted with the median and trimmed mean: a small number of catastrophic positive-click recoveries can contribute a large fraction of the average gain.
