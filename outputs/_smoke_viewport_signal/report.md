# Viewport signal controlled ablation

All four trainable rows are Viewport-only VIMR models with identical initialization per seed, split, schedule, and frozen baseline-derived click coordinates. Only the viewport trajectory changes.

`correct` is a GT-instance-bbox oracle camera trajectory, not recorded user zoom/pan data; it is an upper bound.
`shuffled` is a no-self instance permutation of complete synthetic trajectories.

| Mode | mIoU@1 | mIoU@2 | mIoU@3 | mIoU@5 | NoC@85 | NoC@90 | Success@85/5 | Success@90/5 | RegressionRate | LateRegressionRate@4-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.413379 | 0.625117 | 0.530454 | 0.780131 | 5.000000 | 5.000000 | 0.500000 | 0.500000 | 0.375000 | 0.250000 |
| default | 0.387015 ± 0.000000 | 0.007155 ± 0.000000 | 0.000088 ± 0.000000 | 0.057225 ± 0.000000 | 6.000000 ± 0.000000 | 6.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.750000 ± 0.000000 | 0.500000 ± 0.000000 |
| synthetic | 0.311498 ± 0.000000 | 0.005464 ± 0.000000 | 0.000437 ± 0.000000 | 0.236495 ± 0.000000 | 6.000000 ± 0.000000 | 6.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.750000 ± 0.000000 | 0.500000 ± 0.000000 |
| shuffled | 0.285579 ± 0.000000 | 0.004425 ± 0.000000 | 0.000000 ± 0.000000 | 0.217778 ± 0.000000 | 6.000000 ± 0.000000 | 6.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.625000 ± 0.000000 | 0.250000 ± 0.000000 |
| correct | 0.313667 ± 0.000000 | 0.004081 ± 0.000000 | 0.000132 ± 0.000000 | 0.058347 ± 0.000000 | 6.000000 ± 0.000000 | 6.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.750000 ± 0.000000 | 0.500000 ± 0.000000 |

## Predeclared viewport-signal check

A positive result requires both paired mIoU@5 effects to be at least 0.01 and positive for every seed: synthetic − default and synthetic − shuffled.

- synthetic − default: +0.179270
- synthetic − shuffled: +0.018717
- Result: **supports aligned viewport information under this synthetic protocol**
