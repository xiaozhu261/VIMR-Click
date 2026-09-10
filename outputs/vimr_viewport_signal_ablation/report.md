# Viewport signal controlled ablation

All four trainable rows are Viewport-only VIMR models with identical initialization per seed, split, schedule, and frozen baseline-derived click coordinates. Only the viewport trajectory changes.

`correct` is a GT-instance-bbox oracle camera trajectory, not recorded user zoom/pan data; it is an upper bound.
`shuffled` is a no-self instance permutation of complete synthetic trajectories.

| Mode | mIoU@1 | mIoU@2 | mIoU@3 | mIoU@5 | NoC@85 | NoC@90 | Success@85/5 | Success@90/5 | RegressionRate | LateRegressionRate@4-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.425946 | 0.626218 | 0.630606 | 0.677160 | 3.585000 | 4.435000 | 0.645000 | 0.430000 | 0.430000 | 0.450000 |
| default | 0.727199 ± 0.028064 | 0.757164 ± 0.019596 | 0.755576 ± 0.019359 | 0.752882 ± 0.023743 | 3.315000 ± 0.230597 | 4.301667 ± 0.161735 | 0.586667 ± 0.042525 | 0.376667 ± 0.024664 | 0.467083 ± 0.018300 | 0.522500 ± 0.006614 |
| synthetic | 0.775872 ± 0.011200 | 0.787822 ± 0.005320 | 0.786075 ± 0.003122 | 0.785753 ± 0.004732 | 3.041667 ± 0.016073 | 4.125000 ± 0.010000 | 0.613333 ± 0.010408 | 0.390000 ± 0.010000 | 0.522917 ± 0.040780 | 0.586667 ± 0.044884 |
| shuffled | 0.766788 ± 0.022429 | 0.789274 ± 0.006824 | 0.788225 ± 0.007913 | 0.784567 ± 0.004971 | 3.090000 ± 0.123794 | 4.176667 ± 0.083116 | 0.611667 ± 0.027538 | 0.383333 ± 0.024664 | 0.506250 ± 0.031549 | 0.561667 ± 0.012829 |
| correct | 0.764287 ± 0.018281 | 0.785395 ± 0.014790 | 0.790081 ± 0.011362 | 0.796875 ± 0.015968 | 3.011667 ± 0.071122 | 4.076667 ± 0.085049 | 0.646667 ± 0.015275 | 0.410000 ± 0.013229 | 0.460833 ± 0.015629 | 0.515000 ± 0.015000 |

## Predeclared viewport-signal check

A positive result requires both paired mIoU@5 effects to be at least 0.01 and positive for every seed: synthetic − default and synthetic − shuffled.

- synthetic − default: +0.032871
- synthetic − shuffled: +0.001186
- Result: **inconclusive: independent viewport information not established**
