# VIMR-Click multi-seed COCO protocol results

- Split: 500 train / 200 validation (split seed 2026)
- Schedule: 3 epochs, 5 clicks, training seeds [42, 43, 44]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.425946 | 0.785054 | 0.006777 | +0.359109 |
| mIoU@2 | 0.626218 | 0.804646 | 0.003451 | +0.178428 |
| mIoU@3 | 0.630606 | 0.811274 | 0.005219 | +0.180668 |
| mIoU@5 | 0.677160 | 0.817281 | 0.003236 | +0.140121 |
| NoC@85 | 3.585000 | 2.981667 | 0.045369 | -0.603333 |
| Success@85/5 | 0.645000 | 0.638333 | 0.015275 | -0.006667 |
| NoC@90 | 4.435000 | 4.045000 | 0.030414 | -0.390000 |
| Success@90/5 | 0.430000 | 0.418333 | 0.005774 | -0.011667 |
| RegressionRate | 0.430000 | 0.365833 | 0.014488 | -0.064167 |
| MeanRegressionMagnitude | 0.112610 | 0.003554 | 0.000647 | -0.109056 |
| FinalSuccess@85/5 | 0.385000 | 0.615000 | 0.020000 | +0.230000 |
| LostAfterSuccess@85/5 | 0.260000 | 0.023333 | 0.005774 | -0.236667 |
| FinalSuccess@90/5 | 0.225000 | 0.405000 | 0.008660 | +0.180000 |
| LostAfterSuccess@90/5 | 0.205000 | 0.013333 | 0.012583 | -0.191667 |
| RegressionRate@4 | 0.450000 | 0.401667 | 0.014434 | -0.048333 |
| RegressionRate@5 | 0.450000 | 0.446667 | 0.055076 | -0.003333 |
| LateRegressionRate@4-5 | 0.450000 | 0.424167 | 0.031258 | -0.025833 |
| LateMeanRegressionMagnitude@4-5 | 0.101800 | 0.002932 | 0.000538 | -0.098869 |
| LateSevereRegressionRate@4-5>0.05 | 0.265000 | 0.010833 | 0.005204 | -0.254167 |
| LateSevereRegressionRate@4-5>0.10 | 0.235000 | 0.003333 | 0.001443 | -0.231667 |
