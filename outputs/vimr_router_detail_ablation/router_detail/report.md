# VIMR-Click multi-seed COCO protocol results

- Split: 500 train / 200 validation (split seed 2026)
- Schedule: 3 epochs, 5 clicks, training seeds [42, 43, 44]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.425946 | 0.764537 | 0.027582 | +0.338591 |
| mIoU@2 | 0.626218 | 0.795748 | 0.003305 | +0.169530 |
| mIoU@3 | 0.630606 | 0.806875 | 0.004638 | +0.176269 |
| mIoU@5 | 0.677160 | 0.815394 | 0.005542 | +0.138234 |
| NoC@85 | 3.585000 | 3.025000 | 0.127574 | -0.560000 |
| Success@85/5 | 0.645000 | 0.645000 | 0.010000 | +0.000000 |
| NoC@90 | 4.435000 | 4.088333 | 0.076376 | -0.346667 |
| Success@90/5 | 0.430000 | 0.415000 | 0.000000 | -0.015000 |
| RegressionRate | 0.430000 | 0.375417 | 0.048611 | -0.054583 |
| MeanRegressionMagnitude | 0.112610 | 0.004342 | 0.001708 | -0.108268 |
| FinalSuccess@85/5 | 0.385000 | 0.610000 | 0.013229 | +0.225000 |
| LostAfterSuccess@85/5 | 0.260000 | 0.035000 | 0.008660 | -0.225000 |
| FinalSuccess@90/5 | 0.225000 | 0.390000 | 0.010000 | +0.165000 |
| LostAfterSuccess@90/5 | 0.205000 | 0.025000 | 0.010000 | -0.180000 |
| RegressionRate@4 | 0.450000 | 0.421667 | 0.082513 | -0.028333 |
| RegressionRate@5 | 0.450000 | 0.503333 | 0.109810 | +0.053333 |
| LateRegressionRate@4-5 | 0.450000 | 0.462500 | 0.092601 | +0.012500 |
| LateMeanRegressionMagnitude@4-5 | 0.101800 | 0.003883 | 0.002584 | -0.097917 |
| LateSevereRegressionRate@4-5>0.05 | 0.265000 | 0.011667 | 0.012583 | -0.253333 |
| LateSevereRegressionRate@4-5>0.10 | 0.235000 | 0.004167 | 0.005204 | -0.230833 |
