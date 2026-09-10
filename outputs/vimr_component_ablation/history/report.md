# VIMR-Click multi-seed COCO protocol results

- Split: 500 train / 200 validation (split seed 2026)
- Schedule: 3 epochs, 5 clicks, training seeds [42, 43, 44]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.425946 | 0.770980 | 0.017362 | +0.345034 |
| mIoU@2 | 0.626218 | 0.790829 | 0.016760 | +0.164611 |
| mIoU@3 | 0.630606 | 0.804311 | 0.015596 | +0.173705 |
| mIoU@5 | 0.677160 | 0.812113 | 0.008490 | +0.134953 |
| NoC@85 | 3.585000 | 3.040000 | 0.079373 | -0.545000 |
| Success@85/5 | 0.645000 | 0.640000 | 0.018028 | -0.005000 |
| NoC@90 | 4.435000 | 4.040000 | 0.056789 | -0.395000 |
| Success@90/5 | 0.430000 | 0.426667 | 0.010408 | -0.003333 |
| RegressionRate | 0.430000 | 0.412917 | 0.007324 | -0.017083 |
| MeanRegressionMagnitude | 0.112610 | 0.006846 | 0.002805 | -0.105764 |
| FinalSuccess@85/5 | 0.385000 | 0.615000 | 0.008660 | +0.230000 |
| LostAfterSuccess@85/5 | 0.260000 | 0.025000 | 0.018028 | -0.235000 |
| FinalSuccess@90/5 | 0.225000 | 0.400000 | 0.008660 | +0.175000 |
| LostAfterSuccess@90/5 | 0.205000 | 0.026667 | 0.011547 | -0.178333 |
| RegressionRate@4 | 0.450000 | 0.473333 | 0.033292 | +0.023333 |
| RegressionRate@5 | 0.450000 | 0.470000 | 0.042720 | +0.020000 |
| LateRegressionRate@4-5 | 0.450000 | 0.471667 | 0.024281 | +0.021667 |
| LateMeanRegressionMagnitude@4-5 | 0.101800 | 0.007234 | 0.003242 | -0.094566 |
| LateSevereRegressionRate@4-5>0.05 | 0.265000 | 0.032500 | 0.017500 | -0.232500 |
| LateSevereRegressionRate@4-5>0.10 | 0.235000 | 0.017500 | 0.011456 | -0.217500 |
