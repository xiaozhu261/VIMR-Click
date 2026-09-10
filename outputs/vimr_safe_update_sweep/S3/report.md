# VIMR-Click multi-seed COCO protocol results

- Split: 500 train / 200 validation (split seed 2026)
- Schedule: 3 epochs, 5 clicks, training seeds [42, 43, 44]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.425946 | 0.783666 | 0.003513 | +0.357721 |
| mIoU@2 | 0.626218 | 0.803115 | 0.010828 | +0.176896 |
| mIoU@3 | 0.630606 | 0.809478 | 0.014061 | +0.178872 |
| mIoU@5 | 0.677160 | 0.817321 | 0.015756 | +0.140161 |
| NoC@85 | 3.585000 | 2.918333 | 0.089768 | -0.666667 |
| Success@85/5 | 0.645000 | 0.656667 | 0.027538 | +0.011667 |
| NoC@90 | 4.435000 | 4.021667 | 0.090738 | -0.413333 |
| Success@90/5 | 0.430000 | 0.418333 | 0.025658 | -0.011667 |
| RegressionRate | 0.430000 | 0.360417 | 0.060484 | -0.069583 |
| MeanRegressionMagnitude | 0.112610 | 0.004898 | 0.001544 | -0.107712 |
| FinalSuccess@85/5 | 0.385000 | 0.615000 | 0.043589 | +0.230000 |
| LostAfterSuccess@85/5 | 0.260000 | 0.041667 | 0.017559 | -0.218333 |
| FinalSuccess@90/5 | 0.225000 | 0.390000 | 0.032787 | +0.165000 |
| LostAfterSuccess@90/5 | 0.205000 | 0.028333 | 0.007638 | -0.176667 |
| RegressionRate@4 | 0.450000 | 0.411667 | 0.080208 | -0.038333 |
| RegressionRate@5 | 0.450000 | 0.460000 | 0.072629 | +0.010000 |
| LateRegressionRate@4-5 | 0.450000 | 0.435833 | 0.066442 | -0.014167 |
| LateMeanRegressionMagnitude@4-5 | 0.101800 | 0.003761 | 0.001635 | -0.098039 |
| LateSevereRegressionRate@4-5>0.05 | 0.265000 | 0.011667 | 0.008036 | -0.253333 |
| LateSevereRegressionRate@4-5>0.10 | 0.235000 | 0.003333 | 0.003819 | -0.231667 |
