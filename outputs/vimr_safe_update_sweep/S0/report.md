# VIMR-Click multi-seed COCO protocol results

- Split: 500 train / 200 validation (split seed 2026)
- Schedule: 3 epochs, 5 clicks, training seeds [42, 43, 44]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.425946 | 0.762481 | 0.028430 | +0.336536 |
| mIoU@2 | 0.626218 | 0.790421 | 0.021562 | +0.164202 |
| mIoU@3 | 0.630606 | 0.806095 | 0.017406 | +0.175489 |
| mIoU@5 | 0.677160 | 0.820110 | 0.013389 | +0.142949 |
| NoC@85 | 3.585000 | 3.013333 | 0.164418 | -0.571667 |
| Success@85/5 | 0.645000 | 0.651667 | 0.028431 | +0.006667 |
| NoC@90 | 4.435000 | 4.095000 | 0.152233 | -0.340000 |
| Success@90/5 | 0.430000 | 0.416667 | 0.017559 | -0.013333 |
| RegressionRate | 0.430000 | 0.356250 | 0.007603 | -0.073750 |
| MeanRegressionMagnitude | 0.112610 | 0.004680 | 0.001092 | -0.107930 |
| FinalSuccess@85/5 | 0.385000 | 0.618333 | 0.041932 | +0.233333 |
| LostAfterSuccess@85/5 | 0.260000 | 0.033333 | 0.017559 | -0.226667 |
| FinalSuccess@90/5 | 0.225000 | 0.386667 | 0.027538 | +0.161667 |
| LostAfterSuccess@90/5 | 0.205000 | 0.030000 | 0.015000 | -0.175000 |
| RegressionRate@4 | 0.450000 | 0.386667 | 0.025166 | -0.063333 |
| RegressionRate@5 | 0.450000 | 0.455000 | 0.010000 | +0.005000 |
| LateRegressionRate@4-5 | 0.450000 | 0.420833 | 0.010408 | -0.029167 |
| LateMeanRegressionMagnitude@4-5 | 0.101800 | 0.004024 | 0.001194 | -0.097776 |
| LateSevereRegressionRate@4-5>0.05 | 0.265000 | 0.013333 | 0.009465 | -0.251667 |
| LateSevereRegressionRate@4-5>0.10 | 0.235000 | 0.005833 | 0.002887 | -0.229167 |
