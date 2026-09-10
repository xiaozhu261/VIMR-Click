# VIMR-Click multi-seed COCO protocol results

- Split: 500 train / 200 validation (split seed 2026)
- Schedule: 3 epochs, 5 clicks, training seeds [42, 43, 44]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.425946 | 0.784169 | 0.007764 | +0.358223 |
| mIoU@2 | 0.626218 | 0.802857 | 0.006129 | +0.176639 |
| mIoU@3 | 0.630606 | 0.813381 | 0.005339 | +0.182775 |
| mIoU@5 | 0.677160 | 0.821115 | 0.005354 | +0.143955 |
| NoC@85 | 3.585000 | 3.020000 | 0.091652 | -0.565000 |
| Success@85/5 | 0.645000 | 0.626667 | 0.020817 | -0.018333 |
| NoC@90 | 4.435000 | 4.081667 | 0.040723 | -0.353333 |
| Success@90/5 | 0.430000 | 0.418333 | 0.010408 | -0.011667 |
| RegressionRate | 0.430000 | 0.366250 | 0.015207 | -0.063750 |
| MeanRegressionMagnitude | 0.112610 | 0.005023 | 0.001307 | -0.107587 |
| FinalSuccess@85/5 | 0.385000 | 0.588333 | 0.017559 | +0.203333 |
| LostAfterSuccess@85/5 | 0.260000 | 0.038333 | 0.007638 | -0.221667 |
| FinalSuccess@90/5 | 0.225000 | 0.398333 | 0.015275 | +0.173333 |
| LostAfterSuccess@90/5 | 0.205000 | 0.020000 | 0.005000 | -0.185000 |
| RegressionRate@4 | 0.450000 | 0.420000 | 0.015000 | -0.030000 |
| RegressionRate@5 | 0.450000 | 0.455000 | 0.052678 | +0.005000 |
| LateRegressionRate@4-5 | 0.450000 | 0.437500 | 0.031225 | -0.012500 |
| LateMeanRegressionMagnitude@4-5 | 0.101800 | 0.003860 | 0.001142 | -0.097940 |
| LateSevereRegressionRate@4-5>0.05 | 0.265000 | 0.019167 | 0.008780 | -0.245833 |
| LateSevereRegressionRate@4-5>0.10 | 0.235000 | 0.005833 | 0.002887 | -0.229167 |
