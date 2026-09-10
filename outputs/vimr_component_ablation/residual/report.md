# VIMR-Click multi-seed COCO protocol results

- Split: 500 train / 200 validation (split seed 2026)
- Schedule: 3 epochs, 5 clicks, training seeds [42, 43, 44]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.425946 | 0.784770 | 0.005433 | +0.358824 |
| mIoU@2 | 0.626218 | 0.804215 | 0.003795 | +0.177997 |
| mIoU@3 | 0.630606 | 0.813367 | 0.001418 | +0.182761 |
| mIoU@5 | 0.677160 | 0.817977 | 0.004954 | +0.140817 |
| NoC@85 | 3.585000 | 2.936667 | 0.007638 | -0.648333 |
| Success@85/5 | 0.645000 | 0.650000 | 0.013229 | +0.005000 |
| NoC@90 | 4.435000 | 4.031667 | 0.022546 | -0.403333 |
| Success@90/5 | 0.430000 | 0.421667 | 0.011547 | -0.008333 |
| RegressionRate | 0.430000 | 0.447083 | 0.017063 | +0.017083 |
| MeanRegressionMagnitude | 0.112610 | 0.007728 | 0.001939 | -0.104882 |
| FinalSuccess@85/5 | 0.385000 | 0.608333 | 0.012583 | +0.223333 |
| LostAfterSuccess@85/5 | 0.260000 | 0.041667 | 0.025658 | -0.218333 |
| FinalSuccess@90/5 | 0.225000 | 0.378333 | 0.018930 | +0.153333 |
| LostAfterSuccess@90/5 | 0.205000 | 0.043333 | 0.007638 | -0.161667 |
| RegressionRate@4 | 0.450000 | 0.508333 | 0.037528 | +0.058333 |
| RegressionRate@5 | 0.450000 | 0.526667 | 0.023094 | +0.076667 |
| LateRegressionRate@4-5 | 0.450000 | 0.517500 | 0.029475 | +0.067500 |
| LateMeanRegressionMagnitude@4-5 | 0.101800 | 0.007334 | 0.001899 | -0.094466 |
| LateSevereRegressionRate@4-5>0.05 | 0.265000 | 0.027500 | 0.006614 | -0.237500 |
| LateSevereRegressionRate@4-5>0.10 | 0.235000 | 0.013333 | 0.010104 | -0.221667 |
