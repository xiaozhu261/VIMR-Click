# VIMR-Click multi-seed COCO protocol results

- Split: 500 train / 200 validation (split seed 2026)
- Schedule: 3 epochs, 5 clicks, training seeds [42, 43, 44]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.425855 | 0.763095 | 0.010994 | +0.337240 |
| mIoU@2 | 0.627116 | 0.784977 | 0.013008 | +0.157861 |
| mIoU@3 | 0.631522 | 0.786473 | 0.012914 | +0.154951 |
| mIoU@5 | 0.679223 | 0.785314 | 0.016404 | +0.106091 |
| NoC@85 | 3.570000 | 3.070000 | 0.063836 | -0.500000 |
| Success@85/5 | 0.645000 | 0.615000 | 0.017321 | -0.030000 |
| NoC@90 | 4.425000 | 4.131667 | 0.050083 | -0.293333 |
| Success@90/5 | 0.425000 | 0.393333 | 0.014434 | -0.031667 |
| RegressionRate | 0.430000 | 0.465000 | 0.023848 | +0.035000 |
| MeanRegressionMagnitude | 0.113270 | 0.011282 | 0.002093 | -0.101988 |
| FinalSuccess@85/5 | 0.365000 | 0.548333 | 0.033292 | +0.183333 |
| LostAfterSuccess@85/5 | 0.280000 | 0.066667 | 0.016073 | -0.213333 |
| FinalSuccess@90/5 | 0.200000 | 0.363333 | 0.012583 | +0.163333 |
| LostAfterSuccess@90/5 | 0.225000 | 0.030000 | 0.008660 | -0.195000 |
| RegressionRate@4 | 0.440000 | 0.530000 | 0.022913 | +0.090000 |
| RegressionRate@5 | 0.460000 | 0.511667 | 0.017559 | +0.051667 |
| LateRegressionRate@4-5 | 0.450000 | 0.520833 | 0.020207 | +0.070833 |
| LateMeanRegressionMagnitude@4-5 | 0.101843 | 0.010632 | 0.002881 | -0.091211 |
| LateSevereRegressionRate@4-5>0.05 | 0.262500 | 0.040833 | 0.005774 | -0.221667 |
| LateSevereRegressionRate@4-5>0.10 | 0.237500 | 0.020833 | 0.003819 | -0.216667 |
