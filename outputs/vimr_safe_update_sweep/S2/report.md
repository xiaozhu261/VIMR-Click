# VIMR-Click multi-seed COCO protocol results

- Split: 500 train / 200 validation (split seed 2026)
- Schedule: 3 epochs, 5 clicks, training seeds [42, 43, 44]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.425946 | 0.776494 | 0.009776 | +0.350548 |
| mIoU@2 | 0.626218 | 0.794807 | 0.008276 | +0.168588 |
| mIoU@3 | 0.630606 | 0.808831 | 0.001044 | +0.178225 |
| mIoU@5 | 0.677160 | 0.817425 | 0.002258 | +0.140264 |
| NoC@85 | 3.585000 | 2.970000 | 0.036056 | -0.615000 |
| Success@85/5 | 0.645000 | 0.641667 | 0.002887 | -0.003333 |
| NoC@90 | 4.435000 | 4.073333 | 0.084311 | -0.361667 |
| Success@90/5 | 0.430000 | 0.411667 | 0.014434 | -0.018333 |
| RegressionRate | 0.430000 | 0.391667 | 0.004390 | -0.038333 |
| MeanRegressionMagnitude | 0.112610 | 0.004918 | 0.000391 | -0.107692 |
| FinalSuccess@85/5 | 0.385000 | 0.605000 | 0.010000 | +0.220000 |
| LostAfterSuccess@85/5 | 0.260000 | 0.036667 | 0.010408 | -0.223333 |
| FinalSuccess@90/5 | 0.225000 | 0.383333 | 0.012583 | +0.158333 |
| LostAfterSuccess@90/5 | 0.205000 | 0.028333 | 0.005774 | -0.176667 |
| RegressionRate@4 | 0.450000 | 0.463333 | 0.027538 | +0.013333 |
| RegressionRate@5 | 0.450000 | 0.495000 | 0.050744 | +0.045000 |
| LateRegressionRate@4-5 | 0.450000 | 0.479167 | 0.016073 | +0.029167 |
| LateMeanRegressionMagnitude@4-5 | 0.101800 | 0.003916 | 0.000699 | -0.097884 |
| LateSevereRegressionRate@4-5>0.05 | 0.265000 | 0.017500 | 0.006614 | -0.247500 |
| LateSevereRegressionRate@4-5>0.10 | 0.235000 | 0.006667 | 0.001443 | -0.228333 |
