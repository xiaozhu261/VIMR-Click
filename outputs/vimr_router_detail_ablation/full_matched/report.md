# VIMR-Click multi-seed COCO protocol results

- Split: 500 train / 200 validation (split seed 2026)
- Schedule: 3 epochs, 5 clicks, training seeds [42, 43, 44]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.425946 | 0.768205 | 0.018613 | +0.342260 |
| mIoU@2 | 0.626218 | 0.784220 | 0.032104 | +0.158002 |
| mIoU@3 | 0.630606 | 0.792635 | 0.039080 | +0.162029 |
| mIoU@5 | 0.677160 | 0.797096 | 0.042351 | +0.119935 |
| NoC@85 | 3.585000 | 3.185000 | 0.329697 | -0.400000 |
| Success@85/5 | 0.645000 | 0.591667 | 0.083865 | -0.053333 |
| NoC@90 | 4.435000 | 4.165000 | 0.294830 | -0.270000 |
| Success@90/5 | 0.430000 | 0.388333 | 0.076540 | -0.041667 |
| RegressionRate | 0.430000 | 0.343333 | 0.021372 | -0.086667 |
| MeanRegressionMagnitude | 0.112610 | 0.002343 | 0.002002 | -0.110267 |
| FinalSuccess@85/5 | 0.385000 | 0.576667 | 0.070770 | +0.191667 |
| LostAfterSuccess@85/5 | 0.260000 | 0.015000 | 0.015000 | -0.245000 |
| FinalSuccess@90/5 | 0.225000 | 0.376667 | 0.066583 | +0.151667 |
| LostAfterSuccess@90/5 | 0.205000 | 0.011667 | 0.010408 | -0.193333 |
| RegressionRate@4 | 0.450000 | 0.398333 | 0.032146 | -0.051667 |
| RegressionRate@5 | 0.450000 | 0.376667 | 0.050083 | -0.073333 |
| LateRegressionRate@4-5 | 0.450000 | 0.387500 | 0.038487 | -0.062500 |
| LateMeanRegressionMagnitude@4-5 | 0.101800 | 0.002416 | 0.002428 | -0.099384 |
| LateSevereRegressionRate@4-5>0.05 | 0.265000 | 0.010000 | 0.015207 | -0.255000 |
| LateSevereRegressionRate@4-5>0.10 | 0.235000 | 0.005000 | 0.006614 | -0.230000 |
