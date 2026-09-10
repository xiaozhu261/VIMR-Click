# VIMR-Click multi-seed COCO protocol results

- Split: 500 train / 200 validation (split seed 2026)
- Schedule: 3 epochs, 5 clicks, training seeds [42, 43, 44]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.425946 | 0.771781 | 0.008190 | +0.345835 |
| mIoU@2 | 0.626218 | 0.793190 | 0.004835 | +0.166972 |
| mIoU@3 | 0.630606 | 0.801401 | 0.006588 | +0.170795 |
| mIoU@5 | 0.677160 | 0.799698 | 0.010288 | +0.122537 |
| NoC@85 | 3.585000 | 3.000000 | 0.065574 | -0.585000 |
| Success@85/5 | 0.645000 | 0.638333 | 0.012583 | -0.006667 |
| NoC@90 | 4.435000 | 4.068333 | 0.016073 | -0.366667 |
| Success@90/5 | 0.430000 | 0.415000 | 0.008660 | -0.015000 |
| RegressionRate | 0.430000 | 0.453750 | 0.034027 | +0.023750 |
| MeanRegressionMagnitude | 0.112610 | 0.008213 | 0.002060 | -0.104397 |
| FinalSuccess@85/5 | 0.385000 | 0.585000 | 0.017321 | +0.200000 |
| LostAfterSuccess@85/5 | 0.260000 | 0.053333 | 0.020207 | -0.206667 |
| FinalSuccess@90/5 | 0.225000 | 0.363333 | 0.020817 | +0.138333 |
| LostAfterSuccess@90/5 | 0.205000 | 0.051667 | 0.012583 | -0.153333 |
| RegressionRate@4 | 0.450000 | 0.528333 | 0.010408 | +0.078333 |
| RegressionRate@5 | 0.450000 | 0.495000 | 0.052678 | +0.045000 |
| LateRegressionRate@4-5 | 0.450000 | 0.511667 | 0.027651 | +0.061667 |
| LateMeanRegressionMagnitude@4-5 | 0.101800 | 0.008493 | 0.002593 | -0.093307 |
| LateSevereRegressionRate@4-5>0.05 | 0.265000 | 0.037500 | 0.015207 | -0.227500 |
| LateSevereRegressionRate@4-5>0.10 | 0.235000 | 0.019167 | 0.008780 | -0.215833 |
