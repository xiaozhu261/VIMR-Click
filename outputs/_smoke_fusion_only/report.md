# VIMR-Click multi-seed COCO protocol results

- Split: 2 train / 2 validation (split seed 2026)
- Schedule: 1 epochs, 5 clicks, training seeds [7]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.417314 | 0.357666 | 0.000000 | -0.059648 |
| mIoU@2 | 0.627931 | 0.004813 | 0.000000 | -0.623117 |
| mIoU@3 | 0.533077 | 0.127088 | 0.000000 | -0.405989 |
| mIoU@5 | 0.781644 | 0.190881 | 0.000000 | -0.590763 |
| NoC@85 | 5.000000 | 6.000000 | 0.000000 | +1.000000 |
| Success@85/5 | 0.500000 | 0.000000 | 0.000000 | -0.500000 |
| NoC@90 | 5.000000 | 6.000000 | 0.000000 | +1.000000 |
| Success@90/5 | 0.500000 | 0.000000 | 0.000000 | -0.500000 |
| RegressionRate | 0.375000 | 0.500000 | 0.000000 | +0.125000 |
| MeanRegressionMagnitude | 0.069627 | 0.120111 | 0.000000 | +0.050484 |
| FinalSuccess@85/5 | 0.500000 | 0.000000 | 0.000000 | -0.500000 |
| LostAfterSuccess@85/5 | 0.000000 | 0.000000 | 0.000000 | +0.000000 |
| FinalSuccess@90/5 | 0.500000 | 0.000000 | 0.000000 | -0.500000 |
| LostAfterSuccess@90/5 | 0.000000 | 0.000000 | 0.000000 | +0.000000 |
| RegressionRate@4 | 0.000000 | 0.500000 | 0.000000 | +0.500000 |
| RegressionRate@5 | 0.500000 | 0.000000 | 0.000000 | -0.500000 |
| LateRegressionRate@4-5 | 0.250000 | 0.250000 | 0.000000 | +0.000000 |
| LateMeanRegressionMagnitude@4-5 | 0.013266 | 0.062532 | 0.000000 | +0.049266 |
| LateSevereRegressionRate@4-5>0.05 | 0.250000 | 0.250000 | 0.000000 | +0.000000 |
| LateSevereRegressionRate@4-5>0.10 | 0.000000 | 0.250000 | 0.000000 | +0.250000 |
