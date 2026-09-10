# VIMR-Click multi-seed COCO protocol results

- Split: 2 train / 2 validation (split seed 2026)
- Schedule: 1 epochs, 3 clicks, training seeds [7]
- Best checkpoint: max validation `mIoU@3` per seed
- NoC failure value: 4
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.413379 | 0.226851 | 0.000000 | -0.186528 |
| mIoU@2 | 0.625117 | 0.206330 | 0.000000 | -0.418787 |
| mIoU@3 | 0.530454 | 0.214696 | 0.000000 | -0.315757 |
| NoC@85 | 4.000000 | 4.000000 | 0.000000 | +0.000000 |
| Success@85/3 | 0.000000 | 0.000000 | 0.000000 | +0.000000 |
| NoC@90 | 4.000000 | 4.000000 | 0.000000 | +0.000000 |
| Success@90/3 | 0.000000 | 0.000000 | 0.000000 | +0.000000 |
| RegressionRate | 0.500000 | 0.750000 | 0.000000 | +0.250000 |
| MeanRegressionMagnitude | 0.124004 | 0.012621 | 0.000000 | -0.111383 |
| FinalSuccess@85/3 | 0.000000 | 0.000000 | 0.000000 | +0.000000 |
| LostAfterSuccess@85/3 | 0.000000 | 0.000000 | 0.000000 | +0.000000 |
| FinalSuccess@90/3 | 0.000000 | 0.000000 | 0.000000 | +0.000000 |
| LostAfterSuccess@90/3 | 0.000000 | 0.000000 | 0.000000 | +0.000000 |
