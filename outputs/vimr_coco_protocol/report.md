# VIMR-Click multi-seed COCO protocol results

- Split: 500 train / 200 validation (split seed 2026)
- Schedule: 3 epochs, 5 clicks, training seeds [42, 43, 44]
- Best checkpoint: max validation `mIoU@5` per seed
- NoC failure value: 6
- Frozen modules: image encoder, prompt encoder, mask decoder

| Metric | Baseline | VIMR mean | VIMR std | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| mIoU@1 | 0.425855 | 0.774947 | 0.012108 | +0.349092 |
| mIoU@2 | 0.627116 | 0.800817 | 0.016514 | +0.173700 |
| mIoU@3 | 0.631522 | 0.811488 | 0.017425 | +0.179966 |
| mIoU@5 | 0.679223 | 0.822679 | 0.017983 | +0.143456 |
| NoC@85 | 3.570000 | 2.961667 | 0.205690 | -0.608333 |
| Success@85/5 | 0.645000 | 0.648333 | 0.048563 | +0.003333 |
| NoC@90 | 4.425000 | 4.048333 | 0.131941 | -0.376667 |
| Success@90/5 | 0.425000 | 0.425000 | 0.031225 | +0.000000 |
| RegressionRate | 0.430000 | 0.361667 | 0.040646 | -0.068333 |
| RegressionRate@4 | 0.440000 | 0.395000 | 0.063836 | -0.045000 |
| RegressionRate@5 | 0.460000 | 0.450000 | 0.052915 | -0.010000 |
| LateRegressionRate@4-5 | 0.450000 | 0.422500 | 0.058256 | -0.027500 |
