# Global/Object/Detail Router and Detail Expert ablation

All variants use the same 500/200 split, 3 epochs, 5-click policy, seeds, and frozen SAM modules.
Safe Update uses lambda_safe=1.0, lambda_monotonic=0.5, gate init=0.1.

| Model | Trainable params | Best epochs | mIoU@1 | mIoU@2 | mIoU@3 | mIoU@5 | NoC@85 | NoC@90 | Success@85/5 | Success@90/5 | RegressionRate | LateRegressionRate@4-5 | LateSevereRegressionRate@4-5>0.05 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full VIMR (matched rerun) | 4,159,588 | [3, 3, 3] | 0.768205 ± 0.018613 | 0.784220 ± 0.032104 | 0.792635 ± 0.039080 | 0.797096 ± 0.042351 | 3.185000 ± 0.329697 | 4.165000 ± 0.294830 | 0.591667 ± 0.083865 | 0.388333 ± 0.076540 | 0.343333 ± 0.021372 | 0.387500 ± 0.038487 | 0.010000 ± 0.015207 |
| +G/O/D Router | 4,750,948 | [3, 3, 3] | 0.785054 ± 0.006777 | 0.804646 ± 0.003451 | 0.811274 ± 0.005219 | 0.817281 ± 0.003236 | 2.981667 ± 0.045369 | 4.045000 ± 0.030414 | 0.638333 ± 0.015275 | 0.418333 ± 0.005774 | 0.365833 ± 0.014488 | 0.424167 ± 0.031258 | 0.010833 ± 0.005204 |
| +G/O/D Router + Detail Expert | 4,754,229 | [2, 3, 3] | 0.764537 ± 0.027582 | 0.795748 ± 0.003305 | 0.806875 ± 0.004638 | 0.815394 ± 0.005542 | 3.025000 ± 0.127574 | 4.088333 ± 0.076376 | 0.645000 ± 0.010000 | 0.415000 ± 0.000000 | 0.375417 ± 0.048611 | 0.462500 ± 0.092601 | 0.011667 ± 0.012583 |

## Router usage on fixed synthetic viewport trajectories

Values are argmax fractions for global/object/detail. Stable semantic routing should not change drastically between seeds.

| Model | Seed | Global | Object | Detail |
| --- | ---: | ---: | ---: | ---: |
| +G/O/D Router | 42 | 0.0000 | 0.0000 | 1.0000 |
| +G/O/D Router | 43 | 0.7520 | 0.2480 | 0.0000 |
| +G/O/D Router | 44 | 0.2000 | 0.4410 | 0.3590 |
| +G/O/D Router + Detail Expert | 42 | 0.0000 | 0.0000 | 1.0000 |
| +G/O/D Router + Detail Expert | 43 | 0.0000 | 0.0000 | 1.0000 |
| +G/O/D Router + Detail Expert | 44 | 0.0000 | 0.0000 | 1.0000 |
