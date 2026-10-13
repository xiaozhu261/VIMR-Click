# CAIS positive-click trajectory gate probe

## Question

Among GT-audited locally correctable positive clicks, can the last 500 ms mouse trajectory identify corrections whose frozen-SAM oracle IoU gain exceeds 0.01?

- Records: 142 / tasks: 126
- Beneficial corrections: 31 (21.83%)
- Evaluation: 5-fold task-grouped OOF, seeds [9527, 1337, 2026]
- Gate selection fraction: 20%

## Seed-ensemble OOF results

| Input/control | ROC-AUC | AP | Policy IoU gain over all records | Selected benefit precision |
|---|---:|---:|---:|---:|
| quality_only | 0.8316 | 0.5703 | +0.010612 | 55.17% |
| context | 0.7370 | 0.4559 | +0.009210 | 51.72% |
| trajectory_real | 0.5065 | 0.2238 | +0.002416 | 24.14% |
| fused_real | 0.7402 | 0.4259 | +0.010458 | 48.28% |
| fused_shuffle | 0.7382 | 0.4122 | +0.010194 | 44.83% |
| fused_null | 0.7414 | 0.4309 | +0.009585 | 41.38% |

## Paired task-bootstrap checks

| Check | Delta | 95% CI | Pass |
|---|---:|---:|---:|
| Fused Real - Context AUC | +0.0032 | [-0.0878, +0.0824] | False |
| Fused Real - Shuffle AUC | +0.0020 | [-0.0914, +0.0869] | False |
| Fused - Context policy IoU | +0.001249 | [-0.002953, +0.005591] | False |

Proceed to learn trajectory offsets: `False`
