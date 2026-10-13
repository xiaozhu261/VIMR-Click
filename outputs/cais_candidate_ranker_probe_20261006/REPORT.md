# CAIS learned local candidate-ranker probe

## Protocol

A small MLP ranks the same fixed candidate grid using only deployable SAM/mask-change features. The fused model additionally receives candidate-wise cursor direction features. All results are task-grouped out-of-fold; Shuffle and Null replace trajectory only at inference using the same fused checkpoint.

- Records/tasks: 142 / 126
- Seeds: [9527, 1337, 2026]
- No-op margin: 0.005

## Quality-gated policy

| Method | Selected mean delta | All-record policy delta | Win | Harm | No-op |
|---|---:|---:|---:|---:|---:|
| ranker_base | +0.024542 | +0.005358 | 55.91% | 37.63% | 6.45% |
| ranker_real | +0.034965 | +0.007633 | 58.06% | 35.48% | 6.45% |
| ranker_shuffle | +0.035289 | +0.007704 | 59.14% | 34.41% | 6.45% |
| ranker_null | +0.029216 | +0.006378 | 52.69% | 34.41% | 12.90% |
| grid_oracle | +0.099578 | +0.021739 | 100.00% | 0.00% | 0.00% |

## Paired task-bootstrap

| Contrast | Mean | 95% CI |
|---|---:|---:|
| Base ranker vs original | +0.005358 | [-0.007210, +0.017943] |
| Real trajectory vs base | +0.002275 | [-0.001061, +0.006370] |
| Real trajectory vs Shuffle | -0.000071 | [-0.003061, +0.002481] |
| Real trajectory vs Null | +0.001255 | [-0.001891, +0.005031] |

## Decision

- Learned base selector is feasible: `False`
- Real trajectory beats Shuffle: `False`
- Real trajectory beats Null: `False`
- Trajectory has independent value: `False`
