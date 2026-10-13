# CAIS quality-gated local candidate search

## Question

On GT-audited, locally correctable positive clicks, can frozen-SAM quality ranking select a better point from a fixed local grid, and does the real cursor-approach axis improve selection beyond shuffled or null trajectory controls?

- Records: 142 / tasks: 126
- Candidate budget: 33.0 points per click
- Quality gate target fraction: 20%
- Evaluation: grouped 5-fold OOF, seeds [9527, 1337, 2026]

## Quality-gated policy results

| Method | Selected fraction | Selected mean IoU delta | All-record policy delta | Win rate | Harm rate |
|---|---:|---:|---:|---:|---:|
| quality_raw | 21.83% | -0.037709 | -0.008232 | 56.99% | 43.01% |
| quality_distance | 21.83% | +0.007973 | +0.001741 | 39.78% | 29.03% |
| trajectory_real | 21.83% | +0.004237 | +0.000925 | 40.86% | 25.81% |
| trajectory_shuffle | 21.83% | +0.008803 | +0.001922 | 40.86% | 26.88% |
| trajectory_null | 21.83% | +0.008519 | +0.001860 | 39.78% | 25.81% |
| grid_oracle | 21.83% | +0.099578 | +0.021739 | 100.00% | 0.00% |

## Paired task-bootstrap checks

| Contrast | Mean delta | 95% CI |
|---|---:|---:|
| Quality-distance policy vs original | +0.001741 | [-0.011332, +0.013197] |
| Trajectory Real vs quality-distance | -0.000816 | [-0.002931, +0.000331] |
| Trajectory Real vs Shuffle | -0.000997 | [-0.003134, +0.000460] |
| Trajectory Real vs Null | -0.000935 | [-0.003039, +0.000111] |

## Predeclared decision

- Local quality search is feasible: `False`
- Real trajectory beats Shuffle: `False`
- Real trajectory beats Null: `False`
- Proceed to a learned trajectory-conditioned ranker: `False`

This is a conditional feasibility study, not an end-to-end claim: all evaluated records were identified as hard and locally correctable using GT during the preceding audit. The quality gate and candidate selectors themselves do not use GT at held-out inference.
