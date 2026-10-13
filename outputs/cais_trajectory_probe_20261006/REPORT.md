# CAIS human-trajectory information probe

## Question

Before changing the full CAIS mask architecture again, test whether the last
500 ms of a real mouse trajectory predicts whether a human will make another
corrective click after clicks 1--4.

The target is derived from later interaction behavior, not from the trajectory
heuristic used by the existing cognitive loss. Click 5 is excluded so that the
probe cannot learn the trivial maximum-click stopping rule.

## Protocol

- Human records only: `coco_click2 + coco_click3`.
- Fixed existing task splits; no task crosses train/validation/test.
- Train: 1,082 records / 450 tasks.
- Validation: 229 records / 99 tasks.
- Test: 216 records / 92 tasks.
- Positive target rate on test: 62.96%.
- Three seeds: 9527, 1337, 2026.
- Probe inputs:
  - Context: click index/type/location, previous-mask IoU and area.
  - Trajectory: fixed-time `dx, dy, dx/dt, dy/dt, speed, valid` samples.
  - Fused: Context + Trajectory.
- Same-checkpoint controls replace test trajectories with cross-task shuffled
  trajectories or an all-zero null trajectory.
- Confidence intervals use 2,000 task-level bootstrap resamples.

## Results

| Probe / control | Test ROC-AUC |
|---|---:|
| Context only, 3-seed mean | 0.5279 ± 0.0049 |
| Trajectory only, 3-seed mean | 0.5178 ± 0.0007 |
| Fused Real, 3-seed mean | 0.5544 ± 0.0035 |
| Fused Shuffle, 3-seed mean | 0.5365 ± 0.0099 |
| Fused Null, 3-seed mean | 0.5534 ± 0.0077 |
| Context ensemble | 0.5278 |
| Trajectory ensemble | 0.5205 |
| Fused Real ensemble | 0.5576 |
| Fused Shuffle ensemble | 0.5326 |
| Fused Null ensemble | 0.5518 |

| Task-bootstrap comparison on seed ensemble | AUC delta | 95% CI |
|---|---:|---:|
| Fused Real - Context | +0.0298 | [-0.0104, +0.0664] |
| Fused Real - Shuffle | +0.0250 | [-0.0141, +0.0643] |
| Fused Real - Null | +0.0058 | [-0.0353, +0.0460] |

## Decision

The experiment does not establish useful trajectory semantics:

1. The trajectory-only probe is near chance (AUC 0.5205).
2. All task-level confidence intervals cross zero.
3. The fused Real and Null results are almost identical; the apparent fused
   improvement can therefore be explained by the extra branch/regularization,
   rather than the trajectory values.
4. Shuffled trajectories are unstable and can even outperform real
   trajectories in the trajectory-only probe, which is inconsistent with a
   learned human-motion relationship.

The predeclared gate for starting a more expensive local-cross-attention SAM
run was not met. The next defensible step is to improve the behavioral target
and data collection (explicit confidence/difficulty, completion reason, or
next-error location) before another mask-fusion architecture experiment.
