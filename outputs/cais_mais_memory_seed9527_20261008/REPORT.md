# CAIS MAIS-style Memory Ablation

| Run | Best epoch | Val IoU | Test click IoU | Test task IoU | Memory gate |
|---|---:|---:|---:|---:|---:|
| native_no_memory | 8 | 0.858996 | 0.851394 | 0.861606 | - |
| click_mask_memory | 4 | 0.862822 | 0.847076 | 0.861002 | 0.104128 |
| click_mask_trajectory_memory | 4 | 0.862377 | 0.847971 | 0.861874 | 0.104858 |

## Paired task-level deltas

- click_mask_minus_native: -0.000604, 95% bootstrap CI [-0.009261, +0.007045] over 92 tasks.
- trajectory_minus_click_mask: +0.000872, 95% bootstrap CI [-0.000454, +0.002441] over 92 tasks.
