# Process-aware EfficientViT-SAM: E0–E4

Result type: matched_replay_experiment

Training epochs: 10; seed: 9527.

| Split | Recorded clicks | Tasks |
| --- | ---: | ---: |
| train | 1122 | 450 |
| val | 240 | 99 |
| test | 228 | 92 |

| Experiment | IoU@1 | IoU@3 | IoU@5 | NoC@90 | Boundary IoU | Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 0.8398 | 0.8455 | 0.8978 | 4.4167 | 0.6471 | 16.1373 |
| E1 | 0.8345 | 0.8438 | 0.8947 | 4.1667 | 0.6457 | 26.5013 |
| E2 | 0.8286 | 0.8475 | 0.8961 | 4.2500 | 0.6447 | 25.8902 |
| E3 | 0.8292 | 0.8473 | 0.8951 | 4.2500 | 0.6439 | 29.5986 |
| E4 | 0.8303 | 0.8452 | 0.8901 | 4.2500 | 0.6439 | 26.1749 |

Paired task-mean IoU delta E3 − E4: 0.003044, bootstrap 95% CI [-0.011391808485639268, 0.018337040015469302].

E4 uses E3's exact weights and events; only process inputs are shuffled within source/history-length strata.
All groups replay the recorded clicks and pre-click masks. NoC uses complete five-click prefixes only; failures score 6.
IoU/Boundary IoU are measured at original resolution. Latency includes prompt/process encoding, fusion and decoding with cached image features; excludes IO, image encoding and mask upsampling.
IoU@k over all available events has different task coverage at each k; complete-prefix IoU@k and coverage are in summary.json.
Test click coverage: {'1': 92, '2': 63, '3': 39, '4': 22, '5': 12}; complete five-click tasks: 12.
