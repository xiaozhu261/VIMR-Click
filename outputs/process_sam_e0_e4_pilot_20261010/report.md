# Process-aware EfficientViT-SAM: E0–E4

Result type: matched_replay_experiment

Training epochs: 1; seed: 9527.

| Split | Recorded clicks | Tasks |
| --- | ---: | ---: |
| train | 1122 | 450 |
| val | 240 | 99 |
| test | 228 | 92 |

| Experiment | IoU@1 | IoU@3 | IoU@5 | NoC@90 | Boundary IoU | Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 0.8221 | 0.8435 | 0.8943 | 4.2500 | 0.6421 | 17.5052 |
| E1 | 0.8183 | 0.8400 | 0.8948 | 3.9167 | 0.6367 | 24.7006 |
| E2 | 0.8142 | 0.8401 | 0.8968 | 4.2500 | 0.6365 | 20.8633 |
| E3 | 0.8139 | 0.8393 | 0.8943 | 4.2500 | 0.6349 | 34.1764 |
| E4 | 0.8127 | 0.8454 | 0.8940 | 4.0000 | 0.6373 | 31.4907 |

Paired task-mean IoU delta E3 − E4: -0.001769, bootstrap 95% CI [-0.017564270441737567, 0.013688409924439654].

E4 uses E3's exact weights and events; only process inputs are shuffled within source/history-length strata.
All groups replay the recorded clicks and pre-click masks. NoC uses complete five-click prefixes only; failures score 6.
IoU/Boundary IoU are measured at original resolution. Latency includes prompt/process encoding, fusion and decoding with cached image features; excludes IO, image encoding and mask upsampling.
IoU@k over all available events has different task coverage at each k; complete-prefix IoU@k and coverage are in summary.json.
Test click coverage: {'1': 92, '2': 63, '3': 39, '4': 22, '5': 12}; complete five-click tasks: 12.
