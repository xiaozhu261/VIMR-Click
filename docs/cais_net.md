# CAIS-Net

CAIS-Net and VIMR are independent interactive-segmentation architectures. They
share the pretrained `EfficientViTSam` image encoder, prompt encoder utilities,
and mask decoder implementation, but CAIS does not import, instantiate, or call
any VIMR module. VIMR likewise contains no CAIS trajectory, cognition, or loss
logic.

```text
image -> EfficientViT-SAM image encoder -> cached [B,256,64,64] feature --------+
                                                                                |
click coordinate + label -> pretrained SAM point prompt ------------------------+
                                                                                |
round-wise click + pre-mask + trace -> interaction memory -> memory attention --+
                                                                                v
                                                                        SAM mask decoder
                                                                                |
                                                                  mask / IoU / cognition heads
```

## Direct model API

`CAISEfficientViTSam.forward` accepts:

- `image_embeddings`: cached EfficientViT features `[B,256,64,64]`, or `images`.
- `point_coords`: SAM-frame point coordinates `[B,N,2]`.
- `point_labels`: foreground/background labels `[B,N]`.
- `click_trajectories`: `[B,N,T,3]` relative `(time,x,y)` samples.
- `trajectory_valid_mask`: optional `[B,N,T]` padding mask.
- `process_features`: optional normalized annotation-process summaries `[B,N,F]`.
- `process_feature_valid`: validity mask for process summaries `[B,N]`.
- `previous_mask`: optional probability mask used only as the next SAM mask prompt.
- `interaction_masks`: per-round masks seen before each click `[B,N,1,256,256]`.
- `interaction_mask_valid`: validity mask for the round-wise mask history `[B,N]`.

The output contains `final_mask`, calibrated `iou_predictions`, the original
`sam_iou_predictions`, `cognitive_friction`, and `cognitive_vector`. The
stateful `CAISEfficientViTSamPredictor` accumulates new clicks and trajectories
while caching image features.

The MAIS-style experimental integration uses `prompt_integration=native`,
`attention_integration=none`, and
`interaction_memory=click_mask|click_mask_trajectory`. It preserves the
pretrained SAM point-prompt path, encodes each `(click, pre-mask, trajectory)`
round as a memory slot, and injects the attended memory only into image
features through a small residual gate. Earlier replacement, residual-token,
and dense-trajectory paths remain available for controlled ablations and
legacy checkpoints.

The `fixed_window_process_v2` trajectory path avoids stretching every recorded
fragment to the same number of fully valid samples. It interpolates only inside
the observed portion of a fixed 2-second pre-click clock, keeps a per-sample
validity mask, and adds train-split-normalized event summaries such as active
inter-click time, observed coverage, raw/unique point counts, path length,
hover/press duration, speed, braking, and endpoint consistency. Missing motion
is never fabricated; process summaries and paths are jointly removed by the
Zero/Shuffle/Wrong-image controls.

## Loss

`cais_multitask_loss` implements:

```text
L_mask = L_dice + mean((1 + alpha * (1 - F_cog) * B_gt) * BCE)
L_total = L_mask + lambda_iou * MSE + lambda_cognitive * SmoothL1
```

The default `lambda_cognitive` is `0.075`. A real `cognitive_friction` target is
used when the dataset provides one. Datasets without cursor telemetry use a
deterministic approach trajectory for segmentation/IoU training and leave the
cognitive regression term disabled rather than inventing a target.

## Commands

```bash
python -m unittest tests.test_cais -v
python tools/count_cais_params.py --model efficientvit-sam-cais-l0
python applications/efficientvit_sam/demo_cais_interactive.py IMAGE \
  --checkpoint assets/checkpoints/efficientvit_sam/distilled_model/efficientvit_sam_l0.pt
python applications/efficientvit_sam/train_efficientvit_sam_model.py \
  applications/efficientvit_sam/configs/efficientvit_sam_cais_l0.yaml --path .exp/cais_l0
```

For the prepared strict COCO-Click behavioral dataset, use its dedicated runner:

```powershell
python tools\train_cais_coco_click.py `
  --dataset-root path\to\cais_training_ready `
  --output-dir outputs\cais_coco_click_20260919 `
  --epochs 20 --train-mask-decoder --accumulation-steps 4 `
  --prompt-integration native --attention-integration none `
  --interaction-memory click_mask_trajectory --memory-size 8 `
  --trajectory-processing fixed_window_process_v2 --trajectory-window-ms 2000
```

Use `--trajectory-mode real|zero|shuffle` for trajectory controls. Shuffle
donors are deterministically selected within the same data source and prompt
history length, so the control changes motion alignment without changing the
number of clicks or the human/synthetic sampling ratio.

This path uses `alignment/modeling_clicks_strict.jsonl`, honors the fixed
image-grouped train/validation/test split, and decodes only the embedded
`ground_truth_rle` as the target. Process masks are used only as previous-round
prompts. Frozen EfficientViT image features are cached during the run.

The dataset has behavioral telemetry but no human-rated `F_cog` label. The
runner therefore records an explicit proxy definition in every checkpoint and
in `run_manifest.json`: seeking time, hover time, hard brakes, direction
changes, mean deceleration, and trajectory inefficiency are robustly scaled by
training-only 5th/95th percentiles and averaged. No mask or IoU value enters
this proxy.
