# VIMR-Click-B MVP

## Repository mapping and baseline data flow

The original implementation remains in `efficientvit/models/efficientvit/sam.py`.
`EfficientViTSamImageEncoder` combines an EfficientViT backbone with `SamNeck`,
which maps selected multi-scale features to a `[B,256,64,64]` SAM image
embedding. `EfficientViTSam.forward` calls Meta SAM's `PromptEncoder` and
`MaskDecoder` for each image. The original predictor already caches the encoder
output in `set_image()` and accepts point, box, and low-resolution mask prompts.

Baseline tensor flow:

```text
preprocessed image [B,3,S,S]
  -> EfficientViT backbone + SamNeck
  -> image embedding [B,256,64,64]
point coords [B,N,2], labels [B,N], optional boxes/mask [B,1,256,256]
  -> SAM PromptEncoder
  -> sparse prompts [B,N(+padding/box),256]
     dense prompt [B,256,64,64]
image embedding + positional encoding + prompts
  -> SAM MaskDecoder
  -> low-resolution logits [B,1|3,256,256], IoU scores [B,1|3]
```

Checkpoint loading lives in `efficientvit/sam_model_zoo.py` and
`efficientvit/apps/setup.py`. Training lives under `efficientvit/samcore`; the
entry point is `applications/efficientvit_sam/train_efficientvit_sam_model.py`.

## VIMR insertion points

VIMR leaves the image encoder and both SAM modules intact. Current clicks still
go through the original prompt encoder. Viewport, history, and residual tokens
are image-conditioned by two light cross-attention layers and appended to SAM's
existing sparse prompt sequence. The unchanged mask decoder produces a draft;
the residual gate safely blends it with the prior probability mask.

```text
image -> cached EfficientViT embedding --------------------------+
current click -> original SAM PromptEncoder -> sparse/dense -----+-> SAM MaskDecoder
viewport -> ViewportIntentEncoder --------------------+           |      |
click/view/mask history -> causal ClickHistoryMemory -+-> fusion -+      v
previous masks -> MaskResidualAnalyzer ---------------+               draft
previous + draft + ROI/click/residual maps -> ResidualGatedSafeUpdate -> final
```

The best places to extend later are sparse-token fusion and the reserved
`DetailRefinement` interface. The EfficientViT backbone, `SamNeck`, SAM prompt
rules, SAM decoder, preprocessing, postprocessing, and baseline predictor remain
unchanged.

## Forward API and ablations

`VIMREfficientViTSam.forward` accepts cached `image_embeddings`, SAM-frame
`point_coords`/`point_labels`, probability `previous_mask`, normalized
`click_history`, viewport state/history, and optional mask history. Missing
viewport state defaults to `[0.5,0.5,0,0,0,0,0]`. Histories longer than 16 keep
the most recent 16.

Pass `vimr_config` to `create_efficientvit_sam_model`:

- Baseline: `enabled: false` (or disable all four `use_*` switches).
- +Viewport: viewport true; history/residual/safe-update false.
- +Viewport+History: viewport/history true; residual/safe-update false.
- +Viewport+History+Residual: viewport/history/residual true; safe-update false.
- Full VIMR: all four switches true.

The default freezes the image encoder, leaves the SAM prompt encoder and mask
decoder trainable, and keeps detail refinement disabled.

For L0, the measured parameter counts are:

| Component | Parameters |
| --- | ---: |
| EfficientViT image encoder | 30,728,224 |
| SAM prompt encoder | 6,220 |
| SAM mask decoder | 4,058,340 |
| Viewport encoder | 159,363 |
| History memory | 3,245,376 |
| Residual analyzer | 167,392 |
| Interaction fusion | 527,360 |
| Residual gate | 60,097 |
| VIMR added total | 4,159,588 |
| Full model total | 38,952,372 |
| Default trainable | 8,224,148 |

The official `efficientvit_sam_l0.pt` checkpoint was loaded strictly by the
baseline and non-strictly by VIMR: all 409 original state-dict entries matched,
with 89 VIMR-only keys reported as randomly initialized and no unexpected keys.

## Commands

```bash
python -m unittest discover -s tests -p "test_vimr.py" -v
python tools/count_vimr_params.py --model efficientvit-sam-vimr-l0
python applications/efficientvit_sam/demo_vimr_interactive.py IMAGE \
  --checkpoint assets/checkpoints/efficientvit_sam/distilled_model/efficientvit-sam-l0.pt
python applications/efficientvit_sam/train_efficientvit_sam_model.py \
  applications/efficientvit_sam/configs/efficientvit_sam_vimr_l0.yaml --path .exp/vimr_l0
```

The demo saves `mask_1.png` through `mask_3.png` and `gate_2.png`/`gate_3.png`.

## COCO training and evaluation protocol

`tools/run_vimr_coco_protocol.py` implements the first-priority research
protocol on a deterministic, image-disjoint COCO 2017 `train2017` holdout. Its
defaults are 500 training instances, 200 validation instances, three epochs,
five clicks, and training seeds 42/43/44. The EfficientViT image encoder, SAM
prompt encoder, and SAM mask decoder remain frozen. The pretrained baseline is
evaluated once on the shared validation split; every VIMR epoch is validated,
and the checkpoint with the highest validation `mIoU@5` is reloaded for final
reporting.

```bash
python tools/run_vimr_coco_protocol.py
```

Use `--epochs 5` for the longer requested schedule. Results are written under
`outputs/vimr_coco_protocol`: the exact split manifest, per-epoch histories,
per-instance click IoUs, one best VIMR-only checkpoint per seed, machine-readable
JSON/CSV, and a baseline-versus-VIMR Markdown table. Reported metrics are
`mIoU@1/2/3/5`, `NoC@85/90`, `Success@85/90`, overall regression rate,
per-click regression after clicks 4 and 5, and their combined late-stage rate.
If an instance does not reach a NoC threshold within five clicks, its NoC is
recorded as six. The generated report also records the deterministic instance
filter and corrective-click policy.

## Current limitations

The MVP does not implement the high-resolution Detail Expert, summary memory
beyond the last 16 interactions, real GUI viewport capture, or a learned
global/object/detail expert bank. Router logits and the detail interface are
present for future experiments. The training adapter uses a simple two-round
corrective-click simulation and is an adjustable starting point rather than a
final interaction-sampling curriculum.
