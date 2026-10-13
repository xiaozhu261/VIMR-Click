$ErrorActionPreference = "Continue"

$python = "E:\miniconda\python.exe"
$output = "outputs\cais_process_quality_gated_clicks_seed9527_20261008\quality_gated_clicks"
$log = "outputs\cais_process_quality_gated_clicks_seed9527_20261008\quality_gated_clicks.log"

New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
& $python tools\train_cais_coco_click.py `
    --dataset-root outputs\cais_synthetic_click2_click3_300_20260921 `
    --output-dir $output `
    --epochs 10 `
    --seed 9527 `
    --train-source human `
    --quality-gate-trajectories `
    --trajectory-encoder tcn_attention `
    --normalize-trajectory-input `
    --trajectory-processing fixed_window_process_v2 `
    --trajectory-window-ms 2000 `
    --process-gate-init 0.05 `
    --prompt-integration native `
    --attention-integration none `
    --interaction-memory click_mask_trajectory `
    --memory-size 8 `
    --memory-layers 1 `
    --memory-heads 8 `
    --memory-ffn-dim 512 `
    --memory-mask-channels 32 `
    --memory-gate-init 0.05 `
    --lambda-cognitive 0.05 `
    --lambda-trajectory-cognitive 0.05 `
    --boundary-alpha 0 `
    --learning-rate 1e-4 `
    --gate-learning-rate 1e-3 `
    --decoder-learning-rate 1e-5 `
    --train-mask-decoder `
    --decoder-warmup-epochs 2 `
    --evaluate-trajectory-controls 2>&1 | Tee-Object -FilePath $log

if ($LASTEXITCODE -ne 0) {
    throw "CAIS click-level quality-gated training failed with exit code $LASTEXITCODE"
}
