param(
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$outputDir = "outputs\cais_dense_trajectory_seed9527_20260922"
$arguments = @(
    "tools\train_cais_coco_click.py",
    "--dataset-root", "outputs\cais_synthetic_click2_click3_300_20260921",
    "--output-dir", $outputDir,
    "--epochs", "10",
    "--seed", "9527",
    "--source-sampling", "balanced",
    "--train-mask-decoder",
    "--trajectory-mode", "real",
    "--normalize-trajectory-input",
    "--trajectory-encoder", "tcn_attention",
    "--prompt-integration", "trajectory_dense",
    "--attention-integration", "none",
    "--prompt-residual-gate-init", "0.05",
    "--trajectory-dense-channels", "32",
    "--trajectory-raster-sigma", "1.5",
    "--decoder-warmup-epochs", "2",
    "--learning-rate", "1e-4",
    "--decoder-learning-rate", "1e-5",
    "--gate-learning-rate", "1e-3",
    "--lambda-cognitive", "0.05",
    "--lambda-trajectory-cognitive", "0.05",
    "--boundary-alpha", "0",
    "--evaluate-trajectory-controls",
    "--log-interval", "100"
)
if ($SmokeTest) {
    $arguments += "--smoke-test"
}

& python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Dense-trajectory experiment failed with exit code $LASTEXITCODE"
}
