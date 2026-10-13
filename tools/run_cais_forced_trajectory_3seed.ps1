param(
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$outputRoot = "outputs\cais_forced_trajectory_3seed_20260922"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$common = @(
    "tools\train_cais_coco_click.py",
    "--dataset-root", "outputs\cais_synthetic_click2_click3_300_20260921",
    "--epochs", "10",
    "--source-sampling", "balanced",
    "--train-mask-decoder",
    "--prompt-integration", "native_residual",
    "--attention-integration", "none",
    "--trajectory-mode", "real",
    "--normalize-trajectory-input",
    "--prompt-residual-gate-init", "0.05",
    "--decoder-warmup-epochs", "2",
    "--learning-rate", "1e-4",
    "--decoder-learning-rate", "1e-5",
    "--gate-learning-rate", "1e-3",
    "--lambda-cognitive", "0.05",
    "--boundary-alpha", "0",
    "--evaluate-trajectory-controls",
    "--log-interval", "100"
)
if ($SmokeTest) {
    $common += "--smoke-test"
}

$seeds = @(9527, 1337, 2026)
foreach ($seed in $seeds) {
    $name = "real_seed${seed}"
    $outputDir = Join-Path $outputRoot $name
    $summary = if ($SmokeTest) {
        Join-Path $outputDir "smoke\training_summary.json"
    } else {
        Join-Path $outputDir "training_summary.json"
    }
    if (Test-Path -LiteralPath $summary) {
        Write-Host "Skipping completed experiment: $name"
        continue
    }
    Write-Host "Starting forced-trajectory experiment: $name"
    & python @common --output-dir $outputDir --seed $seed
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment $name failed with exit code $LASTEXITCODE"
    }
    Write-Host "Completed forced-trajectory experiment: $name"
}

Write-Host "All forced-trajectory experiments completed."
