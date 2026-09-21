param(
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$outputRoot = "outputs\cais_trajectory_controls_3seed_20260921"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$common = @(
    "tools\train_cais_coco_click.py",
    "--dataset-root", "outputs\cais_synthetic_click2_click3_300_20260921",
    "--epochs", "10",
    "--source-sampling", "balanced",
    "--train-mask-decoder",
    "--prompt-integration", "native_residual",
    "--attention-integration", "none",
    "--lambda-cognitive", "0",
    "--boundary-alpha", "0",
    "--log-interval", "100"
)
if ($SmokeTest) {
    $common += "--smoke-test"
}

$seeds = @(9527, 1337, 2026)
$modes = @("real", "zero", "shuffle")

foreach ($seed in $seeds) {
    foreach ($mode in $modes) {
        $name = "${mode}_seed${seed}"
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
        Write-Host "Starting trajectory-control experiment: $name"
        & python @common --output-dir $outputDir --seed $seed --trajectory-mode $mode
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment $name failed with exit code $LASTEXITCODE"
        }
        Write-Host "Completed trajectory-control experiment: $name"
    }
}

Write-Host "All trajectory-control experiments completed."
