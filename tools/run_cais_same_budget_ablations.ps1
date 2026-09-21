$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$outputRoot = "outputs\cais_same_budget_ablations_20260921"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$common = @(
    "tools\train_cais_coco_click.py",
    "--dataset-root", "outputs\cais_synthetic_click2_click3_300_20260921",
    "--epochs", "10",
    "--source-sampling", "balanced",
    "--log-interval", "100"
)

$experiments = @(
    @{
        Name = "no_trajectory"
        Arguments = @("--train-mask-decoder", "--disable-trajectory-input")
    },
    @{
        Name = "no_cognitive_loss"
        Arguments = @("--train-mask-decoder", "--lambda-cognitive", "0", "--boundary-alpha", "0")
    },
    @{
        Name = "coordinate_only"
        Arguments = @(
            "--train-mask-decoder",
            "--disable-trajectory-input",
            "--lambda-cognitive", "0",
            "--boundary-alpha", "0"
        )
    },
    @{
        Name = "standard_sam"
        Arguments = @("--standard-sam-ablation")
    }
)

foreach ($experiment in $experiments) {
    $name = $experiment.Name
    $outputDir = Join-Path $outputRoot $name
    $summary = Join-Path $outputDir "training_summary.json"
    if (Test-Path -LiteralPath $summary) {
        Write-Host "Skipping completed experiment: $name"
        continue
    }
    Write-Host "Starting same-budget ablation: $name"
    & python @common --output-dir $outputDir @($experiment.Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw "Ablation $name failed with exit code $LASTEXITCODE"
    }
    Write-Host "Completed same-budget ablation: $name"
}

Write-Host "All same-budget ablations completed."
