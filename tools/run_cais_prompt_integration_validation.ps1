param(
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$outputRoot = "outputs\cais_integration_validation_20260921"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$common = @(
    "tools\train_cais_coco_click.py",
    "--dataset-root", "outputs\cais_synthetic_click2_click3_300_20260921",
    "--epochs", "10",
    "--source-sampling", "balanced",
    "--lambda-cognitive", "0",
    "--boundary-alpha", "0",
    "--log-interval", "100"
)
if ($SmokeTest) {
    $common += "--smoke-test"
}

$experiments = @(
    @{
        Name = "replacement_no_attention"
        Description = "Custom coordinate/type tokens replace native SAM points; no trajectory or CAIS attention."
        Arguments = @(
            "--train-mask-decoder",
            "--disable-trajectory-input",
            "--prompt-integration", "replace",
            "--attention-integration", "none"
        )
    },
    @{
        Name = "native_trajectory_residual"
        Description = "Native SAM points retained; trajectory enters through a zero-init prompt residual; no CAIS attention."
        Arguments = @(
            "--train-mask-decoder",
            "--prompt-integration", "native_residual",
            "--attention-integration", "none"
        )
    },
    @{
        Name = "native_gated_attention"
        Description = "Native SAM points retained; trajectory and two-way attention enter through zero-init residual gates."
        Arguments = @(
            "--train-mask-decoder",
            "--prompt-integration", "native_residual",
            "--attention-integration", "gated_residual"
        )
    },
    @{
        Name = "native_gated_cais_only"
        Description = "Parameter-matched test: native gated CAIS modules train while the pretrained mask decoder stays frozen."
        Arguments = @(
            "--prompt-integration", "native_residual",
            "--attention-integration", "gated_residual"
        )
    }
)

foreach ($experiment in $experiments) {
    $name = $experiment.Name
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
    Write-Host "Starting prompt-integration validation: $name"
    Write-Host $experiment.Description
    & python @common --output-dir $outputDir @($experiment.Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment $name failed with exit code $LASTEXITCODE"
    }
    Write-Host "Completed prompt-integration validation: $name"
}

Write-Host "All prompt-integration validation experiments completed."
