param(
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$rootOutput = "outputs\cais_human_only_constant_token_3seed_20261005"
$seeds = @(9527, 1337, 2026)

foreach ($seed in $seeds) {
    $outputDir = Join-Path $rootOutput "seed_$seed"
    $arguments = @(
        "tools\train_cais_coco_click.py",
        "--dataset-root", "outputs\cais_synthetic_click2_click3_300_20260921",
        "--output-dir", $outputDir,
        "--epochs", "10",
        "--seed", "$seed",
        "--train-source", "human",
        "--source-sampling", "natural",
        "--train-mask-decoder",
        "--trajectory-mode", "real",
        "--normalize-trajectory-input",
        "--trajectory-encoder", "tcn_attention",
        "--trajectory-token-content", "constant",
        "--prompt-integration", "trajectory_token",
        "--attention-integration", "none",
        "--prompt-residual-gate-init", "0.05",
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
        throw "Human-only constant-token seed $seed failed with exit code $LASTEXITCODE"
    }
}
