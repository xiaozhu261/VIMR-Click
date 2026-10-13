$ErrorActionPreference = "Continue"

$python = "E:\miniconda\python.exe"
$trainer = "tools\train_cais_coco_click.py"
$dataset = "outputs\cais_synthetic_click2_click3_300_20260921"
$root = "outputs\cais_mais_memory_seed9527_20261008"

$common = @(
    $trainer,
    "--dataset-root", $dataset,
    "--epochs", "10",
    "--seed", "9527",
    "--train-source", "human",
    "--trajectory-encoder", "tcn_attention",
    "--normalize-trajectory-input",
    "--prompt-integration", "native",
    "--attention-integration", "none",
    "--memory-size", "8",
    "--memory-layers", "1",
    "--memory-heads", "8",
    "--memory-ffn-dim", "512",
    "--memory-mask-channels", "32",
    "--memory-gate-init", "0.05",
    "--lambda-cognitive", "0.05",
    "--lambda-trajectory-cognitive", "0.05",
    "--boundary-alpha", "0",
    "--learning-rate", "1e-4",
    "--gate-learning-rate", "1e-3",
    "--decoder-learning-rate", "1e-5",
    "--train-mask-decoder",
    "--decoder-warmup-epochs", "2",
    "--evaluate-trajectory-controls"
)

$runs = @(
    @{ Name = "native_no_memory"; Mode = "none" },
    @{ Name = "click_mask_memory"; Mode = "click_mask" },
    @{ Name = "click_mask_trajectory_memory"; Mode = "click_mask_trajectory" }
)

New-Item -ItemType Directory -Force -Path $root | Out-Null
foreach ($run in $runs) {
    $output = Join-Path $root $run.Name
    $log = Join-Path $root ($run.Name + ".log")
    if (Test-Path (Join-Path $output "training_summary.json")) {
        Write-Output ("SKIP completed {0} {1}" -f $run.Name, (Get-Date -Format o))
        continue
    }
    Write-Output ("START {0} {1}" -f $run.Name, (Get-Date -Format o))
    & $python @common --output-dir $output --interaction-memory $run.Mode 2>&1 |
        Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Error "Training failed for $($run.Name) with exit code $exitCode"
        exit $exitCode
    }
    Write-Output ("DONE {0} {1}" -f $run.Name, (Get-Date -Format o))
}

& $python tools\summarize_cais_mais_memory.py $root
