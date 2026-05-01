param(
    [string]$RootDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RootDir)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RootDir = Join-Path "figs" ("requested_cpd_large_" + $timestamp)
}

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$root = Join-Path $projectRoot $RootDir
$suddenRoot = Join-Path $root "sudden"
$mixedRoot = Join-Path $root "mixed_theta"
$plantRoot = Join-Path $root "plantSim"
$logRoot = Join-Path $root "logs"

New-Item -ItemType Directory -Force -Path $root, $suddenRoot, $mixedRoot, $plantRoot, $logRoot | Out-Null

function Invoke-LoggedConda {
    param(
        [string]$Name,
        [string[]]$CommandArgs
    )
    $logPath = Join-Path $logRoot ($Name + ".log")
    Write-Host "=== [$Name] ==="
    Write-Host ("conda run " + ($CommandArgs -join " "))
    $prevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & conda run @CommandArgs 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevErrorAction
    }
    if ($exitCode -ne 0) {
        throw "Command failed for $Name. See $logPath"
    }
}

$methods = @(
    "half_refit",
    "shared_onlineBPC_proxyStableMean_sigmaObs",
    "shared_onlineBPC_proxyStableMean_sigmaObs_wCUSUM",
    "shared_onlineBPC_exact_sigmaObs",
    "shared_onlineBPC_exact_sigmaObs_wCUSUM",
    "shared_onlineBPC_fixedSupport_sigmaObs",
    "shared_onlineBPC_fixedSupport_sigmaObs_wCUSUM"
)
$seeds = 0..9 | ForEach-Object { "$_" }
$magnitudes = @("0.5", "1.0", "2.0", "3.0")
$segLens = @("80", "120", "200")

foreach ($mag in $magnitudes) {
    foreach ($seg in $segLens) {
        $tagMag = $mag.Replace(".", "p")
        $outDir = Join-Path $suddenRoot ("sudden_mag" + $tagMag + "_seg" + $seg)
        if (Test-Path (Join-Path $outDir 'mechanism_metric_summary.csv')) {
            Write-Host "Skipping sudden_mag${tagMag}_seg${seg}; summary already exists."
            continue
        }
        $cmd = @(
            "-n", "jumpGP", "python", "-m", "calib.run_synthetic_mechanism_figures",
            "--scenarios", "sudden",
            "--seeds"
        ) + $seeds + @(
            "--batch-size", "20",
            "--sudden-mag", $mag,
            "--sudden-seg-len", $seg,
            "--num-particles", "1024",
            "--out_dir", $outDir,
            "--methods"
        ) + $methods
        Invoke-LoggedConda -Name ("sudden_mag" + $tagMag + "_seg" + $seg) -CommandArgs $cmd
    }
}

if (-not (Test-Path (Join-Path $mixedRoot 'all_metrics.csv'))) {
    Invoke-LoggedConda -Name "mixed_theta_cpd_ablation" -CommandArgs @(
        "-n", "jumpGP", "python", "-m", "calib.run_synthetic_mixed_thetaCmp",
        "--profile", "cpd_ablation",
        "--num_particles", "1024",
        "--out_dir", $mixedRoot
    )
} else {
    Write-Host "Skipping mixed_theta_cpd_ablation; all_metrics.csv already exists."
}

$plantDone = ((Get-ChildItem $plantRoot -Filter 'plantSim_results_mode*.pt' -ErrorAction SilentlyContinue | Measure-Object).Count -ge 3)
if (-not $plantDone) {
    Invoke-LoggedConda -Name "plantSim_cpd_ablation" -CommandArgs @(
        "-n", "jumpGP", "python", "-m", "calib.run_plantSim_v3_std",
        "--profile", "cpd_ablation",
        "--num_particles", "1024",
        "--modes", "0", "1", "2",
        "--out_dir", $plantRoot
    )
} else {
    Write-Host "Skipping plantSim_cpd_ablation; mode result files already exist."
}

Invoke-LoggedConda -Name "aggregate_summary" -CommandArgs @(
    "-n", "jumpGP", "python", "tools/summarize_requested_cpd_experiments.py",
    "--root", $root
)

Set-Content -Path (Join-Path $root "RUN_COMPLETE.txt") -Value ("Completed at " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Host "All requested experiments completed."
Write-Host "Root output: $root"
