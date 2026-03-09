# =============================================================================
# Experimento B: ejecución controlada (log + monitor de recursos)
# conda tfm, Windows, GPU
# =============================================================================

$condaExe = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
if (-not (Test-Path $condaExe)) {
    Write-Error "No se encuentra conda en $condaExe."
    exit 1
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
Set-Location $projectRoot
& $condaExe run -n tfm python experiments/run_experiment_b_controlled.py
exit $LASTEXITCODE
