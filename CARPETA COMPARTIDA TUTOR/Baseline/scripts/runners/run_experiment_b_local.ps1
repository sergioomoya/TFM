# =============================================================================
# Experimento B: ejecucion local con entorno conda tfm (sin Docker)
# Usa GPU directamente en Windows, evita errores NCCL de Docker
# =============================================================================

$condaExe = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
if (-not (Test-Path $condaExe)) {
    Write-Error "No se encuentra conda en $condaExe. Ajusta la ruta si Anaconda esta en otra ubicacion."
    exit 1
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
Set-Location $projectRoot
& $condaExe run -n tfm python experiments/run_experiment_b_standalone.py
exit $LASTEXITCODE
