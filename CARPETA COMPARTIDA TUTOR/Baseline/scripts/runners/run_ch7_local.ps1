# =============================================================================
# Capítulo 7 Deep Learning: ejecución local con entorno conda tfm (sin Docker)
# Usa GPU directamente en Windows, evita errores NCCL de Docker
# =============================================================================

$condaExe = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
if (-not (Test-Path $condaExe)) {
    $condaExe = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
}
if (-not (Test-Path $condaExe)) {
    Write-Error "No se encuentra conda (Anaconda/Miniconda). Esperado en: $env:USERPROFILE\anaconda3\Scripts\conda.exe"
    exit 1
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
Set-Location $projectRoot

Write-Host "Ejecutando Capítulo 7 (Deep Learning) con entorno tfm..." -ForegroundColor Cyan
& $condaExe run -n tfm python run_unified_notebooks.py --notebook Chapter_7_DeepLearning/Chapter_7_Unified.ipynb --timeout 0
exit $LASTEXITCODE
