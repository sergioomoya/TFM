# Ejecuta Ch7 tras Ch5. Requiere que performances_model_selection.pkl exista (de Ch5).
# Uso: .\run_ch7_post_ch5.ps1

$ErrorActionPreference = "Continue"
$ch5Pkl = "Chapter_5_ModelValidationAndSelection\performances_model_selection.pkl"

if (-not (Test-Path $ch5Pkl)) {
    Write-Host "ERROR: Ejecuta Ch5 primero. Falta: $ch5Pkl" -ForegroundColor Red
    Write-Host "Ejecutar: docker compose run --rm unified-notebooks python run_unified_notebooks.py --notebook Chapter_5_ModelValidationAndSelection/Chapter_5_Unified.ipynb --timeout 0" -ForegroundColor Yellow
    exit 1
}

Write-Host "performances_model_selection.pkl encontrado. Iniciando Ch7 GPU..." -ForegroundColor Green
docker ps -q --filter "name=ch7-gpu" 2>$null | ForEach-Object { docker stop $_ 2>$null }
if (Test-Path "execution.lock") { Remove-Item "execution.lock" -Force }

$logPath = "ch7_supervision_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
docker compose run --rm ch7-gpu 2>&1 | Tee-Object -FilePath $logPath
exit $LASTEXITCODE
