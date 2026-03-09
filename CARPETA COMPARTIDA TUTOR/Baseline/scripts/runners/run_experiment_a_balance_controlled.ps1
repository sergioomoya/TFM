# =============================================================================
# Experimento A variantes de balance: ejecución controlada con entorno tfm
# - Activa conda tfm (obligatorio)
# - Lanza monitor de recursos en background
# - Ejecuta experimento con log y monitoreo integrado
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

# Crear directorio de resultados si no existe
$null = New-Item -Path "experiments\results" -ItemType Directory -Force

# Iniciar monitor en background (cada 10s, hasta 60 min)
$monitorJob = Start-Job -ScriptBlock {
    param($wd)
    Set-Location $wd
    & powershell -File ".\scripts\monitoring\monitor_recursos_a_balance.ps1" -Interval 10 -DurationMin 60
} -ArgumentList (Get-Location).Path

Write-Host "Monitor de recursos iniciado (Job ID: $($monitorJob.Id))" -ForegroundColor Cyan
Write-Host "Ejecutando Experimento A variantes de balance con entorno tfm..." -ForegroundColor Cyan
Write-Host ""

# Ejecutar experimento
& $condaExe run -n tfm python experiments/run_experiment_a_balance_controlled.py
$exitCode = $LASTEXITCODE

# Detener monitor
Stop-Job -Job $monitorJob -ErrorAction SilentlyContinue
Remove-Job -Job $monitorJob -Force -ErrorAction SilentlyContinue

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "Experimento completado correctamente." -ForegroundColor Green
    Write-Host "  Resultados: experiments\results\experiment_a_balance_variants_comparison.csv"
    Write-Host "  Log: experiments\results\experiment_a_balance_run_*.log"
} else {
    Write-Host "Experimento finalizado con codigo: $exitCode" -ForegroundColor Yellow
}
exit $exitCode
