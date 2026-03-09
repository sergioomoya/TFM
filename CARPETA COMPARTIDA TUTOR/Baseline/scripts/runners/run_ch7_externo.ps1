# Ejecutar Chapter 7 SIN abrir Cursor - evita OOM
# Uso: Abre PowerShell, cd al proyecto, ejecuta: .\scripts\runners\run_ch7_externo.ps1
# IMPORTANTE: Cierra Cursor antes de ejecutar para liberar RAM

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
Set-Location $projectRoot

Write-Host "Deteniendo contenedores ch7 previos..." -ForegroundColor Yellow
docker ps -q --filter "name=ch7-gpu" 2>$null | ForEach-Object { docker stop $_ }
docker compose -f docker-compose.yml down --remove-orphans 2>$null

Write-Host "`nEjecutando Chapter 7 (1 contenedor, ~8GB RAM)..." -ForegroundColor Cyan
$logPath = "execution_output.txt"
docker compose run --rm ch7-gpu 2>&1 | Tee-Object -FilePath $logPath
$exitCode = $LASTEXITCODE

Write-Host "`nSalida guardada en: $logPath" -ForegroundColor Green
if ($exitCode -ne 0) { Write-Host "ERROR: Codigo $exitCode - revisa execution_progress.txt y $logPath" -ForegroundColor Red }
exit $exitCode
