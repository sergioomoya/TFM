# =============================================================================
# Capítulo 7: Ejecución controlada con logs y monitorización
# - Logs en ch7_execution_YYYYMMDD_HHMMSS.log
# - Progreso en execution_progress.txt
# - Métricas en ch7_monitor_*.log
# =============================================================================

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
Set-Location $projectRoot

$condaExe = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
if (-not (Test-Path $condaExe)) { $condaExe = "$env:USERPROFILE\miniconda3\Scripts\conda.exe" }
if (-not (Test-Path $condaExe)) {
    Write-Error "Conda no encontrado"
    exit 1
}

# Limpiar lock de ejecución previa
if (Test-Path "execution.lock") { Remove-Item "execution.lock" -Force }
if (Test-Path "execution_progress.txt") { Remove-Item "execution_progress.txt" -Force }

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "Chapter_7_DeepLearning\execution_logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$mainLog = "$logDir/ch7_execution_$timestamp.log"
$monitorLog = "$logDir/ch7_monitor_$timestamp.log"

Write-Host "Iniciando ejecución Ch7 - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "Log principal: $mainLog" -ForegroundColor Gray
Write-Host "Monitor: $monitorLog" -ForegroundColor Gray

# Lanzar monitor en segundo plano (cada 2 min)
$monitorLogPath = Join-Path $projectRoot $monitorLog
$progressPath = Join-Path $projectRoot "execution_progress.txt"
$monitorJob = Start-Job -ScriptBlock {
    param($monLog, $progPath, $intervalSec)
    $start = Get-Date
    for ($i = 0; $i -lt 600; $i++) {  # max ~200 horas
        try {
            $elapsed = ((Get-Date) - $start).TotalMinutes
            $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>$null
            $cpu = (Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue).CounterSamples.CookedValue
            $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
            $ramStr = if ($os) { "$([math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory)*1MB/1GB,1))/$([math]::Round($os.TotalVisibleMemorySize*1MB/1GB,1))GB" } else { "?" }
            $progress = if (Test-Path $progPath) { (Get-Content $progPath -Raw -ErrorAction SilentlyContinue).Trim().Replace("`n"," ") } else { "?" }
            $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | ${elapsed:F0}min | GPU:$gpu | CPU:${cpu}% | RAM:$ramStr | $progress"
            Add-Content -Path $monLog -Value $line -ErrorAction SilentlyContinue
        } catch {}
        Start-Sleep -Seconds $intervalSec
    }
} -ArgumentList $monitorLogPath, $progressPath, 120

# Ejecutar notebook (PYTHONUNBUFFERED para log en tiempo real)
$env:PYTHONUNBUFFERED = "1"
& $condaExe run -n tfm --no-capture-output python run_unified_notebooks.py --notebook Chapter_7_DeepLearning/Chapter_7_Unified.ipynb --timeout 0 2>&1 | Tee-Object -FilePath $mainLog

$exitCode = $LASTEXITCODE

Stop-Job $monitorJob -ErrorAction SilentlyContinue
Remove-Job $monitorJob -ErrorAction SilentlyContinue

Write-Host "`nEjecución finalizada - Exit: $exitCode - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor $(if ($exitCode -eq 0) { "Green" } else { "Red" })
Write-Host "Log: $mainLog | Monitor: $monitorLog" -ForegroundColor Gray
exit $exitCode
