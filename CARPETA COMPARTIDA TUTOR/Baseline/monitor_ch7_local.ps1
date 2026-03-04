# Monitor Ch7 - Ejecución local (sin Docker)
# Uso: .\monitor_ch7_local.ps1   (ejecutar en otra terminal mientras corre el notebook)

$intervalSec = 30
$start = Get-Date
Write-Host "Monitor Ch7 - Ctrl+C para salir. Actualización cada ${intervalSec}s" -ForegroundColor Cyan

while ($true) {
    Clear-Host
    Write-Host "=== Monitor Ch7 - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" -ForegroundColor Cyan
    Write-Host "Tiempo transcurrido: $([math]::Round(((Get-Date) - $start).TotalMinutes, 0)) min" -ForegroundColor Gray
    Write-Host ""
    
    $progress = if (Test-Path "execution_progress.txt") { Get-Content "execution_progress.txt" -Raw } else { "No iniciado" }
    Write-Host "Progreso:" -ForegroundColor Yellow
    Write-Host $progress.Trim()
    Write-Host ""
    
    Write-Host "GPU:" -ForegroundColor Yellow
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>$null
    Write-Host ""
    
    Write-Host "RAM:" -ForegroundColor Yellow
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    if ($os) {
        $usedGB = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 1)
        $totalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
        Write-Host "${usedGB} / ${totalGB} GB"
    }
    Write-Host ""
    
    if (-not (Test-Path "execution.lock")) {
        Write-Host "Ejecución finalizada (no hay execution.lock)" -ForegroundColor Green
        break
    }
    
    Start-Sleep -Seconds $intervalSec
}
