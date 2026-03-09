# Monitor de recursos durante ejecucion del Experimento A variantes de balance
# Uso: .\monitor_recursos_a_balance.ps1 [intervalo_segundos] [duracion_minutos]
# Ejemplo: .\monitor_recursos_a_balance.ps1 10 15  (cada 10s, durante 15 min)

param(
    [int]$Interval = 10,
    [int]$DurationMin = 30
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$endTime = (Get-Date).AddMinutes($DurationMin)
$logPath = "experiments\results\monitor_a_balance_log.txt"
$progressPath = "experiments\results\experiment_a_balance_progress.txt"
$null = New-Item -Path (Split-Path $logPath) -ItemType Directory -Force

"=== Monitor Experimento A Balance | Iniciado $(Get-Date) | Intervalo: ${Interval}s | Duracion: ${DurationMin} min ===" | Tee-Object -FilePath $logPath -Append

while ((Get-Date) -lt $endTime) {
    $ts = Get-Date -Format "HH:mm:ss"
    
    # GPU
    $gpu = nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader 2>$null
    $memUsed = 0
    if ($gpu) {
        $gpuLine = "[$ts] GPU: $gpu"
        $gpuParts = $gpu -split ","
        if ($gpuParts.Count -ge 3) { $memUsed = [int]($gpuParts[2] -replace "MiB","").Trim() }
        if ($memUsed -gt 500) { $gpuStatus = "GPU EN USO (VRAM: ${memUsed} MiB)" } else { $gpuStatus = "GPU idle" }
    } else {
        $gpuLine = "[$ts] GPU: nvidia-smi no disponible"
        $gpuStatus = "N/A"
    }
    
    # RAM
    $os = Get-CimInstance Win32_OperatingSystem
    $ramFree = [math]::Round($os.FreePhysicalMemory/1MB, 1)
    $ramTotal = [math]::Round($os.TotalVisibleMemorySize/1MB, 1)
    $ramLine = "[$ts] RAM: ${ramFree} GB libres / ${ramTotal} GB total"
    
    # Progreso
    $progress = ""
    if (Test-Path $progressPath) {
        $progress = " | " + (Get-Content $progressPath -Raw -ErrorAction SilentlyContinue).Trim()
    }
    if (Test-Path "experiments\results\experiment_a_balance_variants_comparison.csv") {
        $progress = " | EXPERIMENTO COMPLETADO"
    }
    
    $procLine = "[$ts] Python: $((Get-Process python* -ErrorAction SilentlyContinue).Count) proc"
    
    $line = "$gpuLine | $ramLine | $procLine$progress"
    Write-Host $line
    Write-Host "  -> $gpuStatus" -ForegroundColor $(if ($memUsed -gt 500) { "Green" } else { "Gray" })
    $line | Out-File -FilePath $logPath -Append
    
    Start-Sleep -Seconds $Interval
}
"=== Monitor finalizado $(Get-Date) ===" | Tee-Object -FilePath $logPath -Append
