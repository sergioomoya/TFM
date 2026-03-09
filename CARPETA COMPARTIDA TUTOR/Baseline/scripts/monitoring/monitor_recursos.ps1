# Monitor de recursos durante ejecucion del Experimento B
# Uso: .\monitor_recursos.ps1 [intervalo_segundos] [duracion_minutos]
# Ejemplo: .\monitor_recursos.ps1 10 30  (cada 10s, durante 30 min)

param(
    [int]$Interval = 10,
    [int]$DurationMin = 60
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$endTime = (Get-Date).AddMinutes($DurationMin)
$logPath = "experiments\results\monitor_log.txt"
$progressPath = "experiments\results\experiment_b_progress.txt"
$null = New-Item -Path (Split-Path $logPath) -ItemType Directory -Force

"=== Monitor iniciado $(Get-Date) | Intervalo: ${Interval}s | Duracion: ${DurationMin} min ===" | Tee-Object -FilePath $logPath -Append

while ((Get-Date) -lt $endTime) {
    $ts = Get-Date -Format "HH:mm:ss"
    
    # GPU (utilizacion, VRAM)
    $gpu = nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader 2>$null
    $memUsed = 0
    if ($gpu) {
        $gpuLine = "[$ts] GPU: $gpu"
        $gpuParts = $gpu -split ","
        if ($gpuParts.Count -ge 3) { $memUsed = [int]($gpuParts[2] -replace "MiB","").Trim() }
        if ($memUsed -gt 500) { $gpuStatus = "GPU EN USO (VRAM: ${memUsed} MiB)" } else { $gpuStatus = "GPU idle o espera" }
    } else {
        $gpuLine = "[$ts] GPU: nvidia-smi no disponible"
        $gpuStatus = "N/A"
    }
    
    # RAM (GB)
    $os = Get-CimInstance Win32_OperatingSystem
    $ramFree = [math]::Round($os.FreePhysicalMemory/1MB, 1)
    $ramTotal = [math]::Round($os.TotalVisibleMemorySize/1MB, 1)
    $ramLine = "[$ts] RAM: ${ramFree} GB libres / ${ramTotal} GB total"
    
    # Progreso
    $progress = ""
    if (Test-Path $progressPath) {
        $progress = " | " + (Get-Content $progressPath -Raw).Trim()
    }
    if (Test-Path "experiments\results\experiment_b_variants_comparison.csv") {
        $progress = " | EXPERIMENTO COMPLETADO"
    }
    
    # Procesos Python
    $py = Get-Process python* -ErrorAction SilentlyContinue
    $pyCount = $py.Count
    $procLine = "[$ts] Python: $pyCount proc"
    
    $line = "$gpuLine | $ramLine | $procLine$progress"
    Write-Host $line
    Write-Host "  -> $gpuStatus" -ForegroundColor $(if ($memUsed -gt 500) { "Green" } else { "Gray" })
    $line | Out-File -FilePath $logPath -Append
    
    Start-Sleep -Seconds $Interval
}
"=== Monitor finalizado $(Get-Date) ===" | Tee-Object -FilePath $logPath -Append
