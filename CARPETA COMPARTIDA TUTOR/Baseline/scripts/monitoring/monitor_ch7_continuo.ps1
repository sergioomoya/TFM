# Monitorización continua Chapter 7 - Ciclo EMRR
# Uso: .\monitor_ch7_continuo.ps1
# Detiene con Ctrl+C

$intervalSec = 60  # comprobación cada 60 segundos
$logFile = "monitor_ch7_continuo_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -ErrorAction SilentlyContinue
}

# Obtener contenedor activo
$container = docker ps -q -f "name=ch7"
if (-not $container) {
    Write-Log "ERROR: No hay contenedor ch7 en ejecución"
    exit 1
}

$containerName = (docker ps --filter "name=ch7" --format "{{.Names}}").Trim()
Write-Log "Monitorizando: $containerName (cada ${intervalSec}s). Log: $logFile"
Write-Log "Ctrl+C para detener"
Write-Log "----------------------------------------"

$startTime = Get-Date

while ($true) {
    $elapsed = (Get-Date) - $startTime
    $elapsedStr = "{0}h {1}m" -f [int]$elapsed.TotalHours, [int]($elapsed.TotalMinutes % 60)

    $progress = docker exec $containerName cat /app/execution_progress.txt 2>$null
    $gpu = docker exec $containerName nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>$null

    if ($progress) {
        $pct = if ($progress -match "Celda (\d+)/(\d+)") {
            [math]::Round(([int]$Matches[1] / [int]$Matches[2]) * 100, 1)
        } else { "?" }
        Write-Log "Progreso: $($progress.Trim()) | $pct% | Transcurrido: $elapsedStr"
    } else {
        Write-Log "Sin progreso (contenedor puede haber terminado)"
    }

    if ($gpu) {
        $parts = $gpu -split ',' | ForEach-Object { $_.Trim() }
        Write-Log "  GPU: $($parts[0]) | VRAM: $($parts[1])/$($parts[2]) | Temp: $($parts[3])"
    }

    # Verificar si terminó
    $running = docker ps -q -f "name=ch7"
    if (-not $running) {
        Write-Log "Contenedor detenido. Monitorización finalizada."
        break
    }

    Write-Log "---"
    Start-Sleep -Seconds $intervalSec
}
