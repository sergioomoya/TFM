# Script de Monitorización Automática para Chapter 7
# Ciclo EMRR - Ejecución Supervisada

param(
    [string]$ContainerName = "baseline-ch7-gpu-run-67ab5a70d20f",
    [int]$IntervalMinutes = 5,
    [string]$LogFile = "/app/monitor_ch7.log"
)

$startTime = Get-Date
$totalCells = 484

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] $Message"
    Write-Host $logEntry

    # Escribir en archivo de log dentro del contenedor si es posible
    try {
        $dockerLog = $Message -replace '"', '\"'
        docker exec $ContainerName sh -c "echo '$logEntry' >> $LogFile" 2>$null
    } catch {}
}

function Get-Progress {
    try {
        $progress = docker exec $ContainerName cat /app/execution_progress.txt 2>$null
        if ($progress -match "Celda (\d+)/(\d+)") {
            return [int]$Matches[1]
        }
    } catch {}
    return $null
}

function Get-GPUStatus {
    try {
        $gpuInfo = docker exec $ContainerName nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>$null
        return $gpuInfo.Trim()
    } catch {
        return "N/A"
    }
}

function Test-Completion {
    try {
        $complete = docker exec $ContainerName test -f /app/execution_complete.txt 2>$null
        if ($LASTEXITCODE -eq 0) { return $true }
    } catch {}
    return $false
}

function Test-Error {
    try {
        $error = docker exec $ContainerName test -f /app/execution_error.txt 2>$null
        if ($LASTEXITCODE -eq 0) { return $true }
    } catch {}

    # Verificar si el contenedor sigue corriendo
    $containerStatus = docker ps --filter "name=$ContainerName" --format "{{.Status}}" 2>$null
    if (-not $containerStatus) { return $true }

    return $false
}

function Get-EstimatedTime {
    param([int]$CurrentCell)

    if ($CurrentCell -eq 0 -or $CurrentCell -ge $totalCells) { return "Calculando..." }

    $elapsed = (Get-Date) - $startTime
    $cellsRemaining = $totalCells - $CurrentCell
    $avgTimePerCell = $elapsed.TotalMinutes / $CurrentCell
    $estimatedRemaining = $avgTimePerCell * $cellsRemaining

    $hours = [math]::Floor($estimatedRemaining / 60)
    $minutes = [math]::Floor($estimatedRemaining % 60)

    return "${hours}h ${minutes}m"
}

# Inicio del monitoreo
Write-Log "========================================"
Write-Log "INICIANDO MONITORIZACIÓN CHAPTER 7"
Write-Log "Contenedor: $ContainerName"
Write-Log "Intervalo: $IntervalMinutes minutos"
Write-Log "Inicio: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Log "Total celdas estimadas: $totalCells"
Write-Log "========================================"

$iteration = 0
$completed = $false
$error = $false

while (-not $completed -and -not $error) {
    $iteration++
    $currentTime = Get-Date
    $elapsed = $currentTime - $startTime

    Write-Log ""
    Write-Log "--- Comprobación #$iteration ---"
    Write-Log "Tiempo transcurrido: $([math]::Floor($elapsed.TotalHours))h $([math]::Floor($elapsed.TotalMinutes % 60))m"

    # Obtener progreso
    $currentCell = Get-Progress
    if ($currentCell) {
        $percentage = [math]::Round(($currentCell / $totalCells) * 100, 1)
        $estimatedRemaining = Get-EstimatedTime -CurrentCell $currentCell
        Write-Log "📊 PROGRESO: Celda $currentCell / $totalCells ($percentage%)"
        Write-Log "⏱️  Tiempo restante estimado: $estimatedRemaining"
    } else {
        Write-Log "⚠️  No se pudo obtener progreso"
    }

    # Obtener estado GPU
    $gpuStatus = Get-GPUStatus
    if ($gpuStatus -ne "N/A") {
        $gpuParts = $gpuStatus -split ',' | ForEach-Object { $_.Trim() }
        Write-Log "🎮 GPU: Utilización: $($gpuParts[0]) | VRAM: $($gpuParts[1]) / $($gpuParts[2]) | Temp: $($gpuParts[3])°C"

        # Alertas si hay problemas
        $gpuUtil = [int]($gpuParts[0] -replace '%', '')
        $vramUsed = [int]($gpuParts[1] -replace ' MiB', '')
        $vramTotal = [int]($gpuParts[2] -replace ' MiB', '')
        $vramPercent = [math]::Round(($vramUsed / $vramTotal) * 100, 1)

        if ($gpuUtil -gt 95) {
            Write-Log "⚠️  ALERTA: GPU al máximo ($gpuUtil%) - posible cuello de botella"
        }
        if ($vramPercent -gt 90) {
            Write-Log "⚠️  ALERTA: VRAM casi llena ($vramPercent%) - riesgo de OOM"
        }
        if ($gpuUtil -lt 10 -and $currentCell -lt $totalCells) {
            Write-Log "⚠️  ALERTA: GPU inactiva durante entrenamiento - posible problema"
        }
    } else {
        Write-Log "⚠️  No se pudo obtener estado de GPU"
    }

    # Verificar finalización
    if (Test-Completion) {
        $completed = $true
        Write-Log ""
        Write-Log "✅ EJECUCIÓN COMPLETADA EXITOSAMENTE"
        break
    }

    # Verificar errores
    if (Test-Error) {
        $error = $true
        Write-Log ""
        Write-Log "❌ DETECTADO ERROR O CONTENEDOR DETENIDO"

        try {
            $errorContent = docker exec $ContainerName cat /app/execution_error.txt 2>$null
            if ($errorContent) {
                Write-Log "Detalle del error: $errorContent"
            }
        } catch {}

        # Obtener logs del contenedor
        Write-Log "Últimos logs del contenedor:"
        docker logs --tail 20 $ContainerName 2>&1 | ForEach-Object { Write-Log "  $_" }

        break
    }

    # Mostrar resumen
    if ($currentCell) {
        $remainingCells = $totalCells - $currentCell
        Write-Log "📈 Resumen: $remainingCells celdas restantes, ~$estimatedRemaining restantes"
    }

    Write-Log "--- Esperando $IntervalMinutes minutos para siguiente comprobación ---"

    # Esperar
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}

# Reporte final
$endTime = Get-Date
$totalDuration = $endTime - $startTime
$totalHours = [math]::Floor($totalDuration.TotalHours)
$totalMinutes = [math]::Floor($totalDuration.TotalMinutes % 60)

Write-Log ""
Write-Log "========================================"
Write-Log "MONITORIZACIÓN FINALIZADA"
Write-Log "========================================"
Write-Log "Inicio: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Log "Fin: $($endTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Log "Duración total: ${totalHours}h ${totalMinutes}m"
Write-Log "Estado: $(if ($completed) { '✅ COMPLETADO' } else { '❌ ERROR/INTERRUMPIDO' })"

# Listar archivos de resultados si existen
Write-Log ""
Write-Log "Archivos generados:"
try {
    $files = docker exec $ContainerName ls -lh /app/results/ 2>$null
    if ($files) {
        $files | ForEach-Object { Write-Log "  $_" }
    } else {
        Write-Log "  (No se encontraron archivos de resultados)"
    }
} catch {
    Write-Log "  (No se pudo listar archivos)"
}

Write-Log "========================================"
