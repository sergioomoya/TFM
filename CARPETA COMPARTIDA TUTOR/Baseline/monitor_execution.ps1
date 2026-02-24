# Monitor automático de ejecución Ch7
# Uso: .\monitor_execution.ps1 [-Interval 60] [-TerminalFile "ruta"] [-MaxMinutes 120]
# Lee execution_progress.txt y la salida del terminal cada Interval segundos.

param(
    [int]$Interval = 60,
    [string]$TerminalFile = "",
    [int]$MaxMinutes = 120
)

$startTime = Get-Date
$projectRoot = $PSScriptRoot
$progressFile = Join-Path $projectRoot "execution_progress.txt"

# Encontrar el terminal más reciente con ch7-gpu
if (-not $TerminalFile) {
    $terminalsDir = "$env:USERPROFILE\.cursor\projects\c-Programacion-GitHub-TFM-CARPETA-COMPARTIDA-TUTOR-Baseline\terminals"
    $ch7Terminal = Get-ChildItem $terminalsDir -Filter "*.txt" -ErrorAction SilentlyContinue | 
        Where-Object { (Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue) -match "ch7-gpu" } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $TerminalFile = $ch7Terminal.FullName
}

if (-not (Test-Path $TerminalFile)) {
    Write-Host "No se encontró terminal de ch7-gpu. Usando solo execution_progress.txt" -ForegroundColor Yellow
    $TerminalFile = $null
}

Write-Host "=== Monitor de Ejecución Ch7 ===" -ForegroundColor Cyan
Write-Host "Intervalo: ${Interval}s | Máximo: ${MaxMinutes} min | Inicio: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
Write-Host ""

$lastProgress = ""
$lastTerminalLines = 0

while ($true) {
    $elapsed = ((Get-Date) - $startTime).TotalMinutes
    if ($elapsed -ge $MaxMinutes) {
        Write-Host "`n[TIMEOUT] Tiempo máximo de monitorización alcanzado ($MaxMinutes min)" -ForegroundColor Yellow
        break
    }

    $timestamp = Get-Date -Format "HH:mm:ss"
    
    # Progreso por celdas
    if (Test-Path $progressFile) {
        try {
            $progress = Get-Content $progressFile -Tail 3 -ErrorAction SilentlyContinue -Raw
            if ($progress -and $progress -ne $lastProgress) {
                $lastProgress = $progress
                Write-Host "[$timestamp] $progress" -ForegroundColor Green
            }
        } catch {}
    }

    # Salida del terminal
    if ($TerminalFile -and (Test-Path $TerminalFile)) {
        $content = Get-Content $TerminalFile -ErrorAction SilentlyContinue
        $lineCount = $content.Count
        if ($lineCount -gt $lastTerminalLines) {
            $newLines = $content | Select-Object -Skip $lastTerminalLines
            foreach ($line in $newLines) {
                if ($line -match "running_for_seconds|Celda|ERROR|Traceback|unified_report|484/484") {
                    Write-Host "[$timestamp] $line" -ForegroundColor $(if ($line -match "ERROR|Traceback") { "Red" } else { "Gray" })
                }
            }
            $lastTerminalLines = $lineCount
        }
        # ¿Terminó?
        $raw = Get-Content $TerminalFile -Raw -ErrorAction SilentlyContinue
        if ($raw -match "last_exit_code:\s*0" -or $raw -match "Executing: Chapter_7.*\n.*\n.*484") {
            Write-Host "`n[COMPLETADO] Ejecución finalizada correctamente" -ForegroundColor Green
            break
        }
        if ($raw -match "last_exit_code:\s*[1-9]") {
            Write-Host "`n[ERROR] La ejecución terminó con errores" -ForegroundColor Red
            Get-Content $TerminalFile -Tail 30
            break
        }
    }

    Start-Sleep -Seconds $Interval
}
