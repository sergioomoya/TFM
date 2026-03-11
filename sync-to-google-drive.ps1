#Requires -Version 5.1
<#
.SYNOPSIS
    Sincroniza el TFM: primero sube el progreso a GitHub, luego copia a Google Drive.

.DESCRIPTION
    Rutina de sincronización que:
    1. Guarda el progreso en GitHub (add, commit, push)
    2. Sincroniza el directorio maestro con Google Drive (robocopy /MIR)

.PARAMETER Mensaje
    Mensaje del commit. Si no se indica, pide introducirlo manualmente.

.EXAMPLE
    .\sync-to-google-drive.ps1
.EXAMPLE
    .\sync-to-google-drive.ps1 -Mensaje "experiments: informe A y figuras SHAP"
#>
param(
    [Parameter(Mandatory=$false)]
    [string]$Mensaje
)

$ErrorActionPreference = "Stop"
$Origen = "C:\Programacion\GitHub\TFM"
$Destino = "G:\Mi unidad\Yo\Estudios\Master Industriales VIU\TFM"

Write-Host "=== Paso 1: Guardar progreso en GitHub ===" -ForegroundColor Cyan
Set-Location $Origen

# Verificar estado del repositorio
$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "No hay cambios pendientes. El repositorio está limpio. Saltando commit/push." -ForegroundColor Yellow
} else {
    git add -A
    Write-Host "Archivos a commitear:" -ForegroundColor Gray
    git diff --cached --name-status | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    Write-Host ""
    if ([string]::IsNullOrWhiteSpace($Mensaje)) {
        $mensajeCommit = Read-Host "Introduce el mensaje del commit (Enter=cancelar)"
    } else {
        $mensajeCommit = $Mensaje
        Write-Host "Mensaje: $mensajeCommit" -ForegroundColor Gray
    }
    if ([string]::IsNullOrWhiteSpace($mensajeCommit)) {
        Write-Host "Commit cancelado. Ejecutando solo sincronización con Google Drive." -ForegroundColor Yellow
        git reset HEAD .
    } else {
        git commit -m $mensajeCommit
        if ($LASTEXITCODE -eq 0) {
            git push
            Write-Host "Progreso guardado en GitHub correctamente." -ForegroundColor Green
        }
    }
}

Write-Host "`n=== Paso 2: Sincronizar con Google Drive ===" -ForegroundColor Cyan
if (-not (Test-Path $Destino)) {
    Write-Error "El directorio destino no existe: $Destino"
}
# /XF excluye: desktop.ini (Windows/GDrive) y ~$* (temporales Office)
robocopy $Origen $Destino /MIR /R:3 /W:5 /NP /NDL /NFL /XF desktop.ini "~$*"

$exitCode = $LASTEXITCODE
if ($exitCode -ge 8) {
    Write-Host "`nATENCION: Robocopy reportó errores (código $exitCode)" -ForegroundColor Red
    exit $exitCode
}
Write-Host "`nSincronización completada." -ForegroundColor Green
