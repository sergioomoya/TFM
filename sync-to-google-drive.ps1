#Requires -Version 5.1
<#
.SYNOPSIS
    Sincroniza el TFM: primero sube el progreso a GitHub, luego copia a Google Drive.

.DESCRIPTION
    Rutina de sincronización que:
    1. Guarda el progreso en GitHub (add, commit, push)
    2. Sincroniza el directorio maestro con Google Drive (robocopy /MIR)

.EXAMPLE
    .\sync-to-google-drive.ps1
#>

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
    $mensajeCommit = "Sync: progreso guardado $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    git commit -m $mensajeCommit
    if ($LASTEXITCODE -eq 0) {
        git push
        Write-Host "Progreso guardado en GitHub correctamente." -ForegroundColor Green
    }
}

Write-Host "`n=== Paso 2: Sincronizar con Google Drive ===" -ForegroundColor Cyan
if (-not (Test-Path $Destino)) {
    Write-Error "El directorio destino no existe: $Destino"
}
robocopy $Origen $Destino /MIR /R:3 /W:5 /NP /NDL /NFL

$exitCode = $LASTEXITCODE
if ($exitCode -ge 8) {
    Write-Host "`nATENCION: Robocopy reportó errores (código $exitCode)" -ForegroundColor Red
    exit $exitCode
}
Write-Host "`nSincronización completada." -ForegroundColor Green
