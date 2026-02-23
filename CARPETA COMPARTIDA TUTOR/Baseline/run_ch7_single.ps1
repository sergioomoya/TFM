# Ejecuta Chapter 7 con un ÚNICO contenedor.
# Detiene cualquier contenedor ch7-gpu en ejecución antes de iniciar.

$ErrorActionPreference = "Continue"
$containers = docker ps -q --filter "name=ch7-gpu" 2>$null
if ($containers) {
    Write-Host "Deteniendo contenedores ch7-gpu existentes..."
    $containers | ForEach-Object { docker stop $_ 2>$null }
}
if (Test-Path "execution.lock") {
    Remove-Item "execution.lock" -Force
}
docker compose run --rm ch7-gpu
exit $LASTEXITCODE
