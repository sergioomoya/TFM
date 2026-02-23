@echo off
REM Ejecutar Chapter 7 SIN Cursor - evita OOM
REM Doble-clic o: run_ch7_externo.bat
echo Deteniendo contenedores previos...
docker compose down 2>nul
docker ps -q --filter "name=ch7" 2>nul | for /f %i in ('docker ps -q --filter "name=ch7"') do docker stop %i 2>nul

echo.
echo Ejecutando Chapter 7...
docker compose run --rm ch7-gpu
pause
