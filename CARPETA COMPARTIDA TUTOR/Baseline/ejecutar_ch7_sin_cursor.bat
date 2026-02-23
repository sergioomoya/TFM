@echo off
chcp 65001 >nul
REM =====================================================
REM EJECUCIÓN DE CHAPTER 7 - SIN CURSOR (evita OOM)
REM =====================================================
REM Este script ejecuta el notebook completamente fuera de
REM Cursor para evitar crashes por consumo de RAM.
REM 
REM INSTRUCCIONES:
REM 1. Cierra Cursor completamente
REM 2. Guarda este archivo
REM 3. Doble clic o ejecuta desde CMD/PowerShell
REM =====================================================

echo.
echo =========================================
echo  EJECUCION CHAPTER 7 - Deep Learning
echo  Modo: Sin Cursor (evita OOM)
echo =========================================
echo.

REM Verificar Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker no esta instalado o no esta en PATH
    pause
    exit /b 1
)

REM Detener cualquier ejecución anterior
echo [1/4] Deteniendo contenedores previos...
docker compose -f "%~dp0docker-compose.yml" down --remove-orphans 2>nul
docker ps -q --filter "name=ch7-gpu" 2>nul | for /f %%i in ('docker ps -q --filter "name=ch7-gpu"') do docker stop %%i 2>nul
timeout /t 2 /nobreak >nul

REM Verificar que no hay bloqueos
echo [2/4] Verificando bloqueos...
if exist "%~dp0execution.lock" (
    echo   Eliminando bloqueo anterior...
    del /f "%~dp0execution.lock" 2>nul
)

REM Limpiar progreso anterior para nueva ejecución
echo [3/4] Preparando nueva ejecución...
if exist "%~dp0execution_progress.txt" (
    echo Celda 0/484 - Inicio | "%~dp0execution_progress.txt"
)

REM Ejecutar
echo [4/4] Ejecutando Chapter 7...
echo   Esto puede tardar 30-60 minutos dependiendo del hardware
echo   No cierres esta ventana hasta que termine
echo.
echo   Progreso en: execution_progress.txt
echo   Salida en: execution_output.txt
echo.

cd /d "%~dp0"
docker compose run --rm ch7-gpu 2>&1 | tee -a execution_output.txt

REM Resultado
if %ERRORLEVEL% equ 0 (
    echo.
    echo =========================================
    echo  EJECUCION COMPLETADA EXITOSAMENTE
echo =========================================
) else (
    echo.
    echo =========================================
    echo  EJECUCION FALLIDA - Revisa errores
    echo  execution_progress.txt para detalles
    echo =========================================
)

pause
