@echo off
REM =============================================================================
REM Experimento B: ejecucion local con entorno conda tfm (sin Docker)
REM Usa GPU directamente en Windows, evita errores NCCL de Docker
REM =============================================================================

set CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe
if not exist "%CONDA_EXE%" (
    echo ERROR: No se encuentra conda. Ajusta CONDA_EXE si Anaconda esta en otra ruta.
    exit /b 1
)

cd /d "%~dp0..\.."
"%CONDA_EXE%" run -n tfm python experiments/run_experiment_b_standalone.py
exit /b %ERRORLEVEL%
