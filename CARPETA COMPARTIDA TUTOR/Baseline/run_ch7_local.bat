@echo off
REM =============================================================================
REM Capítulo 7 Deep Learning: ejecución local con entorno conda tfm (sin Docker)
REM Usa GPU directamente en Windows, evita errores NCCL de Docker
REM =============================================================================

set CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe
if not exist "%CONDA_EXE%" set CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe
if not exist "%CONDA_EXE%" (
    echo ERROR: No se encuentra conda (Anaconda/Miniconda).
    exit /b 1
)

cd /d "%~dp0"
echo Ejecutando Capítulo 7 (Deep Learning) con entorno tfm...
"%CONDA_EXE%" run -n tfm python run_unified_notebooks.py --notebook Chapter_7_DeepLearning/Chapter_7_Unified.ipynb --timeout 0
exit /b %ERRORLEVEL%
