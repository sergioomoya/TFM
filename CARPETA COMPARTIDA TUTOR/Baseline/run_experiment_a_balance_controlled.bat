@echo off
REM Experimento A variantes de balance: ejecucion controlada con entorno tfm

set CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe
if not exist "%CONDA_EXE%" set CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe
if not exist "%CONDA_EXE%" (
    echo Error: No se encuentra conda. Esperado en %%USERPROFILE%%\anaconda3\Scripts\conda.exe
    exit /b 1
)

cd /d "%~dp0"
echo Ejecutando Experimento A variantes de balance con entorno tfm...
"%CONDA_EXE%" run -n tfm python experiments/run_experiment_a_balance_controlled.py
exit /b %ERRORLEVEL%
