@echo off
echo Iniciando monitorizacion de Chapter 7...
echo.
set "PROJECT_ROOT=%~dp0..\.."
powershell -ExecutionPolicy Bypass -File "%~dp0monitor_ch7.ps1" > "%PROJECT_ROOT%\Chapter_7_DeepLearning\execution_logs\monitor_output.log" 2>&1
echo Monitorizacion finalizada.
pause