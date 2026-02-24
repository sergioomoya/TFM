@echo off
echo Iniciando monitorizacion de Chapter 7...
echo.
powershell -ExecutionPolicy Bypass -File "C:\Programacion\GitHub\TFM\CARPETA COMPARTIDA TUTOR\Baseline\monitor_ch7.ps1" > "C:\Programacion\GitHub\TFM\CARPETA COMPARTIDA TUTOR\Baseline\monitor_output.log" 2>&1
echo Monitorizacion finalizada.
pause