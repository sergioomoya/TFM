$ErrorActionPreference = "Continue" # Changed to Continue to see all errors

Write-Host "Directorio actual: $(Get-Location)"
Write-Host "Listando archivos:"
Get-ChildItem

$composeFile = "C:\Programacion\GitHub\TFM\CARPETA COMPARTIDA TUTOR\Baseline\docker-compose.yml"

Write-Host "Iniciando construcción de imágenes Docker..."
docker-compose -f $composeFile build unified-notebooks ch7-gpu

Write-Host "Ejecutando Capítulo 3 (CPU)..."
docker-compose -f $composeFile run --rm unified-notebooks python run_unified_notebooks.py --notebook Chapter_3_GettingStarted/Chapter_3_Unified.ipynb

Write-Host "Ejecutando Capítulo 4 (CPU)..."
docker-compose -f $composeFile run --rm unified-notebooks python run_unified_notebooks.py --notebook Chapter_4_PerformanceMetrics/Chapter_4_Unified.ipynb

Write-Host "Ejecutando Capítulo 5 (CPU)..."
docker-compose -f $composeFile run --rm unified-notebooks python run_unified_notebooks.py --notebook Chapter_5_ModelValidationAndSelection/Chapter_5_Unified.ipynb

Write-Host "Ejecutando Capítulo 6 (CPU)..."
docker-compose -f $composeFile run --rm unified-notebooks python run_unified_notebooks.py --notebook Chapter_6_ImbalancedLearning/Chapter_6_Unified.ipynb

Write-Host "Ejecutando Capítulo 7 (GPU)..."
docker-compose -f $composeFile run --rm ch7-gpu python run_unified_notebooks.py --notebook Chapter_7_DeepLearning/Chapter_7_Unified.ipynb --timeout 0

Write-Host "Ejecución completada."