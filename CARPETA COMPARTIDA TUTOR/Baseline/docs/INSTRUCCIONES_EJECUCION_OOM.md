# Evitar OOM al ejecutar Chapter 7

Si Cursor se cierra por falta de memoria al ejecutar el notebook:

## Solución recomendada: Ejecutar SIN Cursor

1. **Cierra Cursor por completo**
2. Abre **PowerShell** o **CMD** en la carpeta del proyecto
3. Ejecuta:
   ```
   .\scripts\runners\run_ch7_externo.ps1
   ```
   (o doble-clic en `scripts\runners\run_ch7_externo.bat`)
4. La salida se guarda en `execution_output.txt`
5. Progreso en `execution_progress.txt`
6. Cuando termine (o falle), abre Cursor para revisar y corregir

## Por qué ayuda

- Cursor + Docker + entrenamiento = mucho uso de RAM
- Ejecutando fuera de Cursor, el pico de RAM ocurre sin el IDE abierto
- El contenedor tiene límite de 8GB; el resto queda para el sistema

## Si quieres ejecutar con Cursor abierto

1. Configura límite WSL: `C:\Users\<tu_usuario>\.wslconfig`
   ```
   [wsl2]
   memory=8GB
   swap=2GB
   ```
2. Después: `wsl --shutdown`
3. El `.cursorignore` ya excluye el notebook grande de la indexación
