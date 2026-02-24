# Supervisión Ch7 GPU – 2026-02-23 22:20 (Actualizado tras reporte OOM)

## Estado: EN EJECUCIÓN

| Métrica | Valor |
|--------|-------|
| Contenedor | `baseline-ch7-gpu-run-67ab5a70d20f` (Up 7 min) |
| Progreso | **Celda 154/484** (~32% completado) |
| Inicio notebook | 22:09:37 |
| Última actualización progreso | 22:16:02 |

## Recursos del Sistema

| Recurso | Valor | Estado |
|---------|-------|--------|
| RAM Total | 31.9 GB | OK |
| RAM Usada | 18.9 GB (59.2%) | ⚠️ Moderado |
| RAM Libre | 13.01 GB | OK |
| Cursor PID 23940 | 1.79 GB WS | ⚠️ Principal consumidor |
| Contenedor Docker | 2.39 GB / 15.57 GB (15%) | OK |
| GPU Utilización | 3% | Baja (I/O bound) |
| VRAM Usada | 4.27 GB / 16.3 GB (26%) | OK |

## Análisis del OOM Reportado

- **Sistema RAM**: A 59% de uso, con 13 GB libres. **No está en estado crítico**.
- **Proceso Cursor**: Un proceso Cursor consume ~1.8 GB (PID 23940). Esto es normal para un IDE con múltiples notebooks abiertos.
- **No se detectaron errores OOM** en los logs del sistema ni en los archivos de log recientes.
- **Docker está funcionando correctamente** con uso de memoria contenido (15%).

### Posibles causas del OOM percibido:

1. **Cursor proceso renderer**: El proceso principal (PID 23940) con 1.8 GB puede haber alcanzado un límite interno de memoria heap de V8/Electron.
2. **Memoria virtual agotada**: Aunque haya RAM física libre, el proceso de Cursor puede haber alcanzado su límite de memoria virtual asignada.
3. **Extensiones de Cursor**: Algunas extensiones pueden tener fugas de memoria.

## Recomendaciones

### Opciones para liberar memoria (orden de prioridad):

1. **Reiniciar Cursor** (más efectivo):
   ```powershell
   # Cerrar Cursor completamente y reiniciar
   Get-Process Cursor | Stop-Process -Force
   # Luego reiniciar Cursor
   ```

2. **Cerrar pestañas de notebooks innecesarias** en Cursor para liberar memoria del proceso renderer.

3. **Configurar batch size más conservador** (si el problema persiste durante el entrenamiento):
   ```python
   # En shared_functions.py o el notebook
   BATCH_SIZE = 4096  # Reducir de 8192 si hay problemas
   ```

4. **Verificar extensiones activas** en Cursor que consuman memoria.

## Próximos pasos de monitorización

- Tiempo estimado restante: ~2-3 horas para completar las 484 celdas.
- Checkpoint actual: Celda 154 (bucle de entrenamiento).
- Próxima verificación recomendada: En 30 minutos o si se reporta otro error.

## Comandos útiles

```powershell
# Ver progreso actual
cat "C:\Programacion\GitHub\TFM\CARPETA COMPARTIDA TUTOR\Baseline\execution_progress.txt"

# Ver uso de recursos del contenedor
docker stats --no-stream baseline-ch7-gpu-run-67ab5a70d20f

# Ver logs recientes
docker logs baseline-ch7-gpu-run-67ab5a70d20f --tail 50

# Ver procesos Python consumiendo memoria
Get-Process python* | Sort-Object WorkingSet64 -Descending | Select-Object Name, Id, @{N="WS(MB)";E={[math]::Round($_.WorkingSet64/1MB,1)}}
```
