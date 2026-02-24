# Supervisión Ch7 GPU – 2026-02-23 21:08 (reinicio con versión actualizada)

## Precondición cumplida: versión actualizada

| Paso | Acción | Estado |
|------|--------|--------|
| 1 | Detener contenedor anterior | ✓ |
| 2 | Eliminar execution.lock | ✓ |
| 3 | Iniciar ejecución nueva | ✓ |

**Lección aplicada:** Los cambios de batch_size y optimizaciones no tenían efecto porque el contenedor anterior cargó el código al arrancar y nunca recargó. Ahora se ejecuta la versión en disco (con todas las correcciones).

---

## Estado: EN EJECUCIÓN

| Métrica | Valor |
|---------|-------|
| Contenedor | `baseline-ch7-gpu-run-c067108322e8` (Up) |
| Progreso | **Celda 136/484** |
| Celda actual | Bucle de entrenamiento (150 épocas) |
| Inicio | 21:08:12 |

## Recursos (primera lectura)

| Recurso | Valor |
|---------|-------|
| VRAM | 2.960 MiB / 16.303 MiB (~18 %) |
| GPU util | 1 % |

---

## Comandos para seguir monitorizando

```powershell
docker exec (docker ps -q -f "name=ch7") cat /app/execution_progress.txt
nvidia-smi
```
