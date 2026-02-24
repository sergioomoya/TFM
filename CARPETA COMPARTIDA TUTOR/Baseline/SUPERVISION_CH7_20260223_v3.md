# Supervisión Ch7 GPU – 2026-02-23 23:42 (Ejecución Supervisada EMRR)

## Precondiciones cumplidas

| Paso | Acción | Estado |
|------|--------|--------|
| 1 | Detener contenedor anterior | ✓ (ninguno en ejecución) |
| 2 | Eliminar execution.lock | ✓ |
| 3 | Iniciar ejecución nueva | ✓ |
| 4 | Versión actualizada | ✓ (contenedor cargado desde disco) |

---

## Comprobación #1 (23:42 – ~25 s transcurridos)

| Métrica | Valor |
|---------|-------|
| Contenedor | `baseline-ch7-gpu-run-663dce4309a5` |
| Progreso | **Celda 122/484** (~25%) |
| Log | `ch7_supervision_20260223_234214.log` |

### Recursos

| Recurso | Valor |
|---------|-------|
| GPU util | 10 % |
| VRAM | 1.450 MiB / 16.303 MiB (~9 %) |
| Temp GPU | 34 °C |
| RAM host | 12,05 GB libres / 31,9 GB total |

**Estado:** EN EJECUCIÓN – Sin errores. Margen amplio en GPU (posibilidad de aumentar batch_size si se desea).

---

## Comprobación #2 (~1 min transcurrido)

| Métrica | Valor |
|---------|-------|
| Progreso | **Celda 136/484** (~28%) |
| Tiempo transcurrido | ~1 min 20 s |

### Recursos

| Recurso | Valor |
|---------|-------|
| GPU util | **62 %** |
| VRAM | 2.469 MiB / 16.303 MiB (~15 %) |
| Temp GPU | 35 °C |

**Resumen:** Entrenamiento activo (GPU 62%). Progreso sostenido (~14 celdas/min en esta fase). Sin alertas.

---

## Comprobación #3 (~33 min transcurridos)

| Métrica | Valor |
|---------|-------|
| Progreso | **Celda 207/484** (~43%) |
| Tiempo transcurrido | ~33 min |
| Velocidad media | ~2,6 celdas/min |

### Recursos

| Recurso | Valor |
|---------|-------|
| GPU util | 4 % (celda ligera / entre entrenamientos) |
| VRAM | 3.879 MiB / 16.303 MiB (~24 %) |
| Temp GPU | 53 °C |
| RAM host | 14,61 GB libres / 31,9 GB total |

**Resumen:** EN EJECUCIÓN. Progreso correcto. GPU en fase de bajo uso (posible celda de evaluación o preprocesado).

---

## Comprobación #4 (~35 min transcurridos)

| Métrica | Valor |
|---------|-------|
| Progreso | **Celda 207/484** (~43%) – *sin cambio* |
| Nota | Celda 207 es de larga duración (entrenamiento con épocas) |
| Tiempo transcurrido | ~35 min |

### Recursos

| Recurso | Valor |
|---------|-------|
| GPU util | 19 % |
| VRAM | 3.881 MiB / 16.303 MiB (~24 %) |
| Temp GPU | 51 °C |

**Resumen:** Ejecutando celda 207 (bucle de entrenamiento). `execution_progress.txt` solo se actualiza al cambiar de celda.

---

## Comprobación #5 (~37 min – Ciclo EMRR)

| Métrica | Valor |
|---------|-------|
| Tiempo transcurrido | ~37 min |
| Progreso | **Celda 207/484** (~43%) |
| Etapa actual | Bucle de entrenamiento (celda larga) |
| Errores | Ninguno |

### Recursos (23:19 UTC)

| Recurso | Valor |
|---------|-------|
| GPU util | 17 % |
| VRAM | 3.902 MiB / 16.303 MiB (~24 %) |
| Temp GPU | 50 °C |
| RAM host | 14,01 GB libres / 31,9 GB total |

### Criterio aceleración

Margen en GPU (util &lt; 50 %, VRAM &lt; 50 %). En futuras ejecuciones podría valorarse `batch_size` mayor; en la actual no se modifica para no alterar la ejecución en curso.

**Resumen:** EN EJECUCIÓN. Sin errores. Celda 207 en curso (entrenamiento FFNN/LSTM).

---

## Comprobación #6 (~38 min)

| Métrica | Valor |
|---------|-------|
| Tiempo | ~38 min |
| Progreso | Celda 207/484 (~43%) |
| GPU | 6 % \| 3.999 MiB VRAM \| 49 °C |
| Errores | Ninguno |

**Resumen:** EN EJECUCIÓN. Celda 207 en curso.

---

## Comprobación #7 (~38 min)

| Progreso | Celda 207/484 | GPU 10% | VRAM 5.295 MiB (~32%) | Temp 51°C |
|----------|---------------|---------|------------------------|-----------|
| Errores  | Ninguno       |         |                        |           |

---

## Comprobación #8 (~40 min)

| Progreso | Celda 207/484 | GPU 9% | VRAM 5.349 MiB (~33%) |
|----------|---------------|--------|----------------------|
| Errores  | Ninguno       |        |                      |

---

## Comprobación #9 (~43 min)

| Progreso | Celda 207/484 | GPU 7% | VRAM 5.276 MiB |
|----------|---------------|--------|----------------|
| Errores  | Ninguno       |        |                |

---

## Comprobación #10 (~48 min)

| Progreso | Celda 207/484 | GPU 7% | VRAM 5.299 MiB | Temp 55°C |
|----------|---------------|--------|----------------|-----------|
| Errores  | Ninguno       |        |                |           |

**Resumen:** Celda 207 sigue en curso (bucle entrenamiento largo). Sin errores.

---

## Comprobación #11 (~53 min)

| Progreso | Celda 207/484 | GPU 9% | VRAM 5.262 MiB |
|----------|---------------|--------|----------------|
| Contenedor | Activo       |        |                |

---

## Comprobación #12 (~54 min)

| Progreso | Celda 207/484 | GPU 4% | VRAM 4.041 MiB |
|----------|---------------|--------|----------------|
| Nota | VRAM bajó (posible fase validación o cambio de modelo) |

---

## Comprobación #13 (~59 min)

| Progreso | Celda 207/484 | GPU 4% | VRAM 4.103 MiB | Temp 50°C |
|----------|---------------|--------|----------------|----------|
| Errores  | Ninguno       |        |                |          |

---

## Comprobación #14 (~64 min)

| Progreso | Celda 207/484 | GPU 9% | VRAM 5.233 MiB |
|----------|---------------|--------|----------------|
| Errores  | Ninguno       |        |                |

---

## Comprobación #15 – ERROR DETECTADO (~67 min)

### Fase 3 EMRR: Resolver errores

| Campo | Valor |
|-------|-------|
| Tiempo hasta fallo | 4.009 s (~67 min) |
| Contenedor | Finalizado (--rm, eliminado) |
| Celda fallida | Carga de `performances_model_selection.pkl` y `performances_model_selection_nn.pkl` |

### Error

```
TypeError: Argument 'placement' has incorrect type (expected pandas._libs.internals.BlockPlacement, got numpy.ndarray)
```

**Archivo:** `pandas/core/internals/blocks.py` en `new_block`  
**Causa raíz:** Incompatibilidad de pickle. Los `.pkl` se crearon con otra versión de pandas; la estructura interna de DataFrame (BlockPlacement) cambió.  
**Archivos afectados:**
- `Chapter_5_ModelValidationAndSelection/performances_model_selection.pkl`
- `Chapter_7_DeepLearning/performances_model_selection_nn.pkl`

### Acciones aplicadas (Fase 4: Reiniciar)

1. **Dockerfile.gpu:** `pandas==1.3.5` para compatibilidad con pickles.
2. **Notebook:** helper `_load_performances_pickle()` con fallback a `pd.read_pickle` ante TypeError BlockPlacement.
3. **Celdas de carga:** sustituido `pickle.load` por `_load_performances_pickle`.
4. **Prerrequisito:** ejecutar Ch5 para generar `performances_model_selection.pkl` antes de Ch7.

---

## Fase 4 – Reinicio (2026-02-24)

| Acción | Estado |
|--------|--------|
| Ch5 | Detenido (pickle ya existía) |
| execution.lock | Eliminado |
| Ch7 supervisado | **EN EJECUCIÓN** |

---

## Comprobación #16 (00:35 – Reinicio EMRR)

| Métrica | Valor |
|---------|-------|
| Contenedor | `baseline-ch7-gpu-run-5405bd89ac27` |
| Progreso | **Celda 136/484** (~28%) |
| Log | `ch7_supervision_20260224_013413.log` |
| GPU | 1 % \| 2.662 MiB / 16.303 MiB \| 37 °C |
| Errores | Ninguno |

**Estado:** EN EJECUCIÓN. Sin errores de pickle.

---

## Comprobación #17 (~3 min)

| Progreso | Celda 151/484 (~31%) | GPU 28% | VRAM 3.544 MiB | Temp 38°C |
|----------|----------------------|---------|----------------|-----------|
| Velocidad | ~7 celdas/min | Errores: Ninguno |

---

## Comprobación #18 (~5 min)

| Progreso | Celda 151/484 | GPU 2% | VRAM 3.552 MiB |
|----------|---------------|--------|----------------|
| Nota | Celda larga (entrenamiento) en curso |

---

## Monitorización continua activa

Script: `monitor_ch7_continuo.ps1` — ejecutándose en segundo plano. Escribe en `monitor_ch7_continuo_*.log` cada 60 s.

Para lanzarlo manualmente: `.\monitor_ch7_continuo.ps1`
