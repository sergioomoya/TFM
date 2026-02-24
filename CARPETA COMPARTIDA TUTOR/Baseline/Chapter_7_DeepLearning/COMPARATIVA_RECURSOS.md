# Comparativa: estado original vs actual (uso de recursos GPU)

## Resumen ejecutivo

El capítulo original del libro y las primeras modificaciones impedían aprovechar correctamente la GPU (VRAM baja, ~3 GB de 16 GB). Tras diagnosticar la causa raíz —`.to(DEVICE)` dentro de `Dataset.__getitem__`— se aplicó el patrón estándar de PyTorch: Datasets en CPU y traslado al dispositivo en el bucle de entrenamiento.

---

## Estado original (libro)

| Parámetro       | Valor  | Comentario                    |
|-----------------|--------|-------------------------------|
| batch_size      | 64     | Fijo en prepare_generators    |
| num_workers     | 0      | Sin prefetch paralelo         |
| pin_memory      | (no)   | Implícito False               |
| GridSearch      | [64,128,256] | Búsqueda en batch pequeños |

## Cambios aplicados (mejor uso de recursos)

| Parámetro       | Original | Actual | Comentario |
|-----------------|----------|--------|------------|
| batch_size      | 64       | 512–8192 según VRAM | `get_dataloader_config()` |
| num_workers     | 0        | 0–8 según CPUs | Solo tras corregir Datasets |
| pin_memory      | False    | True si GPU | Acelera CPU→GPU |
| GridSearch batch| [64,128,256] | [batch dinámico] | Un solo valor por recursos |

## Causa principal que impedía mejor uso: `.to(DEVICE)` en `Dataset.__getitem__`

El notebook definía Datasets que movían los datos a GPU **dentro** de `__getitem__`:

```python
def __getitem__(self, index):
    return self.x[index].to(DEVICE), self.y[index].to(DEVICE)  # ❌ Problema
```

### Por qué bloqueaba un mejor uso de recursos

1. **num_workers > 0**  
   Los workers del DataLoader se crean con `fork`. Cada worker hereda el proceso y, al acceder a CUDA (por `.to(DEVICE)`), intenta inicializar CUDA en un subproceso clonado → error *"Cannot re-initialize CUDA in forked subprocess"*.

2. **pin_memory = True**  
   Solo funciona con tensores en CPU. Si `__getitem__` devuelve tensores en GPU, el DataLoader intenta hacer pin de tensores ya en GPU → error *"cannot pin 'torch.cuda.FloatTensor'"*.

### Patrón recomendado en PyTorch

- **Dataset** devuelve tensores en **CPU**.
- **DataLoader** usa `num_workers` y `pin_memory=True`.
- **Bucle de entrenamiento** mueve cada batch: `x_batch, y_batch = x_batch.to(device), y_batch.to(device)`.

---

## Correcciones realizadas

### 1. shared_functions.py
- `FraudDataset.__getitem__`: devuelve `self.x[index], self.y[index]` (sin `.to(DEVICE)`).
- `get_dataloader_config()`: batch_size según VRAM; num_workers y pin_memory activos cuando hay GPU.
- `evaluate_model`, `training_loop`, `per_sample_mse`: añadido `device = next(model.parameters()).device` y `x_batch, y_batch = x_batch.to(device), y_batch.to(device)`.

### 2. Chapter_7_Unified.ipynb
- **Celda 135**: bucle de entrenamiento inline sin `.to(device)` → añadido.
- **Celda 150**: `training_loop` sin `.to(device)` → añadido `device` y `.to(device)`.
- **Celda 246**: `FraudDatasetUnsupervised.__getitem__` con `.to(DEVICE)` → eliminado.
- **Celda 255**: `per_sample_mse` sin `.to(device)` → añadido `device` y `.to(device)`.
- **Celda 280**: bucle `compute_representation` sin `.to(device)` → añadido.

### 3. FraudDataset y FraudSequenceDataset
- `FraudDataset`: ya devolvía tensores en CPU (sin `.to(DEVICE)` en `__getitem__`).
- `FraudSequenceDataset`: revisado; no usaba `.to(DEVICE)` en `__getitem__`.
- `FraudDatasetUnsupervised`: corregido en celda 246.

---

## Estado final esperado

Con estas correcciones:
- `num_workers > 0` y `pin_memory=True` pueden usarse sin errores de CUDA fork ni de pin.
- El batch_size se escala automáticamente según VRAM (hasta 8192 en GPUs de 16 GB).
- El entrenamiento debería usar más VRAM y completarse más rápido.
