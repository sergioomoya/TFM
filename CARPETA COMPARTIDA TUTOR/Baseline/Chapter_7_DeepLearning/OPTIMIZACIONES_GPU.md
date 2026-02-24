# Optimizaciones GPU para Chapter 7

## Resumen de Optimizaciones Implementadas

### 1. Mixed Precision (FP16) - AMP
- **Archivo**: `shared_functions.py` - `training_loop()`, `evaluate_model()`, `per_sample_mse()`
- **Beneficio**: 2-3x velocidad en RTX 5080 (Tensor Cores)
- **Uso**: Automático cuando `use_amp=True` y hay GPU disponible

### 2. Gradient Accumulation
- **Archivo**: `shared_functions.py` - `training_loop()`
- **Beneficio**: Simula batches más grandes sin usar más VRAM
- **Parámetro**: `gradient_accumulation_steps` (default 1)
- **Ejemplo**: `gradient_accumulation_steps=2` duplica el batch efectivo

### 3. One-Cycle Learning Rate
- **Archivo**: `shared_functions.py` - `training_loop()`
- **Beneficio**: Convergencia más rápida, menos epochs necesarios
- **Parámetro**: `use_one_cycle_lr=True`
- **Comportamiento**: LR sube 10x durante 30% del entrenamiento, luego decae cosenoidal

### 4. DataLoader Optimizado
- **Archivo**: `shared_functions.py` - `get_dataloader_config()`
- **Cambios**:
  - `num_workers=2` (antes 0) - prefetching activo
  - `prefetch_factor=4` - mantiene GPU ocupada
  - Batch sizes mayores con AMP:
    - 4GB VRAM: 16384
    - 8GB VRAM: 32768
    - 16GB VRAM: 32768-49152

### 5. Parallel Training
- **Archivo**: `shared_functions.py` - `parallel_training_loop()`
- **Beneficio**: Entrena múltiples modelos pequeños simultáneamente
- **Uso**: Para modelos que individualmente no saturan GPU

### 6. Fast Training
- **Archivo**: `shared_functions.py` - `fast_training_loop()`
- **Beneficio**: Configuración agresiva para modelos pequeños
- **Características**:
  - AMP activado
  - Gradient accumulation (steps=2)
  - OneCycleLR
  - Menos epochs (default 50 vs 100)
  - Weight decay optimizado

## Uso Recomendado

### Para un solo modelo pequeño:
```python
# Opción 1: Usar training_loop con optimizaciones
model, time, train_losses, valid_losses = training_loop(
    model, train_gen, valid_gen, optimizer, criterion,
    max_epochs=50, use_amp=True, gradient_accumulation_steps=2, 
    use_one_cycle_lr=True
)

# Opción 2: Usar fast_training_loop (más simple)
model, time, train_losses, valid_losses = fast_training_loop(
    model, train_gen, valid_gen, criterion, max_epochs=50, lr=0.001
)
```

### Para múltiples modelos pequeños:
```python
# Entrenar en paralelo para saturar GPU
configs = [
    (model1, optimizer1, "Modelo_1"),
    (model2, optimizer2, "Modelo_2"),
    (model3, optimizer3, "Modelo_3"),
]
results = parallel_training_loop(configs, train_gen, valid_gen, criterion)
```

## Configuración RTX 5080 Optimizada

| Parámetro | Valor |
|-----------|-------|
| batch_size | 49152 (con AMP) |
| num_workers | 2 |
| pin_memory | True |
| prefetch_factor | 4 |
| gradient_accumulation_steps | 2 |
| use_amp | True |
| use_one_cycle_lr | True |

## Resultados Esperados

- **Velocidad de entrenamiento**: 2-4x más rápida
- **Uso de VRAM**: ~60-80% de 16GB (vs ~20% sin optimizaciones)
- **Uso de GPU**: 30-70% (vs ~5-10% sin optimizaciones)
- **Convergencia**: 30-50% menos epochs necesarios
