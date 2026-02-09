# Errores Encontrados en la Ejecución de Notebooks Unificados

## Fecha de Ejecución
2026-02-09 19:15:38

## Estado General
❌ **Todos los notebooks fallaron durante la ejecución**

- **Notebooks ejecutados**: 5
- **Notebooks exitosos**: 0
- **Notebooks con errores**: 5

## Errores Detallados por Notebook

### 1. Chapter_3_GettingStarted/Chapter_3_Unified.ipynb
- **Estado**: ❌ ERROR
- **Celdas ejecutadas**: 0/121
- **Tipo de error**: `IndentationError`
- **Celda con error**: Celda 30 (In[30])
- **Descripción**: 
  - Error de indentación en la función `get_model_selection_performance_plot`
  - El problema está en la línea 12: `(mean_performances_dictionary,std_performances_dictionary) = \`
  - Python espera un bloque indentado después de la definición de la función, pero encuentra una línea de continuación sin indentación adecuada

**Código problemático:**
```python
def get_model_selection_performance_plot(...):
    
    
    (mean_performances_dictionary,std_performances_dictionary) = \
        model_selection_performances(...)
```

**Solución**: Agregar indentación correcta o `pass` después de la definición de la función.

---

### 2. Chapter_4_PerformanceMetrics/Chapter_4_Unified.ipynb
- **Estado**: ❌ ERROR
- **Celdas ejecutadas**: 0/99
- **Tipo de error**: `NameError`
- **Celda con error**: Celda 46 (In[46])
- **Descripción**: 
  - Variable `confusion_matrix_plots` no está definida
  - La celda intenta acceder a una variable que no existe en el contexto de ejecución

**Código problemático:**
```python
confusion_matrix_plots
```

**Solución**: 
- Verificar que la variable se haya definido en una celda anterior
- O comentar/eliminar esta celda si es solo para visualización

---

### 3. Chapter_5_ModelValidationAndSelection/Chapter_5_Unified.ipynb
- **Estado**: ❌ ERROR
- **Celdas ejecutadas**: 0/166
- **Tipo de error**: `AttributeError`
- **Celda con error**: Celda 84 (In[84])
- **Descripción**: 
  - `GridSearchCV` no tiene el atributo `cv_results_`
  - Este es un problema de versión de scikit-learn
  - En versiones antiguas de scikit-learn, el atributo puede tener un nombre diferente o requerir que se llame a `fit()` primero

**Código problemático:**
```python
grid_search.cv_results_
```

**Solución**: 
- Verificar que `grid_search.fit()` se haya ejecutado antes
- O usar `grid_search.cv_results_` solo después de `fit()`
- Verificar compatibilidad con scikit-learn 1.0.0

---

### 4. Chapter_6_ImbalancedLearning/Chapter_6_Unified.ipynb
- **Estado**: ❌ ERROR
- **Celdas ejecutadas**: 0/218
- **Tipo de error**: `NameError`
- **Celda con error**: Celda 44 (In[44])
- **Descripción**: 
  - Variable `fig_distribution` no está definida
  - La celda intenta acceder a una figura que no existe en el contexto de ejecución

**Código problemático:**
```python
fig_distribution
```

**Solución**: 
- Verificar que la figura se haya creado en una celda anterior
- O comentar/eliminar esta celda si es solo para visualización

---

### 5. Chapter_7_DeepLearning/Chapter_7_Unified.ipynb
- **Estado**: ❌ ERROR
- **Celdas ejecutadas**: 0/249
- **Tipo de error**: `NameError`
- **Celda con error**: Celda 51 (In[51])
- **Descripción**: 
  - Variable `fig_activation` no está definida
  - La celda intenta acceder a una figura que no existe en el contexto de ejecución

**Código problemático:**
```python
fig_activation
```

**Solución**: 
- Verificar que la figura se haya creado en una celda anterior
- O comentar/eliminar esta celda si es solo para visualización

---

## Resumen de Tipos de Errores

| Tipo de Error | Cantidad | Notebooks Afectados |
|---------------|----------|---------------------|
| `IndentationError` | 1 | Chapter_3 |
| `NameError` | 3 | Chapter_4, Chapter_6, Chapter_7 |
| `AttributeError` | 1 | Chapter_5 |

## Análisis de Errores

### Errores de Sintaxis (1)
- **Chapter_3**: Error de indentación que impide la ejecución completa

### Errores de Variables No Definidas (3)
- **Chapter_4, 6, 7**: Variables/figuras que se intentan mostrar pero no están definidas
- Probablemente son celdas de visualización que dependen de celdas anteriores que no se ejecutaron

### Errores de API/Versionado (1)
- **Chapter_5**: Problema con la API de scikit-learn, posiblemente relacionado con la versión

## Recomendaciones

### Correcciones Inmediatas Necesarias

1. **Chapter_3**: Corregir indentación en función `get_model_selection_performance_plot`
2. **Chapter_4**: Verificar o comentar celda que accede a `confusion_matrix_plots`
3. **Chapter_5**: Verificar que `grid_search.fit()` se ejecute antes de acceder a `cv_results_`
4. **Chapter_6**: Verificar o comentar celda que accede a `fig_distribution`
5. **Chapter_7**: Verificar o comentar celda que accede a `fig_activation`

### Estrategia de Corrección

1. **Errores de sintaxis**: Corregir directamente en los notebooks
2. **Variables no definidas**: 
   - Verificar si las celdas anteriores se ejecutaron correctamente
   - Comentar celdas de visualización que dependen de variables no definidas
   - O asegurar que las variables se definan antes de usarlas
3. **Problemas de API**: Verificar compatibilidad con versiones de librerías

## Archivos de Reporte Generados

- **JSON**: `execution_results/execution_report_20260209_191538.json`
- **Texto**: `execution_results/execution_report_20260209_191538.txt`

## Notas

- Todos los notebooks comenzaron a ejecutarse correctamente (el problema de dependencias se resolvió)
- Los errores encontrados son errores reales en el código de los notebooks
- Estos errores impiden la ejecución completa de los notebooks
- Se requiere corrección manual de estos errores para que los notebooks se ejecuten completamente
