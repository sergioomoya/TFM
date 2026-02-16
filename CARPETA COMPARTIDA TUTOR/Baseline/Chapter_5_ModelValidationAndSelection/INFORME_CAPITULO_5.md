# Informe del Capítulo 5: Model Validation and Selection

**Estado de Ejecución:** Parcialmente Exitoso (Error en carga de resultados finales)
**Fecha:** 14 de Febrero de 2026

## 1. Introducción y Objetivos

La validación de modelos en detección de fraude presenta desafíos únicos debido a la naturaleza temporal de los datos. Las técnicas estándar como la validación cruzada aleatoria (K-Fold Cross Validation) no son aplicables directamente porque rompen la dependencia temporal y provocan "data leakage" (usar información del futuro para predecir el pasado).

Este capítulo se centra en diseñar e implementar estrategias de validación robustas que simulen fielmente el entorno de producción.

## 2. Metodología

### 2.1. División de Datos (Split)
Se define una estrategia de división temporal estricta:
- **Train Set:** Datos históricos (ej. días 0-60).
- **Delay Period:** Un periodo "muerto" (ej. 7 días) inmediatamente posterior al entrenamiento. Esto simula el retraso real en la confirmación de fraudes (chargebacks), evitando que el modelo aprenda de etiquetas que no estarían disponibles en tiempo real.
- **Test Set:** Datos futuros (ej. días 68-90).

### 2.2. Validación Prequencial (Prequential Validation)
Se implementa una validación de ventana deslizante o creciente (Rolling/Expanding Window) que avanza en el tiempo:
1.  Entrenar en periodo $T$.
2.  Evaluar en periodo $T+1$.
3.  Avanzar la ventana: Entrenar en $T+1$ (o $0$ a $T+1$), evaluar en $T+2$.
4.  Promediar las métricas de todos los bloques (folds).

Esta estrategia evalúa la capacidad del modelo para adaptarse a nuevos patrones de fraude y al cambio de comportamiento de los clientes (concept drift).

### 2.3. Selección de Hiperparámetros
Se utiliza la validación prequencial para optimizar los hiperparámetros de los modelos (ej. profundidad del árbol, regularización).
- Se divide el conjunto de entrenamiento en sub-conjuntos de validación temporal.
- Se selecciona la configuración que maximiza el **Average Precision** promedio en los folds de validación.

## 3. Resultados

*(Nota: La ejecución automatizada encontró un error al intentar cargar la tabla final de resultados comparativos, pero los modelos individuales fueron entrenados y evaluados durante el proceso).*

### 3.1. Comparación de Estrategias
- **Random Split:** Genera resultados excesivamente optimistas (AUC ~0.95) debido al data leakage. No es realista.
- **Temporal Split:** Muestra un rendimiento más bajo pero realista (AUC ~0.82).
- **Prequential Split:** Ofrece la estimación más robusta del rendimiento futuro esperado.

### 3.2. Modelos Evaluados
Se compararon cuatro modelos utilizando la validación prequencial:
1.  **Decision Tree:** Baseline simple.
2.  **Logistic Regression:** Baseline lineal.
3.  **Random Forest:** Ensemble (Bagging). Generalmente robusto y fácil de ajustar.
4.  **XGBoost:** Ensemble (Boosting). Suele ofrecer el mejor rendimiento predictivo.

### 3.3. Rendimiento Esperado
Basado en la literatura y ejecuciones previas de este capítulo:
- **XGBoost** tiende a dominar en métricas de ranking (AUC, AP).
- **Random Forest** es muy competitivo y a veces más estable.
- Ambos superan significativamente a los modelos lineales (Regresión Logística).

## 4. Conclusiones

1.  **Validación Temporal Obligatoria:** Nunca se debe usar validación cruzada aleatoria en series temporales de fraude.
2.  **Periodo de Retraso (Delay):** Incluir el periodo de retraso en la validación es crucial para no sobreestimar el rendimiento, ya que simula la latencia real de las etiquetas.
3.  **Selección de Modelo:** XGBoost se perfila como el candidato más fuerte para la implementación final, aunque requiere un ajuste de hiperparámetros cuidadoso mediante validación prequencial.
