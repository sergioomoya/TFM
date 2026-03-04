# Informe del Experimento A: Baseline Puro

**Estado:** Implementado, ejecutado y documentado  
**Ubicación:** `experiments/experiment_a_baseline.ipynb`  
**Metodología:** Capítulo 5 (validación prequential + GridSearchCV)  
**Ejecución:**  
- Notebook: `experiments/experiment_a_baseline.ipynb`  
- Script standalone: `python experiments/run_experiment_a_standalone.py` (modo completo ~15–30 min; modo rápido con `QUICK=True` ~3–5 min)  
- Docker: `docker compose run --rm experiments python experiments/run_experiment_a_standalone.py`

---

## 1. Introducción

El objetivo del Experimento A es establecer una **línea base de rendimiento** utilizando modelos de Machine Learning estándar **sin aplicar técnicas específicas para el desbalanceo de clases** (sin `class_weight`, sin SMOTE). Este punto de partida es crucial para cuantificar las mejoras que aportarán técnicas más avanzadas en experimentos posteriores y para evidenciar la **paradoja del desbalance**: métricas engañosas como la Accuracy frente a métricas apropiadas como AUPRC y CP@100.

---

## 2. Metodología

### 2.1. Validación Prequential (Capítulo 5)

Se sigue el protocolo del **Capítulo 5** del *Fraud Detection Handbook*:

- **4 folds prequential** con desplazamiento temporal
- **GridSearchCV** con búsqueda de hiperparámetros por modelo
- **Split validation/test:** fechas diferenciadas (`START_DATE_TRAINING_FOR_VALID`, `START_DATE_TRAINING_FOR_TEST`)
- **Reporte:** media ± desviación estándar para AUC ROC, AUPRC y Card Precision@100

### 2.2. Datos

- **Dataset:** Transacciones simuladas transformadas (Capítulo 3).
- **Features:** 15 variables (ver `INPUT_FEATURES` en `config.py`): `TX_AMOUNT`, `TX_DURING_WEEKEND`, `TX_DURING_NIGHT`, ventanas de 1/7/30 días para cliente y terminal.
- **Parámetros temporales:** `DELTA_TRAIN=7`, `DELTA_DELAY=7`, `DELTA_TEST=7` días.

### 2.3. Modos de Ejecución

- **Modo completo** (`QUICK=False`): 4 folds, grid completo. Tiempo estimado: ~15–30 min.
- **Modo rápido** (`QUICK=True` en `run_experiment_a_standalone.py`): 2 folds, grid reducido. Tiempo estimado: ~3–5 min.

### 2.4. Modelos y Grids de Hiperparámetros

Se entrenan tres clasificadores con **búsqueda de hiperparámetros** (sin `class_weight`/`scale_pos_weight`):

1. **Regresión Logística:** Grid sobre `C` ∈ {0.1, 1, 10, 100}
2. **Random Forest:** Grid sobre `max_depth` ∈ {10, 20, 50}, `n_estimators` ∈ {50, 100}
3. **XGBoost:** Grid sobre `max_depth` ∈ {3, 6, 9}, `n_estimators` ∈ {50, 100}, `learning_rate` = 0.3

Pipeline: `StandardScaler` + clasificador. Selección del mejor modelo por AUPRC en validación.

### 2.5. Métricas

- **AUC ROC, AUPRC, Card Precision@100:** reportadas como media ± desv. estándar sobre los 4 folds prequential.

---

## 3. Resultados Obtenidos

Los resultados se obtienen con **validación prequential** y se reportan como **media ± desviación estándar** sobre los 4 folds.

### 3.1. Métricas principales (orientadas al ranking)

| Modelo | AUC ROC | AUPRC | CP@100 |
|--------|---------|-------|--------|
| **Logistic Regression** | 0.8688 ± 0.0158 | 0.6350 ± 0.0163 | 0.2929 ± 0.0141 |
| **Random Forest** | **0.8729 ± 0.0108** | 0.6846 ± 0.0103 | **0.2971 ± 0.0144** |
| **XGBoost** | 0.8692 ± 0.0091 | **0.6904 ± 0.0084** | 0.2961 ± 0.0139 |

*Resultados con validación prequential completa: 4 folds, grid completo de hiperparámetros.*

### 3.2. Paradoja de la Accuracy: demostración empírica del desbalance

Para evidenciar la **inutilidad de la Accuracy** en detección de fraude, se calcularon métricas de clasificación (threshold=0.5) con los mismos splits prequential:

| Modelo | Accuracy | Recall (Fraude) | Precision (Fraude) | F1 (Fraude) | Specificity |
|--------|----------|-----------------|--------------------| ------------|-------------|
| Logistic Regression | **99.62%** ± 0.06 | 51.82% ± 3.73 | 88.29% ± 5.80 | 65.28% ± 4.19 | 99.952% |
| Random Forest | **99.69%** ± 0.03 | 58.94% ± 1.63 | 93.99% ± 2.92 | 72.42% ± 1.11 | 99.974% |
| XGBoost | **99.70%** ± 0.02 | 61.75% ± 0.92 | 92.07% ± 1.59 | 73.92% ± 0.71 | 99.963% |

**Lectura clave para el TFM:** Los tres modelos superan el 99.6% de Accuracy, una cifra que sugiere un rendimiento casi perfecto. Sin embargo, el **Recall de fraude** (la proporción de fraudes reales que el modelo detecta) oscila entre el 52% y el 62% — es decir, **entre 4 y 5 de cada 10 fraudes pasan completamente desapercibidos**. Un clasificador trivial que predijera siempre "legítimo" obtendría 99.16% de Accuracy (1 − 0.0084), evidenciando que esta métrica es inservible cuando las clases están desbalanceadas (ratio ~118:1).

### 3.3. Matriz de confusión (agregada sobre 4 folds, threshold=0.5)

#### XGBoost (mejor AUPRC)

|  | **Predicho: Legítimo** | **Predicho: Fraude** |
|--|------------------------|----------------------|
| **Real: Legítimo** | TN = 230,485 | FP = 85 |
| **Real: Fraude** | FN = 609 | TP = 982 |

- **Total evaluado:** 232,161 transacciones (4 folds × ~58k)
- **Fraudes reales:** 1,591 → **Detectados:** 982 (61.7%) | **No detectados:** 609 (38.3%)
- **Falsas alarmas:** 85 (solo 0.037% de las transacciones legítimas)

#### Random Forest

|  | **Predicho: Legítimo** | **Predicho: Fraude** |
|--|------------------------|----------------------|
| **Real: Legítimo** | TN = 230,509 | FP = 61 |
| **Real: Fraude** | FN = 654 | TP = 937 |

- Fraudes reales: 1,591 → Detectados: 937 (58.9%) | No detectados: 654 (41.1%)

#### Logistic Regression

|  | **Predicho: Legítimo** | **Predicho: Fraude** |
|--|------------------------|----------------------|
| **Real: Legítimo** | TN = 230,459 | FP = 111 |
| **Real: Fraude** | FN = 768 | TP = 823 |

- Fraudes reales: 1,591 → Detectados: 823 (51.7%) | No detectados: 768 (48.3%)

### 3.4. Hiperparámetros ganadores (seleccionados por mejor AUPRC en validación)

| Modelo | Hiperparámetros óptimos |
|--------|-------------------------|
| Logistic Regression | `C=10` (regularización inversa) |
| Random Forest | `max_depth=50`, `n_estimators=100` |
| XGBoost | `max_depth=3`, `n_estimators=100`, `learning_rate=0.3` |

Todos los modelos usan `StandardScaler` como preprocesamiento y `random_state=42`.

> **Nota sobre XGBoost:** El valor bajo de `max_depth=3` indica que el modelo favorece árboles poco profundos, compensando con 100 estimadores. Esto produce un modelo más conservador pero robusto, coherente con el bajo número de FP (85 falsas alarmas sobre 230k transacciones legítimas).

### 3.5. Tiempos de entrenamiento (GridSearch completo, 4 folds prequential)

| Modelo | Tiempo total | Equivalencia |
|--------|-------------|--------------|
| Logistic Regression | 17.4 s | — |
| Random Forest | 124.8 s | ~2.1 min |
| XGBoost | 3,310.8 s | ~55.2 min |
| **Total Experimento A** | **3,453.0 s** | **~57.6 min** |

*Tiempos medidos en contenedor Docker (CPU), incluyendo validación + test con GridSearchCV completo.*

### 3.6. Visualización

![Resultados Experimento A](results/figures/experiment_a_baseline_results.png)

**Figura 1.** Barras con media ± desviación estándar para AUC ROC, AUPRC y Card Precision@100 (4 folds prequential).  
*Se genera en `experiments/results/figures/` al ejecutar el experimento.*

---

## 4. Análisis

1. **AUC ROC:** Random Forest obtiene el mejor valor (**0.8729 ± 0.0108**), seguido de XGBoost (0.8692) y Logistic Regression (0.8688).
2. **AUPRC (métrica prioritaria):** XGBoost destaca con **0.6904 ± 0.0084**, seguido de Random Forest (0.6846) y Logistic Regression (0.6350).
3. **CP@100:** Valores en el rango 0.29–0.30. Random Forest alcanza 0.2971 ± 0.0144.
4. **Paradoja de la Accuracy:** Con un 99.70% de Accuracy, XGBoost parece perfecto. Sin embargo, un Recall de fraude del 61.75% implica que **609 de 1,591 fraudes no fueron detectados** (38.3%). Un clasificador trivial "siempre legítimo" lograría 99.16% de Accuracy. La diferencia marginal (0.54 puntos porcentuales) demuestra que la Accuracy es una métrica engañosa con datasets desbalanceados (~0.84% de fraude).
5. **Ventaja de la metodología prequential:** El reporte de media ± desv. estándar permite evaluar la robustez; las desviaciones (~0.009–0.016) reflejan la variabilidad entre los 4 folds temporales.
6. **Robustez de XGBoost:** La combinación `max_depth=3` + 100 estimadores genera un modelo conservador con la mayor precisión en detección de fraude (92.07% Precision) manteniendo el mejor Recall entre los tres modelos.
7. **Tiempo de ejecución:** La búsqueda de hiperparámetros de XGBoost (3,310.8 s) domina el coste computacional total (95.9%), lo que justifica el uso de grids reducidos en validaciones preliminares.

---

## 5. Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `experiments/results/experiment_a_results.csv` | Métricas (AUC ROC, AUPRC, CP@100) con media y desv. estándar |
| `experiments/results/experiment_a_predictions.pkl` | Predicciones, probabilidades y metadatos de cada modelo |
| `experiments/results/experiment_a_detailed_metrics.json` | Accuracy, Confusion Matrix, Recall, F1, Precision, tiempos por fold |
| `experiments/results/figures/experiment_a_confusion_matrices.png` | Gráfico de matrices de confusión (heatmaps) por modelo |
| `experiments/results/figures/experiment_a_baseline_results.png` | Gráfico de barras con media ± desv. estándar |

---

## 6. Reproducibilidad

- **Semilla:** `SEED=42` (config.py)
- **Fechas:** `START_DATE_TRAINING_FOR_VALID=2018-08-01`, `START_DATE_TRAINING_FOR_TEST=2018-08-22`
- **Splits temporales:** `DELTA_TRAIN=7`, `DELTA_DELAY=7`, `DELTA_TEST=7` días
- **Datos:** `simulated-data-transformed` (Capítulo 3, 2018-04-01 a 2018-09-30)

---

## 7. Conclusiones

- **La Accuracy es engañosa:** Los tres modelos superan el 99.6% de Accuracy, pero dejan sin detectar entre el 38% y el 48% de los fraudes reales. Un clasificador trivial ("siempre legítimo") alcanzaría 99.16% de Accuracy. Esto demuestra empíricamente la necesidad de métricas como AUPRC y CP@100 en problemas desbalanceados.
- Los modelos basados en árboles (XGBoost, RF) ofrecen mejor AUPRC y balance general que la Regresión Logística.
- **XGBoost** obtiene la mejor AUPRC (**0.6904**), el mejor Recall de fraude (**61.75%**) y el mejor F1 de fraude (**73.92%**); **Random Forest** la mejor AUC ROC y CP@100.
- **Hiperparámetros:** XGBoost con `max_depth=3` y 100 estimadores demuestra que un modelo conservador (árboles poco profundos) es preferible a uno complejo, logrando alta Precision (92.07%) con el mejor Recall.
- La metodología temporal (prequential) y el pipeline sin leakage garantizan resultados honestos para comparación con experimentos posteriores.

---

## 8. Variante Undersampling (10:1)

Existe una variante del Experimento A que aplica **undersampling de legítimas** (ratio 10:1) en train/valid. Ver `INFORME_EXPERIMENTO_A_UNDERSAMPLED.md` para metodología, resultados (AUPRC 0,659, CP@100 0,279, Recall 66,31 %, FP 579, tiempo XGBoost 16,1 s) y comparativa con el baseline.
