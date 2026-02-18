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

Los resultados se obtienen con **validación prequential** y se reportan como **media ± desviación estándar** sobre los folds.

| Modelo | AUC ROC | AUPRC | CP@100 |
|--------|---------|-------|--------|
| **Logistic Regression** | 0.8688 ± 0.0158 | 0.6350 ± 0.0163 | 0.2929 ± 0.0141 |
| **Random Forest** | **0.8729 ± 0.0108** | 0.6846 ± 0.0103 | **0.2971 ± 0.0144** |
| **XGBoost** | 0.8692 ± 0.0091 | **0.6904 ± 0.0084** | 0.2961 ± 0.0139 |

*Resultados con validación prequential completa: 4 folds, grid completo de hiperparámetros.*

### 3.1. Visualización

![Resultados Experimento A](results/figures/experiment_a_baseline_results.png)

**Figura 1.** Barras con media ± desviación estándar para AUC ROC, AUPRC y Card Precision@100 (4 folds prequential).  
*Se genera en `experiments/results/figures/` al ejecutar el experimento.*

---

## 4. Análisis

1. **AUC ROC:** Random Forest obtiene el mejor valor (**0.8729 ± 0.0108**), seguido de XGBoost (0.8692) y Logistic Regression (0.8688).
2. **AUPRC (métrica prioritaria):** XGBoost destaca con **0.6904 ± 0.0084**, seguido de Random Forest (0.6846) y Logistic Regression (0.6350).
3. **CP@100:** Valores en el rango 0.29–0.30. Random Forest alcanza 0.2971 ± 0.0144.
4. **Ventaja de la metodología prequential:** El reporte de media ± desv. estándar permite evaluar la robustez; las desviaciones (~0.009–0.016) reflejan la variabilidad entre los 4 folds temporales.
5. **Tiempo de ejecución:** LR ~10 s, RF ~55 s, XGBoost ~30–40 min (modo completo con 4 folds).

---

## 5. Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `experiments/results/experiment_a_results.csv` | Métricas (AUC ROC, AUPRC, CP@100) con media y desv. estándar |
| `experiments/results/experiment_a_predictions.pkl` | Predicciones, probabilidades y metadatos de cada modelo |
| `experiments/results/figures/experiment_a_baseline_results.png` | Gráfico de barras con media ± desv. estándar |

---

## 6. Reproducibilidad

- **Semilla:** `SEED=42` (config.py)
- **Fechas:** `START_DATE_TRAINING_FOR_VALID=2018-08-01`, `START_DATE_TRAINING_FOR_TEST=2018-08-22`
- **Splits temporales:** `DELTA_TRAIN=7`, `DELTA_DELAY=7`, `DELTA_TEST=7` días
- **Datos:** `simulated-data-transformed` (Capítulo 3, 2018-04-01 a 2018-09-30)

---

## 7. Conclusiones

- Los modelos basados en árboles (XGBoost, RF) ofrecen mejor AUPRC y balance general que la Regresión Logística.
- **XGBoost** obtiene la mejor AUPRC (0.6904); **Random Forest** la mejor AUC ROC y CP@100.
- La metodología temporal y el pipeline sin leakage garantizan resultados honestos para comparación con experimentos posteriores.
