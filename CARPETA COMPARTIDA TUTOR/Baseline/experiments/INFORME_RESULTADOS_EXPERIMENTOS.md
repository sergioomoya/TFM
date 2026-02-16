# Informe Detallado de Resultados — Experimentos Tentativos

**Proyecto:** TFM — Detección de Fraude en Transacciones con Tarjeta de Crédito  
**Fecha de ejecución:** 16 de febrero de 2026  
**Entorno:** Docker (Python 3.11, scikit-learn, XGBoost, SHAP, imbalanced-learn)  
**Datos:** Simulación temporal de transacciones (Capítulo 3 del libro *Fraud Detection Handbook*)

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Experimento A — Establecimiento del Baseline Puro](#2-experimento-a--establecimiento-del-baseline-puro)
3. [Experimento C — Validación de Integridad Metodológica (Test Anti-Leakage)](#3-experimento-c--validación-de-integridad-metodológica-test-anti-leakage)
4. [Experimento D — Interpretabilidad y Caja Blanca (XAI)](#4-experimento-d--interpretabilidad-y-caja-blanca-xai)
5. [Tabla Comparativa Global](#5-tabla-comparativa-global)
6. [Conclusiones Generales](#6-conclusiones-generales)
7. [Anexo Técnico](#7-anexo-técnico)

---

## 1. Resumen Ejecutivo

Se ejecutaron con éxito los **tres experimentos tentativos** planificados para las primeras semanas del TFM:

| Experimento | Estado | Tiempo de ejecución | Celdas |
|---|---|---|---|
| **A — Baseline Puro** | ✅ Éxito | 15.15 s | 8/8 |
| **C — Test Anti-Leakage** | ✅ Éxito | 204.27 s (~3.4 min) | 7/7 |
| **D — Interpretabilidad (XAI)** | ✅ Éxito | 12.65 s | 7/7 |

**Tiempo total de ejecución:** ~232 segundos (~3.9 minutos)

### Hallazgos clave

1. **Random Forest** obtiene la mejor AUPRC (0.6634) del baseline, seguido de XGBoost (0.6389) y Logistic Regression (0.6057).
2. **El data leakage infla las métricas de forma catastrófica**: una pipeline incorrecta con SMOTE global y split aleatorio muestra AUPRC ≈ 1.000 vs. 0.611 con la metodología correcta, una diferencia del **63.5%**.
3. **`TERMINAL_ID_RISK_7DAY_WINDOW`** es, con diferencia, la variable más importante para la detección de fraude (41.9% de la ganancia en XGBoost), seguida de `TX_AMOUNT` (12.9%).
4. La **paradoja del desbalance** queda evidenciada: todos los modelos superan 99.6% de Accuracy, pero su Recall para fraude oscila entre 46.7% y 54.0%.

---

## 2. Experimento A — Establecimiento del Baseline Puro

### 2.1 Objetivo

Establecer métricas de referencia para tres modelos clásicos de Machine Learning aplicados a la detección de fraude, utilizando la metodología correcta del libro (división temporal, sin data leakage).

### 2.2 Modelos evaluados

- **Logistic Regression** (con `max_iter=1000`)
- **Random Forest** (100 estimadores)
- **XGBoost** (100 estimadores, `eval_metric='logloss'`)

### 2.3 Protocolo experimental

- **División temporal:** días 0–40 para entrenamiento, días 41–50 para test (ventana de retraso de 7 días incluida)
- **Features:** 17 variables derivadas de ingeniería de características del Capítulo 3
- **Sin resampling ni ponderación de clases** (baseline puro)

### 2.4 Resultados

| Modelo | AUC ROC | AUPRC | CP@100 | Accuracy | Recall (Fraude) | Precision | F1-Score |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.8705 | 0.6057 | 0.2914 | 0.9962 | 0.4675 | 0.9045 | 0.6164 |
| **Random Forest** | 0.8643 | **0.6634** | **0.2900** | 0.9966 | 0.5065 | **0.9653** | 0.6644 |
| **XGBoost** | 0.8618 | 0.6389 | 0.2729 | **0.9968** | **0.5403** | 0.9455 | **0.6876** |

### 2.5 Visualización

![Resultados Experimento A](results/figures/experiment_a_baseline_results.png)

**Figura 1.** Panel izquierdo: Curva Precision-Recall (Random Forest con AP=0.663 es superior). Panel central: Curvas ROC (todas similares ~0.86-0.87). Panel derecho: Paradoja del desbalance — Accuracy >99.6% pero Recall del fraude <55%.

### 2.6 Análisis

- **AUC ROC:** Los tres modelos muestran rendimiento muy similar (~0.86), lo que sugiere que la capacidad de discriminación global es comparable. Logistic Regression tiene una ligera ventaja (0.8705).
- **AUPRC (métrica prioritaria en desbalance):** Random Forest destaca con 0.6634, un 9.5% superior a Logistic Regression. Esta métrica es más informativa que AUC ROC para problemas con desbalance severo.
- **Card Precision@100 (CP@100):** Las tres métricas son cercanas (~0.27-0.29), indicando que de las 100 tarjetas más sospechosas cada día, aproximadamente 29 están realmente comprometidas.
- **Paradoja del desbalance:** A pesar de Accuracy >99.6%, el Recall para la clase de fraude es bajo (46.7%-54.0%), lo que demuestra que la Accuracy es una métrica engañosa en este contexto.
- **XGBoost** ofrece el mejor compromiso F1-Score (0.6876) y el mayor Recall (54.0%), lo que lo hace preferible si se busca detectar la mayor cantidad de fraudes posible.

---

## 3. Experimento C — Validación de Integridad Metodológica (Test Anti-Leakage)

### 3.1 Objetivo

Demostrar cuantitativamente el impacto devastador del **data leakage** en las métricas de evaluación, comparando una pipeline metodológicamente correcta con una incorrecta. Este experimento tiene un valor pedagógico fundamental para el TFM.

### 3.2 Diseño experimental

Se comparan dos ramas con **Logistic Regression** como modelo base:

| Aspecto | Rama Correcta | Rama Incorrecta |
|---|---|---|
| **División de datos** | Temporal (días 0-40 train, 41-50 test) | Aleatoria (`train_test_split`) |
| **Escalado** | `StandardScaler` ajustado solo en train | `StandardScaler` ajustado en TODOS los datos |
| **Resampling** | SMOTE aplicado solo a train | SMOTE aplicado ANTES de dividir |
| **Fuentes de leakage** | 0 | 3 |

### 3.3 Resultados

| Pipeline | AUC ROC | AUPRC | CP@100 |
|---|---|---|---|
| **C-Correcta** (temporal + SMOTE en train) | 0.8658 | 0.6115 | 0.29 |
| **C-Incorrecta** (SMOTE global + split aleatorio) | **0.9999** ⚠️ | **0.9999** ⚠️ | N/A |

### 3.4 Visualización

![Comparación Leakage](results/figures/experiment_c_leakage_comparison.png)

**Figura 2.** Panel izquierdo: Comparativa de AUPRC y AUC ROC — la pipeline incorrecta muestra métricas "perfectas" (≈1.0) que son completamente artificiales. Panel central: Curva PR de la rama correcta (AP=0.611). Panel derecho: Curva PR de la rama incorrecta (AP=1.000) con advertencia de *Data Leakage*.

### 3.5 Comparación con Experimento A

| Experimento | AUC ROC | AUPRC | CP@100 |
|---|---|---|---|
| A — Logistic Regression (baseline) | 0.8705 | 0.6057 | 0.2914 |
| A — Random Forest (baseline) | 0.8643 | 0.6634 | 0.2900 |
| A — XGBoost (baseline) | 0.8618 | 0.6389 | 0.2729 |
| **C-Correcta** (LR + SMOTE en train) | 0.8658 | 0.6115 | 0.2900 |
| **C-Incorrecta** (con Leakage) ⚠️ | 0.9999 | 0.9999 | N/A |

### 3.6 Análisis

1. **Evidencia irrefutable de data leakage:**
   - La pipeline incorrecta obtiene AUPRC = **0.9999** frente a 0.6115 de la correcta. Esto representa una **inflación del 63.5%** que es completamente artificial.
   - AUC ROC pasa de 0.8658 a 0.9999, una diferencia de **+15.5 puntos porcentuales**.

2. **Tres fuentes de contaminación identificadas:**
   - **Escalado global:** `StandardScaler` ajustado con datos de test filtra información de la distribución futura al modelo.
   - **SMOTE global:** Generar muestras sintéticas antes de la división crea puntos de test derivados directamente de los de entrenamiento, causando que el modelo "memorice" el test.
   - **Split aleatorio vs. temporal:** Romper el orden temporal permite al modelo aprender de eventos futuros, algo imposible en producción.

3. **Consistencia con el Baseline (Exp. A):**
   - La rama correcta del Experimento C (LR con SMOTE, AUC ROC=0.866, AUPRC=0.611) es consistente con el baseline de Logistic Regression sin SMOTE (AUC ROC=0.871, AUPRC=0.606). La adición de SMOTE proporciona una mejora marginal en AUPRC (+0.9%) sin degradar significativamente el AUC ROC.

4. **Valor para el TFM:**
   - Este experimento constituye una **evidencia empírica robusta** de por qué la integridad metodológica es esencial en la investigación de detección de fraude.
   - Cualquier trabajo que reporte métricas cercanas a 1.0 en este tipo de problema debe ser examinado críticamente en busca de data leakage.

---

## 4. Experimento D — Interpretabilidad y Caja Blanca (XAI)

### 4.1 Objetivo

Analizar qué variables impulsan las predicciones del modelo de detección de fraude, utilizando técnicas de **Explainable AI (XAI)**: Feature Importance nativa de XGBoost y valores SHAP (SHapley Additive exPlanations).

### 4.2 Modelo utilizado

- **XGBoost cost-sensitive** (`scale_pos_weight` = ratio de desbalance ≈ 111.4)
- Métricas del modelo: AUC ROC = 0.8320, AUPRC = 0.5988, CP@100 = 0.2557

> **Nota:** Las métricas del XGBoost cost-sensitive son ligeramente inferiores a las del baseline (Exp. A) porque la ponderación agresiva de la clase minoritaria cambia el punto de operación del modelo hacia mayor Recall a costa de Precision.

### 4.3 Feature Importance (Ganancia)

| Ranking | Variable | Importancia (Gain) |
|---|---|---|
| 1 | `TERMINAL_ID_RISK_7DAY_WINDOW` | **0.4186** |
| 2 | `TX_AMOUNT` | **0.1294** |
| 3 | `CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW` | 0.0552 |
| 4 | `CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW` | 0.0549 |
| 5 | `CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW` | 0.0428 |
| 6 | `TERMINAL_ID_RISK_30DAY_WINDOW` | 0.0375 |
| 7 | `TERMINAL_ID_NB_TX_1DAY_WINDOW` | 0.0362 |
| 8 | `TERMINAL_ID_RISK_1DAY_WINDOW` | 0.0330 |
| 9 | `CUSTOMER_ID_NB_TX_30DAY_WINDOW` | 0.0306 |
| 10 | `CUSTOMER_ID_NB_TX_7DAY_WINDOW` | 0.0290 |

### 4.4 Visualizaciones

#### Feature Importance

![Feature Importance](results/figures/experiment_d_feature_importance.png)

**Figura 3.** Las 10 variables más importantes según la ganancia (Gain) de XGBoost. `TERMINAL_ID_RISK_7DAY_WINDOW` domina con el 41.9% de la ganancia total, más del triple que la segunda variable.

#### SHAP Beeswarm Plot

![SHAP Beeswarm](results/figures/experiment_d_shap_beeswarm.png)

**Figura 4.** Gráfico Beeswarm de SHAP (500 muestras del test). Los puntos rojos representan valores altos de la variable, los azules valores bajos. Se observa claramente cómo los valores altos de las variables de riesgo del terminal empujan las predicciones hacia fraude (SHAP > 0).

### 4.5 Análisis

1. **Dominio del riesgo del terminal:**
   - `TERMINAL_ID_RISK_7DAY_WINDOW` concentra el **41.9%** de toda la ganancia, lo cual tiene sentido dominio: un terminal que ha procesado muchas transacciones fraudulentas en los últimos 7 días es altamente indicativo de fraude.
   - Las tres ventanas temporales de riesgo del terminal (1, 7 y 30 días) suman el **48.9%** de la ganancia total.

2. **Importancia del monto de transacción:**
   - `TX_AMOUNT` es la segunda variable más importante (12.9%). En el gráfico SHAP se observa que **montos altos (puntos rojos) empujan la predicción hacia fraude**, lo que es coherente con el escenario de fraude simulado.

3. **Variables de comportamiento del cliente:**
   - Las variables de importe medio del cliente (`CUSTOMER_ID_AVG_AMOUNT_*`) en diferentes ventanas temporales contribuyen colectivamente un ~15.3%, indicando que desviaciones del comportamiento habitual de gasto son indicadores de fraude.

4. **Discrepancia entre Feature Importance y SHAP:**
   - Aunque `TERMINAL_ID_RISK_7DAY_WINDOW` domina la Feature Importance, en el gráfico SHAP la variable que más dispersión muestra (mayor impacto medio absoluto) es `CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW`. Esto se debe a que la Feature Importance mide la ganancia total en los árboles, mientras que SHAP mide el impacto marginal en cada predicción individual.
   - Esta discrepancia es una observación metodológica valiosa: **no existe una única respuesta a "qué variable es más importante"**, depende del enfoque de medición.

5. **Force Plots (análisis local):**
   - Se han generado exitosamente los force plots para transacciones individuales (fraudulenta y normal) como imágenes estáticas (`experiment_d_shap_force_fraud.png`, `experiment_d_shap_force_normal.png`).
   - Estos gráficos muestran cómo cada característica empuja la predicción desde el valor base (base value) hacia la puntuación final del modelo. Las barras rojas indican características que aumentan el riesgo de fraude, mientras que las azules lo disminuyen.

---

## 5. Tabla Comparativa Global

| Experimento | Modelo | AUC ROC | AUPRC | CP@100 | Observación |
|---|---|---|---|---|---|
| A — Baseline | Logistic Regression | 0.8705 | 0.6057 | 0.2914 | Mejor AUC ROC del baseline |
| A — Baseline | Random Forest | 0.8643 | **0.6634** | 0.2900 | **Mejor AUPRC del baseline** |
| A — Baseline | XGBoost | 0.8618 | 0.6389 | 0.2729 | Mejor F1 y Recall |
| C — Correcta | LR + SMOTE (train) | 0.8658 | 0.6115 | 0.2900 | Consistente con Exp. A |
| C — Incorrecta ⚠️ | LR + SMOTE (global) | 0.9999 | 0.9999 | N/A | **Data leakage catastrófico** |
| D — XAI | XGBoost cost-sensitive | 0.8320 | 0.5988 | 0.2557 | Modelo para interpretabilidad |

---

## 6. Conclusiones Generales

### 6.1 Respecto al rendimiento del baseline

- Los tres modelos del baseline muestran un rendimiento **razonable pero no excepcional** (AUPRC entre 0.61-0.66), lo cual es esperado para datos simulados y modelos sin hiperparámetro tunning avanzado.
- **Random Forest** es el modelo baseline más sólido según AUPRC, mientras que **XGBoost** tiene el mejor balance F1.
- La **Card Precision@100** (~0.29) indica que se podrían priorizar correctamente unas 29 de cada 100 tarjetas sospechosas por día, una cifra operacionalmente útil como punto de partida.

### 6.2 Respecto a la integridad metodológica

- El Experimento C **demuestra cuantitativamente** que el data leakage puede inflar la AUPRC de 0.61 a 1.00, haciendo que un modelo parezca "perfecto" cuando en realidad no generalizaría en producción.
- Este hallazgo refuerza la importancia de la **división temporal** y del **aislamiento estricto** de los pasos de preprocesamiento al conjunto de entrenamiento.

### 6.3 Respecto a la interpretabilidad

- El riesgo histórico del terminal (`TERMINAL_ID_RISK_*DAY_WINDOW`) es el predictor dominante, lo cual valida que las features de ingeniería del Capítulo 3 capturan señales relevantes.
- SHAP proporciona una comprensión más matizada que la Feature Importance nativa, revelando que el impacto de las variables varía significativamente según la transacción individual.

### 6.4 Próximos pasos sugeridos

1. **Hyperparameter tuning** de los modelos baseline para mejorar las métricas de referencia.
2. **Experimento B (Cost-Sensitive Learning):** Evaluar el impacto de la ponderación de clases de forma sistemática en todos los modelos.
3. **Validación cruzada temporal** para obtener estimaciones más robustas de las métricas.
4. **Ampliar el análisis SHAP** con ejecución en JupyterLab para visualizar correctamente los force plots y dependence plots.

---

## 7. Anexo Técnico

### 7.1 Archivos de resultados generados

| Archivo | Descripción |
|---|---|
| `experiments/results/experiment_a_results.csv` | Tabla de métricas del Exp. A |
| `experiments/results/experiment_a_predictions.pkl` | Predicciones y probabilidades de todos los modelos |
| `experiments/results/experiment_c_comparison.csv` | Comparación correcta vs. incorrecta |
| `experiments/results/experiment_c_vs_a_comparison.csv` | Comparación Exp. C vs. Exp. A |
| `experiments/results/experiment_c_results.pkl` | Resultados completos con metadatos |
| `experiments/results/experiment_d_feature_importance.csv` | Ranking de importancia de variables |
| `experiments/results/experiment_d_results.pkl` | Valores SHAP, métricas y metadatos |
| `experiments/results/execution_report.json` | Reporte de ejecución (tiempos, estado) |

### 7.2 Figuras generadas

| Figura | Descripción |
|---|---|
| `experiment_a_baseline_results.png` | Curvas PR, ROC y paradoja del desbalance |
| `experiment_c_leakage_comparison.png` | Impacto del data leakage en métricas |
| `experiment_d_feature_importance.png` | Top-10 variables por ganancia (XGBoost) |
| `experiment_d_shap_beeswarm.png` | Distribución de valores SHAP por variable |
| `experiment_d_shap_force_fraud.png` | Force plot de una transacción fraudulenta |
| `experiment_d_shap_force_normal.png` | Force plot de una transacción normal |

### 7.3 Reproducibilidad

Para reproducir los resultados:

```bash
# 1. Ejecutar los cuadernos unificados (generan los datos)
docker compose run --rm unified-notebooks

# 2. Ejecutar los experimentos
docker compose run --rm experiments python experiments/run_experiment.py --all
```

**Semilla aleatoria:** 42 (fijada en todos los modelos y operaciones de splitting)  
**Versión de Python:** 3.11 (contenedor Docker)
