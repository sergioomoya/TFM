# Informe Detallado de Resultados — Experimentos Tentativos

**Proyecto:** TFM — Detección de Fraude en Transacciones con Tarjeta de Crédito  
**Fecha de ejecución:** 19 de febrero de 2026  
**Entorno:** Docker (Python 3.8, scikit-learn, XGBoost, SHAP, imbalanced-learn)  
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
| **A — Baseline Puro** | ✅ Éxito | 16.1 s | 8/8 |
| **C — Test Anti-Leakage** | ✅ Éxito | 181.9 s (~3.0 min) | 7/7 |
| **D — Interpretabilidad (XAI)** | ✅ Éxito | 17.0 s | 8/8 |

**Tiempo total de ejecución:** ~212 segundos (~3.5 minutos)

### Hallazgos clave

1. **Random Forest** obtiene la mejor AUPRC (0.6634) del baseline, seguido de XGBoost (0.6389) y Logistic Regression (0.6057).
2. **El data leakage infla las métricas de forma catastrófica**: una pipeline incorrecta con SMOTE global y split aleatorio muestra AUPRC ≈ 1.000 vs. 0.611 con la metodología correcta, una diferencia del **63.5%**.
3. **`TERMINAL_ID_RISK_7DAY_WINDOW`** es, con diferencia, la variable más importante para la detección de fraude (38.8% de la ganancia en XGBoost baseline), seguida de `TX_AMOUNT` (12.9%).
4. La **paradoja del desbalance** queda evidenciada: todos los modelos superan 99.6% de Accuracy, pero su Recall para fraude oscila entre 46.7% y 54.0%.

---

## 2. Experimento A — Establecimiento del Baseline Puro

### 2.1 Objetivo

Establecer métricas de referencia para tres modelos clásicos de Machine Learning aplicados a la detección de fraude, utilizando la **metodología del Capítulo 5** (validación prequential, búsqueda de hiperparámetros, división temporal sin data leakage).

### 2.2 Modelos evaluados

- **Logistic Regression** (Grid sobre `C` ∈ {0.1, 1, 10, 100})
- **Random Forest** (Grid sobre `max_depth`, `n_estimators`)
- **XGBoost** (Grid sobre `max_depth`, `n_estimators`, `learning_rate`)

### 2.3 Protocolo experimental

- **Validación prequential:** 4 folds temporales
- **GridSearchCV** con selección por AUPRC en validación
- **Features:** 15 variables (ingeniería de características del Capítulo 3)
- **Sin resampling ni ponderación de clases** (baseline puro)
- **Reporte:** media ± desviación estándar para AUC ROC, AUPRC, CP@100

### 2.4 Resultados

*Media ± desviación estándar sobre 4 folds prequential (grid completo).*

| Modelo | AUC ROC | AUPRC | CP@100 |
|---|---|---|---|
| **Logistic Regression** | 0.8688 ± 0.016 | 0.6350 ± 0.016 | 0.2929 ± 0.014 |
| **Random Forest** | **0.8729 ± 0.011** | 0.6846 ± 0.010 | **0.2971 ± 0.014** |
| **XGBoost** | 0.8692 ± 0.009 | **0.6904 ± 0.008** | 0.2961 ± 0.014 |

### 2.5 Visualización

![Resultados Experimento A](results/figures/experiment_a_baseline_results.png)

**Figura 1.** Barras con media ± desviación estándar para AUC ROC, AUPRC y Card Precision@100 (metodología Capítulo 5).

### 2.6 Análisis

- **Rigor estadístico:** La validación prequential y el reporte de desviación estándar permiten cuantificar la incertidumbre de las estimaciones.
- **Búsqueda de hiperparámetros:** La selección por AUPRC en validación mejora la robustez de las comparaciones entre modelos.
- **Métricas priorizadas:** AUC ROC, AUPRC y CP@100 son las métricas apropiadas para problemas desbalanceados; la Accuracy queda relegada por su carácter engañoso.

---

## 3. Experimento C — Validación de Integridad Metodológica (Test Anti-Leakage)

### 3.1 Objetivo

Demostrar cuantitativamente el impacto del **data leakage** y **desglosar el efecto de cada fuente** (split aleatorio, escalado global, SMOTE global), replicando el análisis con **LR, RF y XGBoost**.

### 3.2 Diseño experimental (refactorizado)

**Modelos:** Logistic Regression, Random Forest, XGBoost  

**Cinco ramas:**

| Rama | Split | Escalado | SMOTE | Fuentes |
|------|-------|----------|-------|---------|
| Correcta | Temporal | Solo train | Solo train | 0 |
| Leak_split | Aleatorio | Solo train | Solo train | 1 |
| Leak_scaler | Temporal | Global | Solo train | 1 |
| Leak_smote | Temporal | Solo train | Global | 1 |
| Leak_todas | Aleatorio | Global | Global | 3 |

**SMOTE:** `k_neighbors=5`, `sampling_strategy='auto'` (config.py)

### 3.3 Resultados

| Modelo | C-Correcta AUC | C-Correcta AUPRC | C-Correcta CP@100 | C-Leak_todas AUC | C-Leak_todas AUPRC |
|--------|---------------|------------------|-------------------|------------------|--------------------|
| Logistic Regression | 0.8692 | 0.5830 | 0.2886 | 0.8979 | **0.9287** ⚠️ |
| Random Forest | 0.8658 | 0.6115 | 0.2900 | **0.9999** ⚠️ | **0.9999** ⚠️ |
| XGBoost | 0.8601 | 0.6163 | 0.2743 | **0.9992** ⚠️ | **0.9995** ⚠️ |

### 3.4 Desglose por fuente de leakage (AUPRC)

| Fuente | LR | RF | XGB |
|--------|--------|--------|--------|
| Correcta | 0.5830 | 0.6115 | 0.6163 |
| Leak_split | 0.6150 | 0.6768 | 0.6910 |
| Leak_scaler | 0.5820 | 0.6070 | 0.6149 |
| Leak_smote | 0.5900 | 0.9038 | 0.7461 |
| Leak_todas | **0.9287** | **0.9999** | **0.9995** |

*El split aleatorio infla moderadamente (+0.03–0.08). El escalado global apenas afecta. El SMOTE global infla mucho en RF (+0.29) y algo en XGBoost (+0.13). Las tres fuentes juntas provocan AUPRC ≈1.0.*

### 3.5 Comparación con Experimento A

| Experimento | AUC ROC | AUPRC | CP@100 |
|---|---|---|---|
| A — Logistic Regression (baseline) | 0.8688 ± 0.016 | 0.6350 ± 0.016 | 0.293 ± 0.014 |
| A — Random Forest (baseline) | 0.8729 ± 0.011 | 0.6846 ± 0.010 | 0.297 ± 0.014 |
| A — XGBoost (baseline) | 0.8692 ± 0.009 | 0.6904 ± 0.008 | 0.296 ± 0.014 |
| C-Correcta — Logistic Regression | 0.8692 | 0.5830 | 0.2886 |
| C-Correcta — Random Forest | 0.8658 | 0.6115 | 0.2900 |
| C-Correcta — XGBoost | 0.8601 | 0.6163 | 0.2743 |
| C-Leak_todas — LR ⚠️ | 0.8979 | **0.9287** | N/A |
| C-Leak_todas — RF ⚠️ | **0.9999** | **0.9999** | N/A |
| C-Leak_todas — XGBoost ⚠️ | **0.9992** | **0.9995** | N/A |

### 3.6 Análisis

1. **Evidencia del data leakage:** La rama Leak_todas obtiene AUPRC ≈1.0 (artificial) frente a valores realistas (~0.6–0.7) en la rama correcta.
2. **Desglose por fuente:** Permite cuantificar el impacto incremental de cada práctica incorrecta.
3. **Generalización:** LR, RF y XGBoost muestran el mismo patrón de inflación.
4. **Valor para el TFM:** Evidencia empírica robusta de la importancia de la integridad metodológica.

---

## 4. Experimento D — Interpretabilidad y Caja Blanca (XAI)

### 4.1 Objetivo

Analizar qué variables impulsan las predicciones del modelo de detección de fraude, utilizando técnicas de **Explainable AI (XAI)**: Feature Importance nativa de XGBoost y valores SHAP (SHapley Additive exPlanations).

### 4.2 Modelo utilizado

- **XGBoost baseline** (mejor AUPRC del Experimento A), en lugar de cost-sensitive
- Métricas del modelo: AUC ROC = 0.8618, AUPRC = 0.6389, CP@100 = 0.2729

> **Mejora (CRITICA_MEJORA_EXPERIMENTOS):** Se usa el modelo con mejor rendimiento del baseline para que la interpretabilidad refleje el modelo desplegado. Incluye Gain/weight/cover, mean \|SHAP\|, force plots con contexto y dependence plots.

### 4.3 Feature Importance (Gain, weight, cover)

| Ranking | Variable | Gain | weight | cover |
|---|---|---|---|---|
| 1 | `TERMINAL_ID_RISK_7DAY_WINDOW` | **0.388** | 76 | 2009 |
| 2 | `TX_AMOUNT` | **0.129** | 330 | 521 |
| 3 | `TERMINAL_ID_RISK_30DAY_WINDOW` | 0.101 | 157 | 34 |
| 4 | `CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW` | 0.056 | 324 | 74 |
| 5 | `TERMINAL_ID_RISK_1DAY_WINDOW` | 0.041 | 21 | 3858 |
| 6 | `CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW` | 0.038 | 254 | 380 |
| 7 | `TERMINAL_ID_NB_TX_1DAY_WINDOW` | 0.038 | 48 | 55 |
| 8 | `CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW` | 0.030 | 251 | 68 |
| 9 | `CUSTOMER_ID_NB_TX_7DAY_WINDOW` | 0.028 | 125 | 40 |
| 10 | `CUSTOMER_ID_NB_TX_30DAY_WINDOW` | 0.027 | 215 | 44 |

### 4.4 Visualizaciones

#### Feature Importance

![Feature Importance](../figuras_experimentos/experiment_d_feature_importance.png)

**Figura 3.** Las 10 variables más importantes según la ganancia (Gain) de XGBoost baseline. `TERMINAL_ID_RISK_7DAY_WINDOW` domina con el 38.8% de la ganancia total, más del triple que la segunda variable.

#### SHAP Beeswarm Plot

![SHAP Beeswarm](../figuras_experimentos/experiment_d_shap_beeswarm.png)

**Figura 4.** Gráfico Beeswarm de SHAP (1000 muestras del test). Los puntos rojos representan valores altos de la variable, los azules valores bajos.

**Tabla mean \|SHAP\|:** `experiment_d_shap_mean_impact.csv` — Top 5: CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW, TX_AMOUNT, CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW, CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW, TERMINAL_ID_NB_TX_30DAY_WINDOW.

**Dependence plots:** TX_AMOUNT y TERMINAL_ID_RISK_7DAY_WINDOW para analizar interacciones.

### 4.5 Análisis

1. **Dominio del riesgo del terminal:**
   - `TERMINAL_ID_RISK_7DAY_WINDOW` concentra el **38.8%** de toda la ganancia, lo cual tiene sentido dominio: un terminal que ha procesado muchas transacciones fraudulentas en los últimos 7 días es altamente indicativo de fraude.
   - Las tres ventanas temporales de riesgo del terminal (1, 7 y 30 días) suman el **~53%** de la ganancia total.

2. **Importancia del monto de transacción:**
   - `TX_AMOUNT` es la segunda variable más importante (12.9%). En el gráfico SHAP se observa que **montos altos (puntos rojos) empujan la predicción hacia fraude**, lo que es coherente con el escenario de fraude simulado.

3. **Variables de comportamiento del cliente:**
   - Las variables de importe medio del cliente (`CUSTOMER_ID_AVG_AMOUNT_*`) en diferentes ventanas temporales contribuyen colectivamente un ~15.3%, indicando que desviaciones del comportamiento habitual de gasto son indicadores de fraude.

4. **Discrepancia entre Feature Importance y SHAP:**
   - Aunque `TERMINAL_ID_RISK_7DAY_WINDOW` domina la Feature Importance, en el gráfico SHAP la variable que más dispersión muestra (mayor impacto medio absoluto) es `CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW`. Esto se debe a que la Feature Importance mide la ganancia total en los árboles, mientras que SHAP mide el impacto marginal en cada predicción individual.
   - Esta discrepancia es una observación metodológica valiosa: **no existe una única respuesta a "qué variable es más importante"**, depende del enfoque de medición.

5. **Force Plots (análisis local) con contexto:**
   - Se han generado force plots para transacciones individuales (fraudulenta y normal) con **descripción de la transacción** (TX_AMOUNT, TERMINAL_ID, CUSTOMER_ID, riesgo del terminal).
   - Las barras rojas indican características que aumentan el riesgo de fraude, las azules lo disminuyen.

### 4.6 Validación por ablación (eliminación de top feature SHAP)

Para validar que la importancia SHAP refleja contribución real al modelo, se ejecutó un **experimento de ablación**: eliminar la característica con mayor mean |SHAP| (`CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW`), reentrenar el XGBoost baseline con las 14 restantes y comparar métricas en el mismo conjunto de test.

| Modelo | AUC ROC | AUPRC | CP@100 |
|--------|---------|-------|--------|
| D (completo, 15 features) | 0.8618 | 0.6389 | 0.2729 |
| D (ablación, 14 features) | 0.8498 | 0.5942 | 0.2714 |
| **Δ** | **−0.0121** | **−0.0447** | −0.0014 |

![Comparación de métricas: modelo completo vs ablación](../figuras_experimentos/experiment_d_ablation_metrics_comparison.png)

**Figura 4a.** Comparación de métricas entre el modelo completo y el modelo ablated. La degradación de AUPRC (−4.47 pp) confirma que la top feature SHAP aporta valor predictivo real.

![Ranking mean |SHAP| tras ablación](../figuras_experimentos/experiment_d_ablation_shap_ranking.png)

**Figura 4b.** Ranking de importancia (mean |SHAP|) del modelo reentrenado sin la top feature. `TX_AMOUNT` pasa a la primera posición; el ranking no se mantiene respecto al modelo original.

**Conclusiones de la ablación:** (1) La top feature SHAP contribuye de forma no redundante al rendimiento; (2) el ranking de las 14 restantes cambia tras reentrenar (redistribución de importancia); (3) la ablación proporciona evidencia causal que complementa el análisis SHAP.

**Script:** `run_experiment_d_ablation.py` — Figuras: `experiment_d_ablation_metrics_comparison.png`, `experiment_d_ablation_shap_ranking.png`, `experiment_d_ablation_shap_beeswarm.png`.

---

## 5. Tabla Comparativa Global

| Experimento | Modelo | AUC ROC | AUPRC | CP@100 | Observación |
|---|---|---|---|---|---|
| A — Baseline | Logistic Regression | 0.8688 ± 0.016 | 0.6350 ± 0.016 | 0.293 ± 0.014 | Prequential + GridSearch |
| A — Baseline | Random Forest | **0.8729 ± 0.011** | 0.6846 ± 0.010 | **0.297 ± 0.014** | Prequential + GridSearch |
| A — Baseline | XGBoost | 0.8692 ± 0.009 | **0.6904 ± 0.008** | 0.296 ± 0.014 | Prequential + GridSearch |
| C — Correcta | Logistic Regression | 0.8692 | 0.5830 | 0.2886 | SMOTE solo en train |
| C — Correcta | Random Forest | 0.8658 | 0.6115 | 0.2900 | SMOTE solo en train |
| C — Correcta | XGBoost | 0.8601 | 0.6163 | 0.2743 | SMOTE solo en train |
| C — Leak_todas ⚠️ | Logistic Regression | 0.8979 | **0.9287** | N/A | Data leakage |
| C — Leak_todas ⚠️ | Random Forest | **0.9999** | **0.9999** | N/A | **Data leakage catastrófico** |
| C — Leak_todas ⚠️ | XGBoost | **0.9992** | **0.9995** | N/A | **Data leakage catastrófico** |
| D — XAI | XGBoost baseline | **0.8618** | **0.6389** | **0.2729** | Interpretabilidad del mejor modelo |
| D — Ablación | XGBoost (14 features) | 0.8498 | 0.5942 | 0.2714 | Sin top SHAP (CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW); validación de importancia |

---

## 6. Conclusiones Generales

### 6.1 Respecto al rendimiento del baseline

- Los tres modelos del baseline muestran un rendimiento **razonable** (AUPRC entre 0.64–0.70) con validación prequential y búsqueda de hiperparámetros.
- **XGBoost** obtiene la mejor AUPRC (0.690 ± 0.008); **Random Forest** la mejor AUC ROC (0.873 ± 0.011) y CP@100 (0.297 ± 0.014).
- La **Card Precision@100** (~0.29) indica que se podrían priorizar correctamente unas 29 de cada 100 tarjetas sospechosas por día, una cifra operacionalmente útil como punto de partida.

### 6.2 Respecto a la integridad metodológica

- El Experimento C **demuestra cuantitativamente** que el data leakage puede inflar la AUPRC de 0.61 a 1.00, haciendo que un modelo parezca "perfecto" cuando en realidad no generalizaría en producción.
- Este hallazgo refuerza la importancia de la **división temporal** y del **aislamiento estricto** de los pasos de preprocesamiento al conjunto de entrenamiento.

### 6.3 Respecto a la interpretabilidad

- El riesgo histórico del terminal (`TERMINAL_ID_RISK_*DAY_WINDOW`) es el predictor dominante, lo cual valida que las features de ingeniería del Capítulo 3 capturan señales relevantes.
- SHAP proporciona una comprensión más matizada que la Feature Importance nativa, revelando que el impacto de las variables varía significativamente según la transacción individual.
- **La ablación** (eliminar la top feature SHAP y reentrenar) confirma que la importancia SHAP refleja valor predictivo real: AUPRC cae ~4.5 pp al eliminar `CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW`.

### 6.4 Próximos pasos sugeridos

1. ~~**Hyperparameter tuning** de los modelos baseline~~ — ✅ Implementado en Experimento A (metodología Cap. 5).
2. **Experimento B (Cost-Sensitive Learning):** Evaluar el impacto de la ponderación de clases de forma sistemática en todos los modelos.
3. ~~**Validación cruzada temporal**~~ — ✅ Experimento A usa validación prequential (4 folds).
4. ~~**Ampliar el análisis SHAP** con dependence plots~~ — ✅ Implementado en Experimento D (mejoras CRITICA).

---

## 7. Anexo Técnico

### 7.1 Archivos de resultados generados

| Archivo | Descripción |
|---|---|
| `experiments/results/experiment_a_results.csv` | Tabla de métricas del Exp. A |
| `experiments/results/experiment_a_predictions.pkl` | Predicciones y probabilidades de todos los modelos |
| `experiments/results/experiment_c_comparison.csv` | Comparación C-Correcta vs. C-Leak_todas por modelo |
| `experiments/results/experiment_c_all_ramas.csv` | Métricas de las 5 ramas (Correcta, Leak_split, Leak_scaler, Leak_smote, Leak_todas) |
| `experiments/results/experiment_c_desglose_leakage.csv` | Desglose por fuente de leakage (AUPRC por rama y modelo) |
| `experiments/results/experiment_c_vs_a_comparison.csv` | Comparación Exp. C vs. Exp. A |
| `experiments/results/experiment_d_feature_importance.csv` | Ranking Gain/weight/cover |
| `experiments/results/experiment_d_feature_importance_all_types.csv` | Feature importance con tipos |
| `experiments/results/experiment_d_shap_mean_impact.csv` | mean \|SHAP\| por variable |
| `experiments/results/experiment_d_results.pkl` | Valores SHAP, métricas y metadatos |
| `experiments/results/experiment_d_ablation_comparison.csv` | Comparación full vs ablated (métricas) |
| `experiments/results/experiment_d_ablation_shap_ranking.csv` | Ranking mean \|SHAP\| del modelo ablated |
| `experiments/results/execution_report.json` | Reporte de ejecución (tiempos, estado) |

### 7.2 Figuras generadas

| Figura | Descripción |
|---|---|
| `experiment_a_baseline_results.png` | Barras AUC ROC, AUPRC, CP@100 (media ± desv.) |
| `experiment_c_leakage_comparison.png` | Impacto del data leakage en métricas |
| `experiment_d_feature_importance.png` | Top-10 variables por Gain (XGBoost baseline) |
| `experiment_d_shap_beeswarm.png` | Beeswarm SHAP (1000 muestras) |
| `experiment_d_shap_force_fraud.png` | Force plot fraude con contexto |
| `experiment_d_shap_force_normal.png` | Force plot transacción normal con contexto |
| `experiment_d_shap_dependence_tx_amount.png` | Dependence plot TX_AMOUNT |
| `experiment_d_shap_dependence_terminal_id_risk_7day_window.png` | Dependence plot riesgo terminal |
| `experiment_d_ablation_metrics_comparison.png` | Comparación de métricas full vs ablated |
| `experiment_d_ablation_shap_ranking.png` | Ranking mean \|SHAP\| del modelo ablated |
| `experiment_d_ablation_shap_beeswarm.png` | Beeswarm SHAP del modelo ablated |

### 7.3 Reproducibilidad

Para reproducir los resultados:

```bash
# 1. Ejecutar los cuadernos unificados (generan los datos)
docker compose run --rm unified-notebooks

# 2. Ejecutar los experimentos
docker compose run --rm experiments python experiments/run_experiment.py --all

# 3. (Opcional) Ejecutar ablación del Experimento D
docker compose run --rm experiments python experiments/run_experiment_d_ablation.py
```

**Semilla aleatoria:** 42 (fijada en todos los modelos y operaciones de splitting)  
**Versión de Python:** 3.8 (contenedor Docker)
