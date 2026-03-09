# Informe del Experimento D: Interpretabilidad y XAI (Explainable AI)

**Estado:** Implementado, ejecutado y documentado (mejoras según CRITICA_MEJORA_EXPERIMENTOS.md aplicadas)  
**Ubicación:** `experiments/experiment_d_interpretability.ipynb`, `experiments/run_experiment_d_standalone.py`, `experiments/run_experiment_d_ablation.py`  
**Fecha de ejecución:** 19 de febrero de 2026 (interpretabilidad), ablación complementaria marzo 2026  
**Tiempo de ejecución:** 17.0 s (8/8 celdas) — contenedor Docker

---

## 1. Introducción

Los modelos de *machine learning* de alto rendimiento como XGBoost suelen comportarse como "cajas negras": proporcionan predicciones precisas pero sin una explicación directa de los factores que las determinan. En el ámbito de la detección de fraude en transacciones con tarjeta de crédito, la **explicabilidad** no es opcional: las normativas (p. ej. GDPR) y las exigencias operativas requieren poder justificar ante el cliente o la autoridad por qué una transacción fue bloqueada o marcada como sospechosa.

El **Experimento D** aborda el Objetivo Específico 6 del TFM: *"Analizar la importancia de las características"*. Se aplican técnicas de **Explainable AI (XAI)** —Feature Importance nativa de XGBoost y valores SHAP— para identificar qué variables impulsan las predicciones de fraude. Adicionalmente, se realiza una **validación por ablación** que confirma empíricamente que la característica más importante según SHAP contribuye de forma medible al rendimiento del modelo.

---

## 2. Metodología

### 2.1. Modelo (mejorado)

- **XGBoost baseline** (mismo que el mejor rendimiento del Experimento A), en lugar de cost-sensitive.
- Entrenado con división temporal estricta.
- **Métricas obtenidas:** AUC ROC 0.8618, AUPRC 0.6389, CP@100 0.2729.

> **Mejora:** Se usa el modelo con mejor AUPRC del baseline para garantizar que la interpretabilidad refleja el modelo realmente desplegado.

### 2.2. Técnicas aplicadas

| Técnica | Descripción |
|---------|-------------|
| **Feature Importance (Gain, weight, cover)** | Métricas de XGBoost: Gain (ganancia de impureza), weight (frecuencia de splits) y cover. |
| **SHAP (TreeExplainer)** | Valores de contribución por característica y predicción (teoría de juegos). |
| **Tabla mean \|SHAP\|** | Análisis cuantitativo: impacto medio absoluto por variable (`experiment_d_shap_mean_impact.csv`). |
| **Beeswarm plot** | Resumen global: dispersión de valores SHAP por variable (**1000** muestras de test). |
| **Force plots con contexto** | Explicaciones locales con descripción de la transacción (monto, terminal, cliente, riesgo). |
| **Dependence plots** | Interacciones: TX_AMOUNT, TERMINAL_ID_RISK_7DAY_WINDOW. |

---

## 3. Resultados Obtenidos

### 3.1. Feature Importance (Top 10 - Gain, weight, cover)

| Ranking | Variable | Gain | weight | cover |
|---------|----------|------|--------|-------|
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

### 3.2. Tabla mean |SHAP| (impacto marginal)

Archivo: `experiments/results/experiment_d_shap_mean_impact.csv`. Top 5 por impacto absoluto medio: CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW (1.29), TX_AMOUNT (0.79), CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW (0.46), CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW (0.34), TERMINAL_ID_NB_TX_30DAY_WINDOW (0.31).

### 3.3. Visualizaciones

Rutas: `experiments/results/figures/`

- `experiment_d_feature_importance.png` — Top-10 variables por Gain (XGBoost baseline).
- `experiment_d_shap_beeswarm.png` — Beeswarm SHAP (**1000** muestras).
- `experiment_d_shap_force_fraud.png` — Force plot fraude **con contexto** (monto, terminal, cliente).
- `experiment_d_shap_force_normal.png` — Force plot transacción normal con contexto.
- `experiment_d_shap_dependence_tx_amount.png` — Dependence plot TX_AMOUNT.
- `experiment_d_shap_dependence_terminal_id_risk_7day_window.png` — Dependence plot riesgo del terminal.

---

## 4. Análisis

1. **Dominio del riesgo del terminal:**
   - `TERMINAL_ID_RISK_7DAY_WINDOW` concentra el **38.8%** de la ganancia total (Gain).
   - Las tres ventanas de riesgo del terminal (1, 7, 30 días) suman ~**53%**.

2. **Importancia del monto:**
   - `TX_AMOUNT` es la segunda variable (12.9%). Montos altos empujan hacia fraude (coherente con el escenario simulado).

3. **Variables de comportamiento del cliente:**
   - `CUSTOMER_ID_AVG_AMOUNT_*` dominan el ranking SHAP (mean |SHAP|), indicando impacto marginal alto en predicciones individuales.

4. **Discrepancia Feature Importance vs SHAP:**
   - La Feature Importance mide ganancia total en árboles; SHAP mide impacto marginal por predicción. No existe una única respuesta a "qué variable es más importante"; depende del enfoque.

5. **Valor operativo:** SHAP permite generar "códigos de razón" para analistas de fraude, facilitando la investigación manual de alertas.

---

## 5. Validación por ablación (eliminación de la top feature SHAP)

### 5.1. Motivación y diseño experimental

Los análisis de importancia (Feature Importance, SHAP) identifican qué variables contribuyen a las predicciones, pero **no garantizan por sí solos** que esa contribución sea causal o irreemplazable. Para validar empíricamente que la característica con mayor impacto SHAP aporta valor real, se diseñó un **experimento de ablación** (ablation study): eliminar la variable top, reentrenar el modelo desde cero con el resto de features y comparar el rendimiento.

Este enfoque es estándar en la literatura de interpretabilidad y selección de características, ya que cuantifica el coste de eliminar información del modelo.

### 5.2. Metodología

1. **Identificación de la top feature:** Se tomó la variable con mayor mean |SHAP| del Experimento D original: `CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW` (media del importe de transacciones del cliente en ventana de 30 días).
2. **Modelo completo (baseline):** XGBoost baseline con las 15 features originales.
3. **Modelo ablated:** Mismo XGBoost, mismas condiciones, pero **sin** la top feature (14 variables).
4. **Comparación:** AUC ROC, AUPRC y Card Precision@100 en el mismo conjunto de test.
5. **Análisis de redistribución:** Cálculo del nuevo ranking mean |SHAP| sobre el modelo ablated para observar cómo cambia la importancia relativa de las variables restantes.

**Script:** `experiments/run_experiment_d_ablation.py`

### 5.3. Resultados: comparación de métricas

| Modelo | AUC ROC | AUPRC | CP@100 |
|--------|---------|-------|--------|
| D (completo, 15 features) | 0.8618 | 0.6389 | 0.2729 |
| D (ablación, 14 features) | 0.8498 | 0.5942 | 0.2714 |
| **Δ (ablated − full)** | **−0.0121** | **−0.0447** | −0.0014 |

![Comparación de métricas: modelo completo vs ablación](results/figures/experiment_d_ablation_metrics_comparison.png)

**Figura 5.** Comparación de AUC ROC, AUPRC y CP@100 entre el modelo con 15 features (completo) y el modelo reentrenado sin `CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW` (ablación). La degradación de AUPRC (−4.47 pp) confirma que esta variable aporta valor predictivo real.

### 5.4. Redistribución del ranking de importancia

Al reentrenar con 14 features, el modelo redistribuye el poder predictivo entre las variables restantes. El ranking mean |SHAP| **no se mantiene**: otras features capturan parte de la información que aportaba la eliminada.

| Rank | Feature | mean_abs_SHAP |
|------|---------|---------------|
| 1 | TX_AMOUNT | 1.027 |
| 2 | CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW | 0.967 |
| 3 | CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW | 0.410 |
| 4 | CUSTOMER_ID_NB_TX_30DAY_WINDOW | 0.336 |
| 5 | TERMINAL_ID_NB_TX_30DAY_WINDOW | 0.290 |
| ... | ... | ... |

![Ranking mean |SHAP| tras ablación](results/figures/experiment_d_ablation_shap_ranking.png)

**Figura 6.** Ranking de importancia (mean |SHAP|) del modelo ablated. `TX_AMOUNT` pasa a la primera posición; `CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW` escala al segundo lugar, evidenciando la redistribución de importancia.

![Beeswarm SHAP modelo ablated](results/figures/experiment_d_ablation_shap_beeswarm.png)

**Figura 7.** Beeswarm SHAP del modelo reentrenado sin la top feature. Muestra la dispersión de contribuciones de cada variable restante (1000 muestras de test).

### 5.5. Interpretación y conclusiones de la ablación

1. **Validación exitosa:** La caída de AUPRC (−4.47 pp) y AUC ROC (−1.21 pp) demuestra que `CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW` aporta información predictiva no redundante con el resto de variables.
2. **Redistribución no trivial:** El ranking de las 14 features restantes cambia respecto al modelo original; no se mantiene el orden relativo. Esto es coherente con la teoría: al eliminar una feature, el modelo reajusta sus splits y las importancias relativas se modifican.
3. **Implicación metodológica:** La ablación complementa el análisis SHAP, proporcionando evidencia causal de que la importancia cuantificada se traduce en pérdida de rendimiento medible.

### 5.6. Archivos y figuras generados

| Archivo | Descripción |
|---------|-------------|
| `experiment_d_ablation_comparison.csv` | Comparación de métricas full vs ablated |
| `experiment_d_ablation_shap_ranking.csv` | Ranking mean \|SHAP\| completo del modelo ablated |
| `experiment_d_ablation_report.md` | Informe resumido en Markdown |
| `experiment_d_ablation_metrics_comparison.png` | Gráfico de barras: comparación de métricas |
| `experiment_d_ablation_shap_ranking.png` | Bar chart: ranking mean \|SHAP\| de las 14 features |
| `experiment_d_ablation_shap_beeswarm.png` | Beeswarm SHAP del modelo ablated |

**Ejecución:** `docker compose run --rm experiments python experiments/run_experiment_d_ablation.py`

---

## 6. Conclusiones

1. **Patrones de negocio coherentes:** El modelo aprovecha variables de alto valor semántico: riesgo histórico del terminal (`TERMINAL_ID_RISK_*`), monto de la transacción (`TX_AMOUNT`) y comportamiento del cliente (`CUSTOMER_ID_AVG_AMOUNT_*`). No se detectan dependencias aparentemente arbitrarias o ruidosas.

2. **Valor de la explicabilidad:** La interpretabilidad permite validar que el modelo refleja conocimiento de dominio razonable y facilita la aceptación por parte de equipos de negocio, auditoría y reguladores.

3. **Complementariedad de técnicas:** La Feature Importance (Gain) y el análisis SHAP ofrecen perspectivas distintas: la primera mide la ganancia total en los árboles; la segunda, el impacto marginal por predicción. Ambas son útiles según el objetivo del análisis.

4. **Validación empírica por ablación:** Al eliminar la característica con mayor mean |SHAP| (`CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW`), el rendimiento empeora de forma significativa (AUPRC −4.47 pp). Esto demuestra que la importancia cuantificada por SHAP se traduce en contribución predictiva real, no en correlación espuria.

5. **Recomendación para producción:** Los force plots y los códigos de razón derivados de SHAP pueden integrarse en sistemas de alerta para soportar la investigación manual de transacciones sospechosas por parte de analistas de fraude.
