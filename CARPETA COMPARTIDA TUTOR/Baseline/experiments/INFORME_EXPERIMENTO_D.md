# Informe del Experimento D: Interpretabilidad y XAI (Explainable AI)

**Estado:** Implementado, ejecutado y documentado (mejoras según CRITICA_MEJORA_EXPERIMENTOS.md aplicadas)  
**Ubicación:** `experiments/experiment_d_interpretability.ipynb`  
**Fecha de ejecución:** 19 de febrero de 2026  
**Tiempo de ejecución:** 17.0 s (8/8 celdas) — contenedor Docker

---

## 1. Introducción

Los modelos de "caja negra" como XGBoost ofrecen alto rendimiento pero **baja explicabilidad**. En el sector financiero es obligatorio justificar por qué se bloquea una transacción (regulaciones como GDPR). Este experimento aplica técnicas de **Explainable AI (XAI)** para abrir la caja negra y analizar qué variables impulsan las predicciones de fraude.

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

## 5. Conclusiones

- El modelo aprende patrones de negocio lógicos: riesgo histórico del terminal, monto, y comportamiento del cliente.
- La explicabilidad valida que el modelo no está capturando ruido.
- SHAP proporciona comprensión tanto global (Beeswarm) como local (Force plots), útil para auditorías y cumplimiento regulatorio.
