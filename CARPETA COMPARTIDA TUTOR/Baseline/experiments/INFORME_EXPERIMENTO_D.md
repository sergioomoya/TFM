# Informe del Experimento D: Interpretabilidad y XAI (Explainable AI)

**Estado:** Implementado, ejecutado y documentado  
**Ubicación:** `experiments/experiment_d_interpretability.ipynb`  
**Fecha de ejecución:** 18 de febrero de 2026  
**Tiempo de ejecución:** 13.7 s (7/7 celdas)

---

## 1. Introducción

Los modelos de "caja negra" como XGBoost ofrecen alto rendimiento pero **baja explicabilidad**. En el sector financiero es obligatorio justificar por qué se bloquea una transacción (regulaciones como GDPR). Este experimento aplica técnicas de **Explainable AI (XAI)** para abrir la caja negra y analizar qué variables impulsan las predicciones de fraude.

---

## 2. Metodología

### 2.1. Modelo

- **XGBoost cost-sensitive** (`scale_pos_weight` = ratio de desbalance ≈ 111.4).
- Entrenado con división temporal estricta.
- Métricas del modelo: AUC ROC ≈ 0.83, AUPRC ≈ 0.60, CP@100 ≈ 0.26.

> Las métricas son ligeramente inferiores al baseline (Exp. A) porque la ponderación de la clase minoritaria cambia el punto de operación hacia mayor Recall a costa de Precision.

### 2.2. Técnicas aplicadas

| Técnica | Descripción |
|---------|-------------|
| **Feature Importance (Gain)** | Métrica intrínseca de XGBoost: ganancia de impureza al dividir por cada variable. |
| **SHAP (TreeExplainer)** | Valores de contribución por característica y predicción (teoría de juegos). |
| **Beeswarm plot** | Resumen global: dispersión de valores SHAP por variable (500 muestras de test). |
| **Force plots** | Explicaciones locales para transacciones individuales (fraude vs normal). |

---

## 3. Resultados Obtenidos

### 3.1. Feature Importance (Top 10)

| Ranking | Variable | Importancia (Gain) |
|---------|----------|-------------------|
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

### 3.2. Visualizaciones

![Feature Importance](../figuras_experimentos/experiment_d_feature_importance.png)  
**Figura 3.** Top-10 variables por ganancia. `TERMINAL_ID_RISK_7DAY_WINDOW` domina con el 41.9% de la ganancia total.

![SHAP Beeswarm](../figuras_experimentos/experiment_d_shap_beeswarm.png)  
**Figura 4.** Beeswarm SHAP (500 muestras). Puntos rojos = valores altos, azules = bajos. Valores altos de riesgo del terminal empujan la predicción hacia fraude (SHAP > 0).

![Force Plot Fraude](../figuras_experimentos/experiment_d_shap_force_fraud.png)  
**Figura 5.** Force plot de una transacción fraudulenta: cómo cada variable empuja la puntuación.

![Force Plot Normal](../figuras_experimentos/experiment_d_shap_force_normal.png)  
**Figura 6.** Force plot de una transacción normal.

---

## 4. Análisis

1. **Dominio del riesgo del terminal:**
   - `TERMINAL_ID_RISK_7DAY_WINDOW` concentra el **41.9%** de la ganancia total.
   - Las tres ventanas de riesgo del terminal (1, 7, 30 días) suman ~**48.9%**.

2. **Importancia del monto:**
   - `TX_AMOUNT` es la segunda variable (12.9%). Montos altos empujan hacia fraude (coherente con el escenario simulado).

3. **Variables de comportamiento del cliente:**
   - `CUSTOMER_ID_AVG_AMOUNT_*` contribuyen ~15.3%, indicando que desviaciones del gasto habitual son indicadores de fraude.

4. **Discrepancia Feature Importance vs SHAP:**
   - La Feature Importance mide ganancia total en árboles; SHAP mide impacto marginal por predicción. No existe una única respuesta a "qué variable es más importante"; depende del enfoque.

5. **Valor operativo:** SHAP permite generar "códigos de razón" para analistas de fraude, facilitando la investigación manual de alertas.

---

## 5. Conclusiones

- El modelo aprende patrones de negocio lógicos: riesgo histórico del terminal, monto, y comportamiento del cliente.
- La explicabilidad valida que el modelo no está capturando ruido.
- SHAP proporciona comprensión tanto global (Beeswarm) como local (Force plots), útil para auditorías y cumplimiento regulatorio.
