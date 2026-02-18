# Crítica y Propuesta de Mejora — Experimentos del TFM

**Proyecto:** TFM — Detección de Fraude en Transacciones con Tarjeta de Crédito  
**Fecha:** 18 de febrero de 2026  
**Objetivo:** Revisión crítica de los experimentos A, C y D, identificando debilidades metodológicas y propuestas concretas de mejora.

---

## 1. Resumen de la Revisión

Los tres experimentos (Baseline, Data Leakage, Interpretabilidad) cumplen su propósito inicial y aportan valor al TFM. Sin embargo, existen **limitaciones metodológicas**, **lagunas en el diseño experimental** y **oportunidades de mejora** que deberían abordarse para reforzar la calidad científica y la aportación del trabajo.

---

## 2. Crítica por Experimento

### 2.1. Experimento A — Baseline Puro

| Aspecto | Valoración | Comentario |
|---------|------------|------------|
| Claridad del objetivo | ✅ | Bien definido |
| Rigor metodológico | ✅ | División temporal + validación prequential |
| Replicabilidad | ✅ | Semilla fijada, pipeline documentado |
| Cobertura de modelos | ⚠️ | Solo 3 modelos; faltan alternativas |

**Debilidades (resueltas en refactorización):**

1. ~~**Sin hiperparámetro tuning**~~ — ✅ **Resuelto.** Se implementó `GridSearchCV` con grids por modelo (LR, RF, XGBoost) siguiendo el Capítulo 5.

2. ~~**Un solo split temporal**~~ — ✅ **Resuelto.** Validación prequential con 4 folds; se reporta media ± desviación estándar.

3. **Comparación incompleta:** No se incluyen LightGBM o CatBoost. Pendiente para futuras iteraciones.

4. ~~**Inconsistencia en nº de features**~~ — Documentado explícitamente en `config.INPUT_FEATURES` (15 features).

**Propuestas de mejora (aplicadas):**

- ✅ Búsqueda de hiperparámetros con validación prequential (metodología Capítulo 5).
- ✅ Reporte de media ± desviación estándar (4 folds prequential).
- Pendiente: Añadir LightGBM u otros modelos.

---

### 2.2. Experimento C — Test Anti-Leakage

| Aspecto | Valoración | Comentario |
|---------|------------|------------|
| Valor pedagógico | ✅ | Excelente para ilustrar leakage |
| Diseño experimental | ✅ | Bien contrastadas rama correcta vs incorrecta |
| Resultados | ✅ | Evidencia clara y cuantitativa |
| Generalización | ✅ | LR, RF y XGBoost |

**Debilidades (resueltas en refactorización):**

1. ~~**Un solo modelo**~~ — ✅ **Resuelto.** LR, RF y XGBoost replicados.

2. **Leakage “artificialmente” extremo:** — ✅ **Resuelto.** Cinco ramas con desglose por fuente.

3. **CP@100 en Leak_todas:** N/A (datos sintéticos sin estructura temporal). Sí se calcula en ramas que preservan el test original.

4. **Tamaño de muestra SMOTE:** — ✅ **Resuelto.** `config.SMOTE_PARAMS`.

---

### 2.3. Experimento D — Interpretabilidad (XAI)

| Aspecto | Valoración | Comentario |
|---------|------------|------------|
| Técnicas aplicadas | ✅ | Feature Importance + SHAP cubren global y local |
| Claridad de resultados | ✅ | Top-10 y gráficos útiles |
| Interpretación | ✅ | Coherente con el dominio |
| Comparabilidad | ⚠️ | Modelo distinto al baseline (cost-sensitive) |

**Debilidades (resueltas):**

1. ~~**Modelo distinto al mejor baseline**~~ — ✅ **Resuelto.** Se usa XGBoost baseline (mejor AUPRC del Exp. A) en lugar de cost-sensitive.

2. ~~**Feature Importance limitada**~~ — ✅ **Resuelto.** Se incluyen Gain, weight y cover; se guarda `experiment_d_feature_importance_all_types.csv`.

3. ~~**Muestra SHAP reducida**~~ — ✅ **Resuelto.** Beeswarm sobre **1000** muestras (antes 500).

4. ~~**Falta de análisis cuantitativo SHAP**~~ — ✅ **Resuelto.** Tabla `mean |SHAP|` por variable en `experiment_d_shap_mean_impact.csv`.

5. ~~**Force plots sin contexto**~~ — ✅ **Resuelto.** Cada force plot incluye descripción de la transacción (TX_AMOUNT, TERMINAL_ID, CUSTOMER_ID, riesgo de terminal).

6. ~~**Sin Dependence plots**~~ — ✅ **Resuelto.** Dependence plots para TX_AMOUNT y TERMINAL_ID_RISK_7DAY_WINDOW.

---

## 3. Deficiencias Transversales

### 3.1. Experimento B ausente

El **Experimento B (Cost-Sensitive Learning)** está implementado (`experiment_b_cost_sensitive.ipynb`) pero **no se ejecuta** en el pipeline actual (`run_experiment.py`). El INFORME_RESULTADOS_EXPERIMENTOS lo menciona como próximo paso, pero su ausencia deja incompleta la cadena:

- A: Baseline (sin tratamiento del desbalance)
- B: Cost-sensitive (class_weight, scale_pos_weight)
- C: Leakage
- D: Interpretabilidad

Sin B, no se cuantifica de forma directa el efecto de la ponderación de clases frente al baseline puro.

**Propuesta:** Incluir el Experimento B en `ALL_EXPERIMENTS` y ejecutarlo para tener una línea de evolución completa.

### 3.2. Datos simulados vs reales

Todos los experimentos usan **datos simulados** (Capítulo 3). Las conclusiones son válidas en ese contexto, pero:

- Los patrones de fraude simulados pueden no reflejar la complejidad real (concept drift, evolución de ataques, etc.).
- Las métricas en producción pueden diferir de las obtenidas en simulación.

**Propuesta:** In la sección de limitaciones del TFM, dejar explícito que los resultados se obtienen sobre datos simulados y que la generalización a datos reales requeriría validación adicional.

### 3.3. Documentación de variabilidad

~~No se reportan intervalos de confianza ni desviaciones estándar.~~ **Resuelto en Experimento A:** Se usa validación prequential (4 folds) y se reporta media ± desv. estándar para AUC ROC, AUPRC y CP@100.

### 3.4. Reproducibilidad

Aunque existe `config.py` y semilla fijada, faltan:

- Versiones exactas de bibliotecas (scikit-learn, XGBoost, SHAP) en un `requirements.txt` o equivalente.
- Registro del hash del dataset o de la fecha de generación de los datos simulados.

**Propuesta:** Añadir `requirements.txt` con versiones fijadas y documentar en el README o en un anexo cómo regenerar los datos y reproducir los experimentos.

---

## 4. Priorización de Mejoras

| Prioridad | Mejora | Esfuerzo | Impacto |
|-----------|--------|----------|---------|
| **Alta** | Incluir Experimento B en la ejecución | Bajo | Alto (cadena completa) |
| ~~**Alta**~~ | ~~Validación prequential y reporte de varianza~~ | — | ✅ Implementado (Exp. A) |
| ~~**Media**~~ | ~~Hiperparámetro tuning del baseline~~ | — | ✅ Implementado (Exp. A) |
| **Media** | Desglose del Exp. C por fuente de leakage | Medio | Alto (didáctico) |
| ~~**Media**~~ | ~~Interpretabilidad del mejor modelo~~ | — | ✅ XGBoost baseline (Exp. D) |
| **Baja** | Añadir LightGBM al baseline | Bajo | Medio (enriquecer comparación) |
| ~~**Baja**~~ | ~~Tabla numérica mean \|SHAP\|~~ | — | ✅ Implementado (Exp. D) |
| ~~**Baja**~~ | ~~Documentar force plots~~ | — | ✅ Contexto en force plots (Exp. D) |

---

## 5. Conclusión

Los experimentos actuales constituyen una base sólida para el TFM y cumplen con los objetivos planteados. Las críticas se centran en **rigor estadístico** (varianza, incertidumbre), **completitud** (Experimento B, desglose del leakage) y **consistencia** (modelo usado en interpretabilidad vs mejor baseline).

Priorizar la incorporación del Experimento B, la validación prequential y el ajuste de hiperparámetros elevaría notablemente la calidad científica del trabajo sin cambiar el diseño conceptual actual.
