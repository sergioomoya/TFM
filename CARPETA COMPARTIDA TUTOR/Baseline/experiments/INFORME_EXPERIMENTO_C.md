# Informe del Experimento C: Prueba de Data Leakage (Fuga de Datos)

**Estado:** Implementado, ejecutado y documentado  
**Ubicación:** `experiments/experiment_c_leakage_test.ipynb`  
**Fecha de ejecución:** 18 de febrero de 2026  
**Tiempo de ejecución:** 181.9 s (~3 min) (7/7 celdas)

---

## 1. Introducción

Este experimento tiene un propósito **educativo y de validación crítica**: demostrar empíricamente cómo las **malas prácticas metodológicas (Data Leakage)** inflan artificialmente las métricas de rendimiento, dando una falsa sensación de seguridad antes de pasar a producción. Constituye evidencia empírica robusta de por qué la integridad metodológica es esencial en la investigación de detección de fraude.

---

## 2. Metodología

Se comparan **dos pipelines** con **Logistic Regression** como modelo base:

### 2.1. Rama Correcta (sin leakage)

- **División temporal:** Train (pasado) vs Test (futuro) con gap de delay.
- **StandardScaler:** Ajustado solo en train, aplicado a test.
- **SMOTE:** Aplicado solo sobre el conjunto de entrenamiento.

### 2.2. Rama Incorrecta (con leakage)

- **División aleatoria:** `train_test_split` mezcla pasado y futuro.
- **StandardScaler:** Ajustado sobre TODOS los datos antes de dividir.
- **SMOTE:** Aplicado antes de la división (genera muestras sintéticas que contaminan test).

| Aspecto | Rama Correcta | Rama Incorrecta |
|---------|---------------|-----------------|
| División de datos | Temporal | Aleatoria |
| Escalado | Solo en train | Global (todo el dataset) |
| Resampling | SMOTE solo en train | SMOTE antes del split |
| Fuentes de leakage | 0 | 3 |

---

## 3. Resultados Obtenidos

| Pipeline | AUC ROC | AUPRC | CP@100 |
|----------|---------|-------|--------|
| **C-Correcta** (temporal + SMOTE en train) | 0.8658 | 0.6115 | 0.29 |
| **C-Incorrecta** (SMOTE global + split aleatorio) | **0.9999** ⚠️ | **0.9999** ⚠️ | N/A |

### 3.1. Comparación con Experimento A (Baseline)

| Experimento | AUC ROC | AUPRC | CP@100 |
|-------------|---------|-------|--------|
| A — Logistic Regression (baseline) | 0.8705 | 0.6057 | 0.2914 |
| A — Random Forest (baseline) | 0.8643 | 0.6634 | 0.2900 |
| A — XGBoost (baseline) | 0.8618 | 0.6389 | 0.2729 |
| **C-Correcta** (LR + SMOTE en train) | 0.8658 | 0.6115 | 0.2900 |
| **C-Incorrecta** (con Leakage) ⚠️ | 0.9999 | 0.9999 | N/A |

### 3.2. Visualización

![Comparación Leakage](../figuras_experimentos/experiment_c_leakage_comparison.png)

**Figura 2.** Comparativa AUPRC/AUC ROC: la pipeline incorrecta muestra métricas "perfectas" (≈1.0) completamente artificiales. Curvas PR de la rama correcta vs incorrecta.

---

## 4. Análisis

1. **Evidencia irrefutable de data leakage:**
   - La pipeline incorrecta obtiene AUPRC = **0.9999** frente a **0.6115** de la correcta → inflación del **63.5%** (artificial).
   - AUC ROC pasa de 0.8658 a 0.9999.

2. **Tres fuentes de contaminación identificadas:**
   - **Escalado global:** Información de la distribución futura filtra al modelo.
   - **SMOTE global:** Puntos de test derivados de train → el modelo "memoriza" el test.
   - **Split aleatorio:** Permite aprender de eventos futuros, imposible en producción.

3. **Consistencia con el Baseline:** La rama correcta (AUPRC=0.611, AUC=0.866) es coherente con Logistic Regression del Exp. A (AUPRC=0.606, AUC=0.871). SMOTE aporta mejora marginal (~+0.9% AUPRC).

4. **Valor para el TFM:** Cualquier trabajo que reporte métricas ≈1.0 en detección de fraude debe examinarse críticamente en busca de data leakage.

---

## 5. Conclusiones

- El data leakage puede inflar la AUPRC de 0.61 a 1.00, haciendo que un modelo parezca "perfecto" cuando no generalizaría en producción.
- Este experimento valida que el pipeline base (Experimento A) es robusto: sus métricas coinciden con la rama correcta, no con la inflada.
- La división temporal y el aislamiento estricto del preprocesamiento al conjunto de entrenamiento son fundamentales.
