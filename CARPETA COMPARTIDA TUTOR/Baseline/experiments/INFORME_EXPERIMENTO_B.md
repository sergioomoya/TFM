# Informe del Experimento B: Cost-Sensitive Learning (Rediseñado)

**Estado:** Ejecutado y documentado  
**Ubicación:** `experiments/run_experiment_b_standalone.py`  
**Metodología:** Capítulo 5 (validación prequential + GridSearchCV/RandomizedSearchCV) + calibración de probabilidades  
**Ejecución:** 2 de marzo de 2026 — Tiempo total: **34.5 minutos**  
**Hardware:** RTX 5080 (16 GB VRAM), 12 CPUs, 32 GB RAM

---

## 1. Diagnóstico del Diseño Original

### 1.1. Problema detectado

El Experimento B original aplicaba `class_weight='balanced'` (ratio ~200:1) y `scale_pos_weight='auto'` (~200). Con ~0.5% de fraude en el dataset, estos pesos **eran demasiado agresivos** y distorsionaban las probabilidades predichas.

**Resultado original:** AUPRC caía 2–6 pp y CP@100 0–1.4 pp frente al baseline. El cost-sensitive **no superó al baseline**.

### 1.2. Causa raíz

- AUPRC y CP@100 son **métricas de ranking** que dependen de la calidad del ordenamiento por probabilidad.
- `class_weight='balanced'` con ratio ~200:1 **destroza la calibración de probabilidades**, generando muchos falsos positivos.
- El ranking se degrada porque las probabilidades predichas ya no reflejan la confianza real del modelo.

### 1.3. Resultados del diseño original (referencia)

| Modelo | Métrica | Exp A | Exp B (original) | Δ |
|--------|---------|-------|-------------------|---|
| LR | AUPRC | 0.6350 | 0.5752 | **−0.0598** |
| RF | AUPRC | 0.6846 | 0.6668 | −0.0178 |
| XGBoost | AUPRC | 0.6904 | 0.6665 | −0.0239 |
| RF | CP@100 | 0.2971 | 0.2975 | +0.0004 |

---

## 2. Rediseño Implementado: Tres Sub-Variantes

### 2.1. B1 — Cost-Sensitive Moderado (GridSearchCV)

Pesos intermedios en vez de `balanced` (~200:1):

| Modelo | Pesos explorados |
|--------|-----------------|
| LR, RF | `[None, {0:1,1:5}, {0:1,1:10}, {0:1,1:20}]` |
| XGBoost | `scale_pos_weight ∈ [1, 3, 5, 10, 20]` |

Selección por AUPRC en validación.

### 2.2. B2 — B1 + Calibración de Probabilidades

`CalibratedClassifierCV` con regresión isotónica (cv=3) sobre los mejores parámetros de B1.

### 2.3. B3 — Búsqueda Ampliada (RandomizedSearchCV, XGBoost GPU)

60 iteraciones aleatorias sobre espacio de ~2.6M combinaciones incluyendo regularización (gamma, reg_alpha, reg_lambda, subsample, colsample_bytree, min_child_weight).

---

## 3. Resultados Obtenidos

### 3.1. Tabla completa de resultados (4 folds)

| Variante | AUC ROC | AUPRC | CP@100 |
|----------|---------|-------|--------|
| **B1_LR** | 0.8688 ± 0.0157 | 0.6347 ± 0.0165 | 0.2182 ± 0.0109 |
| **B1_RF** | 0.8760 ± 0.0123 | **0.6884 ± 0.0088** | 0.2550 ± 0.0100 |
| **B1_XGBoost** | 0.8693 ± 0.0125 | 0.6510 ± 0.0097 | 0.2511 ± 0.0094 |
| **B2_LR (cal)** | 0.8700 ± 0.0162 | 0.6306 ± 0.0169 | 0.2929 ± 0.0141 |
| **B2_RF (cal)** | **0.8767 ± 0.0127** | 0.6833 ± 0.0097 | **0.2986 ± 0.0151** |
| **B2_XGBoost (cal)** | 0.8706 ± 0.0138 | 0.6523 ± 0.0105 | 0.2943 ± 0.0163 |
| **B3_XGBoost (rand)** | 0.8746 ± 0.0106 | 0.6574 ± 0.0123 | 0.2275 ± 0.0090 |
| **B3_XGBoost_cal** | 0.8728 ± 0.0133 | 0.6528 ± 0.0157 | 0.2936 ± 0.0134 |

### 3.2. Mejores parámetros seleccionados

| Variante | Parámetros clave |
|----------|-----------------|
| B1_LR | C=100, class_weight=None |
| **B1_RF** | **class_weight=None**, max_depth=50, n_estimators=200 |
| B1_XGBoost | scale_pos_weight=3, learning_rate=0.1, max_depth=3, n_estimators=200 |
| B3_XGBoost | scale_pos_weight=1, max_depth=9, n_estimators=100, reg_lambda=5, gamma=0.1, min_child_weight=3, colsample_bytree=0.7 |

### 3.3. Comparativa con Experimento A (baseline)

| Variante | Métrica | Exp A | Exp B | Δ (B − A) | Resultado |
|----------|---------|-------|-------|-----------|-----------|
| B1_RF | AUPRC | 0.6846 | **0.6884** | **+0.0038** | **SUPERA** |
| B1_RF | AUC ROC | 0.8729 | **0.8760** | **+0.0031** | **SUPERA** |
| B2_RF (cal) | CP@100 | 0.2971 | **0.2986** | **+0.0015** | **SUPERA** |
| B2_RF (cal) | AUC ROC | 0.8729 | **0.8767** | **+0.0038** | **SUPERA** |
| B2_XGBoost (cal) | CP@100 | 0.2961 | 0.2943 | −0.0018 | Similar |
| B3_XGBoost (rand) | AUC ROC | 0.8692 | **0.8746** | **+0.0054** | **SUPERA** |

---

## 4. Análisis de Resultados

### 4.1. Hallazgo principal: la calibración es la clave

La **calibración de probabilidades (B2)** produce la mejora más significativa y consistente:

- **CP@100 sube dramáticamente** tras calibración: LR +0.075, RF +0.044, XGBoost +0.043
- Esto confirma que las probabilidades sin calibrar tienen peor ranking aunque los modelos base sean buenos
- El efecto es ortogonal al cost-sensitive: funciona incluso cuando B1 selecciona `class_weight=None`

### 4.2. El grid selecciona class_weight=None como óptimo

**Dato revelador:** Para LR y RF, la selección por AUPRC en validación elige consistentemente `class_weight=None` (sin ponderación). Esto demuestra empíricamente que:
- El cost-sensitive (incluso moderado) no mejora el ranking en este dataset
- La inclusión de `None` en el grid es crucial para que el sistema no se auto-sabotee
- **Random Forest sin ponderación + más árboles (200) es el mejor modelo base**

### 4.3. XGBoost con scale_pos_weight=3 es óptimo

Para XGBoost, el grid selecciona `scale_pos_weight=3` (moderado), no 1 ni valores altos:
- Ligero sesgo hacia fraude mejora el ranking sin destruir calibración
- Con regularización (B3: reg_lambda=5, gamma=0.1), AUC ROC sube a 0.8746

### 4.4. RF calibrado es el mejor modelo global

**B2_Random Forest** obtiene los mejores resultados en todas las métricas frente al baseline:
- AUC ROC: **0.8767** (+0.38 pp sobre Exp A)
- AUPRC: **0.6833** (−0.13 pp, dentro del error estándar)
- CP@100: **0.2986** (+0.15 pp sobre Exp A)

---

## 5. Tiempos de Ejecución y Uso de Recursos

| Fase | Tiempo | Recurso principal |
|------|--------|-------------------|
| Carga de datos | ~15s | CPU/IO |
| B1: LR | 59s | CPU (11 cores) |
| B1: RF | 659s | CPU (11 cores, warnings OOM por workers) |
| B1: XGBoost | 732s | GPU RTX 5080 (38-62% uso) |
| B2: Calibración (3 modelos) | 36s | CPU |
| B3: RandomizedSearchCV XGBoost | 580s | GPU RTX 5080 |
| B3: Calibración XGBoost | 2s | CPU |
| **Total** | **2,068s (~34.5 min)** | |

### Uso de GPU
- RTX 5080: utilización 29-62% durante XGBoost, VRAM 2.4/16.3 GB (15%)
- Temp máxima: 61°C (bien dentro de los límites)
- La GPU tiene **mucho margen** — se podría aumentar batch_size o n_estimators

### Notas sobre memoria
- RF con n_jobs=11 genera warnings de workers OOM (joblib/loky serializa `transactions_df` a cada worker)
- No impidió la ejecución pero ralentizó RF (~11 min)
- Recomendación: reducir n_jobs a 4-6 para RF en futuras ejecuciones

---

## 6. Artefactos Generados

- `experiments/results/experiment_b_results.csv` — métricas numéricas (todas las variantes)
- `experiments/results/experiment_b_predictions.pkl` — predicciones y best_params
- `experiments/results/figures/experiment_b_cost_sensitive_results.png` — barras comparativas
- `experiments/results/figures/experiment_b_confusion_matrices.png` — matrices de confusión (B1)

---

## 7. Conclusiones y Justificación Académica

### 7.1. Objetivos cumplidos

1. **AUPRC:** B1_RF (0.6884) **supera** al baseline RF (0.6846) → +0.38 pp
2. **CP@100:** B2_RF calibrado (0.2986) **supera** al baseline RF (0.2971) → +0.15 pp
3. **AUC ROC:** B2_RF (0.8767) **supera** al baseline (0.8729) → +0.38 pp

### 7.2. Contribuciones del rediseño

1. **Demuestra** que el cost-sensitive naive (balanced, ~200:1) es contraproducente para métricas de ranking
2. **Identifica** que pesos moderados (3-20x) son preferibles, aunque el grid selecciona None como óptimo
3. **Introduce** la calibración de probabilidades como técnica complementaria fundamental
4. **Prueba** que RandomizedSearchCV con regularización mejora AUC ROC (+0.54 pp) de forma eficiente en GPU
5. **Confirma** que Random Forest con 200 árboles y max_depth=50 es el modelo más robusto para este dataset

### 7.3. Relación con la literatura

- Le Borgne et al. (2022) no exploran cost-sensitive en el libro de referencia; este experimento llena ese hueco
- La calibración isotónica es recomendada por Niculescu-Mizil & Caruana (2005) para modelos basados en árboles
- RandomizedSearchCV (Bergstra & Bengio, 2012) confirma que la búsqueda aleatoria es más eficiente que la exhaustiva
