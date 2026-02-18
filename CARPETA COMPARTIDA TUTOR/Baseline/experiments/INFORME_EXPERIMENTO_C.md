# Informe del Experimento C: Validación de Integridad Metodológica (Anti-Leakage Test)

**Estado:** Refactorizado, ejecutado y documentado  
**Ubicación:** `experiments/experiment_c_leakage_test.ipynb`, `experiments/run_experiment_c_standalone.py`  
**Ejecución:** `docker compose run --rm experiments python experiments/run_experiment_c_standalone.py`

---

## 1. Introducción

Este experimento demuestra empíricamente cómo el **data leakage** infla las métricas y **cuantifica el impacto de cada fuente por separado**, replicando el análisis con **tres modelos** (LR, RF, XGBoost) para comprobar que el efecto se mantiene.

---

## 2. Metodología

### 2.1. Modelos

- **Logistic Regression** (max_iter=1000)
- **Random Forest** (n_estimators=100)
- **XGBoost** (n_estimators=100)

### 2.2. Cinco ramas experimentales

| Rama | Split | Escalado | SMOTE | Fuentes de leakage |
|------|-------|----------|-------|--------------------|
| **Correcta** | Temporal | Solo train | Solo train | 0 |
| **Leak_split** | Aleatorio | Solo train | Solo train | 1 (split) |
| **Leak_scaler** | Temporal | Global (train+test) | Solo train | 1 (escalado) |
| **Leak_smote** | Temporal | Solo train | Global (train+test) | 1 (SMOTE) |
| **Leak_todas** | Aleatorio | Global | Global | 3 |

### 2.3. Parámetros SMOTE (config.py)

- `k_neighbors=5` — Vecinos para interpolación (imblearn default)
- `sampling_strategy='auto'` — Balanceo a la clase minoritaria
- `random_state=SEED` — Reproducibilidad

### 2.4. Métricas

- **AUC ROC**, **AUPRC**, **Card Precision@100** (donde aplica; Leak_todas no tiene estructura temporal para CP@100)

---

## 3. Resultados

### 3.1. Resumen: Correcta vs Leak_todas por modelo

| Modelo | C-Correcta AUC | C-Correcta AUPRC | C-Correcta CP@100 | C-Leak_todas AUC | C-Leak_todas AUPRC |
|--------|----------------|------------------|-------------------|------------------|--------------------|
| Logistic Regression | 0.8692 | 0.5830 | 0.2886 | 0.8979 | **0.9287** ⚠️ |
| Random Forest | 0.8658 | 0.6115 | 0.2900 | **0.9999** ⚠️ | **0.9999** ⚠️ |
| XGBoost | 0.8601 | 0.6163 | 0.2743 | **0.9992** ⚠️ | **0.9995** ⚠️ |

### 3.2. Desglose: impacto incremental por fuente de leakage (AUPRC)

| Modelo | Correcta | Leak_split | Leak_scaler | Leak_smote | Leak_todas |
|--------|----------|------------|-------------|------------|------------|
| Logistic Regression | 0.5830 | +0.032 (0.615) | −0.001 (0.582) | +0.007 (0.590) | **+0.346** (0.929) ⚠️ |
| Random Forest | 0.6115 | +0.066 (0.677) | −0.004 (0.607) | +0.292 (0.904) | **+0.389** (1.000) ⚠️ |
| XGBoost | 0.6163 | +0.075 (0.691) | −0.001 (0.615) | +0.130 (0.746) | **+0.383** (0.999) ⚠️ |

*Observación:* El **split aleatorio** aporta inflación moderada (+0.03 a +0.08). El **escalado global** apenas afecta. El **SMOTE global** infla mucho en RF (+0.29) y algo en XGBoost (+0.13). Las **tres fuentes juntas** provocan AUPRC ≈1.0.

---

## 4. Visualización

![Comparación Leakage](results/figures/experiment_c_leakage_comparison.png)

**Figura.** AUPRC por rama y modelo. El desglose permite identificar qué fuente de leakage contribuye más a la inflación.

---

## 5. Conclusiones

- **Generalización:** El efecto del leakage se evalúa con LR, RF y XGBoost para comprobar que el patrón es consistente.
- **Desglose por fuente:** Cuantifica el impacto de split aleatorio, escalado global y SMOTE global por separado.
- **Parámetros documentados:** SMOTE queda explícitamente configurado en `config.SMOTE_PARAMS`.
- **Valor para el TFM:** Evidencia empírica robusta de por qué la integridad metodológica es esencial; métricas ≈1.0 deben examinarse críticamente.
