# Revisión de Resultados del TFM y Propuestas de Mejora

**Proyecto:** TFM — Detección de Fraude en Transacciones con Tarjeta de Crédito  
**Fecha del informe:** 9 de marzo de 2026  
**Alcance:** Revisión global de métricas obtenidas en Capítulos 3–7 y Experimentos A–D; identificación de áreas de mejora y propuestas concretas.

---

## Índice

1. [Contexto y Metodología](#1-contexto-y-metodología)
2. [Tabla Comparativa Global de Resultados](#2-tabla-comparativa-global-de-resultados)
3. [Diagnóstico: Hallazgos Clave](#3-diagnóstico-hallazgos-clave)
4. [Propuestas de Mejora](#4-propuestas-de-mejora)
5. [Matriz Prioridad vs Esfuerzo](#5-matriz-prioridad-vs-esfuerzo)
6. [Conclusiones](#6-conclusiones)

---

## 1. Contexto y Metodología

### 1.1. Métricas utilizadas

El TFM adopta tres métricas principales para evaluar los modelos de detección de fraude:

| Métrica | Tipo | Descripción | Rol en el TFM |
|---|---|---|---|
| **AUC ROC** | Ranking global | Área bajo la curva ROC (TPR vs FPR). Mide la capacidad general de discriminación. | Métrica técnica secundaria |
| **AUPRC (Average Precision)** | Ranking con foco en la clase minoritaria | Área bajo la curva Precision–Recall. Más informativa que AUC ROC en datasets desbalanceados. | **Métrica técnica principal** |
| **CP@100 (Card Precision@100)** | Negocio / operativa | Precisión media diaria entre las 100 tarjetas más sospechosas. Simula la capacidad operativa real de un equipo de investigación. | **Métrica operativa principal** |

### 1.2. Protocolo de evaluación

- **Validación prequential (temporal):** 4 folds con desplazamiento temporal (`DELTA_TRAIN=7`, `DELTA_DELAY=7`, `DELTA_TEST=7` días).
- **Búsqueda de hiperparámetros:** GridSearchCV / RandomizedSearchCV con selección por AUPRC en validación.
- **Reporte:** Media ± desviación estándar sobre los 4 folds.
- **Semilla:** 42 (fijada en todos los modelos y operaciones de splitting).

### 1.3. Fuentes de resultados analizadas

| Fuente | Descripción |
|---|---|
| **Experimento A** | Baseline puro: LR, RF, XGBoost sin técnicas de desbalanceo (validación prequential + GridSearchCV). |
| **Experimento A (Undersampled)** | Variante con submuestreo de legítimas (ratio 10:1 en train, test intacto). |
| **Experimento B** | Cost-Sensitive Learning rediseñado: pesos moderados (B1), calibración de probabilidades (B2), búsqueda ampliada con GPU (B3). |
| **Experimento C** | Test anti-leakage: 5 ramas experimentales × 3 modelos para cuantificar data leakage. |
| **Experimento D** | Interpretabilidad XAI: Feature Importance, SHAP, ablación de features. |
| **Capítulo 7** | Deep Learning: FFNN, CNN 1D, Autoencoder (no supervisado y semi-supervisado), LSTM, LSTM+Attention. |

---

## 2. Tabla Comparativa Global de Resultados

### 2.1. Modelos de Machine Learning Clásico

*Resultados obtenidos con validación prequential (4 folds). Se reporta media ± desviación estándar.*

| Fuente | Modelo | AUC ROC | AUPRC | CP@100 | Observación |
|---|---|---|---|---|---|
| Exp A | Logistic Regression | 0.869 ± 0.016 | 0.635 ± 0.016 | 0.293 ± 0.014 | Baseline lineal |
| Exp A | Random Forest | 0.873 ± 0.011 | 0.685 ± 0.010 | 0.297 ± 0.014 | Mejor AUC ROC del baseline |
| Exp A | **XGBoost** | 0.869 ± 0.009 | **0.690 ± 0.008** | 0.296 ± 0.014 | **Mejor AUPRC global** |
| Exp A (undersamp) | XGBoost (10:1) | 0.868 | 0.659 | 0.279 | Más recall (+4.5 pp), más FP (579 vs 85) |
| Exp B (B1) | RF (200 trees, sin pesos) | 0.876 ± 0.012 | 0.688 ± 0.009 | 0.255 ± 0.010 | El grid selecciona `class_weight=None` |
| Exp B (B2) | **RF calibrado** | **0.877 ± 0.013** | 0.683 ± 0.010 | **0.299 ± 0.015** | **Mejor CP@100 de ML clásico** |
| Exp B (B1) | XGBoost (`scale_pos_weight=3`) | 0.869 ± 0.013 | 0.651 ± 0.010 | 0.251 ± 0.009 | Peso moderado mejora AUC ROC |
| Exp B (B2) | XGBoost calibrado | 0.871 ± 0.014 | 0.652 ± 0.011 | 0.294 ± 0.016 | Calibración sube CP@100 +0.043 |
| Exp B (B3) | XGBoost GPU (rand+reg) | 0.875 ± 0.011 | 0.657 ± 0.012 | 0.228 ± 0.009 | Regularización mejora AUC ROC |
| Exp B (B3) | XGBoost GPU calibrado | 0.873 ± 0.013 | 0.653 ± 0.016 | 0.294 ± 0.013 | Calibración recupera CP@100 |

### 2.2. Modelos de Deep Learning (Capítulo 7)

*Resultados de la selección de modelo con validación prequential. Parámetros: lr/batch_size/epochs/dropout/hidden_layers o seq_len/num_layers/batch_size/hidden_size/dropout.*

| Modelo | AUC ROC Test | AUPRC Test | CP@100 Test | Observación |
|---|---|---|---|---|
| **FFNN** (grid search) | 0.876 ± 0.01 | 0.675 ± 0.01 | **0.303 ± 0.02** | **Mejor CP@100 global**; red densa con early stopping |
| CNN 1D | 0.872 ± 0.01 | 0.599 ± 0.01 | 0.288 ± 0.01 | Peor AUPRC de los modelos supervisados |
| LSTM | 0.876 ± 0.02 | 0.665 ± 0.02 | 0.297 ± 0.01 | Captura secuencialidad; competitiva con FFNN |
| LSTM + Attention | 0.874 ± 0.02 | 0.667 ± 0.01 | 0.295 ± 0.01 | Atención aporta interpretabilidad pero no mejora métricas |
| Autoencoder (no supervisado) | 0.894 ± 0.01 | 0.084 ± 0.02 | 0.236 ± 0.04 | AUC ROC alto pero AUPRC catastrófico |
| Autoencoder (semi-supervisado) | 0.898 ± 0.01 | 0.105 ± 0.02 | 0.232 ± 0.03 | Mejora marginal sobre no supervisado |

### 2.3. Resultados del Experimento C (Integridad Metodológica)

*Demostración cuantitativa del impacto del data leakage.*

| Rama | LR (AUPRC) | RF (AUPRC) | XGB (AUPRC) |
|---|---|---|---|
| **Correcta** (temporal, escalado local, SMOTE local) | 0.583 | 0.612 | 0.616 |
| Leak_split (aleatorio) | 0.615 (+0.03) | 0.677 (+0.07) | 0.691 (+0.08) |
| Leak_scaler (escalado global) | 0.582 (≈0) | 0.607 (≈0) | 0.615 (≈0) |
| Leak_smote (SMOTE global) | 0.590 (+0.01) | 0.904 (+0.29) | 0.746 (+0.13) |
| **Leak_todas** (3 fuentes) | **0.929** ⚠️ | **1.000** ⚠️ | **1.000** ⚠️ |

**Conclusión del Exp C:** La pipeline incorrecta (split aleatorio + escalado global + SMOTE global) infla la AUPRC de ~0.6 a ~1.0. La mayor fuente individual de leakage es SMOTE aplicado antes del split, especialmente para RF (+0.29 pp).

### 2.4. Resultados del Experimento D (Interpretabilidad)

*Top-10 features por ganancia (Gain) del XGBoost baseline.*

| Ranking | Variable | Gain (%) | Interpretación |
|---|---|---|---|
| 1 | `TERMINAL_ID_RISK_7DAY_WINDOW` | **38.8** | Riesgo histórico del terminal (7 días) |
| 2 | `TX_AMOUNT` | **12.9** | Monto de la transacción |
| 3 | `TERMINAL_ID_RISK_30DAY_WINDOW` | 10.1 | Riesgo del terminal (30 días) |
| 4 | `CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW` | 5.6 | Gasto medio del cliente (30 días) |
| 5 | `TERMINAL_ID_RISK_1DAY_WINDOW` | 4.1 | Riesgo del terminal (1 día) |
| 6 | `CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW` | 3.8 | Gasto medio del cliente (7 días) |
| 7 | `TERMINAL_ID_NB_TX_1DAY_WINDOW` | 3.8 | Nº transacciones en terminal (1 día) |
| 8 | `CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW` | 3.0 | Gasto medio del cliente (1 día) |
| 9 | `CUSTOMER_ID_NB_TX_7DAY_WINDOW` | 2.8 | Nº transacciones del cliente (7 días) |
| 10 | `CUSTOMER_ID_NB_TX_30DAY_WINDOW` | 2.7 | Nº transacciones del cliente (30 días) |

**Validación por ablación:** Eliminar la top feature SHAP (`CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW`) y reentrenar degrada la AUPRC en **4.47 pp** (de 0.639 a 0.594), confirmando contribución real.

**Discrepancia Feature Importance vs SHAP:** La variable que más domina por Gain (TERMINAL_ID_RISK_7DAY_WINDOW, 38.8%) no coincide con la de mayor mean |SHAP| (CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW). Esto evidencia que la importancia depende del enfoque de medición — observación metodológica relevante para la memoria del TFM.

---

## 3. Diagnóstico: Hallazgos Clave

### 3.1. El ML clásico supera al Deep Learning en la métrica principal (AUPRC)

| Comparación | Modelo | AUPRC |
|---|---|---|
| **Mejor ML clásico** | XGBoost baseline (Exp A) | **0.690** |
| **Mejor DL** | FFNN grid search (Ch7) | **0.675** |
| **Diferencia** | | **−1.5 pp** |

Los modelos basados en árboles superan consistentemente a las redes neuronales en AUPRC. Esto es coherente con la literatura para **datos tabulares de fraude** (Le Borgne et al., 2022; Grinsztajn et al., 2022 "Why do tree-based models still outperform deep learning on typical tabular data?").

### 3.2. El CP@100 cuenta una historia diferente

| Comparación | Modelo | CP@100 |
|---|---|---|
| **Mejor ML clásico** | RF calibrado (Exp B) | **0.299** |
| **Mejor DL** | FFNN grid search (Ch7) | **0.303** |
| **Diferencia** | | **+0.4 pp a favor de DL** |

La FFNN supera a todos los modelos clásicos en la métrica operativa (CP@100). Esto sugiere que la red neuronal produce un **ranking más preciso en el extremo superior** de la distribución de probabilidades, lo cual es exactamente lo que importa para el equipo de investigación de fraude que revisa las alertas top-100 del día.

### 3.3. Los Autoencoders fracasan estrepitosamente en métricas de precisión

| Modelo | AUC ROC | AUPRC | CP@100 |
|---|---|---|---|
| Autoencoder (no supervisado) | 0.894 | **0.084** | 0.236 |
| Autoencoder (semi-supervisado) | 0.898 | **0.105** | 0.232 |

A pesar de tener el AUC ROC más alto de todo el TFM (~0.90), los autoencoders obtienen una AUPRC 8× peor que el baseline (0.08 vs 0.69). Esto significa que detectan anomalías generales (alto AUC ROC = buena separabilidad global) pero generan una cantidad masiva de falsos positivos al intentar rankear transacciones sospechosas. El error de reconstrucción como score de fraude no es suficientemente discriminativo para la clase minoritaria.

### 3.4. La calibración de probabilidades es una técnica con alto retorno y bajo coste

El Experimento B demuestra que `CalibratedClassifierCV` (regresión isotónica) produce mejoras sustanciales y consistentes:

| Modelo | CP@100 sin calibrar | CP@100 calibrado | Δ |
|---|---|---|---|
| LR | 0.218 | 0.293 | **+0.075** |
| RF | 0.255 | **0.299** | **+0.044** |
| XGBoost | 0.251 | 0.294 | **+0.043** |

La calibración corrige la distorsión de las probabilidades predichas, mejorando el ranking sin modificar el modelo base. **Ningún modelo de Deep Learning ha sido calibrado todavía** — oportunidad de mejora directa.

### 3.5. Solo 15 features básicas están siendo usadas

El Experimento D revela que las 15 features actuales (ingeniería de características del Capítulo 3) son:
- 1 monto: `TX_AMOUNT`
- 2 temporales: `TX_DURING_WEEKEND`, `TX_DURING_NIGHT`
- 6 de cliente: conteo y media de monto por ventana (1, 7, 30 días)
- 6 de terminal: conteo y riesgo por ventana (1, 7, 30 días)

Se trata de perfiles RFM (Recency, Frequency, Monetary) elementales. **No se han explorado features más sofisticadas** como ratios, codificaciones cíclicas, velocidad transaccional ni features de interacción.

### 3.6. El cost-sensitive naive es contraproducente

El Exp B original con `class_weight='balanced'` (ratio ~200:1) **empeoró** todas las métricas frente al baseline:
- LR: AUPRC −0.060
- RF: AUPRC −0.018
- XGBoost: AUPRC −0.024

La causa raíz: pesos excesivos destruyen la calibración de probabilidades. Los pesos moderados (`scale_pos_weight` 3–5 para XGBoost) son preferibles, y para LR y RF el grid seleccionó `class_weight=None` como óptimo. **El desbalanceo no se combate mejor con pesos agresivos, sino con calibración posterior.**

---

## 4. Propuestas de Mejora

### Mejora 1: Calibración de probabilidades para modelos Deep Learning

**Justificación:** El Exp B demuestra mejoras de +4 a +7 pp en CP@100 mediante calibración isotónica. Los modelos DL (FFNN, LSTM, LSTM+Attention) no han sido calibrados. Dado que la FFNN ya obtiene el mejor CP@100 global (0.303) sin calibrar, la calibración podría elevarla a ~0.33-0.35.

**Implementación:**
1. Reservar un subconjunto de validación para calibrar (separado del usado para early stopping).
2. Aplicar `sklearn.calibration.CalibratedClassifierCV` con método isotónico o, alternativamente, calibración Platt (sigmoid).
3. Comparar métricas antes y después de calibración para FFNN, LSTM y LSTM+Attention.

**Impacto esperado:** +2–5 pp en CP@100; mejora potencial en AUPRC.  
**Complejidad:** Baja (pocas líneas de código adicional).

---

### Mejora 2: Ensemble (stacking) de modelos heterogéneos

**Justificación:** Los modelos explotan señales diferentes:
- **XGBoost** trabaja con features tabulares estáticas (mejor AUPRC = 0.690).
- **FFNN** aprende representaciones no lineales de las mismas features (mejor CP@100 = 0.303).
- **LSTM** captura patrones secuenciales en el comportamiento del cliente.

La combinación de modelos con fortalezas complementarias suele superar a cualquier modelo individual (Wolpert, 1992; Breiman, 1996).

**Implementación:**
1. Obtener probabilidades de fraude de los 3 mejores modelos (XGBoost calibrado, FFNN calibrada, LSTM) en el conjunto de validación.
2. Entrenar un meta-learner (Logistic Regression o RF calibrado) que tome como entrada las 3 probabilidades y prediga la etiqueta final.
3. Evaluar el ensemble con validación prequential para evitar data leakage.

**Impacto esperado:** +1–3 pp en AUPRC.  
**Complejidad:** Media.

---

### Mejora 3: Feature Engineering avanzado

**Justificación:** Solo se utilizan 15 features básicas (perfiles RFM). El Exp D demuestra que `TERMINAL_ID_RISK_7DAY_WINDOW` concentra el 38.8% de la ganancia, lo que indica una fuerte dependencia de una sola variable. Más features ricas podrían distribuir mejor la información predictiva y mejorar la robustez del modelo.

**Features propuestas:**

| Feature | Tipo | Descripción |
|---|---|---|
| `TX_HOUR_SIN`, `TX_HOUR_COS` | Cíclica | Codificación cíclica de la hora del día (evita discontinuidad 23h→0h) |
| `TX_DAY_OF_WEEK` | Categórica | Día de la semana (0=lunes, 6=domingo) |
| `TX_AMOUNT_RATIO_1D` | Ratio | TX_AMOUNT / CUSTOMER_AVG_AMOUNT_1DAY. Cuántas veces supera el gasto habitual reciente. |
| `TX_AMOUNT_RATIO_7D` | Ratio | TX_AMOUNT / CUSTOMER_AVG_AMOUNT_7DAY. Ídem para ventana semanal. |
| `TX_AMOUNT_RATIO_30D` | Ratio | TX_AMOUNT / CUSTOMER_AVG_AMOUNT_30DAY. Ídem para ventana mensual. |
| `TX_AMOUNT_ZSCORE` | Estadística | (TX_AMOUNT − mean) / std del cliente. Desviación estandarizada del monto. |
| `TX_VELOCITY` | Temporal | Tiempo (en segundos) desde la transacción anterior del mismo cliente. Transacciones muy rápidas son sospechosas. |
| `TERMINAL_UNIQUE_CUSTOMERS_1D` | Agregada | Nº de clientes únicos en el terminal en las últimas 24h. Diversidad alta puede indicar terminal comprometido. |
| `CUSTOMER_UNIQUE_TERMINALS_1D` | Agregada | Nº de terminales únicos usados por el cliente en las últimas 24h. Dispersión geográfica anómala. |
| `TERMINAL_AMOUNT_STD_7D` | Variabilidad | Desviación estándar de los montos en el terminal en 7 días. Alta variabilidad puede indicar fraude. |

**Impacto esperado:** +2–5 pp en AUPRC.  
**Complejidad:** Media (requiere modificar el pipeline de feature engineering del Capítulo 3).

---

### Mejora 4: LightGBM / CatBoost como alternativas a XGBoost

**Justificación:** El TFM solo evalúa XGBoost como modelo de boosting. Existen alternativas modernas con ventajas potenciales:
- **LightGBM** (Ke et al., 2017): Histograma-based, más rápido y a menudo superior en datos tabulares. Soporta `is_unbalance` y `scale_pos_weight`.
- **CatBoost** (Prokhorenkova et al., 2018): Manejo nativo de features categóricas (CUSTOMER_ID, TERMINAL_ID), menos overfitting por defecto (ordered boosting).

**Implementación:**
1. Añadir LightGBM y CatBoost al pipeline del Experimento A.
2. Incluir en la búsqueda de hiperparámetros con validación prequential.
3. Aplicar calibración de probabilidades (B2) al mejor modelo.
4. Comparar con los resultados existentes.

**Impacto esperado:** +0.5–2 pp en AUPRC.  
**Complejidad:** Baja (misma estructura de pipeline, solo cambia el estimador).

---

### Mejora 5: Optimización de hiperparámetros con Bayesian/Optuna

**Justificación:** Actualmente se usa `GridSearchCV` (exhaustivo, lento) y `RandomizedSearchCV` (60 iteraciones, aleatorio). La búsqueda bayesiana (Tree-structured Parzen Estimator, TPE) converge a mejores hiperparámetros en menos iteraciones al aprender de evaluaciones previas.

**Implementación:**
1. Integrar **Optuna** (Akiba et al., 2019) en el pipeline de experimentación.
2. Definir el espacio de búsqueda para cada modelo (XGBoost, RF, FFNN, LSTM).
3. Usar `study.optimize()` con `n_trials=100-200` y `pruning` (early stopping de trials malos).
4. Mantener la validación prequential como evaluación interna.

**Impacto esperado:** +1–2 pp en múltiples métricas.  
**Complejidad:** Media (requiere refactorización del pipeline de búsqueda).

---

### Mejora 6: Arquitectura Transformer para secuencias

**Justificación:** La LSTM+Attention captura secuencialidad (AUPRC=0.667) pero no supera a la FFNN simple (0.675). Los Transformers (Vaswani et al., 2017), con self-attention multi-head, pueden capturar dependencias temporales más complejas sin las limitaciones de las RNN (vanishing gradients, procesamiento secuencial).

Referencia relevante: "TabTransformer" (Huang et al., 2020) y "FT-Transformer" (Gorishniy et al., 2021) para datos tabulares.

**Implementación:**
1. Implementar un módulo `FraudTransformer` con:
   - Embedding de posición para la secuencia temporal.
   - 2–4 capas de self-attention multi-head.
   - Capa feed-forward final para clasificación binaria.
2. Integrar en el pipeline de validación prequential del Capítulo 7.
3. Comparar con LSTM+Attention en las mismas condiciones.

**Impacto esperado:** +1–3 pp en métricas de DL.  
**Complejidad:** Alta (nueva arquitectura, requiere tuning cuidadoso).

---

### Mejora 7: Focal Loss para desbalanceo en Deep Learning

**Justificación:** Los modelos DL actuales usan `BCELoss` estándar, que trata todas las muestras por igual. Con un ratio de fraude de ~0.84%, la inmensa mayoría de los gradientes provienen de ejemplos legítimos (fáciles). **Focal Loss** (Lin et al., 2017) reduce automáticamente el peso de los ejemplos fáciles y focaliza el aprendizaje en los difíciles:

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Con $\gamma > 0$, los ejemplos bien clasificados (alta $p_t$) reciben menos peso. Esto es equivalente a un cost-sensitive automático y adaptativo.

**Implementación:**
1. Implementar `FocalLoss` como módulo PyTorch.
2. Sustituir `BCELoss` en el entrenamiento de FFNN, LSTM y LSTM+Attention.
3. Buscar $\alpha$ (peso de la clase positiva, rango 0.25–0.75) y $\gamma$ (focusing, rango 1–5) mediante validación.

**Impacto esperado:** +1–2 pp en AUPRC.  
**Complejidad:** Baja (pocas líneas de código).

---

### Mejora 8: Arreglar los Autoencoders

**Justificación:** Los autoencoders actuales obtienen AUPRC ~0.08-0.11, lo que los hace inútiles como sistema de detección autónomo. El fallo radica en que el error de reconstrucción no discrimina bien entre fraudes y anomalías legítimas (transacciones genuinas con características inusuales).

**Propuestas:**

| Variante | Descripción | Beneficio esperado |
|---|---|---|
| **VAE** (Variational Autoencoder) | Regulariza el espacio latente con KL-divergence; genera distribución continua. | Mejor generalización en detección de anomalías |
| **Denoising Autoencoder** | Añadir ruido gaussiano o dropout a las entradas; reconstruir la entrada limpia. | Representaciones más robustas |
| **Entrenamiento solo con legítimas** | Entrenar el autoencoder exclusivamente con transacciones legítimas del train set. | Mayor error de reconstrucción en fraudes (la "normalidad" está mejor definida) |
| **Score combinado** | Usar el error de reconstrucción como feature adicional de un modelo supervisado (XGBoost o FFNN). | Combina detección de anomalías con supervisión (enfoque semi-supervisado mejorado) |

**Impacto esperado:** AUPRC de ~0.08 → 0.20-0.40 (mejora sustancial pero difícilmente competitiva con modelos supervisados).  
**Complejidad:** Alta (requiere reestructurar el pipeline de autoencoders y experimentar con múltiples variantes).

---

## 5. Matriz Prioridad vs Esfuerzo

| # | Mejora | Impacto esperado (AUPRC/CP@100) | Complejidad | Prioridad |
|---|---|---|---|---|
| 1 | **Calibración de modelos DL** | +2–5 pp CP@100 | Baja | **Muy Alta** |
| 7 | **Focal Loss en DL** | +1–2 pp AUPRC | Baja | **Alta** |
| 4 | **LightGBM / CatBoost** | +0.5–2 pp AUPRC | Baja | **Alta** |
| 3 | **Feature Engineering avanzado** | +2–5 pp AUPRC | Media | **Alta** |
| 2 | **Ensemble / Stacking** | +1–3 pp AUPRC | Media | **Alta** |
| 5 | **Optuna para hiperparámetros** | +1–2 pp global | Media | Media |
| 6 | **Transformer para secuencias** | +1–3 pp DL | Alta | Media |
| 8 | **Arreglar Autoencoders** | AUPRC 0.08→0.20+ | Alta | Baja |

**Recomendación de orden de ejecución:**
1. **Fase 1 (quick wins):** Mejoras 1, 7 y 4 — bajo esfuerzo, alto retorno.
2. **Fase 2 (investigación media):** Mejoras 3 y 2 — mayor impacto potencial, requieren diseño.
3. **Fase 3 (investigación avanzada):** Mejoras 5 y 6 — optimización fina y nuevas arquitecturas.
4. **Fase 4 (opcional):** Mejora 8 — interesante académicamente pero impacto limitado frente a modelos supervisados.

---

## 6. Conclusiones

### 6.1. Estado actual de los resultados

Los resultados del TFM son **sólidos y metodológicamente correctos**. La validación prequential, la demostración de data leakage (Exp C) y el análisis de interpretabilidad (Exp D) aportan valor académico significativo. Los modelos de ML clásico (AUPRC ~0.69) y Deep Learning (AUPRC ~0.68, CP@100 ~0.30) representan un baseline competente para datos simulados de fraude.

### 6.2. Margen de mejora

Existe un margen de mejora estimado de **3–10 pp en AUPRC** y **5–10 pp en CP@100** combinando las técnicas propuestas. Las tres palancas principales son:
1. **Calibración de probabilidades** (aplicable a todos los modelos, especialmente DL).
2. **Feature Engineering** (15 features actuales son insuficientes para capturar toda la señal disponible).
3. **Ensemble de modelos heterogéneos** (combinar las fortalezas de ML clásico y DL).

### 6.3. Relación con la literatura del TFM

Las mejoras propuestas están alineadas con la literatura de referencia:
- Calibración: Niculescu-Mizil & Caruana (2005), Platt (1999).
- Ensemble: Wolpert (1992), Le Borgne et al. (2022).
- Focal Loss: Lin et al. (2017), adaptado a detección de fraude.
- Transformers tabulares: Gorishniy et al. (2021), Huang et al. (2020).
- Feature Engineering en fraude: Carcillo et al. (2018), Le Borgne et al. (2022).

Cada mejora implementada con éxito constituiría una **aportación adicional** del TFM respecto al repositorio de referencia del libro, fortaleciendo la sección de "Diferencias y Aportaciones".
