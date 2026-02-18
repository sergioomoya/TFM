# Informe del Experimento A: Baseline Puro

**Estado:** Implementado, ejecutado y documentado  
**Ubicación:** `experiments/experiment_a_baseline.ipynb`  
**Fecha de ejecución:** 18 de febrero de 2026  
**Tiempo de ejecución:** 16.1 s (8/8 celdas)

---

## 1. Introducción

El objetivo del Experimento A es establecer una **línea base de rendimiento** utilizando modelos de Machine Learning estándar **sin aplicar técnicas específicas para el desbalanceo de clases** (sin `class_weight`, sin SMOTE). Este punto de partida es crucial para cuantificar las mejoras que aportarán técnicas más avanzadas en experimentos posteriores y para evidenciar la **paradoja del desbalance**: métricas engañosas como la Accuracy frente a métricas apropiadas como AUPRC y CP@100.

---

## 2. Metodología

### 2.1. Datos

- **Dataset:** Transacciones simuladas transformadas (Capítulo 3 del *Fraud Detection Handbook*).
- **Features:** 15 variables de ingeniería de características: `TX_AMOUNT`, `TX_DURING_WEEKEND`, `TX_DURING_NIGHT`, ventanas de 1/7/30 días para cliente y terminal.
- **Split temporal estricto** para evitar data leakage:
  - **Entrenamiento:** 7 días (configuración prequential).
  - **Test:** 7 días posteriores (con ventana de delay de 7 días).
  - Fechas según protocolo: `START_DATE_TRAINING`, `DELTA_TRAIN=7`, `DELTA_DELAY=7`, `DELTA_TEST=7`.

### 2.2. Modelos

Se entrenan tres clasificadores con **configuración por defecto** (sin ajustes de desbalance):

1. **Regresión Logística:** `max_iter=1000`, sin `class_weight`.
2. **Random Forest:** 100 estimadores, `max_depth=None`, sin `class_weight`.
3. **XGBoost:** 100 estimadores, `eval_metric='logloss'`, sin `scale_pos_weight`.

Todos usan un pipeline `StandardScaler` + clasificador, con división temporal estricta.

### 2.3. Métricas

- **AUC ROC:** Capacidad de discriminación global.
- **Average Precision (AUPRC):** Métrica prioritaria para clases desbalanceadas.
- **Card Precision@100 (CP@100):** Precisión en el top 100 transacciones más sospechosas por día (protocolo del libro).
- **Accuracy, Recall, Precision, F1-Score:** Para evidenciar la paradoja del desbalance.

---

## 3. Resultados Obtenidos

| Modelo | AUC ROC | AUPRC | CP@100 | Accuracy | Recall (Fraude) | Precision | F1-Score |
|--------|---------|-------|--------|----------|-----------------|------------|----------|
| **Logistic Regression** | **0.8705** | 0.6057 | **0.2914** | 0.9962 | 0.4675 | 0.9045 | 0.6164 |
| **Random Forest** | 0.8643 | **0.6634** | **0.2900** | 0.9966 | 0.5065 | **0.9653** | 0.6644 |
| **XGBoost** | 0.8618 | 0.6389 | 0.2729 | **0.9968** | **0.5403** | 0.9455 | **0.6876** |

### 3.1. Visualización

![Resultados Experimento A](../figuras_experimentos/experiment_a_baseline_results.png)

**Figura 1.** Panel izquierdo: Curva Precision-Recall (Random Forest con AP=0.663 es superior). Panel central: Curvas ROC (similares ~0.86-0.87). Panel derecho: Paradoja del desbalance — Accuracy >99.6% pero Recall del fraude &lt;55%.

---

## 4. Análisis

1. **AUC ROC:** Los tres modelos muestran rendimiento muy similar (~0.86). Logistic Regression tiene ligera ventaja (0.8705).
2. **AUPRC (métrica prioritaria):** Random Forest destaca con **0.6634** (9.5% superior a Logistic Regression).
3. **CP@100:** Valores cercanos (~0.27–0.29). De cada 100 tarjetas más sospechosas por día, ~29 están realmente comprometidas.
4. **Paradoja del desbalance:** A pesar de Accuracy &gt;99.6%, el Recall para fraude es bajo (46.7%–54.0%). La Accuracy es engañosa en este contexto.
5. **XGBoost** ofrece el mejor F1-Score (0.6876) y el mayor Recall (54.0%).

---

## 5. Conclusiones

- Los modelos basados en árboles (XGBoost, RF) ofrecen mejor AUPRC y balance general que la Regresión Logística.
- **Random Forest** es el modelo baseline más sólido según AUPRC.
- La metodología temporal y el pipeline sin leakage garantizan resultados honestos para comparación con experimentos posteriores.
