# Informe del Capítulo 4: Performance Metrics

**Estado de Ejecución:** Exitoso (✓)
**Tiempo de Ejecución:** 0.3 min
**Fecha:** 14 de Febrero de 2026

## 1. Introducción y Objetivos

En la detección de fraude, evaluar correctamente los modelos es tan importante como construirlos. Debido al extremo desbalanceo de clases (los fraudes son <1% de las transacciones), las métricas tradicionales como la "Exactitud" (Accuracy) son inútiles e incluso engañosas.

Este capítulo tiene como objetivo definir, implementar y comparar las métricas de rendimiento más relevantes para sistemas de detección de fraude en entornos reales.

## 2. Metodología

Se analizan tres familias de métricas utilizando las predicciones generadas por los modelos del Capítulo 3:

### 2.1. Métricas Basadas en Umbral (Threshold-Based)
Requieren definir un punto de corte (umbral) en la probabilidad predicha para clasificar una transacción como fraude (1) o legítima (0).
- **Matriz de Confusión:** TP (True Positives), FP (False Positives), TN, FN.
- **Precision (Precisión):** De los que predije como fraude, ¿cuántos lo son realmente? ($TP / (TP+FP)$).
- **Recall (Sensibilidad):** De todos los fraudes reales, ¿cuántos detecté? ($TP / (TP+FN)$).
- **F1-Score:** Media armónica entre Precision y Recall.

### 2.2. Métricas Libres de Umbral (Threshold-Free)
Evalúan la capacidad del modelo a través de todos los umbrales posibles.
- **Curva ROC (Receiver Operating Characteristic):** TPR vs FPR.
- **AUC ROC:** Área bajo la curva ROC. Mide la capacidad de ranking global.
- **Curva Precision-Recall (PR):** Precision vs Recall. Más informativa que la ROC en datasets desbalanceados.
- **Average Precision (AP):** Área bajo la curva PR.

### 2.3. Métricas de Negocio (Top-K)
Diseñadas para reflejar la capacidad operativa real de los equipos de investigación de fraude, que tienen recursos limitados (tiempo/personal).
- **Card Precision@k (CP@k):** Precisión calculada considerando solo las $k$ transacciones más sospechosas de cada día.
- **Interpretación:** Si los investigadores solo pueden revisar 100 alertas al día, ¿cuántas de ellas son realmente fraudes?

## 3. Resultados y Comparativa

### 3.1. La Falacia del Accuracy
Se demuestra que un clasificador "tonto" que predice siempre "No Fraude" obtiene un Accuracy del 99.x%, pero un Recall de 0%. Esto confirma que el Accuracy no debe usarse.

### 3.2. ROC vs Precision-Recall
- **ROC:** Tiende a ser optimista. Un AUC de 0.95 puede parecer excelente.
- **PR:** Muestra una realidad más dura. Un modelo con AUC ROC de 0.95 podría tener un Average Precision de solo 0.60, indicando que genera muchos falsos positivos para alcanzar un alto recall.

### 3.3. Card Precision@100 (CP@100)
Esta métrica se establece como el estándar de oro (Gold Standard) para este proyecto.
- Permite comparar modelos en función de su utilidad diaria.
- Es robusta frente a cambios en la distribución de clases a lo largo del tiempo.
- Se implementa la función `card_precision_top_k` que calcula la precisión media diaria en el top $k$.

## 4. Conclusiones

1.  **Métrica Principal:** Se adopta **Card Precision@100 (CP@100)** como la métrica principal para la toma de decisiones y comparación de modelos en los siguientes capítulos.
2.  **Métricas Secundarias:** **AUC ROC** y **Average Precision** se mantendrán como métricas técnicas para monitorear la estabilidad y capacidad de ranking.
3.  **Impacto Operativo:** Optimizar CP@100 impacta directamente en la eficiencia del equipo de fraude, maximizando los fraudes detectados por hora de investigación invertida.
