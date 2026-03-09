# Informe de Ejecución de Cuadernos Unificados - TFM Detección de Fraude

**Fecha:** 14 de Febrero de 2026
**Autor:** Sergio Moya
**Estado:** En Ejecución

## 1. Introducción

Este informe detalla el contenido y los resultados de la ejecución de los cuadernos unificados del proyecto "Fraud Detection Handbook". El objetivo es validar el correcto funcionamiento de todos los componentes del sistema, desde la carga y transformación de datos hasta el entrenamiento de modelos avanzados de Deep Learning.

La ejecución se ha realizado en un entorno contenerizado (Docker) para garantizar la reproducibilidad, utilizando aceleración por GPU (NVIDIA) para los modelos de Deep Learning del Capítulo 7.

## 2. Resumen de Ejecución

| Capítulo | Notebook | Estado | Tiempo de Ejecución | Notas |
|----------|----------|--------|---------------------|-------|
| 3 | `Chapter_3_GettingStarted` | *Pendiente* | - | Baseline y Transformación de datos |
| 4 | `Chapter_4_PerformanceMetrics` | *Pendiente* | - | Definición de métricas (CP@k) |
| 5 | `Chapter_5_ModelValidationAndSelection` | *Pendiente* | - | Estrategias de validación temporal |
| 6 | `Chapter_6_ImbalancedLearning` | *Pendiente* | - | Técnicas para datos desbalanceados |
| 7 | `Chapter_7_DeepLearning` | *Pendiente* | - | CNN, LSTM, Attention (GPU) |

---

## 3. Detalle por Capítulo

### Capítulo 3: Getting Started
**Objetivo:** Establecer la línea base del proyecto.
**Contenido:**
- Carga del dataset de transacciones simuladas.
- Transformación de características (Feature Engineering):
    - Variables temporales (día, hora).
    - Variables de comportamiento del cliente (promedios, conteos).
- Entrenamiento de modelos simples: Regresión Logística, Árboles de Decisión.
- Evaluación inicial.

### Capítulo 4: Performance Metrics
**Objetivo:** Definir y calcular métricas relevantes para la detección de fraude.
**Contenido:**
- Limitaciones de la precisión (Accuracy) en datos desbalanceados.
- Curvas ROC y Precision-Recall.
- **Card Precision@k (CP@k):** Métrica de negocio clave que mide la precisión en las top-k transacciones más sospechosas por día.

### Capítulo 5: Model Validation and Selection
**Objetivo:** Implementar estrategias de validación robustas para series temporales.
**Contenido:**
- División Train/Test/Delay.
- **Validación Prequencial (Prequential Split):** Evaluación secuencial que simula el entorno de producción real, reentrenando o evaluando periódicamente.
- Selección de hiperparámetros.

### Capítulo 6: Imbalanced Learning
**Objetivo:** Abordar el problema del desbalanceo de clases (pocos fraudes vs muchas transacciones legítimas).
**Contenido:**
- **Undersampling:** Reducir la clase mayoritaria.
- **Oversampling (SMOTE):** Generar fraudes sintéticos.
- **Cost-Sensitive Learning:** Asignar pesos mayores a la clase minoritaria en la función de pérdida (XGBoost, Random Forest).
- Comparación de técnicas.

### Capítulo 7: Deep Learning
**Objetivo:** Aplicar modelos de aprendizaje profundo a secuencias de transacciones.
**Contenido:**
- Preparación de datos secuenciales (ventanas temporales).
- **CNN (Convolutional Neural Networks):** Convoluciones 1D para detectar patrones locales en la secuencia.
- **LSTM (Long Short-Term Memory):** Redes recurrentes para capturar dependencias temporales a largo plazo.
- **Attention Mechanism:** Mecanismo para enfocar el modelo en las transacciones más relevantes de la secuencia histórica.
- Comparación de rendimiento con modelos tradicionales (XGBoost).

## 4. Resultados Preliminares

*Esta sección se actualizará automáticamente al finalizar la ejecución de los scripts.*

Se espera obtener:
1.  **Archivos de rendimiento (`performances_*.pkl`):** Dataframes con las métricas de todos los modelos probados.
2.  **Figuras:** Gráficos comparativos de AUC, AUPRC y CP@100.
3.  **Logs de ejecución:** Tiempos de entrenamiento y posibles errores.

## 5. Conclusiones

La ejecución unificada permite verificar la integridad del código y la consistencia de los resultados a través de todos los capítulos. El uso de GPU en el Capítulo 7 es crítico para reducir los tiempos de entrenamiento de las redes neuronales recurrentes.

---
*Nota: Los resultados detallados se adjuntaran en el archivo `logs/execution_results/unified_report_*.txt` y `*.json` generados por el script de automatizacion.*
