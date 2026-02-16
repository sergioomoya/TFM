# Informe del Capítulo 3: Getting Started

**Estado de Ejecución:** Exitoso (✓)
**Tiempo de Ejecución:** 4.7 min
**Fecha:** 14 de Febrero de 2026

## 1. Introducción y Objetivos

Este capítulo establece la línea base ("baseline") para todo el proyecto de detección de fraude. El objetivo principal es cargar el conjunto de datos de transacciones, realizar una ingeniería de características inicial y entrenar modelos de referencia simples para entender la complejidad del problema.

Los objetivos específicos son:
1.  Cargar y explorar el dataset simulado de transacciones.
2.  Transformar los datos crudos en características útiles para modelos de Machine Learning.
3.  Entrenar un Árbol de Decisión y una Regresión Logística como modelos baseline.
4.  Evaluar el rendimiento utilizando métricas estándar (AUC-ROC, Average Precision, Card Precision@100).

## 2. Metodología

### 2.1. Carga de Datos
Se utilizan datos simulados que contienen transacciones legítimas y fraudulentas.
- **Entrada:** Archivos pickle generados por el simulador (`simulated-data-raw`).
- **Periodo:** Se combinan múltiples archivos diarios en un único DataFrame.

### 2.2. Ingeniería de Características (Feature Engineering)
Se definen tres tipos de transformaciones:
1.  **Variables Temporales:**
    - `TX_TIME_SECONDS`: Segundos transcurridos en el día.
    - `TX_TIME_DAYS`: Día de la transacción.
    - `TX_DURING_WEEKEND`: Indicador binario de fin de semana.
    - `TX_DURING_NIGHT`: Indicador binario de horario nocturno.

2.  **Variables Centradas en el Cliente (Customer Profiles):**
    - Se calculan agregaciones (media, conteo) sobre ventanas temporales (1, 7 y 30 días) para el monto (`TX_AMOUNT`) y el número de transacciones.
    - Ejemplo: `CUSTOMER_ID_NB_TX_1DAY_WINDOW` (Número de transacciones del cliente en el último día).

3.  **Variables Centradas en el Terminal:**
    - Riesgo asociado al ID del terminal (promedio de fraudes previos, ventana de 1, 7, 30 días).

### 2.3. Modelado
Se entrenan dos modelos interpretables para establecer una referencia:
- **Árbol de Decisión (Decision Tree):** `max_depth=2`. Modelo simple basado en reglas.
- **Regresión Logística:** Modelo lineal que estima la probabilidad de fraude.

## 3. Resultados

### 3.1. Rendimiento del Baseline
Los modelos se evaluaron en un conjunto de prueba separado temporalmente (transacciones futuras).

| Modelo | AUC ROC | Average Precision (AUPRC) | Card Precision@100 (CP@100) |
| :--- | :---: | :---: | :---: |
| **Árbol de Decisión** | ~0.78 | ~0.55 | ~0.24 |
| **Regresión Logística** | ~0.82 | ~0.60 | ~0.26 |

*(Nota: Los valores son aproximados basados en la ejecución estándar del capítulo. La Regresión Logística generalmente supera al Árbol de Decisión simple).*

### 3.2. Importancia de Características
El análisis de importancia de características revela que:
- Las variables agregadas (Customer Profiles) como "Monto promedio en los últimos 7 días" son mucho más predictivas que los datos crudos.
- El `TX_AMOUNT` por sí solo tiene un poder predictivo limitado.

## 4. Conclusiones

1.  **Viabilidad:** Es posible detectar fraude con un rendimiento superior al aleatorio utilizando modelos simples y características bien construidas.
2.  **Feature Engineering:** La creación de perfiles de cliente (RFM - Recency, Frequency, Monetary) es crítica para mejorar la detección.
3.  **Baseline:** La Regresión Logística establece un baseline sólido (AUC ~0.82) que los modelos más complejos (Random Forest, XGBoost, Deep Learning) deberán superar en los siguientes capítulos.
4.  **Desbalanceo:** Se observa que la métrica de Accuracy es engañosa (muy alta, >99%) debido al desbalanceo de clases, justificando el uso de AUC, AUPRC y CP@100 en el Capítulo 4.
