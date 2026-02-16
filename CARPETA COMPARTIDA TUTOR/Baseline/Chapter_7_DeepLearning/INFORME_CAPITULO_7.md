# Informe del Capítulo 7: Deep Learning for Fraud Detection

**Estado de Ejecución:** En Progreso (Entrenamiento intensivo en GPU)
**Fecha:** 14 de Febrero de 2026

## 1. Introducción y Objetivos

El aprendizaje profundo (Deep Learning) ha revolucionado campos como la visión por computador y el procesamiento del lenguaje natural (NLP). Este capítulo investiga su aplicación en la detección de fraude, tratando las transacciones no como eventos aislados, sino como **secuencias temporales** (similar a una frase en un texto).

El objetivo es capturar patrones secuenciales complejos que los modelos tradicionales (basados en variables agregadas estáticas) podrían perder.

## 2. Metodología

### 2.1. Representación de Datos
- **Secuencias:** Se transforman los datos tabulares en secuencias de longitud fija (ej. 5 transacciones consecutivas).
- **Input Shape:** `(Batch_Size, Sequence_Length, Num_Features)`.

### 2.2. Arquitecturas Modeladas
Se implementan y comparan tres arquitecturas de redes neuronales utilizando **PyTorch**:

1.  **CNN (Convolutional Neural Network) 1D:**
    - Inspirada en procesamiento de audio/texto.
    - Utiliza filtros convolucionales unidimensionales para detectar patrones locales en la secuencia temporal.
    - Es rápida de entrenar y efectiva para dependencias a corto plazo.

2.  **LSTM (Long Short-Term Memory):**
    - Red Neuronal Recurrente (RNN) diseñada para capturar dependencias a largo plazo.
    - Mantiene un "estado oculto" que evoluciona con cada transacción de la secuencia.
    - Ideal para modelar la evolución del comportamiento del cliente.

3.  **LSTM con Mecanismo de Atención (Attention):**
    - Incorpora un mecanismo que permite a la red "enfocarse" en las transacciones más relevantes del pasado para clasificar la actual.
    - Mejora la interpretabilidad (sabemos qué transacción pasada disparó la alerta) y el rendimiento en secuencias largas.

## 3. Resultados Esperados

*(Basado en la literatura y experimentos previos del libro)*

### 3.1. Rendimiento Comparativo
- **CNN:** Suele tener un rendimiento inferior a los modelos de boosting (XGBoost) en datos tabulares, pero es muy eficiente.
- **LSTM:** Competitiva con XGBoost. Capaz de igualar o superar ligeramente en métricas de precisión (AUPRC).
- **Attention:** Proporciona una ligera mejora sobre LSTM estándar y añade explicabilidad.

### 3.2. Coste Computacional
- Los modelos de Deep Learning requieren tiempos de entrenamiento significativamente mayores (horas vs minutos) y hardware especializado (GPU) para ser viables.
- En este proyecto, se utiliza una **GPU NVIDIA** para acelerar este proceso.

## 4. Conclusiones Preliminares

1.  **Complejidad vs Beneficio:** Para datos tabulares de fraude, los modelos de árboles (XGBoost/LightGBM) siguen siendo el "estado del arte" en relación costo-beneficio. El Deep Learning es una alternativa válida pero costosa.
2.  **Potencial:** El verdadero valor del Deep Learning surge cuando se tienen volúmenes masivos de datos (millones de transacciones) o cuando se integran datos no estructurados (texto de conceptos, geolocalización, grafos).
3.  **Secuencialidad:** Modelar la secuencia explícitamente (LSTM) aporta valor, confirmando que el orden de las transacciones contiene información de riesgo.
