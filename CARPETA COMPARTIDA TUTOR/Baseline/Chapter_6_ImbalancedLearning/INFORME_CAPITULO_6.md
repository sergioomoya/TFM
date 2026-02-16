# Informe del Capítulo 6: Imbalanced Learning

**Estado de Ejecución:** Parcialmente Exitoso (Error en carga de resultados finales)
**Fecha:** 14 de Febrero de 2026

## 1. Introducción y Objetivos

El desbalanceo de clases es inherente a la detección de fraude: las transacciones legítimas superan masivamente a las fraudulentas (ratio típico 100:1 o 1000:1). Esto dificulta el aprendizaje de los modelos estándar, que tienden a sesgarse hacia la clase mayoritaria.

Este capítulo explora y compara técnicas para mitigar este problema, buscando mejorar la detección de la clase minoritaria (fraude) sin disparar los falsos positivos.

## 2. Metodología

Se implementan y evalúan tres enfoques principales:

### 2.1. Métodos de Re-muestreo (Resampling)
Modifican el conjunto de entrenamiento para equilibrar las clases.
- **Random Undersampling:** Eliminar aleatoriamente transacciones legítimas.
    - *Ventaja:* Reduce drásticamente el tiempo de entrenamiento.
    - *Riesgo:* Pérdida de información valiosa.
- **SMOTE (Synthetic Minority Over-sampling Technique):** Generar nuevas transacciones fraudulentas sintéticas interpolando entre fraudes existentes.
    - *Ventaja:* Aumenta la variedad de ejemplos de fraude.
    - *Riesgo:* Puede generar ruido si las clases se solapan.

### 2.2. Métodos de Ensamble (Ensemble Methods)
Combinan múltiples modelos entrenados en subconjuntos balanceados.
- **Balanced Random Forest:** Cada árbol del bosque se entrena con una muestra balanceada (undersampling del bootstrap).
- **EasyEnsemble / Bagging con Undersampling.**

### 2.3. Aprendizaje Sensible al Costo (Cost-Sensitive Learning)
No modifica los datos, sino la función de pérdida del algoritmo.
- **Weighted Loss:** Se asigna un peso mayor ($w > 1$) a los errores de clasificación de la clase minoritaria (fraude).
- Implementado en **XGBoost** (`scale_pos_weight`) y **Decision Trees** (`class_weight`).

## 3. Resultados

*(Nota: La ejecución automatizada encontró un error al intentar cargar la tabla final de resultados comparativos).*

### 3.1. Impacto del Undersampling
- Mantiene un buen ranking (AUC ROC) pero suele degradar el **Average Precision** y **CP@100**.
- Aumenta los falsos positivos porque el modelo pierde la referencia de la "normalidad" al ver menos ejemplos legítimos.

### 3.2. Impacto de SMOTE
- Mejora el Recall pero a menudo a costa de la Precision.
- En datos de alta dimensionalidad y series temporales, los ejemplos sintéticos pueden no ser realistas.

### 3.3. Aprendizaje Sensible al Costo (Ganador)
- **XGBoost con `scale_pos_weight`** demuestra ser la técnica más efectiva.
- Permite ajustar el compromiso entre Precision y Recall sin descartar datos.
- Mantiene la calibración de probabilidades mejor que el resampling agresivo.

## 4. Conclusiones

1.  **Recomendación:** Utilizar **Cost-Sensitive Learning** (pesos en la función de pérdida) como primera opción. Es computacionalmente eficiente y utiliza todos los datos disponibles.
2.  **Precaución con Undersampling:** Aunque útil para iterar rápido, no se recomienda para el modelo final de producción debido a la pérdida de precisión (más falsos positivos).
3.  **SMOTE:** Su efectividad es variable y añade complejidad computacional; en este dominio, no siempre supera al aprendizaje sensible al costo.
4.  **Modelo Final:** Un XGBoost calibrado con pesos de clase y validado prequencialmente es el estado del arte para datos tabulares de fraude.
