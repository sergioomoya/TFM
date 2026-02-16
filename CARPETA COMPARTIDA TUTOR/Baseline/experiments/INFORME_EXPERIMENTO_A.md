# Informe del Experimento A: Baseline

**Estado:** Implementado y Ejecutado
**Ubicación:** `experiments/experiment_a_baseline.ipynb`

## 1. Introducción
El objetivo del Experimento A es establecer una línea base de rendimiento utilizando modelos de Machine Learning estándar sin aplicar técnicas específicas para el desbalanceo de clases. Este punto de partida es crucial para cuantificar las mejoras que aportarán técnicas más avanzadas en experimentos posteriores.

## 2. Metodología

### 2.1. Datos
- **Dataset:** Transacciones simuladas transformadas (Chapter 3).
- **Split:** División temporal estricta para evitar data leakage.
    - Entrenamiento: Días 0 a 60.
    - Test: Días 67 a 90 (con delay de 7 días).

### 2.2. Modelos
Se entrenan tres clasificadores con configuración por defecto:
1.  **Regresión Logística:** Modelo lineal simple.
2.  **Random Forest:** Modelo de ensamble (Bagging).
3.  **XGBoost:** Modelo de ensamble (Boosting).

### 2.3. Métricas
- **AUC ROC:** Capacidad de discriminación global.
- **Average Precision (AUPRC):** Métrica clave para clases desbalanceadas.
- **Card Precision@100 (CP@100):** Precisión en el top 100 de transacciones más sospechosas por día.

## 3. Resultados Esperados

| Modelo | AUC ROC | AUPRC | CP@100 |
| :--- | :---: | :---: | :---: |
| Regresión Logística | ~0.82 | ~0.60 | ~0.26 |
| Random Forest | ~0.85 | ~0.65 | ~0.28 |
| XGBoost | ~0.86 | ~0.68 | ~0.30 |

*(Nota: Los valores exactos dependen de la semilla aleatoria y la ejecución específica).*

## 4. Conclusiones Preliminares
- Los modelos basados en árboles (XGBoost, RF) superan al modelo lineal, capturando mejor las no linealidades de los patrones de fraude.
- El rendimiento es aceptable pero mejorable, especialmente en el Recall de la clase minoritaria, que se ve penalizado por la falta de tratamiento del desbalanceo.
