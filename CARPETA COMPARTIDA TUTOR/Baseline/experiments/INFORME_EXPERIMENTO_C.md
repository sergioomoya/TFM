# Informe del Experimento C: Prueba de Leakage (Fuga de Datos)

**Estado:** Implementado y Ejecutado
**Ubicación:** `experiments/experiment_c_leakage_test.ipynb`

## 1. Introducción
Este experimento tiene un propósito educativo y de validación crítica: demostrar empíricamente cómo las malas prácticas metodológicas (Data Leakage) inflan artificialmente las métricas de rendimiento, dando una falsa sensación de seguridad antes de pasar a producción.

## 2. Metodología

Se comparan dos pipelines de entrenamiento:

### 2.1. Rama Incorrecta (Con Leakage)
Simula errores comunes en ciencia de datos:
1.  **Split Aleatorio:** Se mezclan datos del futuro con el pasado para entrenar y testear.
2.  **Transformación Global:** Se calculan estadísticas (media, desviación estándar) para la normalización usando *todo* el dataset antes de dividirlo.
3.  **Resultado:** El modelo "ve" el futuro indirectamente.

### 2.2. Rama Correcta (Sin Leakage)
Sigue las mejores prácticas para series temporales:
1.  **Split Temporal:** Entrenamiento (Pasado) vs Test (Futuro) con gap de seguridad.
2.  **Transformación Aislada:** El escalador (`StandardScaler`) se ajusta (`fit`) solo en el conjunto de entrenamiento y se aplica (`transform`) al test.

## 3. Resultados Comparativos

| Métrica | Rama Incorrecta (Inflada) | Rama Correcta (Realista) | Diferencia |
| :--- | :---: | :---: | :---: |
| **AUC ROC** | **0.99+** | ~0.86 | -0.13 |
| **AUPRC** | **0.95+** | ~0.68 | -0.27 |
| **CP@100** | **0.90+** | ~0.30 | -0.60 |

## 4. Análisis y Conclusiones
- **Peligro Silencioso:** La rama incorrecta produce métricas casi perfectas. Si esto se llevara a producción, el rendimiento real colapsaría, causando pérdidas financieras no previstas.
- **Validación del Pipeline:** Este experimento confirma que nuestro pipeline base (Experimento A) es robusto y honesto, ya que sus métricas coinciden con la "Rama Correcta" y no con la inflada.
- **Origen del Leakage:** En fraude, el leakage más grave suele ser el uso de variables agregadas (ej. "número de transacciones hoy") calculadas sobre todo el dataset antes del split, filtrando información de fraudes futuros al presente.
