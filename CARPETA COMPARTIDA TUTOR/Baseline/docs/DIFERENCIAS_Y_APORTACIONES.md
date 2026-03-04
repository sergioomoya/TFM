# Diferencias con el Repositorio de Referencia y Aportaciones del TFM

Este documento detalla las diferencias técnicas, arquitectónicas y metodológicas entre el repositorio de referencia (basado en el libro *"Reproducible Machine Learning for Credit Card Fraud Detection"*) y la implementación actual desarrollada para el Trabajo de Fin de Máster (TFM).

El objetivo principal de estas modificaciones ha sido transformar una colección de cuadernos con fines educativos en un **entorno de experimentación robusto, reproducible y escalable**.

---

## 1. Infraestructura y Entorno (MLOps)

La mayor diferencia estructural es la adopción de prácticas de **DevOps y MLOps** para garantizar que los resultados sean reproducibles independientemente de la máquina anfitriona.

| Característica | Repositorio de Referencia | Repositorio TFM (Actual) |
| :--- | :--- | :--- |
| **Entorno de Ejecución** | Local (depende del OS y Python del usuario). | **Dockerizado**. Uso de contenedores aislados (`Dockerfile`). |
| **Gestión de Servicios** | Ejecución manual de Jupyter Notebook. | **Docker Compose**. Orquestación de servicios (Jupyter, Experimentos, GPU). |
| **Dependencias** | Lista de librerías (a veces desactualizada). | `requirements.txt` con **versiones pineadas** para garantizar compatibilidad histórica (Python 3.8). |
| **Aceleración Hardware** | No configurada explícitamente. | **Soporte GPU NVIDIA** nativo mediante `Dockerfile.gpu` para entrenamiento de Deep Learning (Capítulo 7). |

### Aportación Clave:
Se ha eliminado el problema *"funciona en mi máquina"*. Cualquier usuario con Docker instalado puede replicar los experimentos con un solo comando (`docker compose up`), garantizando el mismo entorno de sistema operativo y librerías.

---

## 2. Arquitectura del Software

Se ha refactorizado el código para separar la lógica de negocio de la presentación, pasando de un enfoque de "Scripting" a uno de "Ingeniería de Software".

### 2.1. Modularización
- **Antes:** La lógica (carga de datos, métricas) se repetía o definía dentro de cada notebook.
- **Ahora:**
    - `experiments/config.py`: Centralización de rutas y constantes.
    - `experiments/data_utils.py`: Funciones reutilizables para carga, transformación y métricas (ej. `card_precision_top_k`).
    - `run_unified_notebooks.py`: Script de automatización que permite ejecutar los capítulos del libro (3 al 7) de forma desatendida, con gestión de timeouts y errores.

### 2.2. Sistema de Experimentos Automatizados
Se ha creado un framework propio para la ejecución de experimentos (`experiments/run_experiment.py`) que permite:
1.  Ejecutar experimentos específicos (A, C, D) por línea de comandos.
2.  Capturar métricas automáticamente.
3.  Generar reportes en JSON y Markdown (`INFORME_RESULTADOS_EXPERIMENTOS.md`) sin intervención humana.

---

## 3. Metodología Experimental

Mientras que el repositorio de referencia se centra en *explicar* conceptos capítulo a capítulo, este repositorio se centra en *validar* hipótesis mediante experimentos aislados y controlados.

### Experimentos Implementados
Se han diseñado ramas y notebooks específicos para validar escenarios concretos:

1.  **Experimento A (Baseline):**
    - Establecimiento de una línea base "pura" sin técnicas de balanceo.
    - **Aportación:** Integración de la métrica de negocio `Card Precision@100` (CP@100) en el flujo de evaluación estándar.
    - **Metodología Capítulo 5:** Validación prequential (4 folds), GridSearchCV con búsqueda de hiperparámetros, reporte de media ± desviación estándar.
    - **Variante A-Undersampled:** Submuestreo de transacciones legítimas en train/valid (ratio configurable, p. ej. 10:1) para reducir el desbalance; test intacto. Script: `run_experiment_a_undersampled.py`. Ver `INFORME_EXPERIMENTO_A_UNDERSAMPLED.md`.

2.  **Experimento B (Cost-Sensitive — Rediseñado):**
    - **Diagnóstico:** El enfoque original (`class_weight='balanced'`, ratio ~200:1) distorsionaba las probabilidades y empeoraba AUPRC/CP@100 frente al baseline.
    - **Rediseño con tres sub-variantes:**
      - **B1 — Pesos moderados:** `class_weight={0:1, 1:w}` con w ∈ {5,10,20} (LR, RF) y `scale_pos_weight` ∈ {1,3,5,10,20} (XGBoost). Preserva calibración de probabilidades.
      - **B2 — Calibración de probabilidades:** `CalibratedClassifierCV` (isotonic regression) sobre los mejores modelos B1 para mejorar la calidad del ranking.
      - **B3 — Búsqueda ampliada (XGBoost GPU):** `RandomizedSearchCV` con 60 iteraciones sobre espacio de ~2.6M combinaciones (max_depth, min_child_weight, subsample, gamma, regularización). Aprovechamiento óptimo de GPU.
    - **Optimización GPU:** De ~1,296 fits exhaustivos a ~480 aleatorizados; early stopping; regularización explícita.
    - Script: `run_experiment_b_standalone.py`.

3.  **Experimento C (Test de Data Leakage):**
    - **Aportación:** Demostración empírica del peligro del filtrado de información.
    - **Refactorizado:** Cinco ramas (Correcta, Leak_split, Leak_scaler, Leak_smote, Leak_todas) para desglosar el impacto por fuente. Tres modelos (LR, RF, XGBoost). Parámetros SMOTE en `config.SMOTE_PARAMS`.

4.  **Experimento D (Interpretabilidad):**
    - **Aportación:** Uso de **SHAP** (SHapley Additive exPlanations) sobre el modelo XGBoost.
    - Permite explicar no solo qué transacciones son fraude, sino *por qué* lo son, aportando valor al analista de fraude.
    - **Mejoras (CRITICA_MEJORA_EXPERIMENTOS):** Modelo XGBoost baseline (mejor AUPRC); Feature Importance Gain/weight/cover; tabla mean \|SHAP\|; Force plots con contexto; Beeswarm 1000 muestras; Dependence plots.

---

## 4. Control de Versiones y Flujo de Trabajo

Se ha impuesto un flujo de trabajo profesional basado en **Git Flow**:

- **Ramas por Experimento:** Cada experimento (A, C, D) se desarrolló en ramas aisladas (`exp/a-baseline`, `exp/c-leakage-test`, etc.) antes de integrarse en `main`.
- **Ignorado de Artefactos:** Configuración estricta de `.gitignore` para evitar subir datos simulados pesados o modelos binarios (`.pkl`), manteniendo el repositorio ligero.
- **Persistencia de Resultados:** Los resultados de las ejecuciones (gráficos, métricas) se guardan en carpetas locales pero no se versionan, salvo los informes finales en Markdown.

---

## 5. Correcciones Técnicas Realizadas

Durante la implementación se resolvieron varios problemas de obsolescencia del código original:

1.  **Compatibilidad Python 3.8:** Ajuste de versiones de `typing-extensions` y `scikit-learn`.
2.  **Rutas de Datos:** Corrección de las rutas relativas para la carga de datos simulados (`simulated-data-transformed`), que en el original asumían una estructura de carpetas plana.
3.  **Lectura de Archivos:** Implementación de soporte para lectura de archivos `.docx` y `.xlsx` para procesar la documentación del proyecto automáticamente.
4.  **Timeouts:** Gestión de tiempos de espera extendidos para el entrenamiento de redes neuronales (Capítulo 7), que fallaban en la ejecución estándar por defecto.
