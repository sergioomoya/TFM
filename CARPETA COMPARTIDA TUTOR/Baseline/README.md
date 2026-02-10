# TFM - Detección de Fraude en Transacciones con Tarjeta de Crédito

Este repositorio contiene el código fuente y los experimentos para el Trabajo de Fin de Máster (TFM) sobre detección de fraude. El proyecto parte del código base del libro *"Reproducible Machine Learning for Credit Card Fraud Detection"* y lo evoluciona hacia una arquitectura robusta, dockerizada y orientada a MLOps.

## 📋 Descripción del Proyecto

El objetivo es desarrollar y validar técnicas de Machine Learning para la detección de fraude en un entorno controlado y reproducible. El proyecto se estructura en dos grandes bloques:

1.  **Baseline Educativo (Capítulos 3-7):** Adaptación de los cuadernos originales del libro, unificados y automatizados.
2.  **Experimentos de Investigación (Carpeta `experiments/`):** Implementaciones propias para validar hipótesis específicas (Baseline, Data Leakage, Interpretabilidad).

## 🚀 Inicio Rápido (Quick Start)

Este proyecto sigue la filosofía **"Docker First"**. No necesitas instalar Python ni librerías en tu máquina local, solo Docker.

### Prerrequisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo.
- (Opcional) Drivers de NVIDIA si deseas usar aceleración GPU.

### Ejecución de Experimentos
Para ejecutar la batería de experimentos (A, C, D) y generar el reporte automáticamente:

```bash
docker compose run --rm experiments
```

Los resultados se guardarán en `experiments/results/` y el informe en `experiments/INFORME_RESULTADOS_EXPERIMENTOS.md`.

### Ejecución de Cuadernos Unificados (Generación de Datos)
Si es la primera vez que ejecutas el proyecto, necesitas generar los datos simulados:

```bash
docker compose run --rm unified-notebooks
```

### Entorno Interactivo (JupyterLab)
Para explorar los cuadernos, editar código o visualizar gráficos interactivamente:

```bash
docker compose up jupyter
```
Accede a [http://localhost:8888](http://localhost:8888) (el token vendrá vacío por configuración).

### Entrenamiento con GPU (Deep Learning)
Para ejecutar el entrenamiento de redes neuronales (Capítulo 7) usando tu GPU NVIDIA:

```bash
docker compose run --rm ch7-gpu
```

---

## 📂 Estructura del Proyecto

```text
.
├── Chapter_X_.../             # Cuadernos originales del libro (adaptados)
├── experiments/               # FRAMEWORK DE EXPERIMENTACIÓN (Aportación TFM)
│   ├── config.py              # Configuración centralizada
│   ├── data_utils.py          # Funciones de carga y métricas
│   ├── run_experiment.py      # Script orquestador
│   ├── experiment_a_...ipynb  # Baseline
│   ├── experiment_c_...ipynb  # Test de Data Leakage
│   └── experiment_d_...ipynb  # Interpretabilidad (XAI)
├── sprints/                   # Documentación de gestión del proyecto (Agile)
├── DIFERENCIAS_Y_APORTACIONES.md # Resumen de cambios respecto al repo original
├── Dockerfile                 # Definición de imagen CPU
├── Dockerfile.gpu             # Definición de imagen GPU (NVIDIA)
├── docker-compose.yml         # Orquestación de servicios
└── requirements.txt           # Dependencias pineadas
```

## 🛠️ Desarrollo y Contribución

Consulta el archivo [CONTRIBUTING.md](CONTRIBUTING.md) para conocer las normas de estilo, flujo de trabajo con Git y gestión de ramas.

## 📊 Resultados

Los resultados detallados de la última ejecución se encuentran en:
- [INFORME_RESULTADOS_EXPERIMENTOS.md](experiments/INFORME_RESULTADOS_EXPERIMENTOS.md)
- [DIFERENCIAS_Y_APORTACIONES.md](DIFERENCIAS_Y_APORTACIONES.md)

## ⚖️ Licencia

Este proyecto se basa en el material de *Fraud Detection Handbook* (Le Borgne et al., 2022).
