#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuración compartida para todos los experimentos del TFM.

Centraliza constantes, rutas, semillas y parámetros comunes para
garantizar reproducibilidad y consistencia entre experimentos.
"""

import os
import sys
import datetime
from pathlib import Path

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

# Raíz del proyecto (Baseline/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directorio de datos simulados (generados por Chapter_3, clonados en Chapter_4+)
# Los datos transformados están dentro de un subdirectorio 'data/' del repo clonado
DATA_DIR_RAW = PROJECT_ROOT / "Chapter_3_GettingStarted" / "simulated-data-raw" / "data"

# Buscar datos transformados en orden de prioridad
_POSSIBLE_TRANSFORMED_DIRS = [
    PROJECT_ROOT / "Chapter_3_GettingStarted" / "simulated-data-transformed" / "data",
    PROJECT_ROOT / "Chapter_3_GettingStarted" / "simulated-data-transformed",
    PROJECT_ROOT / "Chapter_4_PerformanceMetrics" / "simulated-data-transformed" / "data",
    PROJECT_ROOT / "Chapter_5_ModelValidationAndSelection" / "simulated-data-transformed" / "data",
    PROJECT_ROOT / "Chapter_6_ImbalancedLearning" / "simulated-data-transformed" / "data",
]

DATA_DIR_TRANSFORMED = None
for _dir in _POSSIBLE_TRANSFORMED_DIRS:
    if _dir.exists():
        DATA_DIR_TRANSFORMED = _dir
        break

if DATA_DIR_TRANSFORMED is None:
    # Fallback: usar la ruta estándar (se generará al ejecutar Chapter 3)
    DATA_DIR_TRANSFORMED = PROJECT_ROOT / "Chapter_3_GettingStarted" / "simulated-data-transformed"

# Directorio de resultados de los experimentos
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Directorio de figuras generadas
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Añadir el directorio del baseline al path para importar shared_functions
BASELINE_CHAPTER3_DIR = str(PROJECT_ROOT / "Chapter_3_GettingStarted")
if BASELINE_CHAPTER3_DIR not in sys.path:
    sys.path.insert(0, BASELINE_CHAPTER3_DIR)

# =============================================================================
# REPRODUCIBILIDAD
# =============================================================================

SEED = 42

# =============================================================================
# FECHAS PARA DIVISIÓN TEMPORAL (PROTOCOLO DEL LIBRO)
# =============================================================================

# El dataset simulado cubre del 2018-04-01 al 2018-09-30 (183 días)
# Protocolo de validación prequential del libro (Chapter 5):
#   - delta_train: días de entrenamiento
#   - delta_delay: días de retraso (fraude reportado)
#   - delta_test:  días de evaluación

DELTA_TRAIN = 7     # 7 días de entrenamiento
DELTA_DELAY = 7     # 7 días de delay (reporte de fraude)
DELTA_TEST = 7      # 7 días de test

# Fecha de inicio para validación prequential
START_DATE_TRAINING = datetime.datetime(2018, 7, 25)

# Fechas para la validación (split temporal)
# Estas fechas siguen el protocolo del Chapter 5 del libro
START_DATE_TRAINING_FOR_VALID = datetime.datetime(2018, 8, 1)
START_DATE_TRAINING_FOR_TEST = datetime.datetime(2018, 8, 22)

# Número de folds para validación prequential
N_FOLDS = 4

# =============================================================================
# FEATURES (VARIABLES)
# =============================================================================

# Variables de entrada base (generadas por el feature engineering del Chapter 3)
OUTPUT_FEATURE = "TX_FRAUD"

# Features disponibles tras la transformación del Chapter 3:
# - TX_AMOUNT: monto de la transacción (se escala con StandardScaler)
# - TX_DURING_WEEKEND: indicador de fin de semana
# - TX_DURING_NIGHT: indicador de transacción nocturna
# - CUSTOMER_ID_NB_TX_*_DAY_WINDOW: número de transacciones del cliente en ventana
# - CUSTOMER_ID_AVG_AMOUNT_*_DAY_WINDOW: monto promedio del cliente en ventana
# - TERMINAL_ID_NB_TX_*_DAY_WINDOW: número de transacciones del terminal en ventana
# - TERMINAL_ID_RISK_*_DAY_WINDOW: riesgo de fraude del terminal en ventana

INPUT_FEATURES = [
    'TX_AMOUNT',
    'TX_DURING_WEEKEND',
    'TX_DURING_NIGHT',
    'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
    'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
    'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
    'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
    'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
    'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW',
    'TERMINAL_ID_NB_TX_1DAY_WINDOW',
    'TERMINAL_ID_RISK_1DAY_WINDOW',
    'TERMINAL_ID_NB_TX_7DAY_WINDOW',
    'TERMINAL_ID_RISK_7DAY_WINDOW',
    'TERMINAL_ID_NB_TX_30DAY_WINDOW',
    'TERMINAL_ID_RISK_30DAY_WINDOW',
]

# =============================================================================
# MODELOS - HIPERPARÁMETROS POR DEFECTO
# =============================================================================

# Experiment A: Parámetros por defecto (sin ajuste para desbalance)
BASELINE_PARAMS = {
    "Logistic Regression": {
        "C": 1.0,  # Regularización por defecto
        "max_iter": 1000,
        "random_state": SEED,
    },
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": SEED,
        "n_jobs": -1,
    },
    "XGBoost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "random_state": SEED,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "n_jobs": -1,
    },
}

# Experiment B: Parámetros con ponderación de clases (cost-sensitive)
# Se calculará scale_pos_weight dinámicamente basado en el ratio de clases
COST_SENSITIVE_PARAMS = {
    "Logistic Regression": {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": SEED,
    },
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": None,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
    },
    "XGBoost": {
        # scale_pos_weight se calcula como: n_negative / n_positive
        # Se asignará dinámicamente en el notebook
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "random_state": SEED,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "n_jobs": -1,
    },
}

# =============================================================================
# MÉTRICAS
# =============================================================================

# Métricas principales del TFM
PERFORMANCE_METRICS_LIST = ['AUC ROC', 'Average precision', 'Card Precision@100']
PERFORMANCE_METRICS_GRID = ['roc_auc', 'average_precision']
TOP_K_LIST = [100]

# =============================================================================
# VISUALIZACIÓN
# =============================================================================

# Estilo de gráficas
PLOT_STYLE = 'darkgrid'
FIGURE_DPI = 150
FIGURE_FORMAT = 'png'

# Paleta de colores para los experimentos
COLORS = {
    "baseline": "#2F4D7E",          # Azul oscuro
    "cost_sensitive": "#CA8035",     # Naranja
    "correct_pipeline": "#008000",   # Verde
    "incorrect_pipeline": "#CC0000", # Rojo
    "shap_positive": "#FF0051",      # Rojo SHAP
    "shap_negative": "#008BFB",      # Azul SHAP
}
