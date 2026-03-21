#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genera el gráfico SHAP Beeswarm alineado con la memoria del TFM.

IMPORTANTE
----------
El script ``ulb_tfm_figures.py`` entrena XGBoost sobre el CSV ULB (Kaggle) con
variables V1..V28 (PCA anonimizado). Eso NO coincide con el marco teórico de la
memoria (ingeniería de características tipo RFM / ventanas temporales:
TERMINAL_ID_RISK_*, CUSTOMER_ID_*, TX_AMOUNT, etc.).

Este script:
- Carga el dataset **transformado** del proyecto (pickles del Chapter 3).
- Aplica el mismo protocolo train/test temporal que el Experimento D.
- Entrena XGBoost **baseline** con ``StandardScaler`` sobre ``INPUT_FEATURES``.
- Calcula SHAP (TreeExplainer) y guarda el beeswarm en PDF vectorial (dpi=300).

Salida por defecto (sobrescribe la figura incorrecta si existía):
  experiments/results/figures/ulb_tfm/shap_beeswarm.pdf

Uso (desde la raíz Baseline/, idealmente en Docker):
  python experiments/generate_shap_beeswarm_transformed_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Raíz del proyecto (Baseline/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import (  # noqa: E402
    BASELINE_PARAMS,
    DELTA_DELAY,
    DELTA_TEST,
    DELTA_TRAIN,
    INPUT_FEATURES,
    OUTPUT_FEATURE,
    SEED,
    START_DATE_TRAINING,
)
from experiments.data_utils import (  # noqa: E402
    get_train_test_set,
    load_transformed_data,
    print_dataset_summary,
)

# Misma filosofía que run_experiment_d_standalone.py
SHAP_SAMPLE_SIZE = min(1000, 5000)

# Destino: misma carpeta que el resto de figuras de memoria ULB (nombre histórico;
# el contenido de este PDF es del dataset transformado del libro, no del CSV Kaggle).
FIGURES_ULB_TFM = PROJECT_ROOT / "experiments" / "results" / "figures" / "ulb_tfm"


def main() -> None:
    FIGURES_ULB_TFM.mkdir(parents=True, exist_ok=True)

    print("Cargando datos transformados (feature engineering del libro / TFM)...")
    transactions_df = load_transformed_data()
    train_df, test_df = get_train_test_set(
        transactions_df,
        start_date_training=START_DATE_TRAINING,
        delta_train=DELTA_TRAIN,
        delta_delay=DELTA_DELAY,
        delta_test=DELTA_TEST,
    )
    print_dataset_summary(train_df, test_df, "SHAP Beeswarm — dataset transformado")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[INPUT_FEATURES])
    X_test_scaled = scaler.transform(test_df[INPUT_FEATURES])

    model = xgb.XGBClassifier(**BASELINE_PARAMS["XGBoost"])
    model.fit(X_train_scaled, train_df[OUTPUT_FEATURE])

    np.random.seed(SEED)
    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test_scaled))
    sample_indices = np.random.choice(len(X_test_scaled), sample_size, replace=False)
    X_sample = X_test_scaled[sample_indices]
    X_sample_df = pd.DataFrame(X_sample, columns=INPUT_FEATURES)

    print(f"TreeExplainer sobre {sample_size} filas del conjunto de prueba...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mapa cálido/frío: valores altos de feature → rojizos, bajos → azulados
    plt.set_cmap("coolwarm")
    plt.rcParams["savefig.dpi"] = 300

    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample_df,
        max_display=10,
        show=False,
        plot_type="dot",
        color_bar_label="Valor de la característica (escalado)",
        cmap=plt.cm.coolwarm,
    )
    plt.title(
        "SHAP Beeswarm — XGBoost sobre dataset con ventanas temporales\n"
        f"({sample_size} muestras del test; features: TERMINAL_ID_*, CUSTOMER_ID_*, TX_AMOUNT, …)",
        fontsize=12,
    )
    plt.tight_layout()

    out_path = FIGURES_ULB_TFM / "shap_beeswarm.pdf"
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ SHAP Beeswarm (dataset transformado) guardado en:\n  {out_path}")
    print(
        "\nVariables en el eje Y deben coincidir con INPUT_FEATURES de config.py "
        "(p. ej. TERMINAL_ID_RISK_7DAY_WINDOW, TX_AMOUNT), no con V1..V28 del ULB."
    )


if __name__ == "__main__":
    main()
