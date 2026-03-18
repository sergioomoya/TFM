#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generación de figuras de alta calidad para la memoria del TFM
usando el dataset de la ULB (Kaggle) de fraude con tarjeta de crédito.

Requisitos:
- Librerías: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, shap
- Dataset: fichero CSV con las columnas V1..V28, Time, Amount y Class.

Salidas (formato vectorial, dpi=300):
- desbalance_clases.svg
- roc_vs_pr.svg
- shap_beeswarm.pdf
- shap_force_plot.svg

Convención de colores (estricta):
- Azul  -> transacciones legítimas (Class = 0)
- Rojo  -> transacciones fraudulentas (Class = 1)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from xgboost import XGBClassifier
import shap


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

# Directorio raíz del proyecto (Baseline/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ruta al fichero CSV descargado de Kaggle dentro del proyecto
# Según la estructura indicada por el usuario:
# C:\Programacion\GitHub\TFM\CARPETA COMPARTIDA TUTOR\Baseline\dataset\creditcard.csv
DATA_PATH = PROJECT_ROOT / "dataset" / "creditcard.csv"

# Directorio de resultados de experimentos (ya utilizado por el framework)
EXPERIMENTS_RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
EXPERIMENTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Subdirectorio específico para estas figuras de la ULB
FIGURES_DIR = EXPERIMENTS_RESULTS_DIR / "figures" / "ulb_tfm"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Estilo gráfico general
sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300

# Semántica de colores
COLOR_LEGIT = "#1f77b4"  # Azul para Class 0
COLOR_FRAUD = "#d62728"  # Rojo para Class 1

# Semilla para reproducibilidad
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Mapa de color global para SHAP y matplotlib (valores altos en rojo, bajos en azul)
plt.set_cmap("coolwarm")


# =============================================================================
# CARGA DEL DATASET
# =============================================================================

def load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Carga el dataset de la ULB desde un CSV.
    Lanza un error explícito si el fichero no existe.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el fichero de datos en: {data_path}\n"
            "Descarga el dataset de Kaggle (ULB) y ajusa la ruta DATA_PATH "
            "al fichero 'creditcard.csv'."
        )

    print(f"Cargando dataset desde: {data_path}")
    df = pd.read_csv(data_path)

    expected_cols = {"Time", "Amount", "Class"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {missing}")

    return df


# =============================================================================
# GRÁFICO 1: DESBALANCE EXTREMO (V1 vs V2)
# =============================================================================

def grafico_desbalance_clases(df: pd.DataFrame, output_path: Path) -> None:
    """
    Scatter plot de V1 vs V2 con submuestreo visual:
    - 50.000 transacciones legítimas (Class = 0)
    - Todos los fraudes (Class = 1)

    Azul  -> legítimo
    Rojo  -> fraude
    """

    print("Generando Gráfico 1: desbalance de clases (V1 vs V2)...")

    legit = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]

    n_legit_sample = min(50000, len(legit))
    legit_sample = legit.sample(n=n_legit_sample, random_state=RANDOM_STATE)

    plt.figure(figsize=(8, 8))

    # Puntos legítimos primero (para que queden debajo)
    plt.scatter(
        legit_sample["V1"],
        legit_sample["V2"],
        c=COLOR_LEGIT,
        alpha=0.2,
        s=8,
        label="Legítima (Class 0)",
        edgecolors="none",
    )

    # Puntos de fraude encima
    plt.scatter(
        fraud["V1"],
        fraud["V2"],
        c=COLOR_FRAUD,
        alpha=0.7,
        s=20,
        label="Fraude (Class 1)",
        edgecolors="k",
        linewidths=0.2,
    )

    plt.xlabel("V1")
    plt.ylabel("V2")
    plt.title("Representación espacial del desbalance extremo (V1 vs V2)")
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico 1 guardado en: {output_path}")


# =============================================================================
# PREPARACIÓN DE DATOS Y ENTRENAMIENTO XGBOOST
# =============================================================================

def preparar_datos(df: pd.DataFrame):
    """
    Prepara X e y, y divide en entrenamiento y prueba (estratificado).
    """
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("Tamaño entrenamiento:", X_train.shape, "Tamaño prueba:", X_test.shape)
    print("Proporción de fraude en train:", y_train.mean())
    print("Proporción de fraude en test:", y_test.mean())

    return X_train, X_test, y_train, y_test


def entrenar_modelo_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Entrena un modelo XGBoost básico para clasificación binaria.
    """
    print("Entrenando modelo XGBoost...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        tree_method="hist",
    )

    model.fit(X_train, y_train)
    print("Modelo XGBoost entrenado.")
    return model


# =============================================================================
# GRÁFICO 2: ROC vs PRECISION-RECALL
# =============================================================================

def grafico_roc_vs_pr(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """
    Figura 1x2:
    - Izquierda: Curva ROC
    - Derecha: Curva Precision-Recall

    Ambos ejes en [0, 1], mostrando AUC-ROC y AUPRC.
    """

    print("Generando Gráfico 2: ROC vs Precision-Recall...")

    y_scores = model.predict_proba(X_test)[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    ax_roc = axes[0]
    ax_roc.plot(fpr, tpr, color=COLOR_FRAUD, lw=2, label=f"AUC-ROC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Azar")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax_roc.set_title("Curva ROC")
    ax_roc.legend(loc="lower right")

    # Precision-Recall
    ax_pr = axes[1]
    ax_pr.plot(recall, precision, color=COLOR_FRAUD, lw=2, label=f"AUPRC = {pr_auc:.3f}")
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Curva Precision-Recall")
    ax_pr.legend(loc="upper right")

    plt.suptitle(
        "Comparativa AUC-ROC vs AUPRC en escenario altamente desbalanceado",
        y=1.02,
    )
    plt.tight_layout()

    plt.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico 2 guardado en: {output_path}")


# =============================================================================
# GRÁFICO 3: SHAP BEESWARM (INTERPRETABILIDAD GLOBAL)
# =============================================================================

def grafico_shap_beeswarm(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_path: Path,
):
    """
    Crea un gráfico SHAP Beeswarm para interpretabilidad global.
    - Usa TreeExplainer sobre XGBoost.
    - Muestra las 10 características más importantes.
    - Colormap 'coolwarm': valores altos en rojos/cálidos, bajos en azules/fríos.
    """

    print("Calculando valores SHAP (puede tardar unos minutos)...")

    background_size = min(2000, len(X_train))
    explain_size = min(5000, len(X_test))

    background = X_train.sample(n=background_size, random_state=RANDOM_STATE)
    X_explain = X_test.sample(n=explain_size, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model, data=background)
    shap_values = explainer(X_explain)

    print("Generando Gráfico 3: SHAP Beeswarm...")

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(
        shap_values,
        max_display=10,
        show=False,
    )
    plt.title("Importancia global de características (SHAP Beeswarm)")
    plt.tight_layout()

    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico 3 guardado en: {output_path}")

    return explainer, X_explain


# =============================================================================
# GRÁFICO 4: AUDITORÍA FORENSE (SHAP WATERFALL PARA TP Y TN)
# =============================================================================

def seleccionar_tp_tn(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Selecciona índices de:
    - Verdadero Positivo (TP): y_test = 1 y predicción = 1
    - Verdadero Negativo (TN): y_test = 0 y predicción = 0
    """
    print("Buscando ejemplos de TP y TN en el conjunto de prueba...")

    y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

    tp_indices = np.where((y_test.values == 1) & (y_pred == 1))[0]
    tn_indices = np.where((y_test.values == 0) & (y_pred == 0))[0]

    if len(tp_indices) == 0:
        raise RuntimeError("No se encontró ningún Verdadero Positivo (TP) en el conjunto de prueba.")
    if len(tn_indices) == 0:
        raise RuntimeError("No se encontró ningún Verdadero Negativo (TN) en el conjunto de prueba.")

    idx_tp = tp_indices[0]
    idx_tn = tn_indices[0]

    print("Índice TP seleccionado:", idx_tp, "Índice TN seleccionado:", idx_tn)
    return idx_tp, idx_tn


def grafico_shap_force_waterfall(
    explainer,
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """
    Genera dos gráficos tipo "force plot" en formato estático utilizando
    shap.plots.waterfall() para:
    - Un Verdadero Positivo (fraude detectado correctamente).
    - Un Verdadero Negativo (legítima correctamente clasificada).

    Los colores siguen la convención SHAP:
    - Rojo: contribuciones que empujan hacia la clase positiva (riesgo alto).
    - Azul: contribuciones que empujan hacia la clase negativa (riesgo bajo).
    """

    print("Generando Gráfico 4: auditoría forense (SHAP waterfall para TP y TN)...")

    idx_tp, idx_tn = seleccionar_tp_tn(model, X_test, y_test)

    x_tp = X_test.iloc[[idx_tp]]
    x_tn = X_test.iloc[[idx_tn]]

    shap_tp = explainer(x_tp)
    shap_tn = explainer(x_tn)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # TP (fraude)
    plt.sca(axes[0])
    shap.plots.waterfall(shap_tp[0], max_display=10, show=False)
    axes[0].set_title("Auditoría SHAP - Verdadero Positivo (Fraude detectado)")

    # TN (legítima)
    plt.sca(axes[1])
    shap.plots.waterfall(shap_tn[0], max_display=10, show=False)
    axes[1].set_title("Auditoría SHAP - Verdadero Negativo (Transacción legítima)")

    plt.tight_layout()
    plt.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico 4 guardado en: {output_path}")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main():
    """
    Ejecuta de principio a fin la generación de las cuatro figuras.
    """
    df = load_dataset(DATA_PATH)

    # Gráfico 1: desbalance extremo
    grafico_desbalance_clases(
        df,
        FIGURES_DIR / "desbalance_clases.svg",
    )

    # Preparación de datos y modelo
    X_train, X_test, y_train, y_test = preparar_datos(df)
    model = entrenar_modelo_xgboost(X_train, y_train)

    # Gráfico 2: ROC vs PR
    grafico_roc_vs_pr(
        model,
        X_test,
        y_test,
        FIGURES_DIR / "roc_vs_pr.svg",
    )

    # Gráfico 3: SHAP Beeswarm (global)
    explainer, X_explain = grafico_shap_beeswarm(
        model,
        X_train,
        X_test,
        FIGURES_DIR / "shap_beeswarm.pdf",
    )

    # Gráfico 4: Auditoría forense (TP y TN)
    grafico_shap_force_waterfall(
        explainer,
        model,
        X_test,
        y_test,
        FIGURES_DIR / "shap_force_plot.svg",
    )

    print("\nProceso completado.")
    print("Figuras generadas en:", FIGURES_DIR)
    print(
        "\nArchivos generados:\n"
        "1) desbalance_clases.svg\n"
        "2) roc_vs_pr.svg\n"
        "3) shap_beeswarm.pdf\n"
        "4) shap_force_plot.svg\n"
    )


if __name__ == "__main__":
    main()

