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
- shap_beeswarm.pdf  (⚠ ver nota abajo)
- shap_waterfall_tp.png, shap_waterfall_tn.png (explicabilidad local; waterfall estático)

NOTA (coherencia con la memoria del TFM)
-----------------------------------------
Este script entrena XGBoost sobre el CSV ULB (V1..V28, Time, Amount). El
``shap_beeswarm.pdf`` generado aquí muestra importancias SHAP sobre esas
variables PCA, NO sobre TERMINAL_ID_RISK_*, CUSTOMER_ID_*, TX_AMOUNT, etc.

Para el beeswarm alineado con el marco teórico de la memoria, ejecutar **después**:
  python experiments/generate_shap_beeswarm_transformed_dataset.py
Ese script sobrescribe ``shap_beeswarm.pdf`` con el modelo sobre el dataset
transformado del proyecto (mismo pipeline que Experimento D).

Convención de colores (estricta):
- Azul  -> transacciones legítimas (Class = 0)
- Rojo  -> transacciones fraudulentas (Class = 1)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
    Visualización del desbalance en (V1, V2):

    - Clase 0 (legítima): densidad con ``hexbin`` y escala de azules para evitar
      overplotting y mostrar dónde se concentra la mayoría de los datos.
    - Clase 1 (fraude): ``scatter`` encima en rojo oscuro con borde blanco fino.

    Mantiene la semántica azul = legítimo, rojo = fraude.
    """

    print("Generando Gráfico 1: desbalance de clases (hexbin + scatter, V1 vs V2)...")

    legit = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]

    n_legit_sample = min(50000, len(legit))
    legit_sample = legit.sample(n=n_legit_sample, random_state=RANDOM_STATE)

    fig, ax = plt.subplots(figsize=(9, 8))

    # Capa 1: densidad de la clase mayoritaria (azules)
    hb = ax.hexbin(
        legit_sample["V1"].values,
        legit_sample["V2"].values,
        gridsize=55,
        cmap="Blues",
        mincnt=1,
        linewidths=0.0,
        edgecolors="face",
        alpha=0.92,
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Cuenta en celda hexagonal (clase legítima)", rotation=270, labelpad=18)

    # Capa 2: fraudes destacados sobre la densidad
    ax.scatter(
        fraud["V1"],
        fraud["V2"],
        c="firebrick",
        s=15,
        alpha=0.95,
        edgecolors="white",
        linewidths=0.6,
        label="Fraude (Class 1)",
        zorder=5,
    )

    ax.set_xlabel("V1")
    ax.set_ylabel("V2")
    ax.set_title(
        "Representación espacial del desbalance: Densidad de la clase mayoritaria vs Fraude"
    )

    # Leyenda explícita (hexbin no aporta handle automático)
    legend_handles = [
        Patch(facecolor=COLOR_LEGIT, edgecolor="navy", linewidth=0.5, alpha=0.7, label="Clase 0 — densidad (mapa azul)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="firebrick",
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7,
            linestyle="None",
            label="Fraude (Class 1)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=True)

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
        use_label_encoder=False,
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

    # Baseline de un clasificador aleatorio en PR: precisión = proporción de positivos
    proporcion_fraude = float((y_test == 1).sum()) / float(len(y_test))

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
    ax_pr.axhline(
        y=proporcion_fraude,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Azar (P(clase positiva) = {proporcion_fraude:.4f})",
    )
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.05)
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


def grafico_shap_waterfall_local(
    explainer,
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """
    Explicabilidad local con **Waterfall** estático (mejor legibilidad que force plot
    al exportar a imagen): una figura por instancia, nombres de variables menos
    comprimidos gracias a ``bbox_inches='tight'`` y figura ancha.

    - Verdadero Positivo (TP): fraude detectado correctamente.
    - Verdadero Negativo (TN): legítima correctamente clasificada.

    Usa ``shap.plots.waterfall(..., max_display=10)`` con el objeto Explanation
    devuelto por ``explainer(fila)``.
    """

    print("Generando Gráfico 4: auditoría forense (SHAP waterfall estático TP / TN)...")

    idx_tp, idx_tn = seleccionar_tp_tn(model, X_test, y_test)

    x_tp = X_test.iloc[[idx_tp]]
    x_tn = X_test.iloc[[idx_tn]]

    shap_tp = explainer(x_tp)
    shap_tn = explainer(x_tn)

    output_dir.mkdir(parents=True, exist_ok=True)

    # TP: figura dedicada (PNG alta resolución para la memoria)
    plt.figure(figsize=(11, 7))
    shap.plots.waterfall(shap_tp[0], max_display=10, show=False)
    plt.title("SHAP Waterfall — Verdadero Positivo (fraude detectado)", fontsize=12)
    plt.tight_layout()
    path_tp = output_dir / "shap_waterfall_tp.png"
    plt.savefig(path_tp, bbox_inches="tight", dpi=300)
    plt.close()

    # TN: figura dedicada
    plt.figure(figsize=(11, 7))
    shap.plots.waterfall(shap_tn[0], max_display=10, show=False)
    plt.title("SHAP Waterfall — Verdadero Negativo (transacción legítima)", fontsize=12)
    plt.tight_layout()
    path_tn = output_dir / "shap_waterfall_tn.png"
    plt.savefig(path_tn, bbox_inches="tight", dpi=300)
    plt.close()

    # Opcional: versión vectorial combinada para impresión
    fig, axes = plt.subplots(2, 1, figsize=(11, 14))
    plt.sca(axes[0])
    shap.plots.waterfall(shap_tp[0], max_display=10, show=False)
    axes[0].set_title("Verdadero Positivo (fraude detectado)")
    plt.sca(axes[1])
    shap.plots.waterfall(shap_tn[0], max_display=10, show=False)
    axes[1].set_title("Verdadero Negativo (transacción legítima)")
    plt.suptitle("Auditoría forense SHAP (Waterfall)", fontsize=13, y=1.01)
    plt.tight_layout()
    path_combo = output_dir / "shap_waterfall_combined.svg"
    plt.savefig(path_combo, format="svg", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico 4 guardado en:\n  {path_tp}\n  {path_tn}\n  {path_combo}")


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

    # Gráfico 4: Auditoría forense (waterfall estático TP / TN)
    grafico_shap_waterfall_local(
        explainer,
        model,
        X_test,
        y_test,
        FIGURES_DIR,
    )

    print("\nProceso completado.")
    print("Figuras generadas en:", FIGURES_DIR)
    print(
        "\nArchivos generados:\n"
        "1) desbalance_clases.svg\n"
        "2) roc_vs_pr.svg\n"
        "3) shap_beeswarm.pdf\n"
        "4) shap_waterfall_tp.png, shap_waterfall_tn.png, shap_waterfall_combined.svg\n"
    )


if __name__ == "__main__":
    main()

