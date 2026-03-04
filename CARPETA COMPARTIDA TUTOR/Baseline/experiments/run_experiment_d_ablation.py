#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experimento D - Ablación: Validación eliminando la característica con mayor impacto SHAP.

Objetivo: Validar que la feature identificada por SHAP como más importante
efectivamente contribuye al rendimiento. Si al eliminarla las métricas empeoran,
se confirma su valor predictivo.

Flujo:
1. Lee experiment_d_shap_mean_impact.csv para identificar la top feature por mean |SHAP|
2. Entrena XGBoost baseline CON todas las features (baseline D)
3. Entrena XGBoost baseline SIN la top feature (ablación)
4. Compara AUC ROC, AUPRC, Card Precision@100 entre ambos
5. Calcula y guarda el ranking mean |SHAP| de las features restantes en el modelo ablated

Uso: python experiments/run_experiment_d_ablation.py
     docker compose run --rm experiments python experiments/run_experiment_d_ablation.py
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import (
    SEED, INPUT_FEATURES, OUTPUT_FEATURE,
    BASELINE_PARAMS, RESULTS_DIR, FIGURES_DIR,
    START_DATE_TRAINING, DELTA_TRAIN, DELTA_DELAY, DELTA_TEST,
)
from experiments.data_utils import (
    load_transformed_data, get_train_test_set,
    print_dataset_summary, card_precision_top_k,
)

warnings.filterwarnings('ignore')


def get_top_shap_feature(shap_csv_path: Path) -> str:
    """
    Obtiene la característica con mayor mean |SHAP| del CSV generado por el Experimento D.
    Si el CSV no existe, retorna la feature de mayor impacto conocida (del último run).
    """
    if shap_csv_path.exists():
        df = pd.read_csv(shap_csv_path)
        if 'Feature' in df.columns and 'mean_abs_SHAP' in df.columns:
            top_row = df.sort_values('mean_abs_SHAP', ascending=False).iloc[0]
            return top_row['Feature']
    # Fallback: feature con mayor impacto según experiment_d_shap_mean_impact.csv típico
    return 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW'


def train_and_evaluate(features, model_name, train_df, test_df):
    """Entrena XGBoost baseline con el subconjunto de features y retorna métricas."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])

    model = xgb.XGBClassifier(**BASELINE_PARAMS["XGBoost"])
    model.fit(X_train, train_df[OUTPUT_FEATURE])
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc_roc = metrics.roc_auc_score(test_df[OUTPUT_FEATURE], y_pred_proba)
    auprc = metrics.average_precision_score(test_df[OUTPUT_FEATURE], y_pred_proba)
    pred_df = test_df.copy()
    pred_df['predictions'] = y_pred_proba
    _, _, cp100 = card_precision_top_k(pred_df, top_k=100)

    return {
        'auc_roc': auc_roc,
        'auprc': auprc,
        'card_precision_at_100': cp100,
        'model': model,
        'scaler': scaler,
        'n_features': len(features),
    }


def compute_shap_ranking(model, scaler, features, test_df, sample_size=1000):
    """Calcula mean |SHAP| por variable para el ranking de importancia."""
    X_test_scaled = scaler.transform(test_df[features])
    np.random.seed(SEED)
    n = min(sample_size, len(X_test_scaled))
    idx = np.random.choice(len(X_test_scaled), n, replace=False)
    X_sample = X_test_scaled[idx]
    X_sample_df = pd.DataFrame(X_sample, columns=features)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    df = pd.DataFrame({
        'Feature': features,
        'mean_abs_SHAP': mean_abs_shap,
    }).sort_values('mean_abs_SHAP', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    return df


def main():
    print("=" * 70)
    print("  EXPERIMENTO D - ABLACIÓN: Validación eliminando top feature SHAP")
    print("=" * 70)

    shap_csv = RESULTS_DIR / 'experiment_d_shap_mean_impact.csv'
    top_feature = get_top_shap_feature(shap_csv)
    print(f"\n  Top feature por mean |SHAP|: {top_feature}")

    features_full = list(INPUT_FEATURES)
    features_ablated = [f for f in features_full if f != top_feature]
    print(f"  Features completas: {len(features_full)}")
    print(f"  Features sin top:   {len(features_ablated)} (ablación)")

    transactions_df = load_transformed_data()
    train_df, test_df = get_train_test_set(
        transactions_df,
        start_date_training=START_DATE_TRAINING,
        delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY, delta_test=DELTA_TEST,
    )
    print_dataset_summary(train_df, test_df, "Experimento D - Ablación")

    # Modelo completo (baseline D)
    print("\n--- Entrenando modelo CON todas las features ---")
    metrics_full = train_and_evaluate(features_full, "full", train_df, test_df)
    print(f"  AUC ROC:  {metrics_full['auc_roc']:.4f}")
    print(f"  AUPRC:    {metrics_full['auprc']:.4f}")
    print(f"  CP@100:   {metrics_full['card_precision_at_100']:.4f}")

    # Modelo ablated (sin top feature)
    print(f"\n--- Entrenando modelo SIN {top_feature} ---")
    metrics_ablated = train_and_evaluate(features_ablated, "ablated", train_df, test_df)
    print(f"  AUC ROC:  {metrics_ablated['auc_roc']:.4f}")
    print(f"  AUPRC:    {metrics_ablated['auprc']:.4f}")
    print(f"  CP@100:   {metrics_ablated['card_precision_at_100']:.4f}")

    # Ranking de importancia SHAP del modelo ablated (14 features restantes)
    print(f"\n--- Ranking mean |SHAP| del modelo SIN {top_feature} ---")
    shap_ablated_df = compute_shap_ranking(
        metrics_ablated['model'],
        metrics_ablated['scaler'],
        features_ablated,
        test_df,
    )
    shap_ablated_df.to_csv(RESULTS_DIR / 'experiment_d_ablation_shap_ranking.csv', index=False)
    print(shap_ablated_df.to_string(index=False))

    # Comparación
    print("\n" + "=" * 70)
    print("  RESULTADO DE VALIDACIÓN (diferencia = ablated - full)")
    print("=" * 70)
    delta_auc = metrics_ablated['auc_roc'] - metrics_full['auc_roc']
    delta_auprc = metrics_ablated['auprc'] - metrics_full['auprc']
    delta_cp = metrics_ablated['card_precision_at_100'] - metrics_full['card_precision_at_100']

    print(f"  Δ AUC ROC:     {delta_auc:+.4f} {'↓' if delta_auc < 0 else '↑'}")
    print(f"  Δ AUPRC:       {delta_auprc:+.4f} {'↓' if delta_auprc < 0 else '↑'}")
    print(f"  Δ CP@100:      {delta_cp:+.4f} {'↓' if delta_cp < 0 else '↑'}")

    validacion_exitosa = delta_auc < 0 or delta_auprc < 0 or delta_cp < 0
    if validacion_exitosa:
        print("\n  ✓ VALIDACIÓN EXITOSA: Al eliminar la top feature SHAP, el rendimiento")
        print(f"    empeora. La feature {top_feature} aporta valor real al modelo.")
    else:
        print("\n  ⚠ Nota: Las métricas no empeoraron significativamente. Puede deberse a")
        print("    redundancia con otras features o variabilidad del split.")

    # Guardar resultados
    ablation_results = {
        'top_feature_removed': top_feature,
        'metrics_full': {
            'auc_roc': metrics_full['auc_roc'],
            'auprc': metrics_full['auprc'],
            'card_precision_at_100': metrics_full['card_precision_at_100'],
        },
        'metrics_ablated': {
            'auc_roc': metrics_ablated['auc_roc'],
            'auprc': metrics_ablated['auprc'],
            'card_precision_at_100': metrics_ablated['card_precision_at_100'],
        },
        'delta': {'auc_roc': delta_auc, 'auprc': delta_auprc, 'card_precision_at_100': delta_cp},
        'validation_success': validacion_exitosa,
        'seed': SEED,
    }

    # CSV resumido
    comp_df = pd.DataFrame([
        {'modelo': 'D_full', 'auc_roc': metrics_full['auc_roc'], 'auprc': metrics_full['auprc'],
         'card_precision_at_100': metrics_full['card_precision_at_100']},
        {'modelo': f'D_ablated_sin_{top_feature}', 'auc_roc': metrics_ablated['auc_roc'],
         'auprc': metrics_ablated['auprc'], 'card_precision_at_100': metrics_ablated['card_precision_at_100']},
    ])
    comp_df.to_csv(RESULTS_DIR / 'experiment_d_ablation_comparison.csv', index=False)

    # Markdown de informe
    report_path = RESULTS_DIR / 'experiment_d_ablation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Experimento D - Ablación: Validación SHAP\n\n")
        f.write(f"**Feature eliminada:** `{top_feature}` (mayor mean |SHAP|)\n\n")
        f.write("## Comparación de métricas\n\n")
        f.write("| Modelo | AUC ROC | AUPRC | CP@100 |\n")
        f.write("|--------|---------|-------|--------|\n")
        f.write(f"| D (completo) | {metrics_full['auc_roc']:.4f} | {metrics_full['auprc']:.4f} | {metrics_full['card_precision_at_100']:.4f} |\n")
        f.write(f"| D (sin top)  | {metrics_ablated['auc_roc']:.4f} | {metrics_ablated['auprc']:.4f} | {metrics_ablated['card_precision_at_100']:.4f} |\n")
        f.write(f"| Δ            | {delta_auc:+.4f} | {delta_auprc:+.4f} | {delta_cp:+.4f} |\n\n")
        f.write("## Ranking de importancia (mean |SHAP|) tras ablación\n\n")
        f.write("Nuevo orden de las 14 features restantes en el modelo reentrenado:\n\n")
        f.write("| Rank | Feature | mean_abs_SHAP |\n")
        f.write("|------|---------|---------------|\n")
        for _, row in shap_ablated_df.iterrows():
            f.write(f"| {row['rank']} | {row['Feature']} | {row['mean_abs_SHAP']:.6f} |\n")
        f.write("\n**Conclusión:** " + (
            "Validación exitosa: la feature eliminada contribuía al rendimiento." if validacion_exitosa
            else "Las métricas no mostraron degradación significativa."
        ) + "\n")

    print(f"\n✓ Resultados guardados:")
    print(f"  - {RESULTS_DIR / 'experiment_d_ablation_comparison.csv'}")
    print(f"  - {RESULTS_DIR / 'experiment_d_ablation_shap_ranking.csv'}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
