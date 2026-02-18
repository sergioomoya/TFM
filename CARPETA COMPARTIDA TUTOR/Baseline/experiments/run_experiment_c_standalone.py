#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ejecuta el Experimento C refactorizado (Anti-Leakage) y guarda resultados.

Uso:
    docker compose run --rm experiments python experiments/run_experiment_c_standalone.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import (
    RESULTS_DIR, FIGURES_DIR, COLORS, SMOTE_PARAMS,
)
from experiments.data_utils import load_transformed_data
from experiments.experiment_c_leakage_test import run_all, INPUT_FEATURES, OUTPUT_FEATURE


def main():
    print("Cargando datos...")
    transactions_df = load_transformed_data()
    print(f"Dataset: {len(transactions_df):,} transacciones")
    print(f"SMOTE: k_neighbors={SMOTE_PARAMS['k_neighbors']}, "
          f"sampling_strategy={SMOTE_PARAMS['sampling_strategy']}\n")

    print("Ejecutando Experimento C (5 ramas × 3 modelos)...")
    results, train_df, test_df = run_all(transactions_df)

    # Construir tablas
    ramas_order = ["Correcta", "Leak_split", "Leak_scaler", "Leak_smote", "Leak_todas"]
    models = ["Logistic Regression", "Random Forest", "XGBoost"]

    rows = []
    for model in models:
        for rama in ramas_order:
            r = results[model][rama]
            rows.append({
                'Modelo': model,
                'Rama': rama,
                'AUC ROC': r['auc_roc'],
                'AUPRC': r['auprc'],
                'CP@100': r.get('cp100', np.nan),
            })
    df_all = pd.DataFrame(rows)

    # Tabla resumen: Correcta vs Leak_todas por modelo
    comparison = []
    for model in models:
        correct = results[model]["Correcta"]
        leak_all = results[model]["Leak_todas"]
        comparison.append({
            'Modelo': model,
            'C-Correcta AUC': correct['auc_roc'],
            'C-Correcta AUPRC': correct['auprc'],
            'C-Correcta CP@100': correct['cp100'],
            'C-Leak_todas AUC': leak_all['auc_roc'],
            'C-Leak_todas AUPRC': leak_all['auprc'],
        })
    df_comparison = pd.DataFrame(comparison)

    # Tabla desglose por fuente de leakage
    desglose_rows = []
    for model in models:
        correct = results[model]["Correcta"]['auprc']
        leak_split = results[model]["Leak_split"]['auprc']
        leak_scaler = results[model]["Leak_scaler"]['auprc']
        leak_smote = results[model]["Leak_smote"]['auprc']
        leak_todas = results[model]["Leak_todas"]['auprc']
        desglose_rows.append({
            'Modelo': model,
            'Correcta': f"{correct:.4f}",
            '+Split aleatorio': f"{leak_split:.4f} (Δ{leak_split-correct:+.4f})",
            '+Escalado global': f"{leak_scaler:.4f} (Δ{leak_scaler-correct:+.4f})",
            '+SMOTE global': f"{leak_smote:.4f} (Δ{leak_smote-correct:+.4f})",
            'Todas (máx. leakage)': f"{leak_todas:.4f} (Δ{leak_todas-correct:+.4f})",
        })
    df_desglose = pd.DataFrame(desglose_rows)

    print("\n" + "=" * 90)
    print("  RESUMEN: Correcta vs Leak_todas por modelo")
    print("=" * 90)
    print(df_comparison.to_string(index=False))

    print("\n" + "=" * 90)
    print("  DESGLOSE: Impacto incremental por fuente de leakage (AUPRC)")
    print("=" * 90)
    print(df_desglose.to_string(index=False))

    # Guardar
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_all.to_csv(RESULTS_DIR / 'experiment_c_all_ramas.csv', index=False)
    df_comparison.to_csv(RESULTS_DIR / 'experiment_c_comparison.csv', index=False)
    df_desglose.to_csv(RESULTS_DIR / 'experiment_c_desglose_leakage.csv', index=False)

    # Guardar PKL para compatibilidad
    correct_lr = results["Logistic Regression"]["Correcta"]
    leak_todas_lr = results["Logistic Regression"]["Leak_todas"]
    with open(RESULTS_DIR / 'experiment_c_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'correct': correct_lr,
            'incorrect': leak_todas_lr,
            'metadata': {
                'smote_params': SMOTE_PARAMS,
                'ramas': ramas_order,
                'models': models,
            },
        }, f)

    # Figura: barras por rama y modelo
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(ramas_order))
    width = 0.25
    colors_ramas = {
        'Correcta': COLORS['correct_pipeline'],
        'Leak_split': '#FFA500',
        'Leak_scaler': '#FF8C00',
        'Leak_smote': '#FF6347',
        'Leak_todas': COLORS['incorrect_pipeline'],
    }
    for i, model in enumerate(models):
        ax = axes[i]
        vals = [results[model][r]['auprc'] for r in ramas_order]
        bars = ax.bar(x, vals, width * 2, color=[colors_ramas[r] for r in ramas_order], edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(['Correcta', 'Leak\nsplit', 'Leak\nscaler', 'Leak\nSMOTE', 'Leak\ntodas'], fontsize=9)
        ax.set_ylabel('AUPRC')
        ax.set_title(model)
        ax.set_ylim([0, 1.05])
        for j, (bar, v) in enumerate(zip(bars, vals)):
            ax.annotate(f'{v:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)
    fig.suptitle('Experimento C: Impacto del Data Leakage por fuente y modelo\n'
                 f'SMOTE: k_neighbors={SMOTE_PARAMS["k_neighbors"]}, sampling_strategy={SMOTE_PARAMS["sampling_strategy"]}',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'experiment_c_leakage_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Figura: {FIGURES_DIR / 'experiment_c_leakage_comparison.png'}")
    print(f"✓ CSV: {RESULTS_DIR / 'experiment_c_all_ramas.csv'}")
    print(f"✓ CSV: {RESULTS_DIR / 'experiment_c_desglose_leakage.csv'}")
    print(f"✓ PKL: {RESULTS_DIR / 'experiment_c_results.pkl'}")

    return results

if __name__ == "__main__":
    main()
