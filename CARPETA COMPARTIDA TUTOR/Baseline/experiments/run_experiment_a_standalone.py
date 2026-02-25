#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ejecuta la lógica del Experimento A (validación prequential + GridSearch)
sin pasar por el notebook. Útil para validación rápida y para obtener
resultados que actualizar en la documentación.

Uso (con Docker):
    docker compose run --rm experiments python experiments/run_experiment_a_standalone.py

Uso (local con venv):
    python experiments/run_experiment_a_standalone.py
"""

import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Añadir path del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import (
    SEED, INPUT_FEATURES, OUTPUT_FEATURE,
    RESULTS_DIR, FIGURES_DIR,
    DELTA_TRAIN, DELTA_DELAY, DELTA_TEST,
    START_DATE_TRAINING_FOR_VALID, START_DATE_TRAINING_FOR_TEST,
    N_FOLDS, COLORS,
)
from experiments.data_utils import (
    load_transformed_data,
    card_precision_top_k_custom,
    model_selection_wrapper,
    compute_confusion_matrices_prequential,
)

# Modo completo: 4 folds y grid completo (~15-30 min)
QUICK = False
n_folds = 2 if QUICK else N_FOLDS

if QUICK:
    param_grids = {
        "Logistic Regression": {'clf__C': [0.1, 1, 10], 'clf__max_iter': [1000], 'clf__random_state': [SEED]},
        "Random Forest": {'clf__max_depth': [10, 50], 'clf__n_estimators': [50], 'clf__random_state': [SEED], 'clf__n_jobs': [-1]},
        "XGBoost": {'clf__max_depth': [3, 6], 'clf__n_estimators': [50], 'clf__learning_rate': [0.3],
                    'clf__random_state': [SEED], 'clf__use_label_encoder': [False],
                    'clf__eval_metric': ['logloss'], 'clf__n_jobs': [-1], 'clf__verbosity': [0]},
    }
else:
    param_grids = {
        "Logistic Regression": {'clf__C': [0.1, 1, 10, 100], 'clf__max_iter': [1000], 'clf__random_state': [SEED]},
        "Random Forest": {'clf__max_depth': [10, 20, 50], 'clf__n_estimators': [50, 100], 'clf__random_state': [SEED], 'clf__n_jobs': [-1]},
        "XGBoost": {'clf__max_depth': [3, 6, 9], 'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.3],
                   'clf__random_state': [SEED], 'clf__use_label_encoder': [False],
                   'clf__eval_metric': ['logloss'], 'clf__n_jobs': [-1], 'clf__verbosity': [0]},
    }

def main():
    print("Cargando datos...")
    transactions_df = load_transformed_data()

    card_precision_scorer = sklearn.metrics.make_scorer(
        card_precision_top_k_custom, needs_proba=True,
        top_k=100, transactions_df=transactions_df,
    )
    scoring = {
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',
        'card_precision@100': card_precision_scorer,
    }
    perf_metrics_grid = ['roc_auc', 'average_precision', 'card_precision@100']
    perf_metrics_names = ['AUC ROC', 'Average precision', 'Card Precision@100']

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": xgb.XGBClassifier(),
    }

    print(f"\nExperimento A — {n_folds} folds, modo {'rápido' if QUICK else 'completo'}\n")
    results_a = {}
    start_total = time.time()

    for name, clf in classifiers.items():
        print(f"  {name}...", end=" ", flush=True)
        start = time.time()
        perf_df = model_selection_wrapper(
            transactions_df, clf, INPUT_FEATURES, OUTPUT_FEATURE,
            param_grids[name], scoring,
            START_DATE_TRAINING_FOR_VALID, START_DATE_TRAINING_FOR_TEST,
            n_folds=n_folds, delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY, delta_assessment=DELTA_TEST,
            performance_metrics_list_grid=perf_metrics_grid, performance_metrics_list=perf_metrics_names,
            n_jobs=-1,
        )
        best_idx = perf_df['Average precision Validation'].idxmax()
        best = perf_df.loc[best_idx]
        results_a[name] = {
            'auc_roc_mean': best['AUC ROC Test'], 'auc_roc_std': best['AUC ROC Test Std'],
            'auprc_mean': best['Average precision Test'], 'auprc_std': best['Average precision Test Std'],
            'cp100_mean': best['Card Precision@100 Test'], 'cp100_std': best['Card Precision@100 Test Std'],
            'best_params': perf_df.loc[best_idx, 'Parameters'],
            'performances_df': perf_df,
        }
        print(f"{time.time()-start:.1f}s — AUC={results_a[name]['auc_roc_mean']:.4f}±{results_a[name]['auc_roc_std']:.4f} "
              f"AUPRC={results_a[name]['auprc_mean']:.4f}±{results_a[name]['auprc_std']:.4f} "
              f"CP100={results_a[name]['cp100_mean']:.4f}±{results_a[name]['cp100_std']:.4f}")

    elapsed = time.time() - start_total
    print(f"\nTiempo total: {elapsed:.1f}s\n")

    # Matrices de confusión
    clf_classes = {
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "XGBoost": xgb.XGBClassifier,
    }
    confusion_matrices = compute_confusion_matrices_prequential(
        transactions_df, results_a, INPUT_FEATURES, OUTPUT_FEATURE,
        clf_classes, START_DATE_TRAINING_FOR_TEST,
        n_folds=n_folds, delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY, delta_assessment=DELTA_TEST,
    )
    for name, cm_data in confusion_matrices.items():
        results_a[name]['confusion_matrix'] = cm_data

    # Tabla y guardado
    model_names = list(results_a.keys())
    results_table = pd.DataFrame({
        'Modelo': model_names,
        'AUC ROC': [f"{results_a[n]['auc_roc_mean']:.4f} ± {results_a[n]['auc_roc_std']:.4f}" for n in model_names],
        'AUPRC': [f"{results_a[n]['auprc_mean']:.4f} ± {results_a[n]['auprc_std']:.4f}" for n in model_names],
        'CP@100': [f"{results_a[n]['cp100_mean']:.4f} ± {results_a[n]['cp100_std']:.4f}" for n in model_names],
    })
    results_table_numeric = pd.DataFrame({
        'Modelo': model_names,
        'AUC ROC': [results_a[n]['auc_roc_mean'] for n in model_names],
        'AUC ROC Std': [results_a[n]['auc_roc_std'] for n in model_names],
        'AUPRC': [results_a[n]['auprc_mean'] for n in model_names],
        'AUPRC Std': [results_a[n]['auprc_std'] for n in model_names],
        'CP@100': [results_a[n]['cp100_mean'] for n in model_names],
        'CP@100 Std': [results_a[n]['cp100_std'] for n in model_names],
    }).set_index('Modelo')

    print("RESULTADOS (media ± desv. estándar):")
    print(results_table.to_string(index=False))
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_table_numeric.to_csv(RESULTS_DIR / 'experiment_a_results.csv')

    results_to_save = {
        name: {
            **{k: v for k, v in res.items() if k != 'performances_df'},
            'card_precision_at_k': {100: res['cp100_mean']},
            'avg_precision': res['auprc_mean'],
            'auc_roc': res['auc_roc_mean'],
            'performances_df': res['performances_df'],
        }
        for name, res in results_a.items()
    }
    with open(RESULTS_DIR / 'experiment_a_predictions.pkl', 'wb') as f:
        pickle.dump(results_to_save, f)

    # Generar figura
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    model_names = list(results_a.keys())
    x_pos = np.arange(len(model_names))
    for ax, (key, label, color) in zip(axes, [
        ('auc_roc', 'AUC ROC', '#2F4D7E'),
        ('auprc', 'AUPRC', '#008000'),
        ('cp100', 'Card Precision@100', '#CA8035'),
    ]):
        means = [results_a[n][f'{key}_mean'] for n in model_names]
        stds = [results_a[n][f'{key}_std'] for n in model_names]
        ax.bar(x_pos, means, 0.5, yerr=stds, capsize=5, color=color, edgecolor='black')
        ax.set_ylabel(label)
        ax.set_title(f'{label}\n(Experimento A)', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.set_ylim([0, 1.05])
    fig.suptitle(f'Validación prequential ({n_folds} folds)', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    (FIGURES_DIR / 'experiment_a_baseline_results.png').parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'experiment_a_baseline_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura: {FIGURES_DIR / 'experiment_a_baseline_results.png'}")

    # Matrices de confusión (heatmaps)
    import seaborn as sns
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(16, 5))
    labels = ['Legítimo', 'Fraude']
    for idx, (name, cm_data) in enumerate(confusion_matrices.items()):
        cm = cm_data['matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
                    ax=axes_cm[idx], cbar=False, annot_kws={'size': 13})
        n_fraud = cm[1, 0] + cm[1, 1]
        recall_pct = 100 * cm[1, 1] / n_fraud if n_fraud > 0 else 0
        axes_cm[idx].set_title(f"{name}\nTP={cm[1,1]:,}  FN={cm[1,0]:,}  (Recall={recall_pct:.1f}%)", fontsize=11)
        axes_cm[idx].set_ylabel('Real')
        axes_cm[idx].set_xlabel('Predicho')
    fig_cm.suptitle(f'Matrices de Confusión — Experimento A ({n_folds} folds, threshold=0.5)', fontsize=13, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_cm.savefig(FIGURES_DIR / 'experiment_a_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Matrices de confusión: {FIGURES_DIR / 'experiment_a_confusion_matrices.png'}")

    print(f"Guardado: {RESULTS_DIR / 'experiment_a_results.csv'}")
    print(f"Guardado: {RESULTS_DIR / 'experiment_a_predictions.pkl'}")
    return results_a

if __name__ == "__main__":
    main()
