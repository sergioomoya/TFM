#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ejecuta todas las variantes del Experimento A en cuanto a balance de clases:

1. Original (~118:1) — sin submuestreo
2. 10:1 — 10 legítimas por 1 fraude
3. 5:1  — 5 legítimas por 1 fraude
4. 1:1  — balanceado

Optimizaciones de recursos:
- XGBoost con GPU: n_jobs=1 (evita conflictos NCCL/serialización en GridSearchCV)
- Random Forest: n_jobs=4-6 (evita OOM de joblib/loky al serializar DataFrame)
- Logistic Regression: n_jobs=-1 (ligero, usa todos los núcleos)

Uso con entorno tfm (recomendado):
    .\run_experiment_a_balance_controlled.ps1

Uso directo:
    conda activate tfm
    python experiments/run_experiment_a_all_balance_variants.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import (
    SEED, INPUT_FEATURES, OUTPUT_FEATURE,
    RESULTS_DIR, FIGURES_DIR,
    DELTA_TRAIN, DELTA_DELAY, DELTA_TEST,
    START_DATE_TRAINING_FOR_VALID, START_DATE_TRAINING_FOR_TEST,
    N_FOLDS,
)
from experiments.data_utils import (
    load_transformed_data,
    card_precision_top_k_custom,
    model_selection_wrapper,
    compute_confusion_matrices_prequential,
)
from experiments.hw_config import get_hw_config, get_xgboost_gpu_params

# Variantes de balance: None = original, float = ratio legítimas:fraudes
BALANCE_VARIANTS = [
    (None, "original"),      # ~118:1, sin submuestreo
    (10.0, "10:1"),
    (5.0, "5:1"),
    (1.0, "1:1"),
]

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


def _write_progress(progress_path: Path, msg: str) -> None:
    """Escribe estado de progreso para monitoreo externo."""
    if progress_path:
        try:
            progress_path.write_text(msg, encoding="utf-8")
        except OSError:
            pass


def run_single_variant(transactions_df, ratio, label, classifiers, param_grids,
                      card_precision_scorer, scoring, perf_metrics_grid, perf_metrics_names,
                      hw, xgb_gpu, n_jobs_per_model, clf_classes, progress_path=None):
    """Ejecuta una variante de balance y retorna resultados."""
    ratio_str = "~118:1" if ratio is None else f"{ratio}:1"
    print(f"\n{'='*70}")
    print(f"  Variante: {label} (ratio legítimas:fraudes = {ratio_str})")
    print(f"{'='*70}\n")
    _write_progress(progress_path, f"Variante {label} | {ratio_str}")

    results = {}
    for name, clf in classifiers.items():
        n_jobs = n_jobs_per_model.get(name, hw["n_jobs"])
        _write_progress(progress_path, f"Variante {label} | {name} (n_jobs={n_jobs})")
        print(f"  {name}...", end=" ", flush=True)
        start = time.time()
        perf_df = model_selection_wrapper(
            transactions_df, clf, INPUT_FEATURES, OUTPUT_FEATURE,
            param_grids[name], scoring,
            START_DATE_TRAINING_FOR_VALID, START_DATE_TRAINING_FOR_TEST,
            n_folds=n_folds, delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY, delta_assessment=DELTA_TEST,
            undersample_legit_ratio=ratio,
            performance_metrics_list_grid=perf_metrics_grid, performance_metrics_list=perf_metrics_names,
            n_jobs=n_jobs,
        )
        best_idx = perf_df['Average precision Validation'].idxmax()
        best = perf_df.loc[best_idx]
        results[name] = {
            'auc_roc_mean': best['AUC ROC Test'], 'auc_roc_std': best['AUC ROC Test Std'],
            'auprc_mean': best['Average precision Test'], 'auprc_std': best['Average precision Test Std'],
            'cp100_mean': best['Card Precision@100 Test'], 'cp100_std': best['Card Precision@100 Test Std'],
            'best_params': perf_df.loc[best_idx, 'Parameters'],
            'performances_df': perf_df,
        }
        print(f"{time.time()-start:.1f}s — AUC={results[name]['auc_roc_mean']:.4f} AUPRC={results[name]['auprc_mean']:.4f} CP100={results[name]['cp100_mean']:.4f}")

    confusion_matrices = compute_confusion_matrices_prequential(
        transactions_df, results, INPUT_FEATURES, OUTPUT_FEATURE,
        clf_classes, START_DATE_TRAINING_FOR_TEST,
        n_folds=n_folds, delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY, delta_assessment=DELTA_TEST,
        undersample_legit_ratio=ratio,
    )
    for name, cm_data in confusion_matrices.items():
        results[name]['confusion_matrix'] = cm_data

    return results


def _get_n_jobs_per_model(hw, xgb_gpu) -> dict:
    """
    n_jobs por modelo para optimizar recursos.
    - XGBoost GPU: 1 (evita NCCL/serialización en GridSearchCV paralelo)
    - RF: 4-6 (evita OOM de joblib/loky al serializar DataFrame)
    - LR: todos los núcleos (ligero)
    """
    n_cpus = hw["n_cpus"]
    n_jobs_rf = min(6, max(2, n_cpus // 2))  # RF serializa DataFrame a cada worker
    n_jobs_lr = -1
    n_jobs_xgb = 1 if xgb_gpu else hw["n_jobs"]
    return {
        "Logistic Regression": n_jobs_lr,
        "Random Forest": n_jobs_rf,
        "XGBoost": n_jobs_xgb,
    }


def main(progress_path=None):
    """Entrada principal. progress_path: Path para archivo de progreso (monitoreo)."""
    hw = get_hw_config()
    xgb_gpu = get_xgboost_gpu_params()
    n_jobs_per_model = _get_n_jobs_per_model(hw, xgb_gpu)

    print(f"Hardware: {'GPU NVIDIA' if hw['gpu_available'] else 'CPU'} | n_cpus={hw['n_cpus']}")
    print(f"  n_jobs: LR={n_jobs_per_model['Logistic Regression']} | RF={n_jobs_per_model['Random Forest']} | XGB={n_jobs_per_model['XGBoost']}")
    if xgb_gpu:
        print(f"  XGBoost GPU: {xgb_gpu}")

    _write_progress(progress_path, "Cargando datos...")
    print("Cargando datos...")
    transactions_df = load_transformed_data()
    print(f"  Total transacciones: {len(transactions_df):,}")

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
        "XGBoost": xgb.XGBClassifier(**xgb_gpu),
    }
    clf_classes = {
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "XGBoost": xgb.XGBClassifier,
    }

    all_results = {}
    start_total = time.time()

    for idx, (ratio, label) in enumerate(BALANCE_VARIANTS):
        variant_start = time.time()
        _write_progress(progress_path, f"Variante {idx+1}/{len(BALANCE_VARIANTS)}: {label}")
        results = run_single_variant(
            transactions_df, ratio, label,
            classifiers, param_grids,
            card_precision_scorer, scoring, perf_metrics_grid, perf_metrics_names,
            hw, xgb_gpu, n_jobs_per_model, clf_classes, progress_path,
        )
        all_results[label] = {'ratio': ratio, 'results': results}
        print(f"  Tiempo variante {label}: {time.time()-variant_start:.1f}s")

    elapsed = time.time() - start_total
    print(f"\n{'='*70}")
    print(f"  TIEMPO TOTAL: {elapsed/60:.1f} min")
    print(f"{'='*70}\n")

    # Guardar cada variante y tabla comparativa
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_names = list(classifiers.keys())

    # Tabla comparativa agregada
    rows = []
    for variant_label, data in all_results.items():
        ratio = data['ratio']
        results = data['results']
        for model in model_names:
            r = results[model]
            rows.append({
                'Variante': variant_label,
                'Ratio': 'original (~118:1)' if ratio is None else f'{ratio}:1',
                'Modelo': model,
                'AUC ROC': r['auc_roc_mean'],
                'AUC ROC Std': r['auc_roc_std'],
                'AUPRC': r['auprc_mean'],
                'AUPRC Std': r['auprc_std'],
                'CP@100': r['cp100_mean'],
                'CP@100 Std': r['cp100_std'],
            })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(RESULTS_DIR / 'experiment_a_balance_variants_comparison.csv', index=False)
    print("Tabla comparativa guardada: experiment_a_balance_variants_comparison.csv")

    # Guardar resultados individuales por variante
    for variant_label, data in all_results.items():
        results = data['results']
        ratio = data['ratio']
        suffix = "original" if ratio is None else f"undersamp_{int(ratio)}"

        results_table_numeric = pd.DataFrame({
            'Modelo': model_names,
            'AUC ROC': [results[n]['auc_roc_mean'] for n in model_names],
            'AUC ROC Std': [results[n]['auc_roc_std'] for n in model_names],
            'AUPRC': [results[n]['auprc_mean'] for n in model_names],
            'AUPRC Std': [results[n]['auprc_std'] for n in model_names],
            'CP@100': [results[n]['cp100_mean'] for n in model_names],
            'CP@100 Std': [results[n]['cp100_std'] for n in model_names],
        }).set_index('Modelo')
        results_table_numeric.to_csv(RESULTS_DIR / f'experiment_a_{suffix}_results.csv')

        results_to_save = {
            name: {
                **{k: v for k, v in res.items() if k != 'performances_df'},
                'card_precision_at_k': {100: res['cp100_mean']},
                'avg_precision': res['auprc_mean'],
                'auc_roc': res['auc_roc_mean'],
                'undersample_legit_ratio': ratio,
                'performances_df': res['performances_df'],
            }
            for name, res in results.items()
        }
        with open(RESULTS_DIR / f'experiment_a_{suffix}_predictions.pkl', 'wb') as f:
            pickle.dump(results_to_save, f)

        # Figuras por variante
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (key, lbl, color) in zip(axes, [
            ('auc_roc', 'AUC ROC', '#2F4D7E'),
            ('auprc', 'AUPRC', '#008000'),
            ('cp100', 'Card Precision@100', '#CA8035'),
        ]):
            means = [results[n][f'{key}_mean'] for n in model_names]
            stds = [results[n][f'{key}_std'] for n in model_names]
            ax.bar(np.arange(len(model_names)), means, 0.5, yerr=stds, capsize=5, color=color, edgecolor='black')
            ax.set_ylabel(lbl)
            ax.set_title(f'{lbl}\n({variant_label})', fontsize=12)
            ax.set_xticks(np.arange(len(model_names)))
            ax.set_xticklabels(model_names, rotation=15, ha='right')
            ax.set_ylim([0, 1.05])
        fig.suptitle(f'Variante {variant_label} ({n_folds} folds)', fontsize=14)
        plt.tight_layout()
        fig.subplots_adjust(top=0.90)
        fig.savefig(FIGURES_DIR / f'experiment_a_{suffix}_results.png', dpi=150, bbox_inches='tight')
        plt.close()

        fig_cm, axes_cm = plt.subplots(1, 3, figsize=(16, 5))
        for idx, (name, res) in enumerate(results.items()):
            cm = res['confusion_matrix']['matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Legítimo', 'Fraude'], yticklabels=['Legítimo', 'Fraude'],
                        ax=axes_cm[idx], cbar=False, annot_kws={'size': 13})
            n_fraud = cm[1, 0] + cm[1, 1]
            recall_pct = 100 * cm[1, 1] / n_fraud if n_fraud > 0 else 0
            axes_cm[idx].set_title(f"{name}\nRecall={recall_pct:.1f}%", fontsize=11)
        fig_cm.suptitle(f'Matrices de Confusión — {variant_label}', fontsize=13)
        plt.tight_layout()
        fig_cm.subplots_adjust(top=0.88)
        fig_cm.savefig(FIGURES_DIR / f'experiment_a_{suffix}_confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Guardado: experiment_a_{suffix}_*")

    # Resumen final
    print("\n" + "="*70)
    print("  RESUMEN COMPARATIVO (AUPRC por modelo y variante)")
    print("="*70)
    pivot = comp_df.pivot_table(index='Modelo', columns='Variante', values='AUPRC')
    print(pivot.round(4).to_string())
    print("\n✓ Todas las variantes completadas.")
    _write_progress(progress_path, "COMPLETADO")


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_path = results_dir / "experiment_a_balance_progress.txt"
    main(progress_path=progress_path)
