#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experimento B: Cost-Sensitive Learning — Rediseñado.

Problema original: class_weight='balanced' y scale_pos_weight='auto' aplican
ratios ~200:1, demasiado agresivos para métricas de ranking (AUPRC, CP@100).
Esto distorsiona las probabilidades y empeora el baseline.

Rediseño con tres sub-variantes:
  B1 — Cost-sensitive moderado (pesos intermedios en [1..20])
  B2 — B1 + calibración de probabilidades (CalibratedClassifierCV)
  B3 — Búsqueda ampliada con RandomizedSearchCV (solo XGBoost GPU)

Cada variante se compara con el Experimento A (baseline sin ponderación).

Uso (Docker GPU):
    docker compose run --rm experiments-gpu python experiments/run_experiment_b_standalone.py

Uso (local con Anaconda, entorno tfm):
    conda activate tfm
    python experiments/run_experiment_b_standalone.py
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
    model_selection_wrapper_randomized,
    compute_confusion_matrices_prequential,
    calibrate_and_evaluate_prequential,
    get_xgboost_cost_sensitive,
)
from experiments.hw_config import get_hw_config, get_xgboost_gpu_params
from experiments.execution_lock import acquire_lock, release_lock

QUICK = False
n_folds = 2 if QUICK else N_FOLDS

# ---------------------------------------------------------------------------
# PESOS MODERADOS: rango [1..20] en vez de 'balanced' (~200:1)
# Rationale: el dataset tiene ~0.5% fraude → balanced = ~200.
# Pesos de 5-20 penalizan FN sin destruir la calibración de probabilidades.
# ---------------------------------------------------------------------------
MODERATE_WEIGHTS_LR_RF = [None, {0: 1, 1: 5}, {0: 1, 1: 10}, {0: 1, 1: 20}]
MODERATE_WEIGHTS_XGB = [1, 3, 5, 10, 20]

# ---------------------------------------------------------------------------
# GRIDS por variante
# ---------------------------------------------------------------------------
if QUICK:
    grid_b1 = {
        "Logistic Regression": {
            'clf__C': [0.1, 1, 10],
            'clf__class_weight': [None, {0: 1, 1: 10}],
            'clf__max_iter': [1000],
            'clf__random_state': [SEED],
        },
        "Random Forest": {
            'clf__max_depth': [10, 50],
            'clf__n_estimators': [50, 100],
            'clf__class_weight': [None, {0: 1, 1: 10}],
            'clf__random_state': [SEED],
            'clf__n_jobs': [-1],
        },
        "XGBoost": {
            'clf__max_depth': [3, 6],
            'clf__n_estimators': [100],
            'clf__learning_rate': [0.1],
            'clf__scale_pos_weight': [1, 5, 10],
            'clf__random_state': [SEED],
            'clf__use_label_encoder': [False],
            'clf__eval_metric': ['logloss'],
            'clf__n_jobs': [-1],
            'clf__verbosity': [0],
        },
    }
    grid_b3_xgb_dist = {
        'clf__max_depth': [3, 6, 9],
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.05, 0.1, 0.2],
        'clf__scale_pos_weight': [1, 3, 5, 10, 20],
        'clf__min_child_weight': [1, 5],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0],
        'clf__random_state': [SEED],
        'clf__use_label_encoder': [False],
        'clf__eval_metric': ['logloss'],
        'clf__n_jobs': [-1],
        'clf__verbosity': [0],
    }
    N_ITER_B3 = 20
else:
    grid_b1 = {
        "Logistic Regression": {
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__class_weight': MODERATE_WEIGHTS_LR_RF,
            'clf__max_iter': [1000],
            'clf__random_state': [SEED],
        },
        "Random Forest": {
            'clf__max_depth': [10, 20, 50],
            'clf__n_estimators': [50, 100, 200],
            'clf__class_weight': MODERATE_WEIGHTS_LR_RF,
            'clf__random_state': [SEED],
            'clf__n_jobs': [-1],
        },
        "XGBoost": {
            'clf__max_depth': [3, 6, 9],
            'clf__n_estimators': [100, 200],
            'clf__learning_rate': [0.05, 0.1, 0.2],
            'clf__scale_pos_weight': MODERATE_WEIGHTS_XGB,
            'clf__random_state': [SEED],
            'clf__use_label_encoder': [False],
            'clf__eval_metric': ['logloss'],
            'clf__n_jobs': [-1],
            'clf__verbosity': [0],
        },
    }
    # B3: espacio ampliado para RandomizedSearchCV (XGBoost GPU)
    grid_b3_xgb_dist = {
        'clf__max_depth': [3, 6, 9, 12],
        'clf__n_estimators': [100, 200, 300, 500],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'clf__scale_pos_weight': [1, 3, 5, 10, 15, 20, 30],
        'clf__min_child_weight': [1, 3, 5, 10],
        'clf__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'clf__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'clf__gamma': [0, 0.1, 0.3, 0.5],
        'clf__reg_alpha': [0, 0.01, 0.1],
        'clf__reg_lambda': [1, 2, 5],
        'clf__random_state': [SEED],
        'clf__use_label_encoder': [False],
        'clf__eval_metric': ['logloss'],
        'clf__n_jobs': [-1],
        'clf__verbosity': [0],
    }
    N_ITER_B3 = 60


def main():
    if not acquire_lock("experiment_b"):
        print("ERROR: Otra ejecución del experimento en curso (execution.lock activo).")
        print("  Espera a que termine o elimina experiments/results/execution.lock si es huérfano.")
        sys.exit(1)
    try:
        return _main_impl()
    finally:
        release_lock()


def _format_metric(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def _print_result(name, res, prefix=""):
    label = f"{prefix}{name}" if prefix else name
    print(f"  {label:40s}  AUC={res['auc_roc_mean']:.4f}±{res['auc_roc_std']:.4f}"
          f"  AUPRC={res['auprc_mean']:.4f}±{res['auprc_std']:.4f}"
          f"  CP100={res['cp100_mean']:.4f}±{res['cp100_std']:.4f}", flush=True)


def _run_b1_model(name, clf, param_grid, transactions_df, scoring,
                   perf_metrics_grid, perf_metrics_names, n_jobs):
    """Entrena un modelo B1 (cost-sensitive moderado) con GridSearchCV."""
    t0 = time.time()
    perf_df = model_selection_wrapper(
        transactions_df, clf, INPUT_FEATURES, OUTPUT_FEATURE,
        param_grid, scoring,
        START_DATE_TRAINING_FOR_VALID, START_DATE_TRAINING_FOR_TEST,
        n_folds=n_folds, delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY,
        delta_assessment=DELTA_TEST,
        performance_metrics_list_grid=perf_metrics_grid,
        performance_metrics_list=perf_metrics_names,
        n_jobs=n_jobs,
    )
    sel_col = 'Average precision Validation'
    if sel_col not in perf_df.columns or perf_df[sel_col].isna().all():
        sel_col = 'AUC ROC Validation'
    best_idx = perf_df[sel_col].idxmax()
    best = perf_df.loc[best_idx]
    return {
        'auc_roc_mean': best['AUC ROC Test'],
        'auc_roc_std': best['AUC ROC Test Std'],
        'auprc_mean': best['Average precision Test'],
        'auprc_std': best['Average precision Test Std'],
        'cp100_mean': best['Card Precision@100 Test'],
        'cp100_std': best['Card Precision@100 Test Std'],
        'best_params': perf_df.loc[best_idx, 'Parameters'],
        'performances_df': perf_df,
    }, time.time() - t0


def _main_impl():
    hw = get_hw_config()
    print(f"Hardware: {'GPU NVIDIA' if hw['gpu_available'] else 'CPU'} | "
          f"n_jobs={hw['n_jobs']} | CPUs={hw['n_cpus']}")
    xgb_gpu = get_xgboost_gpu_params()
    n_jobs_xgb = 1
    if xgb_gpu:
        print(f"  XGBoost GPU: {xgb_gpu} | GridSearch n_jobs={n_jobs_xgb}")

    print("Cargando datos...")
    transactions_df = load_transformed_data()

    XGBoostCostSensitive = get_xgboost_cost_sensitive()

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

    n_jobs_cpu = hw['n_jobs']

    results_all = {}
    start_total = time.time()

    # ===================================================================
    # FASE B1: Cost-Sensitive MODERADO (GridSearchCV, secuencial)
    # Ejecución secuencial para evitar OOM: cada modelo usa todos los CPUs.
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  FASE B1: Cost-Sensitive Moderado ({n_folds} folds)")
    print(f"  Pesos LR/RF: {MODERATE_WEIGHTS_LR_RF}")
    print(f"  Pesos XGBoost: {MODERATE_WEIGHTS_XGB}")
    print(f"{'='*70}\n")

    classifiers_b1 = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBoostCostSensitive(**xgb_gpu),
    }

    results_b1 = {}
    for name, clf in classifiers_b1.items():
        n_jobs = n_jobs_xgb if (name == "XGBoost" and xgb_gpu) else n_jobs_cpu
        print(f"  Entrenando {name} (n_jobs={n_jobs})...", flush=True)
        res, elapsed = _run_b1_model(
            name, clf, grid_b1[name],
            transactions_df, scoring, perf_metrics_grid, perf_metrics_names, n_jobs,
        )
        results_b1[name] = res
        _print_result(name, res, prefix=f"  B1 ({elapsed:.0f}s) ")

    for name, res in results_b1.items():
        results_all[f"B1_{name}"] = res

    # ===================================================================
    # FASE B2: Calibración de probabilidades sobre los mejores modelos B1
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  FASE B2: Calibración de Probabilidades (isotonic regression)")
    print(f"{'='*70}\n")

    clf_classes_map = {
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "XGBoost": XGBoostCostSensitive,
    }

    results_b2 = {}
    for name, res_b1 in results_b1.items():
        t0 = time.time()
        cal_result = calibrate_and_evaluate_prequential(
            transactions_df,
            classifier_class=clf_classes_map[name],
            best_params=res_b1['best_params'],
            input_features=INPUT_FEATURES,
            output_feature=OUTPUT_FEATURE,
            start_date_training=START_DATE_TRAINING_FOR_TEST,
            n_folds=n_folds,
            delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY,
            delta_assessment=DELTA_TEST,
            calibration_method='isotonic',
            calibration_cv=3,
        )
        cal_result['best_params'] = res_b1['best_params']
        results_b2[name] = cal_result
        _print_result(name, cal_result, prefix="  B2 cal ")
        print(f"    ({time.time() - t0:.0f}s)", flush=True)

    for name, res in results_b2.items():
        results_all[f"B2_{name}"] = res

    # ===================================================================
    # FASE B3: RandomizedSearchCV ampliado para XGBoost (GPU)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  FASE B3: RandomizedSearchCV XGBoost ({N_ITER_B3} iter, espacio ampliado)")
    print(f"{'='*70}\n")

    t0 = time.time()
    xgb_clf = XGBoostCostSensitive(**xgb_gpu)
    perf_df_b3 = model_selection_wrapper_randomized(
        transactions_df, xgb_clf, INPUT_FEATURES, OUTPUT_FEATURE,
        grid_b3_xgb_dist, scoring,
        START_DATE_TRAINING_FOR_VALID, START_DATE_TRAINING_FOR_TEST,
        n_folds=n_folds, n_iter=N_ITER_B3,
        delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY,
        delta_assessment=DELTA_TEST,
        performance_metrics_list_grid=perf_metrics_grid,
        performance_metrics_list=perf_metrics_names,
        n_jobs=n_jobs_xgb if xgb_gpu else n_jobs_cpu,
    )
    sel_col = 'Average precision Validation'
    if sel_col not in perf_df_b3.columns or perf_df_b3[sel_col].isna().all():
        sel_col = 'AUC ROC Validation'
    best_idx = perf_df_b3[sel_col].idxmax()
    best_b3 = perf_df_b3.loc[best_idx]
    results_b3_xgb = {
        'auc_roc_mean': best_b3['AUC ROC Test'],
        'auc_roc_std': best_b3['AUC ROC Test Std'],
        'auprc_mean': best_b3['Average precision Test'],
        'auprc_std': best_b3['Average precision Test Std'],
        'cp100_mean': best_b3['Card Precision@100 Test'],
        'cp100_std': best_b3['Card Precision@100 Test Std'],
        'best_params': perf_df_b3.loc[best_idx, 'Parameters'],
        'performances_df': perf_df_b3,
    }
    elapsed_b3 = time.time() - t0
    _print_result("XGBoost", results_b3_xgb, prefix="  B3 rand ")
    print(f"    ({elapsed_b3:.0f}s)", flush=True)
    results_all["B3_XGBoost"] = results_b3_xgb

    # B3 + calibración
    t0 = time.time()
    cal_b3 = calibrate_and_evaluate_prequential(
        transactions_df,
        classifier_class=XGBoostCostSensitive,
        best_params=results_b3_xgb['best_params'],
        input_features=INPUT_FEATURES,
        output_feature=OUTPUT_FEATURE,
        start_date_training=START_DATE_TRAINING_FOR_TEST,
        n_folds=n_folds,
        delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY,
        delta_assessment=DELTA_TEST,
        calibration_method='isotonic', calibration_cv=3,
    )
    cal_b3['best_params'] = results_b3_xgb['best_params']
    results_all["B3_XGBoost_cal"] = cal_b3
    _print_result("XGBoost", cal_b3, prefix="  B3 rand+cal ")
    print(f"    ({time.time() - t0:.0f}s)", flush=True)

    elapsed_total = time.time() - start_total
    print(f"\nTiempo total: {elapsed_total:.1f}s\n")

    # ===================================================================
    # TABLA RESUMEN COMPARATIVA
    # ===================================================================
    print(f"\n{'='*70}")
    print("  RESULTADOS COMPARATIVOS (media ± desv. estándar)")
    print(f"{'='*70}\n")

    summary_rows = []
    for key, res in results_all.items():
        summary_rows.append({
            'Variante': key,
            'AUC ROC': _format_metric(res['auc_roc_mean'], res['auc_roc_std']),
            'AUPRC': _format_metric(res['auprc_mean'], res['auprc_std']),
            'CP@100': _format_metric(res['cp100_mean'], res['cp100_std']),
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print()

    # Best params por variante
    for key, res in results_all.items():
        if 'best_params' in res:
            params = res['best_params']
            relevant = {k.replace('clf__', ''): v for k, v in params.items()
                        if k.startswith('clf__') and k not in (
                            'clf__random_state', 'clf__use_label_encoder',
                            'clf__eval_metric', 'clf__n_jobs', 'clf__verbosity',
                            'clf__max_iter',
                        )}
            print(f"  {key}: {relevant}")

    # ===================================================================
    # MATRICES DE CONFUSIÓN (mejores B1)
    # ===================================================================
    confusion_matrices = compute_confusion_matrices_prequential(
        transactions_df, results_b1, INPUT_FEATURES, OUTPUT_FEATURE,
        clf_classes_map, START_DATE_TRAINING_FOR_TEST,
        n_folds=n_folds, delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY,
        delta_assessment=DELTA_TEST,
    )
    for name, cm_data in confusion_matrices.items():
        results_b1[name]['confusion_matrix'] = cm_data

    # ===================================================================
    # GUARDAR RESULTADOS
    # ===================================================================
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_numeric = pd.DataFrame([
        {
            'Variante': key,
            'AUC ROC': res['auc_roc_mean'],
            'AUC ROC Std': res['auc_roc_std'],
            'AUPRC': res['auprc_mean'],
            'AUPRC Std': res['auprc_std'],
            'CP@100': res['cp100_mean'],
            'CP@100 Std': res['cp100_std'],
        }
        for key, res in results_all.items()
    ]).set_index('Variante')
    summary_numeric.to_csv(RESULTS_DIR / 'experiment_b_results.csv')

    with open(RESULTS_DIR / 'experiment_b_predictions.pkl', 'wb') as f:
        pickle.dump(results_all, f)

    # ===================================================================
    # FIGURAS
    # ===================================================================
    _generate_figures(results_all, results_b1, confusion_matrices)

    print(f"\nGuardado: {RESULTS_DIR / 'experiment_b_results.csv'}")
    print(f"Guardado: {RESULTS_DIR / 'experiment_b_predictions.pkl'}")
    return results_all


def _generate_figures(results_all, results_b1, confusion_matrices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figura 1: Comparativa de todas las variantes ---
    variants = list(results_all.keys())
    x_pos = np.arange(len(variants))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    palette = plt.cm.Set2(np.linspace(0, 1, len(variants)))

    for ax, (key, label) in zip(axes, [
        ('auc_roc', 'AUC ROC'),
        ('auprc', 'AUPRC'),
        ('cp100', 'Card Precision@100'),
    ]):
        means = [results_all[v][f'{key}_mean'] for v in variants]
        stds = [results_all[v][f'{key}_std'] for v in variants]
        bars = ax.bar(x_pos, means, 0.6, yerr=stds, capsize=4,
                       color=palette, edgecolor='black', linewidth=0.5)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=13)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variants, rotation=35, ha='right', fontsize=8)
        ax.set_ylim([0, 1.05])
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle(f'Cost-Sensitive — Comparativa de Variantes ({n_folds} folds)',
                 fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(FIGURES_DIR / 'experiment_b_cost_sensitive_results.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura: {FIGURES_DIR / 'experiment_b_cost_sensitive_results.png'}")

    # --- Figura 2: Matrices de confusión (B1) ---
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(16, 5))
    labels = ['Legítimo', 'Fraude']
    for idx, (name, cm_data) in enumerate(confusion_matrices.items()):
        cm = cm_data['matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=axes_cm[idx], cbar=False, annot_kws={'size': 13})
        n_fraud = cm[1, 0] + cm[1, 1]
        recall_pct = 100 * cm[1, 1] / n_fraud if n_fraud > 0 else 0
        axes_cm[idx].set_title(
            f"{name}\nTP={cm[1,1]:,}  FN={cm[1,0]:,}  (Recall={recall_pct:.1f}%)",
            fontsize=11)
        axes_cm[idx].set_ylabel('Real')
        axes_cm[idx].set_xlabel('Predicho')
    fig_cm.suptitle(
        f'Matrices de Confusión — Cost-Sensitive Moderado ({n_folds} folds, threshold=0.5)',
        fontsize=13)
    plt.tight_layout()
    fig_cm.subplots_adjust(top=0.88)
    fig_cm.savefig(FIGURES_DIR / 'experiment_b_confusion_matrices.png',
                   dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Matrices de confusión: {FIGURES_DIR / 'experiment_b_confusion_matrices.png'}")


if __name__ == "__main__":
    main()
