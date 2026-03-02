#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experimento B: variantes para abordar el desbalanceo de clases.

Ejecuta el Experimento B con distintas estrategias y compara resultados:
  B1: Selección por AUPRC (actual)
  B2: Selección por CP@100 (métrica de negocio)
  B3: scale_pos_weight más agresivo [1, 100, 200, 500, 'auto']
  B4: Solo cost-sensitive (sin baseline) — fuerza la técnica

Uso: conda activate tfm && python experiments/run_experiment_b_variants.py
"""

import sys
import time
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    compute_confusion_matrices_prequential,
    get_xgboost_cost_sensitive,
)
from experiments.hw_config import get_hw_config, get_xgboost_gpu_params
from experiments.execution_lock import acquire_lock, release_lock

# Modo: False=completo (~25 min/variante), True=rapido (~5 min/variante)
QUICK = False
n_folds = 2 if QUICK else N_FOLDS

# Definición de variantes
VARIANTS = {
    "B1_AUPRC": {
        "selection_metric": "Average precision Validation",
        "description": "Selección por AUPRC (baseline + cost-sensitive en grid)",
        "param_grids": None,  # usa el grid estándar
    },
    "B2_CP100": {
        "selection_metric": "Card Precision@100 Validation",
        "description": "Selección por CP@100 (métrica de negocio)",
        "param_grids": None,
    },
    "B3_agresivo": {
        "selection_metric": "Average precision Validation",
        "description": "scale_pos_weight agresivo [1,100,200,500,auto]",
        "param_grids": "aggressive",
    },
    "B4_solo_cs": {
        "selection_metric": "Average precision Validation",
        "description": "Solo cost-sensitive (sin baseline en grid)",
        "param_grids": "cost_sensitive_only",
    },
}


def get_param_grids(variant_key):
    """Retorna los param_grids según la variante."""
    base_lr = {
        'clf__C': [0.01, 0.1, 1, 10, 100] if not QUICK else [0.1, 1, 10],
        'clf__class_weight': [None, 'balanced'],
        'clf__max_iter': [1000],
        'clf__random_state': [SEED],
    }
    base_rf = {
        'clf__max_depth': [10, 20, 50] if not QUICK else [10, 50],
        'clf__n_estimators': [50, 100, 150] if not QUICK else [50, 100],
        'clf__class_weight': [None, 'balanced'],
        'clf__random_state': [SEED],
        'clf__n_jobs': [-1],
    }
    base_xgb = {
        'clf__max_depth': [3, 6, 9] if not QUICK else [3, 6],
        'clf__n_estimators': [50, 100],
        'clf__learning_rate': [0.3],
        'clf__scale_pos_weight': [1, 50, 100, 'auto'],
        'clf__random_state': [SEED],
        'clf__use_label_encoder': [False],
        'clf__eval_metric': ['logloss'],
        'clf__n_jobs': [-1],
        'clf__verbosity': [0],
    }

    if variant_key == "aggressive":
        base_xgb['clf__scale_pos_weight'] = [1, 100, 200, 500, 'auto']
    elif variant_key == "cost_sensitive_only":
        base_lr['clf__class_weight'] = ['balanced']
        base_rf['clf__class_weight'] = ['balanced']
        base_xgb['clf__scale_pos_weight'] = [50, 100, 200, 'auto']

    return {
        "Logistic Regression": base_lr,
        "Random Forest": base_rf,
        "XGBoost": base_xgb,
    }


def run_variant(variant_name, config, transactions_df, card_precision_scorer,
                perf_metrics_grid, perf_metrics_names, hw, xgb_gpu):
    """Ejecuta una variante y retorna resultados."""
    nf = n_folds

    selection = config["selection_metric"]
    param_grids = get_param_grids(config.get("param_grids"))

    XGBoostCostSensitive = get_xgboost_cost_sensitive()
    scoring = {
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',
        'card_precision@100': card_precision_scorer,
    }

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBoostCostSensitive(**xgb_gpu),
    }
    n_jobs_cpu = max(2, hw['n_cpus'] // 3) if hw['gpu_available'] else hw['n_jobs']

    def _train_one(name):
        clf = classifiers[name]
        n_jobs = 1 if (name == "XGBoost" and xgb_gpu) else n_jobs_cpu
        perf_df = model_selection_wrapper(
            transactions_df, clf, INPUT_FEATURES, OUTPUT_FEATURE,
            param_grids[name], scoring,
            START_DATE_TRAINING_FOR_VALID, START_DATE_TRAINING_FOR_TEST,
            n_folds=nf, delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY, delta_assessment=DELTA_TEST,
            performance_metrics_list_grid=perf_metrics_grid, performance_metrics_list=perf_metrics_names,
            n_jobs=n_jobs,
        )
        if selection not in perf_df.columns or perf_df[selection].isna().all():
            best_idx = perf_df['Average precision Validation'].idxmax()
        else:
            best_idx = perf_df[selection].idxmax()
        best = perf_df.loc[best_idx]
        return name, {
            'auc_roc_mean': best['AUC ROC Test'], 'auc_roc_std': best['AUC ROC Test Std'],
            'auprc_mean': best['Average precision Test'], 'auprc_std': best['Average precision Test Std'],
            'cp100_mean': best['Card Precision@100 Test'], 'cp100_std': best['Card Precision@100 Test Std'],
            'best_params': perf_df.loc[best_idx, 'Parameters'],
        }

    results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_train_one, n): n for n in classifiers}
        for future in as_completed(futures):
            name, res = future.result()
            results[name] = res
            print(f"    {name}: AUPRC={res['auprc_mean']:.4f} CP100={res['cp100_mean']:.4f}", flush=True)
    return results


def main():
    if not acquire_lock("experiment_b_variants"):
        print("ERROR: Otra ejecución del experimento en curso (execution.lock activo).")
        print("  Espera a que termine o elimina experiments/results/execution.lock si es huérfano.")
        sys.exit(1)
    try:
        return _main_impl()
    finally:
        release_lock()


def _main_impl():
    hw = get_hw_config()
    print(f"Hardware: {'GPU NVIDIA' if hw['gpu_available'] else 'CPU'} | n_jobs={hw['n_jobs']} | CPUs={hw['n_cpus']}")
    xgb_gpu = get_xgboost_gpu_params()
    if xgb_gpu:
        print(f"  XGBoost GPU: {xgb_gpu}")

    print("Cargando datos...")
    transactions_df = load_transformed_data()

    card_precision_scorer = sklearn.metrics.make_scorer(
        card_precision_top_k_custom, needs_proba=True,
        top_k=100, transactions_df=transactions_df,
    )
    perf_metrics_grid = ['roc_auc', 'average_precision', 'card_precision@100']
    perf_metrics_names = ['AUC ROC', 'Average precision', 'Card Precision@100']

    all_results = {}
    start_total = time.time()

    progress_file = RESULTS_DIR / "experiment_b_progress.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for vname, config in VARIANTS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Variante {vname}: {config['description']}", flush=True)
        print(f"{'='*60}", flush=True)
        progress_file.write_text(f"Ejecutando: {vname} - {config['description']}\n", encoding='utf-8')
        t0 = time.time()
        all_results[vname] = run_variant(
            vname, config, transactions_df, card_precision_scorer,
            perf_metrics_grid, perf_metrics_names, hw, xgb_gpu,
        )
        for name, res in all_results[vname].items():
            print(f"  {name}: AUPRC={res['auprc_mean']:.4f}  CP100={res['cp100_mean']:.4f}", flush=True)
        elapsed_v = time.time() - t0
        print(f"  Tiempo: {elapsed_v:.1f}s", flush=True)
        progress_file.write_text(f"Completado: {vname} en {elapsed_v:.1f}s\n", encoding='utf-8')

    elapsed = time.time() - start_total
    print(f"\nTiempo total variantes: {elapsed:.1f}s")

    # Comparativa
    print("\n" + "="*80)
    print("COMPARATIVA DE VARIANTES")
    print("="*80)

    comp_data = []
    for vname, res_dict in all_results.items():
        for model, res in res_dict.items():
            comp_data.append({
                'Variante': vname,
                'Modelo': model,
                'AUC ROC': res['auc_roc_mean'],
                'AUPRC': res['auprc_mean'],
                'CP@100': res['cp100_mean'],
            })

    comp_df = pd.DataFrame(comp_data)
    comp_pivot = comp_df.pivot_table(
        index='Modelo', columns='Variante',
        values=['AUPRC', 'CP@100'],
        aggfunc='first'
    )
    print("\nAUPRC por variante y modelo:")
    print(comp_pivot['AUPRC'].to_string())
    print("\nCP@100 por variante y modelo:")
    print(comp_pivot['CP@100'].to_string())

    # Mejor por métrica
    print("\n--- Mejor variante por métrica (XGBoost) ---")
    xgb_df = comp_df[comp_df['Modelo'] == 'XGBoost']
    best_auprc = xgb_df.loc[xgb_df['AUPRC'].idxmax()]
    best_cp100 = xgb_df.loc[xgb_df['CP@100'].idxmax()]
    print(f"Mejor AUPRC: {best_auprc['Variante'].values[0]} ({best_auprc['AUPRC'].values[0]:.4f})")
    print(f"Mejor CP@100: {best_cp100['Variante'].values[0]} ({best_cp100['CP@100'].values[0]:.4f})")

    # Guardar
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(RESULTS_DIR / 'experiment_b_variants_comparison.csv', index=False)
    with open(RESULTS_DIR / 'experiment_b_variants_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nGuardado: {RESULTS_DIR / 'experiment_b_variants_comparison.csv'}")

    return all_results


if __name__ == "__main__":
    main()
