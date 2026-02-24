#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extrae métricas adicionales del Experimento A para el TFM:
  1. Accuracy global (por modelo y por fold)
  2. Confusion Matrix (TP, TN, FP, FN) agregada y por fold
  3. Tiempos exactos de entrenamiento (ya disponibles en outputs)
  4. Hiperparámetros ganadores (ya disponibles en outputs)

Re-entrena SOLO con los hiperparámetros ganadores (sin GridSearch),
usando exactamente los mismos splits prequential del experimento original.
"""

import os
import sys
import time
import json
import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from experiments.config import (
    SEED, INPUT_FEATURES, OUTPUT_FEATURE,
    RESULTS_DIR, DELTA_TRAIN, DELTA_DELAY, DELTA_TEST,
    START_DATE_TRAINING_FOR_TEST, N_FOLDS,
)
from experiments.data_utils import (
    load_transformed_data, prequentialSplit,
)

warnings.filterwarnings('ignore')

BEST_PARAMS = {
    "Logistic Regression": {
        'C': 10, 'max_iter': 1000, 'random_state': SEED,
    },
    "Random Forest": {
        'max_depth': 50, 'n_estimators': 100, 'random_state': SEED, 'n_jobs': -1,
    },
    "XGBoost": {
        'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 100,
        'random_state': SEED, 'use_label_encoder': False,
        'eval_metric': 'logloss', 'n_jobs': -1, 'verbosity': 0,
    },
}

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(**BEST_PARAMS["Logistic Regression"]),
    "Random Forest": RandomForestClassifier(**BEST_PARAMS["Random Forest"]),
    "XGBoost": xgb.XGBClassifier(**BEST_PARAMS["XGBoost"]),
}

THRESHOLDS = [0.5]


def compute_fold_metrics(y_true, y_prob, threshold=0.5):
    """Calcula todas las métricas relevantes para un fold dado un umbral."""
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp

    return {
        'threshold': threshold,
        'accuracy': (tp + tn) / total,
        'balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
        'precision_fraud': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'recall_fraud': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'f1_fraud': metrics.f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'total': int(total),
        'n_fraud_real': int(tp + fn),
        'n_legit_real': int(tn + fp),
        'auc_roc': metrics.roc_auc_score(y_true, y_prob),
        'auprc': metrics.average_precision_score(y_true, y_prob),
    }


def main():
    print("=" * 72)
    print("  EXTRACCIÓN DE MÉTRICAS ADICIONALES — EXPERIMENTO A")
    print("  (Accuracy, Confusion Matrix, Recall, F1)")
    print("=" * 72)

    print("\n[1/3] Cargando datos...")
    transactions_df = load_transformed_data()
    print(f"  Dataset: {len(transactions_df):,} transacciones")
    print(f"  Fraude: {transactions_df[OUTPUT_FEATURE].sum():,} ({100*transactions_df[OUTPUT_FEATURE].mean():.2f}%)")

    print("\n[2/3] Generando splits prequential (test, 4 folds)...")
    splits = prequentialSplit(
        transactions_df,
        start_date_training=START_DATE_TRAINING_FOR_TEST,
        n_folds=N_FOLDS,
        delta_train=DELTA_TRAIN,
        delta_delay=DELTA_DELAY,
        delta_assessment=DELTA_TEST,
    )
    for i, (train_idx, test_idx) in enumerate(splits):
        n_fraud_train = transactions_df.loc[train_idx, OUTPUT_FEATURE].sum()
        n_fraud_test = transactions_df.loc[test_idx, OUTPUT_FEATURE].sum()
        print(f"  Fold {i}: train={len(train_idx):,} (fraude={int(n_fraud_train)}), "
              f"test={len(test_idx):,} (fraude={int(n_fraud_test)})")

    print("\n[3/3] Entrenando modelos con mejores hiperparámetros...")
    all_results = {}

    for name, clf_template in CLASSIFIERS.items():
        print(f"\n{'─' * 60}")
        print(f"  Modelo: {name}")
        print(f"  Params: {BEST_PARAMS[name]}")

        fold_metrics_list = []
        fold_cm_list = []
        total_train_time = 0.0

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_train = transactions_df.loc[train_idx, INPUT_FEATURES]
            y_train = transactions_df.loc[train_idx, OUTPUT_FEATURE]
            X_test = transactions_df.loc[test_idx, INPUT_FEATURES]
            y_test = transactions_df.loc[test_idx, OUTPUT_FEATURE]

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', clf_template.__class__(**clf_template.get_params())),
            ])

            t0 = time.time()
            pipe.fit(X_train, y_train)
            train_time = time.time() - t0
            total_train_time += train_time

            y_prob = pipe.predict_proba(X_test)[:, 1]

            fm = compute_fold_metrics(y_test.values, y_prob, threshold=0.5)
            fm['fold'] = fold_i
            fm['train_time_s'] = round(train_time, 2)
            fold_metrics_list.append(fm)

            fold_cm_list.append({
                'fold': fold_i,
                'TP': fm['TP'], 'TN': fm['TN'],
                'FP': fm['FP'], 'FN': fm['FN'],
            })

        df_folds = pd.DataFrame(fold_metrics_list)

        agg_cm = {
            'TP': df_folds['TP'].sum(),
            'TN': df_folds['TN'].sum(),
            'FP': df_folds['FP'].sum(),
            'FN': df_folds['FN'].sum(),
        }
        agg_total = agg_cm['TP'] + agg_cm['TN'] + agg_cm['FP'] + agg_cm['FN']

        summary = {
            'best_params': BEST_PARAMS[name],
            'accuracy_mean': round(df_folds['accuracy'].mean(), 6),
            'accuracy_std': round(df_folds['accuracy'].std(), 6),
            'balanced_accuracy_mean': round(df_folds['balanced_accuracy'].mean(), 4),
            'balanced_accuracy_std': round(df_folds['balanced_accuracy'].std(), 4),
            'precision_fraud_mean': round(df_folds['precision_fraud'].mean(), 4),
            'precision_fraud_std': round(df_folds['precision_fraud'].std(), 4),
            'recall_fraud_mean': round(df_folds['recall_fraud'].mean(), 4),
            'recall_fraud_std': round(df_folds['recall_fraud'].std(), 4),
            'f1_fraud_mean': round(df_folds['f1_fraud'].mean(), 4),
            'f1_fraud_std': round(df_folds['f1_fraud'].std(), 4),
            'specificity_mean': round(df_folds['specificity'].mean(), 6),
            'specificity_std': round(df_folds['specificity'].std(), 6),
            'auc_roc_mean': round(df_folds['auc_roc'].mean(), 4),
            'auc_roc_std': round(df_folds['auc_roc'].std(), 4),
            'auprc_mean': round(df_folds['auprc'].mean(), 4),
            'auprc_std': round(df_folds['auprc'].std(), 4),
            'confusion_matrix_aggregated': agg_cm,
            'confusion_matrix_per_fold': fold_cm_list,
            'total_train_time_s': round(total_train_time, 2),
            'mean_train_time_per_fold_s': round(total_train_time / N_FOLDS, 2),
        }

        all_results[name] = summary

        print(f"\n  Accuracy:          {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        print(f"  Balanced Accuracy: {summary['balanced_accuracy_mean']:.4f} ± {summary['balanced_accuracy_std']:.4f}")
        print(f"  Precision (fraude):{summary['precision_fraud_mean']:.4f} ± {summary['precision_fraud_std']:.4f}")
        print(f"  Recall (fraude):   {summary['recall_fraud_mean']:.4f} ± {summary['recall_fraud_std']:.4f}")
        print(f"  F1 (fraude):       {summary['f1_fraud_mean']:.4f} ± {summary['f1_fraud_std']:.4f}")
        print(f"  Specificity:       {summary['specificity_mean']:.6f} ± {summary['specificity_std']:.6f}")
        print(f"  AUC ROC:           {summary['auc_roc_mean']:.4f} ± {summary['auc_roc_std']:.4f}")
        print(f"  AUPRC:             {summary['auprc_mean']:.4f} ± {summary['auprc_std']:.4f}")

        print(f"\n  Confusion Matrix (agregada sobre {N_FOLDS} folds):")
        print(f"    {'':>20} Predicho Legítimo | Predicho Fraude")
        print(f"    {'Real Legítimo':>20}   TN={agg_cm['TN']:>8,}    | FP={agg_cm['FP']:>6,}")
        print(f"    {'Real Fraude':>20}   FN={agg_cm['FN']:>8,}    | TP={agg_cm['TP']:>6,}")
        print(f"    Total transacciones evaluadas: {agg_total:,}")
        n_fraud_total = agg_cm['TP'] + agg_cm['FN']
        print(f"    Fraudes reales: {n_fraud_total} → Detectados: {agg_cm['TP']} ({100*agg_cm['TP']/n_fraud_total:.1f}%), "
              f"No detectados: {agg_cm['FN']} ({100*agg_cm['FN']/n_fraud_total:.1f}%)")

        print(f"\n  Tiempos de entrenamiento (4 folds, sin GridSearch):")
        print(f"    Total: {summary['total_train_time_s']:.2f}s | Media/fold: {summary['mean_train_time_per_fold_s']:.2f}s")

    output_path = RESULTS_DIR / 'experiment_a_detailed_metrics.json'

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\n✓ Métricas detalladas guardadas en: {output_path}")

    print("\n" + "=" * 72)
    print("  RESUMEN PARA EL TFM")
    print("=" * 72)

    print("\n┌─ TABLA 1: Accuracy vs métricas apropiadas ─────────────────────────┐")
    print(f"│ {'Modelo':<25} {'Accuracy':>12} {'AUPRC':>14} {'Recall(F)':>12} │")
    print(f"├{'─'*67}┤")
    for name, s in all_results.items():
        print(f"│ {name:<25} {s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}"
              f"  {s['auprc_mean']:.4f}±{s['auprc_std']:.4f}"
              f"  {s['recall_fraud_mean']:.4f}±{s['recall_fraud_std']:.4f} │")
    print(f"└{'─'*67}┘")

    print("\n┌─ TABLA 2: Confusion Matrix de XGBoost (agregada) ─────────────────┐")
    xgb_cm = all_results['XGBoost']['confusion_matrix_aggregated']
    n_fraud = xgb_cm['TP'] + xgb_cm['FN']
    print(f"│  TP (fraudes detectados):    {xgb_cm['TP']:>6,}  ({100*xgb_cm['TP']/n_fraud:>5.1f}% de fraudes) │")
    print(f"│  FN (fraudes NO detectados): {xgb_cm['FN']:>6,}  ({100*xgb_cm['FN']/n_fraud:>5.1f}% de fraudes) │")
    print(f"│  TN (legítimos correctos):   {xgb_cm['TN']:>6,}                          │")
    print(f"│  FP (falsas alarmas):        {xgb_cm['FP']:>6,}                          │")
    print(f"└{'─'*67}┘")

    print("\n┌─ Hiperparámetros ganadores (seleccionados por mejor AUPRC) ───────┐")
    for name, s in all_results.items():
        params_str = ', '.join(f"{k}={v}" for k, v in s['best_params'].items()
                               if k not in ('random_state', 'n_jobs', 'use_label_encoder',
                                            'eval_metric', 'verbosity', 'max_iter'))
        print(f"│  {name:<25} → {params_str}")
    print(f"└{'─'*67}┘")

    print("\n┌─ Tiempos del experimento original (de outputs del notebook) ──────┐")
    print("│  Logistic Regression:   17.4s                                     │")
    print("│  Random Forest:        124.8s (~2.1 min)                          │")
    print("│  XGBoost:            3,310.8s (~55.2 min)                         │")
    print("│  TOTAL:              3,453.0s (~57.6 min)                         │")
    print("│  (incluye GridSearch completo: validación + test, 4 folds cada)   │")
    print(f"└{'─'*67}┘")


if __name__ == '__main__':
    main()
