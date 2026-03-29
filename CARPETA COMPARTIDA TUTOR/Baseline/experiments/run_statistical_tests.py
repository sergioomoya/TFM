#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests de significancia estadística, latencia de inferencia XAI
y análisis de multicolinealidad para la memoria del TFM.

Genera:
  - Per-fold metrics para Tabla 1 (3 modelos × 4 folds)
  - Per-fold metrics para Tabla 5 (full vs ablated × 4 folds)
  - Friedman + Wilcoxon pairwise tests
  - Latencia de predict_proba y TreeSHAP por transacción
  - Matriz de correlación entre variables RFM

Uso:
  .venv/Scripts/python experiments/run_statistical_tests.py
  docker compose run --rm experiments python experiments/run_statistical_tests.py
"""

import sys
import time
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as sk_metrics
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import (
    SEED, INPUT_FEATURES, OUTPUT_FEATURE,
    RESULTS_DIR, FIGURES_DIR,
    DELTA_TRAIN, DELTA_DELAY, DELTA_TEST,
    START_DATE_TRAINING_FOR_TEST, START_DATE_TRAINING,
    N_FOLDS, BASELINE_PARAMS,
)
from experiments.data_utils import (
    load_transformed_data,
    prequentialSplit,
    card_precision_top_k,
)

warnings.filterwarnings('ignore')

BEST_PARAMS_TABLE1 = {
    "Logistic Regression": {"C": 10, "max_iter": 1000, "random_state": SEED},
    "Random Forest": {"max_depth": 50, "n_estimators": 100, "random_state": SEED, "n_jobs": -1},
    "XGBoost": {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.3,
                "random_state": SEED, "use_label_encoder": False,
                "eval_metric": "logloss", "n_jobs": -1, "verbosity": 0},
}


def evaluate_fold(
    transactions_df: pd.DataFrame,
    train_idx: list,
    test_idx: list,
    features: list,
    clf,
) -> dict[str, float]:
    """Entrena y evalúa un clasificador en un fold prequential."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(transactions_df.loc[train_idx, features])
    y_train = transactions_df.loc[train_idx, OUTPUT_FEATURE]
    X_test = scaler.transform(transactions_df.loc[test_idx, features])
    y_test = transactions_df.loc[test_idx, OUTPUT_FEATURE]

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    auc_roc = sk_metrics.roc_auc_score(y_test, y_prob)
    auprc = sk_metrics.average_precision_score(y_test, y_prob)

    pred_df = transactions_df.loc[test_idx].copy()
    pred_df['predictions'] = y_prob
    _, _, cp100 = card_precision_top_k(pred_df, top_k=100)

    return {"auc_roc": auc_roc, "auprc": auprc, "cp100": cp100,
            "model": clf, "scaler": scaler}


def make_clf(name: str):
    """Instancia un clasificador con los mejores hiperparámetros de la Tabla 1."""
    p = BEST_PARAMS_TABLE1[name]
    if name == "Logistic Regression":
        return LogisticRegression(**p)
    elif name == "Random Forest":
        return RandomForestClassifier(**p)
    elif name == "XGBoost":
        return xgb.XGBClassifier(**p)


def run_table1_folds(transactions_df: pd.DataFrame, splits: list) -> pd.DataFrame:
    """Evalúa los 3 modelos en cada fold y devuelve un DataFrame de métricas."""
    rows = []
    for name in BEST_PARAMS_TABLE1:
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            clf = make_clf(name)
            m = evaluate_fold(transactions_df, train_idx, test_idx, INPUT_FEATURES, clf)
            rows.append({"model": name, "fold": fold_i,
                         "auc_roc": m["auc_roc"], "auprc": m["auprc"], "cp100": m["cp100"]})
            print(f"  {name} fold {fold_i}: AUPRC={m['auprc']:.4f}  CP@100={m['cp100']:.4f}")
    return pd.DataFrame(rows)


def run_table5_folds(transactions_df: pd.DataFrame, splits: list) -> pd.DataFrame:
    """Evalúa modelo completo vs ablated en cada fold (Tabla 5)."""
    import shap as shap_lib
    shap_csv = RESULTS_DIR / 'experiment_d_shap_mean_impact.csv'
    if shap_csv.exists():
        top_feat = pd.read_csv(shap_csv).sort_values('mean_abs_SHAP', ascending=False).iloc[0]['Feature']
    else:
        top_feat = 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW'

    features_ablated = [f for f in INPUT_FEATURES if f != top_feat]
    print(f"\n  Top feature ablated: {top_feat}")

    rows = []
    for variant, feats in [("full", INPUT_FEATURES), ("ablated", features_ablated)]:
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            clf = xgb.XGBClassifier(**BASELINE_PARAMS["XGBoost"])
            m = evaluate_fold(transactions_df, train_idx, test_idx, feats, clf)
            rows.append({"variant": variant, "fold": fold_i,
                         "auc_roc": m["auc_roc"], "auprc": m["auprc"], "cp100": m["cp100"]})
            print(f"  {variant} fold {fold_i}: AUPRC={m['auprc']:.4f}  CP@100={m['cp100']:.4f}")
    return pd.DataFrame(rows)


def statistical_tests_table1(fold_df: pd.DataFrame) -> str:
    """Friedman + Wilcoxon pairwise sobre métricas por fold (Tabla 1)."""
    lines = []
    models = list(BEST_PARAMS_TABLE1.keys())

    for metric in ["auprc", "cp100", "auc_roc"]:
        metric_label = {"auprc": "AUPRC", "cp100": "CP@100", "auc_roc": "AUC ROC"}[metric]
        samples = [fold_df[fold_df["model"] == m][metric].values for m in models]

        stat_f, p_friedman = stats.friedmanchisquare(*samples)
        lines.append(f"\n### {metric_label}")
        lines.append(f"Friedman chi2={stat_f:.4f}, p={p_friedman:.4f}"
                      f" {'(significativo p<0.05)' if p_friedman < 0.05 else '(no significativo)'}")

        for (i, m1), (j, m2) in combinations(enumerate(models), 2):
            stat_w, p_wilcoxon = stats.wilcoxon(samples[i], samples[j])
            lines.append(f"  Wilcoxon {m1} vs {m2}: W={stat_w:.1f}, p={p_wilcoxon:.4f}"
                          f" {'*' if p_wilcoxon < 0.05 else ''}")
    return "\n".join(lines)


def statistical_tests_table5(fold_df: pd.DataFrame) -> str:
    """Wilcoxon signed-rank entre modelo completo y ablated (Tabla 5)."""
    lines = []
    for metric in ["auprc", "cp100", "auc_roc"]:
        metric_label = {"auprc": "AUPRC", "cp100": "CP@100", "auc_roc": "AUC ROC"}[metric]
        full_vals = fold_df[fold_df["variant"] == "full"][metric].values
        ablated_vals = fold_df[fold_df["variant"] == "ablated"][metric].values

        try:
            stat_w, p_val = stats.wilcoxon(full_vals, ablated_vals)
            lines.append(f"  {metric_label}: W={stat_w:.1f}, p={p_val:.4f}"
                          f" {'(significativo)' if p_val < 0.05 else '(no significativo)'}")
        except ValueError as e:
            lines.append(f"  {metric_label}: No se pudo calcular Wilcoxon ({e})")
    return "\n".join(lines)


def measure_latency(transactions_df: pd.DataFrame) -> dict:
    """Mide latencia de predict_proba y TreeSHAP por transacción."""
    import shap as shap_lib

    train_df_full, test_df_full = None, None
    splits = prequentialSplit(
        transactions_df, start_date_training=START_DATE_TRAINING,
        n_folds=1, delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY,
        delta_assessment=DELTA_TEST,
    )
    train_idx, test_idx = splits[0]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(transactions_df.loc[train_idx, INPUT_FEATURES])
    y_train = transactions_df.loc[train_idx, OUTPUT_FEATURE]
    X_test = scaler.transform(transactions_df.loc[test_idx, INPUT_FEATURES])

    model = xgb.XGBClassifier(**BASELINE_PARAMS["XGBoost"])
    model.fit(X_train, y_train)

    n_test = len(X_test)
    batch_sizes = [1, 10, 100, 1000]
    results = {"predict_proba": {}, "shap": {}}

    for bs in batch_sizes:
        if bs > n_test:
            continue
        X_batch = X_test[:bs]

        times_pred = []
        for _ in range(20):
            t0 = time.perf_counter()
            model.predict_proba(X_batch)
            times_pred.append(time.perf_counter() - t0)
        median_pred = np.median(times_pred)
        per_tx_pred_ms = (median_pred / bs) * 1000
        results["predict_proba"][bs] = {
            "total_ms": median_pred * 1000,
            "per_tx_ms": per_tx_pred_ms,
        }

        explainer = shap_lib.TreeExplainer(model)
        times_shap = []
        n_rep = max(3, 20 // bs)
        for _ in range(n_rep):
            t0 = time.perf_counter()
            explainer.shap_values(X_batch)
            times_shap.append(time.perf_counter() - t0)
        median_shap = np.median(times_shap)
        per_tx_shap_ms = (median_shap / bs) * 1000
        results["shap"][bs] = {
            "total_ms": median_shap * 1000,
            "per_tx_ms": per_tx_shap_ms,
        }

    return results


def correlation_rfm(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la matriz de correlación de Pearson entre las features RFM."""
    splits = prequentialSplit(
        transactions_df, start_date_training=START_DATE_TRAINING,
        n_folds=1, delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY,
        delta_assessment=DELTA_TEST,
    )
    train_idx, _ = splits[0]
    return transactions_df.loc[train_idx, INPUT_FEATURES].corr()


def main():
    print("=" * 70)
    print("  TESTS DE SIGNIFICANCIA ESTADÍSTICA + LATENCIA XAI")
    print("=" * 70)

    print("\nCargando datos...")
    transactions_df = load_transformed_data()
    print(f"  {len(transactions_df):,} transacciones cargadas")

    # --- Splits prequential para Tabla 1 (usa START_DATE_TRAINING_FOR_TEST) ---
    splits_t1 = prequentialSplit(
        transactions_df,
        start_date_training=START_DATE_TRAINING_FOR_TEST,
        n_folds=N_FOLDS,
        delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY,
        delta_assessment=DELTA_TEST,
    )

    # --- Splits prequential para Tabla 5 (usa START_DATE_TRAINING) ---
    splits_t5 = prequentialSplit(
        transactions_df,
        start_date_training=START_DATE_TRAINING,
        n_folds=N_FOLDS,
        delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY,
        delta_assessment=DELTA_TEST,
    )

    # ===================== TABLA 1 =====================
    print(f"\n{'='*70}")
    print("  TABLA 1: Métricas por fold (3 modelos × {N_FOLDS} folds)")
    print(f"{'='*70}")
    t1_fold_df = run_table1_folds(transactions_df, splits_t1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t1_fold_df.to_csv(RESULTS_DIR / 'table1_per_fold_metrics.csv', index=False)
    print(f"\n  Guardado: {RESULTS_DIR / 'table1_per_fold_metrics.csv'}")

    print(f"\n--- Tests estadísticos (Tabla 1) ---")
    t1_stats = statistical_tests_table1(t1_fold_df)
    print(t1_stats)

    # ===================== TABLA 5 =====================
    print(f"\n{'='*70}")
    print(f"  TABLA 5: Ablación por fold (full vs ablated × {N_FOLDS} folds)")
    print(f"{'='*70}")
    t5_fold_df = run_table5_folds(transactions_df, splits_t5)
    t5_fold_df.to_csv(RESULTS_DIR / 'table5_per_fold_metrics.csv', index=False)
    print(f"\n  Guardado: {RESULTS_DIR / 'table5_per_fold_metrics.csv'}")

    print(f"\n--- Tests estadísticos (Tabla 5: full vs ablated) ---")
    t5_stats = statistical_tests_table5(t5_fold_df)
    print(t5_stats)

    # ===================== LATENCIA =====================
    print(f"\n{'='*70}")
    print("  LATENCIA DE INFERENCIA: predict_proba + TreeSHAP")
    print(f"{'='*70}")
    latency = measure_latency(transactions_df)
    print("\n  predict_proba (XGBoost):")
    for bs, vals in latency["predict_proba"].items():
        print(f"    batch={bs}: {vals['total_ms']:.2f} ms total, {vals['per_tx_ms']:.4f} ms/tx")
    print("\n  TreeSHAP:")
    for bs, vals in latency["shap"].items():
        print(f"    batch={bs}: {vals['total_ms']:.2f} ms total, {vals['per_tx_ms']:.4f} ms/tx")

    # ===================== CORRELACIÓN =====================
    print(f"\n{'='*70}")
    print("  CORRELACIÓN ENTRE VARIABLES RFM")
    print(f"{'='*70}")
    corr = correlation_rfm(transactions_df)
    corr.to_csv(RESULTS_DIR / 'rfm_correlation_matrix.csv')
    print(f"\n  Guardado: {RESULTS_DIR / 'rfm_correlation_matrix.csv'}")

    avg_amount_cols = [c for c in INPUT_FEATURES if 'AVG_AMOUNT' in c]
    if len(avg_amount_cols) > 1:
        sub_corr = corr.loc[avg_amount_cols, avg_amount_cols]
        print(f"\n  Correlación entre ventanas AVG_AMOUNT:")
        print(sub_corr.to_string())

    risk_cols = [c for c in INPUT_FEATURES if 'RISK' in c]
    if len(risk_cols) > 1:
        sub_corr_risk = corr.loc[risk_cols, risk_cols]
        print(f"\n  Correlación entre ventanas RISK:")
        print(sub_corr_risk.to_string())

    # ===================== RESUMEN CONSOLIDADO =====================
    report_path = RESULTS_DIR / 'statistical_tests_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Tests de significancia estadística — TFM\n\n")

        f.write("## Tabla 1: Comparativa algorítmica (3 modelos × {} folds)\n\n".format(N_FOLDS))
        f.write("### Métricas por fold\n\n")
        f.write(t1_fold_df.to_markdown(index=False))
        f.write("\n\n### Resultados de los tests\n\n")
        f.write("```\n" + t1_stats + "\n```\n\n")

        f.write("## Tabla 5: Ablación (full vs ablated × {} folds)\n\n".format(N_FOLDS))
        f.write("### Métricas por fold\n\n")
        f.write(t5_fold_df.to_markdown(index=False))
        f.write("\n\n### Resultados de los tests\n\n")
        f.write("```\n" + t5_stats + "\n```\n\n")

        f.write("## Latencia de inferencia XAI\n\n")
        f.write("| Batch | predict_proba (ms/tx) | TreeSHAP (ms/tx) | Total (ms/tx) |\n")
        f.write("|-------|----------------------|-------------------|---------------|\n")
        for bs in latency["predict_proba"]:
            pp = latency["predict_proba"][bs]["per_tx_ms"]
            sh = latency["shap"][bs]["per_tx_ms"]
            f.write(f"| {bs} | {pp:.4f} | {sh:.4f} | {pp+sh:.4f} |\n")

        f.write("\n## Correlación RFM\n\n")
        if len(avg_amount_cols) > 1:
            f.write("### Ventanas AVG_AMOUNT\n\n")
            sub = corr.loc[avg_amount_cols, avg_amount_cols]
            f.write(sub.to_markdown())
            f.write("\n\n")

    print(f"\n  Reporte: {report_path}")
    print("\n  COMPLETADO.")


if __name__ == "__main__":
    main()
