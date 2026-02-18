#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ejecuta el Experimento D (Interpretabilidad/XAI) con mejoras según CRITICA_MEJORA_EXPERIMENTOS.md:
- Modelo: XGBoost baseline (mejor AUPRC que cost-sensitive)
- Feature Importance: Gain, weight, cover
- Tabla mean |SHAP| por variable
- Contexto de transacciones en force plots
- Muestra SHAP: 1000
- Dependence plot

Uso: python experiments/run_experiment_d_standalone.py
"""

import os
import sys
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import (
    SEED, INPUT_FEATURES, OUTPUT_FEATURE,
    BASELINE_PARAMS, RESULTS_DIR, FIGURES_DIR, COLORS,
    START_DATE_TRAINING, DELTA_TRAIN, DELTA_DELAY, DELTA_TEST,
)
from experiments.data_utils import (
    load_transformed_data, get_train_test_set,
    print_dataset_summary, card_precision_top_k,
)

warnings.filterwarnings('ignore')
sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

# Mejora: muestra SHAP más grande para mayor estabilidad
SHAP_SAMPLE_SIZE = min(1000, 5000)


def get_xgb_importance_all_types(model, feature_names):
    """Obtiene Gain, weight y cover de XGBoost."""
    booster = model.get_booster()
    importance_types = ['gain', 'weight', 'cover']
    result = {}
    for imp_type in importance_types:
        scores = booster.get_score(importance_type=imp_type)
        # scores usa f0, f1, ... si no hay feature_names
        if not scores and hasattr(model, 'feature_names_in_'):
            continue
        # Mapear f0->feature[0], etc.
        feat_map = {f'f{i}': name for i, name in enumerate(feature_names)}
        result[imp_type] = {feat_map.get(k, k): v for k, v in scores.items()}
    return result


def main():
    print("=" * 60)
    print("  EXPERIMENTO D: INTERPRETABILIDAD Y XAI (mejorado)")
    print("=" * 60)
    print(f"  Modelo: XGBoost baseline (mejor rendimiento que cost-sensitive)")
    print(f"  Mejoras: Gain/weight/cover, mean|SHAP|, contexto force plots, dependence")
    print()

    transactions_df = load_transformed_data()
    train_df, test_df = get_train_test_set(
        transactions_df,
        start_date_training=START_DATE_TRAINING,
        delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY, delta_test=DELTA_TEST,
    )
    print_dataset_summary(train_df, test_df, "Experimento D - Interpretabilidad")

    # Mejora 1: Usar XGBoost BASELINE (mejor AUPRC), no cost-sensitive
    model = xgb.XGBClassifier(**BASELINE_PARAMS["XGBoost"])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[INPUT_FEATURES])
    X_test_scaled = scaler.transform(test_df[INPUT_FEATURES])

    model.fit(X_train_scaled, train_df[OUTPUT_FEATURE])
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    auprc = metrics.average_precision_score(test_df[OUTPUT_FEATURE], y_pred_proba)
    auc_roc = metrics.roc_auc_score(test_df[OUTPUT_FEATURE], y_pred_proba)
    predictions_df = test_df.copy()
    predictions_df['predictions'] = y_pred_proba
    _, _, cp100 = card_precision_top_k(predictions_df, top_k=100)

    print(f"\n  XGBoost baseline entrenado:")
    print(f"    AUC ROC:  {auc_roc:.4f}")
    print(f"    AUPRC:    {auprc:.4f}")
    print(f"    CP@100:   {cp100:.4f}")

    # Mejora 2: Feature Importance - Gain, weight, cover
    importances_gain = model.feature_importances_
    imp_all = get_xgb_importance_all_types(model, INPUT_FEATURES)

    fi_df = pd.DataFrame({'Feature': INPUT_FEATURES, 'Gain': importances_gain})
    if 'weight' in imp_all:
        fi_df['Weight'] = fi_df['Feature'].map(imp_all['weight']).fillna(0)
    if 'cover' in imp_all:
        fi_df['Cover'] = fi_df['Feature'].map(imp_all['cover']).fillna(0)

    fi_df = fi_df.sort_values('Gain', ascending=False)
    top_10 = fi_df.head(10)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_10)), top_10['Gain'].values, color=COLORS.get('baseline', '#2F4D7E'))
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10['Feature'].values, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Importancia (Gain)', fontsize=12)
    ax.set_title('Top-10 Variables Más Importantes\n(XGBoost Baseline - mejor AUPRC)', fontsize=14)
    for i, val in enumerate(top_10['Gain']):
        ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'experiment_d_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Feature Importance (Gain) guardada")

    # Comparación Gain vs weight vs cover (si están disponibles)
    if 'Weight' in fi_df.columns and 'Cover' in fi_df.columns:
        fi_df_norm = fi_df.copy()
        for col in ['Gain', 'Weight', 'Cover']:
            if col in fi_df_norm.columns and fi_df_norm[col].sum() > 0:
                fi_df_norm[col + '_norm'] = fi_df_norm[col] / fi_df_norm[col].sum()
        fi_df.to_csv(RESULTS_DIR / 'experiment_d_feature_importance_all_types.csv', index=False)
        print("✓ Comparación Gain/weight/cover guardada en CSV")

    # SHAP
    np.random.seed(SEED)
    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test_scaled))
    sample_indices = np.random.choice(len(X_test_scaled), sample_size, replace=False)
    X_sample = X_test_scaled[sample_indices]
    X_sample_df = pd.DataFrame(X_sample, columns=INPUT_FEATURES)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mejora 3: Tabla mean |SHAP| por variable
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_impact_df = pd.DataFrame({
        'Feature': INPUT_FEATURES,
        'mean_abs_SHAP': mean_abs_shap,
    }).sort_values('mean_abs_SHAP', ascending=False)

    shap_impact_df.to_csv(RESULTS_DIR / 'experiment_d_shap_mean_impact.csv', index=False)
    print("\n✓ Tabla mean |SHAP| por variable:")
    print(shap_impact_df.head(10).to_string(index=False))

    # Beeswarm
    fig = plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_values, X_sample_df, show=False)
    plt.title(f'SHAP Beeswarm - XGBoost Baseline ({sample_size} muestras)\n', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'experiment_d_shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Beeswarm SHAP guardado")

    # Mejora 4: Force plots con contexto de la transacción
    y_test_sample = test_df[OUTPUT_FEATURE].iloc[sample_indices].values
    preds_sample = model.predict_proba(X_sample)[:, 1]
    test_sample_df = test_df.iloc[sample_indices].reset_index(drop=True)

    fraud_indices = np.where(y_test_sample == 1)[0]
    fraud_idx = fraud_indices[np.argmax(preds_sample[fraud_indices])] if len(fraud_indices) > 0 else None

    normal_indices = np.where(y_test_sample == 0)[0]
    normal_idx = normal_indices[np.argmin(preds_sample[normal_indices])]

    def describe_transaction(idx, label):
        row = test_sample_df.iloc[idx]
        ctx = []
        if 'TX_AMOUNT' in row:
            ctx.append(f"TX_AMOUNT={row['TX_AMOUNT']:.2f}")
        if 'TERMINAL_ID' in row:
            ctx.append(f"TERMINAL_ID={row['TERMINAL_ID']}")
        if 'CUSTOMER_ID' in row:
            ctx.append(f"CUSTOMER_ID={row['CUSTOMER_ID']}")
        for feat in ['TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_RISK_1DAY_WINDOW']:
            if feat in row:
                ctx.append(f"{feat}={row[feat]:.4f}")
        return f"{label}: " + ", ".join(ctx)

    if fraud_idx is not None:
        fraud_ctx = describe_transaction(fraud_idx, "FRAUDE")
        print(f"\n  Transacción FRAUDE (force plot): {fraud_ctx}")
        print(f"    Prob. predicción: {preds_sample[fraud_idx]:.4f}")

        fig = plt.figure(figsize=(16, 3))
        shap.force_plot(
            explainer.expected_value, shap_values[fraud_idx],
            X_sample_df.iloc[fraud_idx], matplotlib=True, show=False
        )
        plt.title(f'Force Plot - Transacción FRAUDULENTA\n({fraud_ctx})', fontsize=11)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'experiment_d_shap_force_fraud.png', dpi=150, bbox_inches='tight')
        plt.close()

    normal_ctx = describe_transaction(normal_idx, "NORMAL")
    print(f"\n  Transacción NORMAL (force plot): {normal_ctx}")
    print(f"    Prob. predicción: {preds_sample[normal_idx]:.4f}")

    fig = plt.figure(figsize=(16, 3))
    shap.force_plot(
        explainer.expected_value, shap_values[normal_idx],
        X_sample_df.iloc[normal_idx], matplotlib=True, show=False
    )
    plt.title(f'Force Plot - Transacción NORMAL\n({normal_ctx})', fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'experiment_d_shap_force_normal.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Mejora 5: Dependence plot TX_AMOUNT vs TERMINAL_ID_RISK_7DAY_WINDOW
    for dep_feat in ['TX_AMOUNT', 'TERMINAL_ID_RISK_7DAY_WINDOW']:
        if dep_feat in INPUT_FEATURES:
            try:
                fig = plt.figure(figsize=(10, 5))
                shap.dependence_plot(dep_feat, shap_values, X_sample_df, show=False)
                plt.title(f'SHAP Dependence: {dep_feat}', fontsize=12)
                plt.tight_layout()
                safe_name = dep_feat.lower().replace(' ', '_')
                fig.savefig(FIGURES_DIR / f'experiment_d_shap_dependence_{safe_name}.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"\n✓ Dependence plot {dep_feat} guardado")
            except Exception as e:
                print(f"  ⚠ Dependence plot {dep_feat}: {e}")
            break

    # Guardar resultados
    fi_df.to_csv(RESULTS_DIR / 'experiment_d_feature_importance.csv', index=False)
    results_d = {
        'feature_importance': fi_df,
        'feature_importance_all_types': imp_all,
        'shap_mean_impact': shap_impact_df,
        'shap_values': shap_values,
        'X_sample': X_sample_df,
        'metrics': {'auc_roc': auc_roc, 'auprc': auprc, 'card_precision_at_100': cp100},
        'metadata': {
            'model': 'XGBoost baseline',
            'seed': SEED,
            'shap_sample_size': sample_size,
            'force_plot_fraud_context': fraud_ctx if fraud_idx is not None else None,
            'force_plot_normal_context': normal_ctx,
        },
    }
    with open(RESULTS_DIR / 'experiment_d_results.pkl', 'wb') as f:
        pickle.dump(results_d, f)

    print("\n✓ Experimento D completado. Archivos:")
    print(f"  - {RESULTS_DIR / 'experiment_d_feature_importance.csv'}")
    print(f"  - {RESULTS_DIR / 'experiment_d_shap_mean_impact.csv'}")
    print(f"  - {RESULTS_DIR / 'experiment_d_results.pkl'}")
    print(f"  - Figuras: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
