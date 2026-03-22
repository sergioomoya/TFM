#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regenera SOLO las figuras de los experimentos A, B, C a partir de los
resultados previamente guardados (CSV / PKL).  No re-ejecuta ningún
experimento.

Uso:
    docker compose run --rm experiments python experiments/regenerate_figures.py
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "correct_pipeline": "#008000",
    "incorrect_pipeline": "#CC0000",
}

METRIC_SPECS = [
    ('auc_roc', 'AUC ROC', '#2F4D7E'),
    ('auprc', 'AUPRC', '#008000'),
    ('cp100', 'Card Precision@100', '#CA8035'),
]


def _bar_figure(results: dict, model_names: list, suptitle: str,
                out_path: Path, subtitle_fn=None):
    """Genera gráfica de barras con 3 subplots (AUC ROC, AUPRC, CP@100)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x_pos = np.arange(len(model_names))
    for ax, (key, label, color) in zip(axes, METRIC_SPECS):
        means = [results[n][f'{key}_mean'] for n in model_names]
        stds = [results[n][f'{key}_std'] for n in model_names]
        ax.bar(x_pos, means, 0.5, yerr=stds, capsize=5,
               color=color, edgecolor='black')
        ax.set_ylabel(label)
        subtitle = subtitle_fn(label) if subtitle_fn else label
        ax.set_title(subtitle, fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.set_ylim([0, 1.05])
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path.name}")


def _cm_figure(confusion_matrices: dict, suptitle: str, out_path: Path):
    """Genera matrices de confusión (heatmaps)."""
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(16, 5))
    labels = ['Legítimo', 'Fraude']
    for idx, (name, cm_data) in enumerate(confusion_matrices.items()):
        cm = cm_data['matrix'] if isinstance(cm_data, dict) else cm_data
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
    fig_cm.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    fig_cm.subplots_adjust(top=0.88)
    fig_cm.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path.name}")


def regenerate_experiment_a():
    """Experimento A baseline."""
    pkl_path = RESULTS_DIR / 'experiment_a_predictions.pkl'
    if not pkl_path.exists():
        print("  ⚠ No existe experiment_a_predictions.pkl — saltando")
        return
    with open(pkl_path, 'rb') as f:
        results_a = pickle.load(f)
    model_names = list(results_a.keys())
    n_folds = 4

    _bar_figure(results_a, model_names,
                f'Validación prequential ({n_folds} folds)',
                FIGURES_DIR / 'experiment_a_baseline_results.png')

    cms = {}
    for name, res in results_a.items():
        if 'confusion_matrix' in res:
            cms[name] = res['confusion_matrix']
    if cms:
        _cm_figure(cms,
                   f'Matrices de Confusión — Baseline ({n_folds} folds, threshold=0.5)',
                   FIGURES_DIR / 'experiment_a_confusion_matrices.png')


def _load_results_from_csv(csv_path: Path) -> dict:
    """Carga resultados desde CSV cuando PKL falla por incompatibilidad NumPy."""
    df = pd.read_csv(csv_path, index_col=0)
    col_map = {
        'AUC ROC': 'auc_roc_mean', 'AUC ROC Std': 'auc_roc_std',
        'AUPRC': 'auprc_mean', 'AUPRC Std': 'auprc_std',
        'CP@100': 'cp100_mean', 'CP@100 Std': 'cp100_std',
    }
    results = {}
    for name in df.index:
        row = df.loc[name]
        results[name] = {col_map[c]: row[c] for c in col_map if c in df.columns}
    return results


def _load_pkl_or_csv(pkl_path: Path, csv_path: Path):
    """Intenta cargar PKL; si falla por NumPy, usa CSV como fallback."""
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f), True
    except (ModuleNotFoundError, ImportError):
        if csv_path.exists():
            print(f"    (PKL incompatible, usando CSV como fallback)")
            return _load_results_from_csv(csv_path), False
        raise


def regenerate_experiment_a_undersampled():
    """Variantes undersampled de A."""
    for ratio in [1, 5, 10]:
        suffix = f"_undersamp_{ratio}"
        pkl_path = RESULTS_DIR / f'experiment_a{suffix}_predictions.pkl'
        csv_path = RESULTS_DIR / f'experiment_a{suffix}_results.csv'
        if not pkl_path.exists() and not csv_path.exists():
            print(f"  ⚠ No existen datos para experiment_a{suffix} — saltando")
            continue

        results_a, has_cm = _load_pkl_or_csv(pkl_path, csv_path)
        model_names = list(results_a.keys())
        n_folds = 4

        _bar_figure(
            results_a, model_names,
            f'Validación prequential ({n_folds} folds) — Submuestreo legítimas {ratio}:1',
            FIGURES_DIR / f'experiment_a{suffix}_baseline_results.png',
            subtitle_fn=lambda lbl, r=ratio: f'{lbl}\n(Undersample {r}:1)',
        )

        if has_cm:
            cms = {}
            for name, res in results_a.items():
                if 'confusion_matrix' in res:
                    cms[name] = res['confusion_matrix']
            if cms:
                _cm_figure(cms,
                           f'Matrices de Confusión — Undersample {ratio}:1 ({n_folds} folds, threshold=0.5)',
                           FIGURES_DIR / f'experiment_a{suffix}_confusion_matrices.png')
        else:
            print(f"    (Matrices de confusión no disponibles desde CSV)")


def regenerate_experiment_a_balance_variants():
    """Variantes original/undersampled del script all_balance_variants."""
    variant_map = {
        'Original (~118:1)': ('original', None),
        'Undersample 10:1': ('undersamp_10', 10),
        'Undersample 5:1': ('undersamp_5', 5),
        'Undersample 1:1': ('undersamp_1', 1),
    }
    n_folds = 4
    for variant_label, (suffix, _ratio) in variant_map.items():
        pkl_path = RESULTS_DIR / f'experiment_a_{suffix}_predictions.pkl'
        csv_path = RESULTS_DIR / f'experiment_a_{suffix}_results.csv'
        if not pkl_path.exists() and not csv_path.exists():
            continue

        results, has_cm = _load_pkl_or_csv(pkl_path, csv_path)
        model_names = list(results.keys())

        _bar_figure(
            results, model_names,
            f'Variante {variant_label} ({n_folds} folds)',
            FIGURES_DIR / f'experiment_a_{suffix}_results.png',
            subtitle_fn=lambda lbl, vl=variant_label: f'{lbl}\n({vl})',
        )

        if has_cm:
            cms = {}
            for name, res in results.items():
                if 'confusion_matrix' in res:
                    cms[name] = res['confusion_matrix']
            if cms:
                _cm_figure(cms,
                           f'Matrices de Confusión — {variant_label}',
                           FIGURES_DIR / f'experiment_a_{suffix}_confusion_matrices.png')
        else:
            print(f"    (Matrices de confusión no disponibles desde CSV)")


def regenerate_experiment_b():
    """Experimento B cost-sensitive."""
    pkl_path = RESULTS_DIR / 'experiment_b_predictions.pkl'
    csv_path = RESULTS_DIR / 'experiment_b_results.csv'
    if not pkl_path.exists() and not csv_path.exists():
        print("  ⚠ No existen datos para experiment_b — saltando")
        return

    results_all, has_cm = _load_pkl_or_csv(pkl_path, csv_path)
    n_folds = 4

    variants = list(results_all.keys())
    x_pos = np.arange(len(variants))
    palette = plt.cm.Set2(np.linspace(0, 1, len(variants)))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, (key, label) in zip(axes, [
        ('auc_roc', 'AUC ROC'), ('auprc', 'AUPRC'), ('cp100', 'Card Precision@100'),
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

    fig.suptitle(f'Cost-Sensitive — Comparativa de Variantes ({n_folds} folds)', fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(FIGURES_DIR / 'experiment_b_cost_sensitive_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ experiment_b_cost_sensitive_results.png")

    if has_cm:
        cms = {}
        for v_name, v_data in results_all.items():
            if 'confusion_matrix' in v_data:
                cms[v_name] = v_data['confusion_matrix']
        if not cms:
            first_variant = next(iter(results_all.values()))
            if 'confusion_matrices' in first_variant:
                cms = first_variant['confusion_matrices']
        if cms:
            _cm_figure(cms,
                       f'Matrices de Confusión — Cost-Sensitive Moderado ({n_folds} folds, threshold=0.5)',
                       FIGURES_DIR / 'experiment_b_confusion_matrices.png')
    else:
        print("    (Matrices de confusión no disponibles desde CSV)")


def regenerate_experiment_c():
    """Experimento C leakage comparison."""
    pkl_path = RESULTS_DIR / 'experiment_c_results.pkl'
    if not pkl_path.exists():
        print("  ⚠ No existe experiment_c_results.pkl — saltando")
        return
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    metadata = data['metadata']
    ramas_order = metadata['ramas']
    models = metadata['models']
    smote_params = metadata['smote_params']

    colors_ramas = {
        'Correcta': COLORS['correct_pipeline'],
        'Leak_split': '#FFA500',
        'Leak_scaler': '#FF8C00',
        'Leak_smote': '#FF6347',
        'Leak_todas': COLORS['incorrect_pipeline'],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(ramas_order))
    width = 0.25
    for i, model in enumerate(models):
        ax = axes[i]
        vals = [results[model][r]['auprc'] for r in ramas_order]
        bars = ax.bar(x, vals, width * 2,
                       color=[colors_ramas[r] for r in ramas_order],
                       edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(
            ['Correcta', 'Leak\nsplit', 'Leak\nscaler', 'Leak\nSMOTE', 'Leak\ntodas'],
            fontsize=9)
        ax.set_ylabel('AUPRC')
        ax.set_title(model)
        ax.set_ylim([0, 1.05])
        for bar, v in zip(bars, vals):
            ax.annotate(f'{v:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)

    fig.suptitle(
        'Impacto del Data Leakage por fuente y modelo\n'
        f'SMOTE: k_neighbors={smote_params["k_neighbors"]}, '
        f'sampling_strategy={smote_params["sampling_strategy"]}',
        fontsize=12)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(FIGURES_DIR / 'experiment_c_leakage_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ experiment_c_leakage_comparison.png")


if __name__ == '__main__':
    print("=" * 60)
    print("Regenerando figuras desde resultados existentes")
    print("=" * 60)

    tasks = [
        ("[Experimento A — Baseline]", regenerate_experiment_a),
        ("[Experimento A — Variantes undersampled]", regenerate_experiment_a_undersampled),
        ("[Experimento A — Balance variants]", regenerate_experiment_a_balance_variants),
        ("[Experimento B — Cost-Sensitive]", regenerate_experiment_b),
        ("[Experimento C — Data Leakage]", regenerate_experiment_c),
    ]
    for label, func in tasks:
        print(f"\n{label}")
        try:
            func()
        except Exception as e:
            print(f"  ⚠ Error: {e}")

    print("\n✓ Regeneración completada.")
