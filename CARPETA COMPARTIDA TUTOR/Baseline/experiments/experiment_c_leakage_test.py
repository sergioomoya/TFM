#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lógica del Experimento C refactorizado: Anti-Leakage Test.
- 3 modelos: LR, RF, XGBoost
- 5 ramas: Correcta, Leak_split, Leak_scaler, Leak_smote, Leak_todas
- Parámetros SMOTE documentados en config
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

from experiments.config import (
    SEED, INPUT_FEATURES, OUTPUT_FEATURE, SMOTE_PARAMS,
    START_DATE_TRAINING, DELTA_TRAIN, DELTA_DELAY, DELTA_TEST,
)
from experiments.data_utils import load_transformed_data, get_train_test_set, card_precision_top_k
from experiments.hw_config import get_hw_config, get_xgboost_gpu_params


def get_classifier(name):
    hw = get_hw_config()
    xgb_gpu = get_xgboost_gpu_params()
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=hw['n_jobs'])
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=100, random_state=SEED, use_label_encoder=False,
                                 eval_metric='logloss', n_jobs=hw['n_jobs'], **xgb_gpu)
    raise ValueError(f"Unknown model: {name}")


def run_correct(train_df, test_df, model_name):
    """Rama correcta: temporal + scaler train + SMOTE train."""
    clf = get_classifier(model_name)
    pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(**SMOTE_PARAMS)),
        ('clf', clf),
    ])
    pipe.fit(train_df[INPUT_FEATURES], train_df[OUTPUT_FEATURE])
    y_pred = pipe.predict_proba(test_df[INPUT_FEATURES])[:, 1]
    auc = metrics.roc_auc_score(test_df[OUTPUT_FEATURE], y_pred)
    auprc = metrics.average_precision_score(test_df[OUTPUT_FEATURE], y_pred)
    pred_df = test_df.copy()
    pred_df['predictions'] = y_pred
    _, _, cp100 = card_precision_top_k(pred_df, 100)
    return {'auc_roc': auc, 'auprc': auprc, 'cp100': cp100, 'y_pred': y_pred, 'test_df': test_df}


def run_leak_split(train_df, test_df, model_name):
    """Solo split aleatorio: mismos datos que temporal pero asignación aleatoria train/test."""
    combined = pd.concat([train_df, test_df], ignore_index=True)
    y_combined = combined[OUTPUT_FEATURE]
    train_idx, test_idx = train_test_split(
        combined.index, test_size=len(test_df) / len(combined), random_state=SEED,
        stratify=y_combined
    )
    train_df = combined.loc[train_idx]
    test_df = combined.loc[test_idx]
    clf = get_classifier(model_name)
    pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(**SMOTE_PARAMS)),
        ('clf', clf),
    ])
    pipe.fit(train_df[INPUT_FEATURES], train_df[OUTPUT_FEATURE])
    y_pred = pipe.predict_proba(test_df[INPUT_FEATURES])[:, 1]
    auc = metrics.roc_auc_score(test_df[OUTPUT_FEATURE], y_pred)
    auprc = metrics.average_precision_score(test_df[OUTPUT_FEATURE], y_pred)
    pred_df = test_df.copy()
    pred_df['predictions'] = y_pred
    _, _, cp100 = card_precision_top_k(pred_df, 100)
    return {'auc_roc': auc, 'auprc': auprc, 'cp100': cp100, 'y_pred': y_pred, 'test_df': test_df}


def run_leak_scaler(train_df, test_df, model_name):
    """Solo escalado global: scaler.fit sobre train+test."""
    scaler = StandardScaler()
    X_all = np.vstack([
        train_df[INPUT_FEATURES].values,
        test_df[INPUT_FEATURES].values,
    ])
    X_all_scaled = scaler.fit_transform(X_all)
    n_train = len(train_df)
    X_train_scaled = X_all_scaled[:n_train]
    X_test = X_all_scaled[n_train:]
    y_train = train_df[OUTPUT_FEATURE].values
    smote = SMOTE(**SMOTE_PARAMS)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    clf = get_classifier(model_name)
    clf.fit(X_train_smote, y_train_smote)
    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = metrics.roc_auc_score(test_df[OUTPUT_FEATURE], y_pred)
    auprc = metrics.average_precision_score(test_df[OUTPUT_FEATURE], y_pred)
    pred_df = test_df.copy()
    pred_df['predictions'] = y_pred
    _, _, cp100 = card_precision_top_k(pred_df, 100)
    return {'auc_roc': auc, 'auprc': auprc, 'cp100': cp100, 'y_pred': y_pred, 'test_df': test_df}


def run_leak_smote(train_df, test_df, model_name):
    """Solo SMOTE global: SMOTE sobre train+test combinados."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_df[INPUT_FEATURES])
    X_test_s = scaler.transform(test_df[INPUT_FEATURES])
    X_combined = np.vstack([X_train_s, X_test_s])
    y_combined = np.concatenate([train_df[OUTPUT_FEATURE].values, test_df[OUTPUT_FEATURE].values])
    smote = SMOTE(**SMOTE_PARAMS)
    X_res, y_res = smote.fit_resample(X_combined, y_combined)
    n_train, n_test = len(train_df), len(test_df)
    X_train_final = np.vstack([X_res[:n_train], X_res[n_train + n_test:]])
    y_train_final = np.concatenate([y_res[:n_train], y_res[n_train + n_test:]])
    X_test_final = X_res[n_train:n_train + n_test]
    clf = get_classifier(model_name)
    clf.fit(X_train_final, y_train_final)
    y_pred = clf.predict_proba(X_test_final)[:, 1]
    auc = metrics.roc_auc_score(test_df[OUTPUT_FEATURE], y_pred)
    auprc = metrics.average_precision_score(test_df[OUTPUT_FEATURE], y_pred)
    pred_df = test_df.copy()
    pred_df['predictions'] = y_pred
    _, _, cp100 = card_precision_top_k(pred_df, 100)
    return {'auc_roc': auc, 'auprc': auprc, 'cp100': cp100, 'y_pred': y_pred, 'test_df': test_df}


def run_leak_todas(transactions_df, model_name):
    """Las 3 fuentes: scaler global + SMOTE global + split aleatorio."""
    scaler = StandardScaler()
    X_all = scaler.fit_transform(transactions_df[INPUT_FEATURES])
    y_all = transactions_df[OUTPUT_FEATURE].values
    smote = SMOTE(**SMOTE_PARAMS)
    X_res, y_res = smote.fit_resample(X_all, y_all)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, random_state=SEED, stratify=y_res
    )
    clf = get_classifier(model_name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = metrics.roc_auc_score(y_test, y_pred)
    auprc = metrics.average_precision_score(y_test, y_pred)
    return {'auc_roc': auc, 'auprc': auprc, 'cp100': np.nan, 'y_pred': y_pred,
            'y_test': y_test, 'test_df': None}


def run_all(transactions_df, verbose=True):
    """Ejecuta todas las ramas para todos los modelos."""
    train_df, test_df = get_train_test_set(
        transactions_df, start_date_training=START_DATE_TRAINING,
        delta_train=DELTA_TRAIN, delta_delay=DELTA_DELAY, delta_test=DELTA_TEST,
    )
    models = ["Logistic Regression", "Random Forest", "XGBoost"]
    ramas = [
        ("Correcta", lambda m: run_correct(train_df, test_df, m)),
        ("Leak_split", lambda m: run_leak_split(train_df, test_df, m)),
        ("Leak_scaler", lambda m: run_leak_scaler(train_df, test_df, m)),
        ("Leak_smote", lambda m: run_leak_smote(train_df, test_df, m)),
        ("Leak_todas", lambda m: run_leak_todas(transactions_df, m)),
    ]
    results = {}
    for model in models:
        results[model] = {}
        for rama_name, run_fn in ramas:
            if verbose:
                print(f"  {model} / {rama_name}...", end=" ", flush=True)
            results[model][rama_name] = run_fn(model)
            if verbose:
                r = results[model][rama_name]
                print(f"AUPRC={r['auprc']:.4f}", flush=True)
    return results, train_df, test_df
