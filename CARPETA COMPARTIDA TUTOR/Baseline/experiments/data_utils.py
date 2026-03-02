#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidades de carga y preparación de datos para los experimentos.

Encapsula la lógica de lectura del dataset simulado y la creación de
los conjuntos train/test con división temporal estricta.
"""

import os
import datetime
import pickle
import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.model_selection
from pathlib import Path

from experiments.config import (
    DATA_DIR_RAW,
    DATA_DIR_TRANSFORMED,
    INPUT_FEATURES,
    OUTPUT_FEATURE,
    DELTA_TRAIN,
    DELTA_DELAY,
    DELTA_TEST,
    START_DATE_TRAINING,
    SEED,
)


def load_raw_data(begin_date: str = "2018-04-01",
                  end_date: str = "2018-09-30") -> pd.DataFrame:
    """
    Carga los datos crudos (sin transformar) desde los archivos pickle diarios.
    
    Args:
        begin_date: Fecha de inicio en formato 'YYYY-MM-DD'
        end_date: Fecha de fin en formato 'YYYY-MM-DD'
    
    Returns:
        DataFrame con todas las transacciones del rango de fechas
    """
    dir_input = str(DATA_DIR_RAW)
    
    files = [
        os.path.join(dir_input, f)
        for f in os.listdir(dir_input)
        if f >= begin_date + '.pkl' and f <= end_date + '.pkl'
    ]

    frames = []
    for f in sorted(files):
        df = pd.read_pickle(f)
        frames.append(df)

    df_final = pd.concat(frames)
    df_final = df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True, inplace=True)
    df_final = df_final.replace([-1], 0)

    return df_final


def load_transformed_data(begin_date: str = "2018-04-01",
                          end_date: str = "2018-09-30") -> pd.DataFrame:
    """
    Carga los datos transformados (con feature engineering) desde los archivos pickle.
    
    Estos datos incluyen las features de ventana temporal calculadas en
    Chapter_3 (BaselineFeatureTransformation).
    
    Args:
        begin_date: Fecha de inicio en formato 'YYYY-MM-DD'
        end_date: Fecha de fin en formato 'YYYY-MM-DD'
    
    Returns:
        DataFrame con transacciones transformadas del rango de fechas
    """
    dir_input = str(DATA_DIR_TRANSFORMED)

    files = [
        os.path.join(dir_input, f)
        for f in os.listdir(dir_input)
        if f >= begin_date + '.pkl' and f <= end_date + '.pkl'
    ]

    frames = []
    for f in sorted(files):
        df = pd.read_pickle(f)
        frames.append(df)

    df_final = pd.concat(frames)
    df_final = df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True, inplace=True)
    df_final = df_final.replace([-1], 0)

    return df_final


def get_train_test_set(transactions_df: pd.DataFrame,
                       start_date_training: datetime.datetime = None,
                       delta_train: int = DELTA_TRAIN,
                       delta_delay: int = DELTA_DELAY,
                       delta_test: int = DELTA_TEST,
                       sampling_ratio: float = 1.0,
                       random_state: int = SEED) -> tuple:
    """
    Obtiene los conjuntos train/test con división temporal estricta.
    
    Implementa el protocolo del libro: división cronológica con periodo
    de delay para simular el retraso real en la detección de fraude.
    
    Args:
        transactions_df: DataFrame completo de transacciones
        start_date_training: Fecha de inicio del entrenamiento
        delta_train: Días de entrenamiento
        delta_delay: Días de delay (reporte de fraude)
        delta_test: Días de test
        sampling_ratio: Ratio de submuestreo (1.0 = sin submuestreo)
        random_state: Semilla para reproducibilidad
    
    Returns:
        Tupla (train_df, test_df)
    """
    if start_date_training is None:
        start_date_training = START_DATE_TRAINING

    # Conjunto de entrenamiento
    train_df = transactions_df[
        (transactions_df.TX_DATETIME >= start_date_training) &
        (transactions_df.TX_DATETIME < start_date_training + datetime.timedelta(days=delta_train))
    ]

    # Conjunto de test con eliminación de tarjetas comprometidas conocidas
    test_df = []
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD == 1].CUSTOMER_ID)
    start_tx_time_days_training = train_df.TX_TIME_DAYS.min()

    for day in range(delta_test):
        test_df_day = transactions_df[
            transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
            delta_train + delta_delay + day
        ]

        test_df_day_delay_period = transactions_df[
            transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
            delta_train + day - 1
        ]

        new_defrauded_customers = set(
            test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD == 1].CUSTOMER_ID
        )
        known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)
        test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]
        test_df.append(test_df_day)

    test_df = pd.concat(test_df)

    # Submuestreo opcional
    if sampling_ratio < 1:
        train_df_frauds = train_df[train_df.TX_FRAUD == 1].sample(
            frac=sampling_ratio, random_state=random_state
        )
        train_df_genuine = train_df[train_df.TX_FRAUD == 0].sample(
            frac=sampling_ratio, random_state=random_state
        )
        train_df = pd.concat([train_df_frauds, train_df_genuine])

    train_df = train_df.sort_values('TRANSACTION_ID')
    test_df = test_df.sort_values('TRANSACTION_ID')

    return (train_df, test_df)


def compute_class_ratio(y: pd.Series) -> float:
    """
    Calcula el ratio negativo/positivo para scale_pos_weight de XGBoost.
    
    Args:
        y: Serie con las etiquetas (0/1)
    
    Returns:
        Ratio n_negative / n_positive
    """
    n_positive = (y == 1).sum()
    n_negative = (y == 0).sum()
    
    if n_positive == 0:
        raise ValueError("No hay muestras positivas en el conjunto de datos")
    
    return n_negative / n_positive


def get_xgboost_cost_sensitive():
    """
    Retorna una clase XGBoost con scale_pos_weight configurable.
    - scale_pos_weight=1: sin ponderación (baseline)
    - scale_pos_weight='auto': calcula n_neg/n_pos en fit
    - scale_pos_weight=N (número): usa N directamente
    """
    import xgboost as xgb

    class XGBoostCostSensitive(xgb.XGBClassifier):
        """XGBClassifier con scale_pos_weight configurable (1, 'auto', o valor numérico)."""

        def fit(self, X, y, **kwargs):
            spw = getattr(self, 'scale_pos_weight', None)
            if spw == 'auto' or spw is None:
                n_pos = max(int((y == 1).sum()), 1)
                n_neg = int((y == 0).sum())
                self.scale_pos_weight = n_neg / n_pos
            return super().fit(X, y, **kwargs)

    return XGBoostCostSensitive


def card_precision_top_k_day(df_day: pd.DataFrame, top_k: int) -> tuple:
    """
    Calcula la Card Precision@k para un solo día.

    Agrupa por CUSTOMER_ID (máximo de predicción y fraude), ordena por
    predicción descendente y toma las top-k tarjetas más sospechosas.

    Args:
        df_day: DataFrame del día con columnas 'predictions', 'CUSTOMER_ID', 'TX_FRAUD'
        top_k: Número de tarjetas más sospechosas a evaluar

    Returns:
        Tupla (lista de tarjetas comprometidas detectadas, card_precision_top_k)
    """
    df_day = (
        df_day.groupby('CUSTOMER_ID')
        .max()
        .sort_values(by="predictions", ascending=False)
        .reset_index(drop=False)
    )

    df_day_top_k = df_day.head(top_k)
    list_detected_compromised_cards = list(
        df_day_top_k[df_day_top_k.TX_FRAUD == 1].CUSTOMER_ID
    )

    precision_top_k = len(list_detected_compromised_cards) / top_k

    return list_detected_compromised_cards, precision_top_k


def card_precision_top_k(predictions_df: pd.DataFrame, top_k: int,
                         remove_detected_compromised_cards: bool = True) -> tuple:
    """
    Calcula la Card Precision@k promedio a lo largo de todos los días.

    Replica el protocolo del libro (Chapter 4): para cada día se evalúan las
    top-k tarjetas más sospechosas, eliminando las ya detectadas como
    comprometidas en días anteriores.

    Args:
        predictions_df: DataFrame con columnas 'predictions', 'CUSTOMER_ID',
                        'TX_FRAUD', 'TX_TIME_DAYS'
        top_k: Número de tarjetas a evaluar por día
        remove_detected_compromised_cards: Si True, elimina tarjetas ya detectadas

    Returns:
        Tupla (nb_compromised_cards_per_day, cp_top_k_per_day, mean_cp_top_k)
    """
    list_days = sorted(predictions_df['TX_TIME_DAYS'].unique())

    list_detected_compromised_cards = []
    card_precision_top_k_per_day_list = []
    nb_compromised_cards_per_day = []

    for day in list_days:
        df_day = predictions_df[predictions_df['TX_TIME_DAYS'] == day]
        df_day = df_day[['predictions', 'CUSTOMER_ID', 'TX_FRAUD']]

        if remove_detected_compromised_cards:
            df_day = df_day[~df_day.CUSTOMER_ID.isin(list_detected_compromised_cards)]

        nb_compromised_cards_per_day.append(
            len(df_day[df_day.TX_FRAUD == 1].CUSTOMER_ID.unique())
        )

        detected_cards, cp_top_k = card_precision_top_k_day(df_day, top_k)

        card_precision_top_k_per_day_list.append(cp_top_k)

        if remove_detected_compromised_cards:
            list_detected_compromised_cards.extend(detected_cards)

    mean_card_precision_top_k = np.array(card_precision_top_k_per_day_list).mean()

    return (
        nb_compromised_cards_per_day,
        card_precision_top_k_per_day_list,
        mean_card_precision_top_k,
    )


def card_precision_top_k_custom(y_true: pd.Series, y_pred: np.ndarray,
                                 top_k: int, transactions_df: pd.DataFrame, **kwargs) -> float:
    """
    Scorer personalizado para GridSearchCV que computa Card Precision@k.

    Compatible con la interfaz de sklearn: recibe y_true (índices del fold)
    y y_pred (probabilidades). Usa transactions_df para construir predictions_df
    con CUSTOMER_ID, TX_TIME_DAYS necesarios para CP@k.
    **kwargs: ignora args extra (needs_proba, etc.) de sklearn 1.3+

    Args:
        y_true: Etiquetas del fold (index = índices de transacciones)
        y_pred: Probabilidades predichas de la clase positiva
        top_k: Número de tarjetas top a evaluar
        transactions_df: DataFrame completo con CUSTOMER_ID, TX_TIME_DAYS

    Returns:
        Media de Card Precision@k a lo largo de los días del fold
    """
    predictions_df = transactions_df.loc[y_true.index].copy()
    predictions_df['predictions'] = y_pred
    _, _, mean_cp = card_precision_top_k(predictions_df, top_k)
    return mean_cp


def prequentialSplit(transactions_df: pd.DataFrame,
                     start_date_training: datetime.datetime,
                     n_folds: int = 4,
                     delta_train: int = None,
                     delta_delay: int = None,
                     delta_assessment: int = None) -> list:
    """
    Genera índices para validación prequential (Capítulo 5).

    Para cada fold, desplaza la fecha de inicio hacia atrás y obtiene
    train/test con división temporal. Devuelve lista de (train_idx, test_idx).

    Args:
        transactions_df: DataFrame de transacciones
        start_date_training: Fecha de inicio del entrenamiento (fold 0)
        n_folds: Número de folds prequential
        delta_train: Días de entrenamiento (usa DELTA_TRAIN si None)
        delta_delay: Días de delay (usa DELTA_DELAY si None)
        delta_assessment: Días de test por fold (usa DELTA_TEST si None)

    Returns:
        Lista de tuplas (índices train, índices test)
    """
    if delta_train is None:
        delta_train = DELTA_TRAIN
    if delta_delay is None:
        delta_delay = DELTA_DELAY
    if delta_assessment is None:
        delta_assessment = DELTA_TEST

    prequential_split_indices = []

    for fold in range(n_folds):
        start_date_fold = start_date_training - datetime.timedelta(
            days=fold * delta_assessment
        )
        train_df, test_df = get_train_test_set(
            transactions_df,
            start_date_training=start_date_fold,
            delta_train=delta_train,
            delta_delay=delta_delay,
            delta_test=delta_assessment,
        )
        prequential_split_indices.append(
            (list(train_df.index), list(test_df.index)))
    return prequential_split_indices


def prequential_grid_search(transactions_df: pd.DataFrame,
                            classifier,
                            input_features: list,
                            output_feature: str,
                            parameters: dict,
                            scoring: dict,
                            start_date_training: datetime.datetime,
                            n_folds: int = 4,
                            expe_type: str = 'Test',
                            delta_train: int = None,
                            delta_delay: int = None,
                            delta_assessment: int = None,
                            performance_metrics_list_grid: list = None,
                            performance_metrics_list: list = None,
                            n_jobs: int = -1) -> pd.DataFrame:
    """
    GridSearchCV con validación prequential (metodología Capítulo 5).

    Args:
        transactions_df: DataFrame de transacciones
        classifier: Clasificador sklearn
        input_features: Lista de features de entrada
        output_feature: Nombre de la variable objetivo
        parameters: Parámetros para GridSearch (dict con clf__*)
        scoring: Dict de scorers (ej. {'roc_auc':'roc_auc', 'card_precision@100': scorer})
        start_date_training: Fecha de inicio para el split
        n_folds: Número de folds prequential
        expe_type: 'Validation' o 'Test' (solo para nombres de columnas)
        delta_train, delta_delay, delta_assessment: Parámetros temporales
        performance_metrics_list_grid: Nombres internos de métricas (ej. ['roc_auc', 'card_precision@100'])
        performance_metrics_list: Nombres para columnas (ej. ['AUC ROC', 'Card Precision@100'])
        n_jobs: Paralelismo

    Returns:
        DataFrame con mean±std por métrica y parámetros
    """
    if performance_metrics_list_grid is None:
        performance_metrics_list_grid = ['roc_auc']
    if performance_metrics_list is None:
        performance_metrics_list = ['AUC ROC']

    pipe = sklearn.pipeline.Pipeline([
        ('scaler', sklearn.preprocessing.StandardScaler()),
        ('clf', classifier),
    ])

    prequential_split_indices = prequentialSplit(
        transactions_df,
        start_date_training=start_date_training,
        n_folds=n_folds,
        delta_train=delta_train,
        delta_delay=delta_delay,
        delta_assessment=delta_assessment,
    )

    grid_search = sklearn.model_selection.GridSearchCV(
        pipe, parameters, scoring=scoring, cv=prequential_split_indices,
        refit=False, n_jobs=n_jobs
    )

    X = transactions_df[input_features]
    y = transactions_df[output_feature]
    grid_search.fit(X, y)

    performances_df = pd.DataFrame()
    for i, metric in enumerate(performance_metrics_list_grid):
        col_mean = f'mean_test_{metric}'
        col_std = f'std_test_{metric}'
        if col_mean in grid_search.cv_results_:
            performances_df[performance_metrics_list[i] + ' ' + expe_type] = \
                grid_search.cv_results_[col_mean]
            performances_df[performance_metrics_list[i] + ' ' + expe_type + ' Std'] = \
                grid_search.cv_results_[col_std]

    performances_df['Parameters'] = grid_search.cv_results_['params']
    performances_df['Execution time'] = grid_search.cv_results_['mean_fit_time']

    return performances_df


def model_selection_wrapper(transactions_df: pd.DataFrame,
                            classifier,
                            input_features: list,
                            output_feature: str,
                            parameters: dict,
                            scoring: dict,
                            start_date_training_for_valid: datetime.datetime,
                            start_date_training_for_test: datetime.datetime,
                            n_folds: int = 4,
                            delta_train: int = None,
                            delta_delay: int = None,
                            delta_assessment: int = None,
                            performance_metrics_list_grid: list = None,
                            performance_metrics_list: list = None,
                            n_jobs: int = -1) -> pd.DataFrame:
    """
    Ejecuta prequential_grid_search en validación y en test (Capítulo 5).

    Combina resultados de ambas evaluaciones en un único DataFrame.
    """
    if performance_metrics_list_grid is None:
        performance_metrics_list_grid = ['roc_auc']
    if performance_metrics_list is None:
        performance_metrics_list = ['AUC ROC']

    perf_valid = prequential_grid_search(
        transactions_df, classifier, input_features, output_feature,
        parameters, scoring,
        start_date_training=start_date_training_for_valid,
        n_folds=n_folds,
        expe_type='Validation',
        delta_train=delta_train,
        delta_delay=delta_delay,
        delta_assessment=delta_assessment,
        performance_metrics_list_grid=performance_metrics_list_grid,
        performance_metrics_list=performance_metrics_list,
        n_jobs=n_jobs,
    )

    perf_test = prequential_grid_search(
        transactions_df, classifier, input_features, output_feature,
        parameters, scoring,
        start_date_training=start_date_training_for_test,
        n_folds=n_folds,
        expe_type='Test',
        delta_train=delta_train,
        delta_delay=delta_delay,
        delta_assessment=delta_assessment,
        performance_metrics_list_grid=performance_metrics_list_grid,
        performance_metrics_list=performance_metrics_list,
        n_jobs=n_jobs,
    )

    perf_valid = perf_valid.drop(
        columns=[c for c in ['Parameters', 'Execution time'] if c in perf_valid.columns],
        errors='ignore'
    )
    performances_df = pd.concat([perf_test, perf_valid], axis=1)
    return performances_df


def compute_confusion_matrices_prequential(
    transactions_df: pd.DataFrame,
    results_dict: dict,
    input_features: list,
    output_feature: str,
    clf_classes: dict,
    start_date_training: datetime.datetime,
    n_folds: int = 4,
    delta_train: int = None,
    delta_delay: int = None,
    delta_assessment: int = None,
    threshold: float = 0.5,
) -> dict:
    """
    Re-entrena cada modelo con best_params y computa matrices de confusión agregadas
    sobre los folds prequential de test.

    Args:
        transactions_df: DataFrame de transacciones
        results_dict: Dict {nombre_modelo: {'best_params': ...}} con params de Pipeline (clf__X)
        input_features: Lista de features
        output_feature: Variable objetivo
        clf_classes: Dict {nombre: ClaseClasificador} (ej. LogisticRegression, RandomForestClassifier)
        start_date_training, n_folds, delta_*: Parámetros del split prequential
        threshold: Umbral de clasificación (default 0.5)

    Returns:
        Dict {nombre: {'TN': int, 'FP': int, 'FN': int, 'TP': int, 'matrix': np.ndarray}}
    """
    import sklearn.pipeline
    import sklearn.preprocessing

    if delta_train is None:
        delta_train = DELTA_TRAIN
    if delta_delay is None:
        delta_delay = DELTA_DELAY
    if delta_assessment is None:
        delta_assessment = DELTA_TEST

    splits = prequentialSplit(
        transactions_df, start_date_training=start_date_training,
        n_folds=n_folds, delta_train=delta_train, delta_delay=delta_delay,
        delta_assessment=delta_assessment,
    )

    output = {}
    for name, res in results_dict.items():
        best_params = res['best_params']
        clf_params = {k.replace('clf__', ''): v for k, v in best_params.items() if k.startswith('clf__')}
        clf = clf_classes[name](**clf_params)
        pipe = sklearn.pipeline.Pipeline([
            ('scaler', sklearn.preprocessing.StandardScaler()),
            ('clf', clf),
        ])
        all_y_true, all_y_pred = [], []
        for train_idx, test_idx in splits:
            pipe.fit(transactions_df.loc[train_idx, input_features], transactions_df.loc[train_idx, output_feature])
            y_prob = pipe.predict_proba(transactions_df.loc[test_idx, input_features])[:, 1]
            all_y_true.extend(transactions_df.loc[test_idx, output_feature].values)
            all_y_pred.extend((y_prob >= threshold).astype(int))

        cm = sklearn.metrics.confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
        output[name] = {
            'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0]), 'TP': int(cm[1, 1]),
            'matrix': cm,
        }
    return output


def prequential_randomized_search(transactions_df: pd.DataFrame,
                                   classifier,
                                   input_features: list,
                                   output_feature: str,
                                   param_distributions: dict,
                                   scoring: dict,
                                   start_date_training: datetime.datetime,
                                   n_folds: int = 4,
                                   n_iter: int = 50,
                                   expe_type: str = 'Test',
                                   delta_train: int = None,
                                   delta_delay: int = None,
                                   delta_assessment: int = None,
                                   performance_metrics_list_grid: list = None,
                                   performance_metrics_list: list = None,
                                   n_jobs: int = -1) -> pd.DataFrame:
    """
    RandomizedSearchCV con validación prequential.

    Muestrea n_iter combinaciones del espacio de hiperparámetros
    en vez de explorar exhaustivamente. Ideal para grids grandes (XGBoost GPU).
    """
    if performance_metrics_list_grid is None:
        performance_metrics_list_grid = ['roc_auc']
    if performance_metrics_list is None:
        performance_metrics_list = ['AUC ROC']

    pipe = sklearn.pipeline.Pipeline([
        ('scaler', sklearn.preprocessing.StandardScaler()),
        ('clf', classifier),
    ])

    prequential_split_indices = prequentialSplit(
        transactions_df,
        start_date_training=start_date_training,
        n_folds=n_folds,
        delta_train=delta_train,
        delta_delay=delta_delay,
        delta_assessment=delta_assessment,
    )

    search = sklearn.model_selection.RandomizedSearchCV(
        pipe, param_distributions, scoring=scoring,
        cv=prequential_split_indices,
        refit=False, n_jobs=n_jobs,
        n_iter=n_iter, random_state=SEED,
    )

    X = transactions_df[input_features]
    y = transactions_df[output_feature]
    search.fit(X, y)

    performances_df = pd.DataFrame()
    for i, metric in enumerate(performance_metrics_list_grid):
        col_mean = f'mean_test_{metric}'
        col_std = f'std_test_{metric}'
        if col_mean in search.cv_results_:
            performances_df[performance_metrics_list[i] + ' ' + expe_type] = \
                search.cv_results_[col_mean]
            performances_df[performance_metrics_list[i] + ' ' + expe_type + ' Std'] = \
                search.cv_results_[col_std]

    performances_df['Parameters'] = search.cv_results_['params']
    performances_df['Execution time'] = search.cv_results_['mean_fit_time']

    return performances_df


def model_selection_wrapper_randomized(transactions_df: pd.DataFrame,
                                       classifier,
                                       input_features: list,
                                       output_feature: str,
                                       param_distributions: dict,
                                       scoring: dict,
                                       start_date_training_for_valid: datetime.datetime,
                                       start_date_training_for_test: datetime.datetime,
                                       n_folds: int = 4,
                                       n_iter: int = 50,
                                       delta_train: int = None,
                                       delta_delay: int = None,
                                       delta_assessment: int = None,
                                       performance_metrics_list_grid: list = None,
                                       performance_metrics_list: list = None,
                                       n_jobs: int = -1) -> pd.DataFrame:
    """
    model_selection_wrapper con RandomizedSearchCV en vez de GridSearchCV.
    """
    if performance_metrics_list_grid is None:
        performance_metrics_list_grid = ['roc_auc']
    if performance_metrics_list is None:
        performance_metrics_list = ['AUC ROC']

    perf_valid = prequential_randomized_search(
        transactions_df, classifier, input_features, output_feature,
        param_distributions, scoring,
        start_date_training=start_date_training_for_valid,
        n_folds=n_folds, n_iter=n_iter,
        expe_type='Validation',
        delta_train=delta_train, delta_delay=delta_delay,
        delta_assessment=delta_assessment,
        performance_metrics_list_grid=performance_metrics_list_grid,
        performance_metrics_list=performance_metrics_list,
        n_jobs=n_jobs,
    )

    perf_test = prequential_randomized_search(
        transactions_df, classifier, input_features, output_feature,
        param_distributions, scoring,
        start_date_training=start_date_training_for_test,
        n_folds=n_folds, n_iter=n_iter,
        expe_type='Test',
        delta_train=delta_train, delta_delay=delta_delay,
        delta_assessment=delta_assessment,
        performance_metrics_list_grid=performance_metrics_list_grid,
        performance_metrics_list=performance_metrics_list,
        n_jobs=n_jobs,
    )

    perf_valid = perf_valid.drop(
        columns=[c for c in ['Parameters', 'Execution time'] if c in perf_valid.columns],
        errors='ignore'
    )
    performances_df = pd.concat([perf_test, perf_valid], axis=1)
    return performances_df


def calibrate_and_evaluate_prequential(
    transactions_df: pd.DataFrame,
    classifier_class,
    best_params: dict,
    input_features: list,
    output_feature: str,
    start_date_training: datetime.datetime,
    n_folds: int = 4,
    delta_train: int = None,
    delta_delay: int = None,
    delta_assessment: int = None,
    calibration_method: str = 'isotonic',
    calibration_cv: int = 3,
) -> dict:
    """
    Entrena con best_params, calibra probabilidades con CalibratedClassifierCV,
    y evalúa AUPRC / CP@100 sobre los folds de test.

    La calibración mejora la calidad del ranking de probabilidades, lo cual
    impacta directamente en AUPRC y CP@100.

    Returns:
        Dict con métricas (auc_roc, auprc, cp100) como media ± std sobre folds.
    """
    from sklearn.calibration import CalibratedClassifierCV

    if delta_train is None:
        delta_train = DELTA_TRAIN
    if delta_delay is None:
        delta_delay = DELTA_DELAY
    if delta_assessment is None:
        delta_assessment = DELTA_TEST

    splits = prequentialSplit(
        transactions_df, start_date_training=start_date_training,
        n_folds=n_folds, delta_train=delta_train, delta_delay=delta_delay,
        delta_assessment=delta_assessment,
    )

    fold_metrics = {'auc_roc': [], 'auprc': [], 'cp100': []}
    clf_params = {k.replace('clf__', ''): v for k, v in best_params.items() if k.startswith('clf__')}

    for train_idx, test_idx in splits:
        X_train = transactions_df.loc[train_idx, input_features]
        y_train = transactions_df.loc[train_idx, output_feature]
        X_test = transactions_df.loc[test_idx, input_features]
        y_test = transactions_df.loc[test_idx, output_feature]

        scaler = sklearn.preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        base_clf = classifier_class(**clf_params)
        calibrated = CalibratedClassifierCV(
            base_clf, method=calibration_method, cv=calibration_cv,
        )
        calibrated.fit(X_train_scaled, y_train)
        y_prob = calibrated.predict_proba(X_test_scaled)[:, 1]

        fold_metrics['auc_roc'].append(
            sklearn.metrics.roc_auc_score(y_test, y_prob))
        fold_metrics['auprc'].append(
            sklearn.metrics.average_precision_score(y_test, y_prob))

        pred_df = transactions_df.loc[test_idx].copy()
        pred_df['predictions'] = y_prob
        _, _, mean_cp = card_precision_top_k(pred_df, top_k=100)
        fold_metrics['cp100'].append(mean_cp)

    return {
        'auc_roc_mean': np.mean(fold_metrics['auc_roc']),
        'auc_roc_std': np.std(fold_metrics['auc_roc']),
        'auprc_mean': np.mean(fold_metrics['auprc']),
        'auprc_std': np.std(fold_metrics['auprc']),
        'cp100_mean': np.mean(fold_metrics['cp100']),
        'cp100_std': np.std(fold_metrics['cp100']),
    }


def performance_assessment(predictions_df: pd.DataFrame,
                           output_feature: str = 'TX_FRAUD',
                           prediction_feature: str = 'predictions',
                           top_k_list: list = None,
                           rounded: bool = True) -> pd.DataFrame:
    """
    Evaluación completa del rendimiento: AUC ROC, Average Precision y Card Precision@k.

    Compatible con el protocolo del libro (Chapter 3/4).

    Args:
        predictions_df: DataFrame con columnas TX_FRAUD, predictions, CUSTOMER_ID, TX_TIME_DAYS
        output_feature: Nombre de la columna de etiquetas
        prediction_feature: Nombre de la columna de predicciones
        top_k_list: Lista de valores k para Card Precision@k
        rounded: Si True, redondea a 3 decimales

    Returns:
        DataFrame con las métricas calculadas
    """
    if top_k_list is None:
        top_k_list = [100]

    from sklearn import metrics as sk_metrics

    auc_roc = sk_metrics.roc_auc_score(
        predictions_df[output_feature],
        predictions_df[prediction_feature],
    )
    avg_precision = sk_metrics.average_precision_score(
        predictions_df[output_feature],
        predictions_df[prediction_feature],
    )

    performances = pd.DataFrame(
        [[auc_roc, avg_precision]],
        columns=['AUC ROC', 'Average precision'],
    )

    for top_k in top_k_list:
        _, _, mean_cp_top_k = card_precision_top_k(predictions_df, top_k)
        performances[f'Card Precision@{top_k}'] = mean_cp_top_k

    if rounded:
        performances = performances.round(3)

    return performances


def print_dataset_summary(train_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          dataset_name: str = "Dataset") -> None:
    """
    Imprime un resumen del dataset: tamaño, distribución de clases, etc.
    
    Args:
        train_df: DataFrame de entrenamiento
        test_df: DataFrame de test
        dataset_name: Nombre descriptivo del dataset
    """
    print(f"\n{'='*60}")
    print(f"  {dataset_name}")
    print(f"{'='*60}")
    
    for name, df in [("Train", train_df), ("Test", test_df)]:
        n_total = len(df)
        n_fraud = df[OUTPUT_FEATURE].sum()
        n_legit = n_total - n_fraud
        pct_fraud = (n_fraud / n_total) * 100 if n_total > 0 else 0
        
        print(f"\n  {name}:")
        print(f"    Total transacciones: {n_total:,}")
        print(f"    Legítimas:           {n_legit:,} ({100 - pct_fraud:.2f}%)")
        print(f"    Fraudulentas:        {int(n_fraud):,} ({pct_fraud:.2f}%)")
        print(f"    Ratio desbalance:    1:{int(n_legit / n_fraud) if n_fraud > 0 else 'inf'}")
    
    print(f"\n{'='*60}")
