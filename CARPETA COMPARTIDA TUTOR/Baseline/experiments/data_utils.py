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
