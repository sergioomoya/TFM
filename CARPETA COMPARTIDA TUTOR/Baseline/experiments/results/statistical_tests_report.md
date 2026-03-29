# Tests de significancia estadística — TFM

## Tabla 1: Comparativa algorítmica (3 modelos × 4 folds)

### Métricas por fold

| model               |   fold |   auc_roc |    auprc |    cp100 |
|:--------------------|-------:|----------:|---------:|---------:|
| Logistic Regression |      0 |  0.875503 | 0.652784 | 0.272857 |
| Logistic Regression |      1 |  0.866252 | 0.621607 | 0.311429 |
| Logistic Regression |      2 |  0.888425 | 0.615182 | 0.297143 |
| Logistic Regression |      3 |  0.845234 | 0.649304 | 0.288571 |
| Random Forest       |      0 |  0.871532 | 0.695944 | 0.278571 |
| Random Forest       |      1 |  0.8729   | 0.690106 | 0.317143 |
| Random Forest       |      2 |  0.888696 | 0.684186 | 0.304286 |
| Random Forest       |      3 |  0.858325 | 0.668299 | 0.288571 |
| XGBoost             |      0 |  0.874432 | 0.626019 | 0.272857 |
| XGBoost             |      1 |  0.831721 | 0.625666 | 0.302857 |
| XGBoost             |      2 |  0.876928 | 0.609198 | 0.297143 |
| XGBoost             |      3 |  0.831367 | 0.536759 | 0.277143 |

### Resultados de los tests

```

### AUPRC
Friedman chi2=6.5000, p=0.0388 (significativo p<0.05)
  Wilcoxon Logistic Regression vs Random Forest: W=0.0, p=0.1250 
  Wilcoxon Logistic Regression vs XGBoost: W=1.0, p=0.2500 
  Wilcoxon Random Forest vs XGBoost: W=0.0, p=0.1250 

### CP@100
Friedman chi2=6.6154, p=0.0366 (significativo p<0.05)
  Wilcoxon Logistic Regression vs Random Forest: W=0.0, p=0.2500 
  Wilcoxon Logistic Regression vs XGBoost: W=0.0, p=0.5000 
  Wilcoxon Random Forest vs XGBoost: W=0.0, p=0.1250 

### AUC ROC
Friedman chi2=3.5000, p=0.1738 (no significativo)
  Wilcoxon Logistic Regression vs Random Forest: W=2.0, p=0.3750 
  Wilcoxon Logistic Regression vs XGBoost: W=0.0, p=0.1250 
  Wilcoxon Random Forest vs XGBoost: W=1.0, p=0.2500 
```

## Tabla 5: Ablación (full vs ablated × 4 folds)

### Métricas por fold

| variant   |   fold |   auc_roc |    auprc |    cp100 |
|:----------|-------:|----------:|---------:|---------:|
| full      |      0 |  0.837651 | 0.563702 | 0.245714 |
| full      |      1 |  0.85745  | 0.622274 | 0.3      |
| full      |      2 |  0.875895 | 0.614457 | 0.268571 |
| full      |      3 |  0.867181 | 0.600872 | 0.25     |
| ablated   |      0 |  0.830208 | 0.525318 | 0.252857 |
| ablated   |      1 |  0.849231 | 0.57774  | 0.295714 |
| ablated   |      2 |  0.857046 | 0.553725 | 0.264286 |
| ablated   |      3 |  0.844932 | 0.576061 | 0.254286 |

### Resultados de los tests

```
  AUPRC: W=0.0, p=0.1250 (no significativo)
  CP@100: W=3.5, p=0.7500 (no significativo)
  AUC ROC: W=0.0, p=0.1250 (no significativo)
```

## Latencia de inferencia XAI

| Batch | predict_proba (ms/tx) | TreeSHAP (ms/tx) | Total (ms/tx) |
|-------|----------------------|-------------------|---------------|
| 1 | 0.3336 | 0.8421 | 1.1756 |
| 10 | 0.0368 | 0.5320 | 0.5688 |
| 100 | 0.0051 | 0.0652 | 0.0703 |
| 1000 | 0.0008 | 0.0458 | 0.0467 |

## Correlación RFM

### Ventanas AVG_AMOUNT

|                                     |   CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW |   CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW |   CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW |
|:------------------------------------|-------------------------------------:|-------------------------------------:|--------------------------------------:|
| CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW  |                             1        |                             0.868367 |                              0.829732 |
| CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW  |                             0.868367 |                             1        |                              0.957295 |
| CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW |                             0.829732 |                             0.957295 |                              1        |

