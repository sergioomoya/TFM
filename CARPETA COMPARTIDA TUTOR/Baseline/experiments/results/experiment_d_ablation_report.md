# Experimento D - Ablación: Validación SHAP

**Feature eliminada:** `CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW` (mayor mean |SHAP|)

## Comparación de métricas

| Modelo | AUC ROC | AUPRC | CP@100 |
|--------|---------|-------|--------|
| D (completo) | 0.8618 | 0.6389 | 0.2729 |
| D (sin top)  | 0.8498 | 0.5942 | 0.2714 |
| Δ            | -0.0121 | -0.0447 | -0.0014 |

## Ranking de importancia (mean |SHAP|) tras ablación

Nuevo orden de las 14 features restantes en el modelo reentrenado:

| Rank | Feature | mean_abs_SHAP |
|------|---------|---------------|
| 1 | TX_AMOUNT | 1.027430 |
| 2 | CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW | 0.966753 |
| 3 | CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW | 0.409711 |
| 4 | CUSTOMER_ID_NB_TX_30DAY_WINDOW | 0.336473 |
| 5 | TERMINAL_ID_NB_TX_30DAY_WINDOW | 0.289565 |
| 6 | CUSTOMER_ID_NB_TX_7DAY_WINDOW | 0.275097 |
| 7 | TERMINAL_ID_NB_TX_7DAY_WINDOW | 0.230642 |
| 8 | TERMINAL_ID_RISK_7DAY_WINDOW | 0.216973 |
| 9 | CUSTOMER_ID_NB_TX_1DAY_WINDOW | 0.134964 |
| 10 | TERMINAL_ID_NB_TX_1DAY_WINDOW | 0.127866 |
| 11 | TERMINAL_ID_RISK_30DAY_WINDOW | 0.071433 |
| 12 | TX_DURING_NIGHT | 0.068030 |
| 13 | TX_DURING_WEEKEND | 0.052220 |
| 14 | TERMINAL_ID_RISK_1DAY_WINDOW | 0.031257 |

## Figuras generadas

- `experiment_d_ablation_metrics_comparison.png` — Comparación de métricas full vs ablated
- `experiment_d_ablation_shap_ranking.png` — Bar chart mean |SHAP| del modelo ablated
- `experiment_d_ablation_shap_beeswarm.png` — Beeswarm SHAP del modelo ablated

**Conclusión:** Validación exitosa: la feature eliminada contribuía al rendimiento.
