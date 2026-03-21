# Figuras `experiments/results/figures/ulb_tfm/`

| Archivo | Origen de datos | Script |
|--------|-----------------|--------|
| `desbalance_clases.svg` | CSV Kaggle ULB (`dataset/creditcard.csv`) | `experiments/ulb_tfm_figures.py` |
| `roc_vs_pr.svg` | Idem (PR con línea de azar = proporción de fraude en test) | Idem |
| `shap_waterfall_tp.png`, `shap_waterfall_tn.png` | Idem (waterfall local, `max_display=10`) | Idem |
| `shap_waterfall_combined.svg` | Idem (opcional, ambos paneles en vectorial) | Idem |
| **`shap_beeswarm.pdf`** | **Dataset transformado del libro** (`TERMINAL_ID_*`, `CUSTOMER_ID_*`, `TX_AMOUNT`, …) | **`experiments/generate_shap_beeswarm_transformed_dataset.py`** |

La memoria del TFM describe el feature engineering del baseline del libro; el **beeswarm SHAP** debe generarse con el segundo script para que el eje Y muestre las variables correctas.

```bash
docker compose run --rm experiments python experiments/generate_shap_beeswarm_transformed_dataset.py
```

Tras ejecutar `ulb_tfm_figures.py` de nuevo, conviene volver a lanzar el comando anterior para no sobrescribir el beeswarm con el modelo ULB.
