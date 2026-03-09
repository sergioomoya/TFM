# Variante Experimento A: Submuestreo de Transacciones Legítimas

**Estado:** Implementado  
**Ubicación:** `experiments/run_experiment_a_undersampled.py`  
**Estrategia:** Undersampling de la clase mayoritaria (legítimas) en train y validación

---

## 1. Motivación

El Experimento A original usa ~118 veces más transacciones legítimas que fraudulentas (~0,84 % de fraude). Esta variante explora si **reducir el desbalance en el entrenamiento** —eliminando parte de las transacciones legítimas— mejora el aprendizaje de patrones de fraude.

### Efectos esperados del undersampling de legítimas

| Aspecto | Resultado esperado |
|---------|--------------------|
| **AUPRC / CP@100** | Posible mejora al dar más peso relativo al fraude durante el entrenamiento |
| **Recall de fraude** | Posible aumento al forzar al modelo a prestar más atención a la clase minoritaria |
| **Falsas alarmas (FP)** | Posible aumento: el modelo verá menos ejemplos de “normalidad”, podría ser más sensible |
| **Tiempo de entrenamiento** | **Reducción** al haber menos muestras totales en train |

---

## 2. Metodología

- **Misma base que Experimento A:** Validación prequential, mismos modelos, mismos grids.
- **Cambio único:** En cada fold, el conjunto de entrenamiento se submuestrea de forma que la relación legítimas:fraudes sea como máximo `UNDERSAMPLE_LEGIT_RATIO` (p. ej. 10:1).
- **Test intacto:** El conjunto de test **no se modifica** para mantener la evaluación honesta.

### Parámetro principal

```python
UNDERSAMPLE_LEGIT_RATIO = 10.0  # 10 legítimas por 1 fraude (vs. ~118:1 original)
```

Valores posibles:
- `10.0` → ratio 10:1 (por defecto)
- `5.0` → ratio 5:1
- `1.0` → balanceado 1:1
- `None` en `data_utils` → sin submuestreo (comportamiento Experimento A)

---

## 3. Ejecución

**Recomendado (entorno tfm + monitor):**
```powershell
.\scripts\runners\run_experiment_a_balance_controlled.ps1
```
- Activa conda tfm automáticamente
- Monitor de GPU/RAM en background
- Log con timestamp en `experiments/results/experiment_a_balance_run_*.log`

**Directo:**
```bash
conda activate tfm
python experiments/run_experiment_a_all_balance_variants.py
```

**Docker:**
```bash
docker compose run --rm experiments-gpu python experiments/run_experiment_a_all_balance_variants.py
```

---

## 4. Archivos generados

| Archivo | Descripción |
|---------|-------------|
| `experiment_a_undersamp_10_results.csv` | Métricas (AUC ROC, AUPRC, CP@100) |
| `experiment_a_undersamp_10_predictions.pkl` | Predicciones y metadatos |
| `experiment_a_undersamp_10_baseline_results.png` | Barras comparativas |
| `experiment_a_undersamp_10_confusion_matrices.png` | Matrices de confusión |

*(El sufijo `10` corresponde a `UNDERSAMPLE_LEGIT_RATIO=10.0`.)*

---

## 5. Resultados obtenidos — XGBoost (variante 10:1)

Métricas extraídas de la ejecución controlada (log `experiment_a_balance_run_20260302_234951.log`) y de los archivos `experiment_a_undersamp_10_results.csv` y `experiment_a_undersamp_10_predictions.pkl`:

| Métrica | Valor | Fuente |
|---------|-------|--------|
| **AUPRC medio** | 0,659 (0,6589 ± 0,0209) | CSV — media sobre 4 folds prequential |
| **CP@100 medio** | 0,279 (0,2786 ± 0,0066) | CSV — Card Precision@100 en top 100 tarjetas |
| **Recall (sensibilidad) del fraude** | 66,31 % | Matriz de confusión agregada: TP/(TP+FN) = 1055/1591 |
| **Falsos positivos (FP)** | 579 | Transacciones legítimas marcadas erróneamente como fraude |
| **Tiempo de entrenamiento XGBoost** | 16,1 s | Log de monitorización (variante 10:1) |

### Matriz de confusión agregada (XGBoost, threshold=0,5)

|  | **Predicho: Legítimo** | **Predicho: Fraude** |
|--|------------------------|----------------------|
| **Real: Legítimo** | TN = 229.991 | FP = 579 |
| **Real: Fraude** | FN = 536 | TP = 1.055 |

- **Total fraudes reales:** 1.591 (TP + FN)
- **Fraudes detectados:** 1.055 (66,31 %)
- **Fraudes no detectados:** 536 (33,69 %)
- **Falsas alarmas:** 579

---

## 6. Comparativa con Experimento A (baseline)

### Tabla comparativa de métricas (4 folds prequential)

| Modelo | AUC ROC (baseline) | AUC ROC (undersamp 10:1) | AUPRC (baseline) | AUPRC (undersamp 10:1) | CP@100 (baseline) | CP@100 (undersamp 10:1) |
|--------|-------------------|--------------------------|------------------|------------------------|-------------------|-------------------------|
| Logistic Regression | 0,869 | 0,871 | 0,635 | 0,624 | 0,218 | 0,265 |
| Random Forest      | 0,873 | 0,875 | 0,685 | 0,648 | 0,256 | 0,285 |
| XGBoost            | 0,846 | 0,868 | 0,600 | **0,659** | 0,235 | **0,279** |

*Fuente: experiment_a_original_results.csv, experiment_a_undersamp_10_results.csv (ejecución run_experiment_a_all_balance_variants.py).*

### XGBoost: baseline vs undersamp 10:1 (detalle)

| Métrica | Baseline A | Undersamp 10:1 | Observación |
|---------|------------|----------------|-------------|
| AUPRC | 0,600 | 0,659 | **Mejora +0,059** |
| CP@100 | 0,235 | 0,279 | **Mejora +0,044** |
| Recall fraude | 61,75 %* | 66,31 % | **Mejora +4,56 pp** |
| FP | 85* | 579 | Aumento de falsas alarmas |
| Tiempo entrenamiento | ~55 min (CPU)* | 16,1 s (GPU) | Reducción notable |

*\* Valores del Experimento A documentado en INFORME_EXPERIMENTO_A.md (ejecución completa).*

- El undersampling 10:1 **mejora AUPRC y CP@100** en XGBoost respecto al baseline de la misma ejecución.
- **Aumenta el Recall de fraude** (66,31 % vs 61,75 %), detectando más fraudes reales.
- **Aumenta los FP** de 85 a 579: el modelo es más sensible pero genera más falsas alarmas.

---

## 7. Conclusiones

- El undersampling de legítimas (10:1) es una alternativa ligera a técnicas como `class_weight='balanced'` o SMOTE.
- La evaluación se mantiene honesta porque el **conjunto de test conserva la distribución natural** de clases.
- Para XGBoost, la variante 10:1 **aumenta el Recall de fraude** (66,31 % vs 61,75 %) a costa de **más falsas alarmas** (579 vs 85 FP).
- Reducir el dataset de entrenamiento acelera el entrenamiento de forma notable.
