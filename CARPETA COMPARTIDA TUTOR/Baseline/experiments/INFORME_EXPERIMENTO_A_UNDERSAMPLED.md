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
.\run_experiment_a_balance_controlled.ps1
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

## 5. Comparativa con Experimento A

Tras ejecutar la variante, se pueden comparar:

1. **AUPRC y CP@100** → ¿Mejora la detección de fraude?
2. **Recall de fraude** → ¿Se detectan más fraudes?
3. **FP (falsos positivos)** → ¿Aumentan las falsas alarmas?
4. **Tiempo de entrenamiento** → ¿Se acelera al reducir el tamaño del train?

---

## 6. Conclusiones (a completar tras ejecución)

- El undersampling de legítimas es una alternativa ligera a técnicas como `class_weight='balanced'` o SMOTE.
- La evaluación se mantiene honesta porque el test conserva la distribución natural de clases.
