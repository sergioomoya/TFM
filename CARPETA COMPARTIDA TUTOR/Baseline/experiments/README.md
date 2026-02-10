# Framework de Experimentación

Este directorio contiene los scripts y notebooks necesarios para ejecutar los experimentos de investigación del TFM.

## Estructura

- **`config.py`**: Archivo de configuración global. Define rutas a datos y parámetros comunes.
- **`data_utils.py`**: Librería de utilidades para cargar datos transformados, calcular métricas (CP@100) y generar splits temporales.
- **`run_experiment.py`**: Script CLI para ejecutar experimentos de forma desatendida.
- **`experiment_*.ipynb`**: Cuadernos individuales para cada experimento.
- **`results/`**: Directorio donde se guardan automáticamente los outputs (JSON, CSV, PKL, PNG).

## Cómo añadir un nuevo experimento

1.  Crea un nuevo notebook `experiment_x_nombre.ipynb`.
2.  Importa las utilidades comunes:
    ```python
    from experiments.data_utils import load_transformed_data, get_train_test_set
    from experiments.config import DATA_DIR_TRANSFORMED
    ```
3.  Asegúrate de que el notebook guarda sus resultados en `experiments/results/`.
4.  Registra el nuevo notebook en la lista `ALL_EXPERIMENTS` dentro de `run_experiment.py`.

## Ejecución

Desde la raíz del proyecto (vía Docker):

```bash
# Ejecutar todos
docker compose run --rm experiments

# Ejecutar uno específico
docker compose run --rm experiments python experiments/run_experiment.py --notebook experiments/experiment_a_baseline.ipynb
```
