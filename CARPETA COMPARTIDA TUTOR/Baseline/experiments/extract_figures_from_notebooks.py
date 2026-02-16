#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extrae las imágenes embebidas (base64) de los notebooks ejecutados
y las guarda como archivos PNG en una carpeta para uso en el TFM.

Uso:
    python experiments/extract_figures_from_notebooks.py
"""

import base64
import json
from pathlib import Path

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR = PROJECT_ROOT / "figuras_experimentos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapeo: nombre del notebook ejecutado -> lista de nombres de figuras (en orden de aparición)
FIGURE_NAMES = {
    "experiment_a_baseline_executed": ["experiment_a_baseline_results.png"],
    "experiment_c_leakage_test_executed": ["experiment_c_leakage_comparison.png"],
    "experiment_d_interpretability_executed": [
        "experiment_d_feature_importance.png",
        "experiment_d_shap_beeswarm.png",
        "experiment_d_shap_force_fraud.png",
        "experiment_d_shap_force_normal.png",
    ],
}


def extract_images_from_notebook(notebook_path: Path, output_dir: Path) -> int:
    """Extrae todas las imágenes PNG de un notebook y las guarda."""
    if not notebook_path.exists():
        print(f"  [!!] No encontrado: {notebook_path}")
        return 0

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    stem = notebook_path.stem.replace("_executed", "")
    figure_names = FIGURE_NAMES.get(
        notebook_path.stem,
        [f"{stem}_fig_{i}.png" for i in range(10)],
    )

    count = 0
    img_index = 0
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                b64 = data["image/png"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                raw = base64.b64decode(b64)
                name = figure_names[img_index] if img_index < len(figure_names) else f"{stem}_fig_{img_index}.png"
                out_path = output_dir / name
                out_path.write_bytes(raw)
                print(f"  OK {name}")
                count += 1
                img_index += 1

    return count


def main():
    print("=" * 60)
    print("  Extracción de figuras de notebooks ejecutados")
    print("=" * 60)
    print(f"\nOrigen:  {RESULTS_DIR}")
    print(f"Destino: {OUTPUT_DIR}\n")

    executed = list(RESULTS_DIR.glob("*_executed.ipynb"))
    if not executed:
        print("[!!] No se encontraron notebooks ejecutados en experiments/results/")
        print("  Ejecuta primero: docker compose run --rm experiments")
        return 1

    total = 0
    for nb_path in sorted(executed):
        print(f"\n{nb_path.name}:")
        n = extract_images_from_notebook(nb_path, OUTPUT_DIR)
        total += n

    print(f"\n{'=' * 60}")
    print(f"  Total: {total} figuras guardadas en {OUTPUT_DIR}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
