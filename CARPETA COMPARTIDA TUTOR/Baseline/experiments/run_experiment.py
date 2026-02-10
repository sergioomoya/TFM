#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para ejecutar un experimento específico dentro del contenedor Docker.

Uso:
    python experiments/run_experiment.py --notebook experiments/experiment_a_baseline.ipynb
    
    O sin argumentos para ejecutar todos los experimentos en orden:
    python experiments/run_experiment.py --all
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


TIMEOUT_PER_CELL = 3600  # 60 min por celda (deep learning puede tardar)

ALL_EXPERIMENTS = [
    "experiments/experiment_a_baseline.ipynb",
    "experiments/experiment_b_cost_sensitive.ipynb",
    "experiments/experiment_c_leakage_test.ipynb",
    "experiments/experiment_d_interpretability.ipynb",
]


def execute_notebook(notebook_path: str) -> dict:
    """Ejecuta un notebook y retorna los resultados."""
    path = Path(notebook_path)
    result = {
        "notebook": str(path),
        "status": "unknown",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "error": None,
        "cells_executed": 0,
        "cells_total": 0,
        "execution_time_seconds": None,
    }

    try:
        print(f"\n{'='*70}")
        print(f"  Ejecutando: {path.name}")
        print(f"{'='*70}\n")

        with open(path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        result["cells_total"] = len([c for c in nb.cells if c.cell_type == 'code'])

        ep = ExecutePreprocessor(
            timeout=TIMEOUT_PER_CELL,
            kernel_name='python3',
            allow_errors=False,
        )

        start = time.time()
        ep.preprocess(nb, {'metadata': {'path': str(path.parent)}})
        elapsed = time.time() - start

        executed = sum(
            1 for c in nb.cells
            if c.cell_type == 'code' and c.get('execution_count') is not None
        )

        result.update({
            "cells_executed": executed,
            "execution_time_seconds": round(elapsed, 2),
            "end_time": datetime.now().isoformat(),
            "status": "success",
        })

        # Guardar notebook ejecutado
        output_path = path.parent / "results" / f"{path.stem}_executed.ipynb"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        print(f"  ✓ Completado en {elapsed:.1f}s  ({executed}/{result['cells_total']} celdas)")
        print(f"  ✓ Guardado en: {output_path}")

    except Exception as exc:
        result.update({
            "status": "error",
            "end_time": datetime.now().isoformat(),
            "error": str(exc),
            "error_traceback": traceback.format_exc(),
        })
        print(f"  ✗ Error: {exc}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Ejecutar experimentos del TFM")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--notebook", type=str, help="Ruta al notebook a ejecutar")
    group.add_argument("--all", action="store_true", help="Ejecutar todos los experimentos")
    args = parser.parse_args()

    notebooks = ALL_EXPERIMENTS if args.all else [args.notebook]
    results = []

    print("="*70)
    print("  EJECUCIÓN DE EXPERIMENTOS - TFM Detección de Fraude")
    print("="*70)

    for nb_path in notebooks:
        if not Path(nb_path).exists():
            print(f"\n  ⚠ No encontrado: {nb_path}")
            results.append({"notebook": nb_path, "status": "not_found"})
            continue
        results.append(execute_notebook(nb_path))

    # Guardar reporte
    report_path = Path("experiments/results/execution_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Resumen
    ok = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    print(f"\n{'='*70}")
    print(f"  RESUMEN: {ok}/{total} experimentos completados con éxito")
    print(f"{'='*70}")

    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
