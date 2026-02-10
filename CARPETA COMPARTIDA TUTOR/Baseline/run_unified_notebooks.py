#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ejecuta todos los cuadernos unificados (Chapter 3 → 7) dentro del contenedor Docker.

Uso:
    python run_unified_notebooks.py              → Ejecutar todos
    python run_unified_notebooks.py --notebook Chapter_3_GettingStarted/Chapter_3_Unified.ipynb

Genera un reporte JSON y de texto con los resultados.
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DEFAULT_TIMEOUT_PER_CELL = 3600  # 60 minutos por celda (por defecto)

UNIFIED_NOTEBOOKS = [
    "Chapter_3_GettingStarted/Chapter_3_Unified.ipynb",
    "Chapter_4_PerformanceMetrics/Chapter_4_Unified.ipynb",
    "Chapter_5_ModelValidationAndSelection/Chapter_5_Unified.ipynb",
    "Chapter_6_ImbalancedLearning/Chapter_6_Unified.ipynb",
    "Chapter_7_DeepLearning/Chapter_7_Unified.ipynb",
]

RESULTS_DIR = Path("execution_results")


# =============================================================================
# FUNCIONES
# =============================================================================

def execute_notebook(notebook_path: str, timeout: int = None) -> dict:
    """Ejecuta un notebook y retorna un diccionario con los resultados.
    
    Args:
        notebook_path: Ruta al notebook .ipynb
        timeout: Timeout por celda en segundos. None = sin límite.
    """
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
        print(f"\n{'=' * 70}")
        print(f"  Ejecutando: {path}")
        print(f"  Inicio: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'=' * 70}\n")

        with open(path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        result["cells_total"] = len(code_cells)

        ep = ExecutePreprocessor(
            timeout=timeout,
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

        print(f"  ✓ Completado en {elapsed:.1f}s")
        print(f"    Celdas ejecutadas: {executed}/{result['cells_total']}")

    except Exception as exc:
        elapsed = time.time() - start if 'start' in dir() else 0
        result.update({
            "status": "error",
            "end_time": datetime.now().isoformat(),
            "execution_time_seconds": round(elapsed, 2),
            "error": str(exc),
            "error_traceback": traceback.format_exc(),
        })
        print(f"  ✗ Error tras {elapsed:.1f}s: {exc}")

    return result


def generate_text_report(results: list, report_path: Path) -> None:
    """Genera un reporte de texto legible."""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  REPORTE DE EJECUCIÓN - CUADERNOS UNIFICADOS\n")
        f.write(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for r in results:
            status_icon = "✓" if r["status"] == "success" else "✗"
            f.write(f"{status_icon} {r['notebook']}\n")
            f.write(f"    Estado: {r['status']}\n")
            f.write(f"    Celdas: {r['cells_executed']}/{r['cells_total']}\n")

            if r.get("execution_time_seconds"):
                minutes = r["execution_time_seconds"] / 60
                f.write(f"    Tiempo: {r['execution_time_seconds']:.1f}s ({minutes:.1f} min)\n")

            if r.get("error"):
                f.write(f"    Error: {r['error']}\n")

            f.write("\n")

        ok = sum(1 for r in results if r["status"] == "success")
        f.write("=" * 70 + "\n")
        f.write(f"  RESUMEN: {ok}/{len(results)} notebooks exitosos\n")
        f.write("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Ejecutar cuadernos unificados")
    parser.add_argument(
        "--notebook", type=str,
        help="Ruta a un notebook específico (si no se indica, ejecuta todos)"
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT_PER_CELL,
        help="Timeout por celda en segundos (0 = sin límite, default: %(default)s)"
    )
    args = parser.parse_args()

    # Timeout: 0 significa sin límite (None para nbconvert)
    cell_timeout = args.timeout if args.timeout > 0 else None

    notebooks = [args.notebook] if args.notebook else UNIFIED_NOTEBOOKS
    results = []

    timeout_str = f"{args.timeout}s ({args.timeout // 60} min)" if args.timeout > 0 else "SIN LÍMITE"
    print("=" * 70)
    print("  EJECUCIÓN DE CUADERNOS UNIFICADOS")
    print(f"  Notebooks a ejecutar: {len(notebooks)}")
    print(f"  Timeout por celda: {timeout_str}")
    print("=" * 70)

    total_start = time.time()

    for nb_path in notebooks:
        if not Path(nb_path).exists():
            print(f"\n  ⚠ No encontrado: {nb_path}")
            results.append({"notebook": nb_path, "status": "not_found",
                            "cells_executed": 0, "cells_total": 0})
            continue
        results.append(execute_notebook(nb_path, timeout=cell_timeout))

    total_elapsed = time.time() - total_start

    # Guardar reportes
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = RESULTS_DIR / f"unified_report_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    txt_path = RESULTS_DIR / f"unified_report_{timestamp}.txt"
    generate_text_report(results, txt_path)

    # Resumen final
    ok = sum(1 for r in results if r["status"] == "success")
    total = len(results)

    print(f"\n{'=' * 70}")
    print(f"  RESUMEN FINAL: {ok}/{total} notebooks exitosos")
    print(f"  Tiempo total: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Reporte JSON: {json_path}")
    print(f"  Reporte texto: {txt_path}")
    print(f"{'=' * 70}")

    if ok < total:
        print("\n  ✗ Notebooks con errores:")
        for r in results:
            if r["status"] != "success":
                print(f"    - {r['notebook']}: {r.get('error', 'desconocido')}")

    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
