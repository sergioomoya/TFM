#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para ejecutar los notebooks unificados y verificar su correcta ejecución.
"""

import json
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import NotebookExporter

# Configurar codificación UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Configuración
NOTEBOOKS_TO_TEST = [
    # "Chapter_3_GettingStarted/Chapter_3_Unified.ipynb",      # OK
    # "Chapter_4_PerformanceMetrics/Chapter_4_Unified.ipynb",   # OK
    # "Chapter_5_ModelValidationAndSelection/Chapter_5_Unified.ipynb",  # OK
    # "Chapter_6_ImbalancedLearning/Chapter_6_Unified.ipynb",   # OK
    "Chapter_7_DeepLearning/Chapter_7_Unified.ipynb",           # Requiere re-ejecución con timeout mayor
]

TIMEOUT_PER_CELL = 3600  # 60 minutos por celda (para entrenamientos de deep learning)
RESULTS_DIR = Path("execution_results")
RESULTS_DIR.mkdir(exist_ok=True)

def execute_notebook(notebook_path: Path) -> dict:
    """
    Ejecuta un notebook y retorna un diccionario con los resultados.
    
    Args:
        notebook_path: Ruta al notebook a ejecutar
        
    Returns:
        Diccionario con información sobre la ejecución
    """
    result = {
        "notebook": str(notebook_path),
        "status": "unknown",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "error": None,
        "cells_executed": 0,
        "cells_total": 0,
        "execution_time_seconds": None
    }
    
    try:
        print(f"\n{'='*70}")
        print(f"Ejecutando: {notebook_path}")
        print(f"{'='*70}")
        
        # Cargar notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        result["cells_total"] = len([cell for cell in nb.cells if cell.cell_type == 'code'])
        
        # Configurar ejecutor
        ep = ExecutePreprocessor(
            timeout=TIMEOUT_PER_CELL,
            kernel_name='python3',
            allow_errors=False
        )
        
        # Ejecutar notebook
        start_time = datetime.now()
        ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        
        # Contar celdas ejecutadas
        executed = sum(1 for cell in nb.cells 
                      if cell.cell_type == 'code' 
                      and 'execution_count' in cell.get('metadata', {})
                      and cell['metadata'].get('execution_count') is not None)
        
        result["cells_executed"] = executed
        result["execution_time_seconds"] = execution_time
        result["end_time"] = end_time.isoformat()
        result["status"] = "success"
        
        # Guardar notebook ejecutado
        output_path = RESULTS_DIR / f"{notebook_path.stem}_executed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        result["output_path"] = str(output_path)
        
        print(f"✓ Ejecución completada exitosamente")
        print(f"  Celdas ejecutadas: {executed}/{result['cells_total']}")
        print(f"  Tiempo de ejecución: {execution_time:.2f} segundos")
        
    except Exception as e:
        result["status"] = "error"
        result["end_time"] = datetime.now().isoformat()
        result["error"] = str(e)
        result["error_traceback"] = traceback.format_exc()
        
        print(f"✗ Error durante la ejecución:")
        print(f"  {str(e)}")
        print(f"\n  Traceback completo guardado en el reporte")
        
    return result

def generate_report(results: list) -> None:
    """
    Genera un reporte de ejecución en formato JSON y texto.
    
    Args:
        results: Lista de resultados de ejecución
    """
    report_path = RESULTS_DIR / f"execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Guardar JSON
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generar reporte en texto
    text_report_path = RESULTS_DIR / f"execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE EJECUCIÓN DE NOTEBOOKS UNIFICADOS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for result in results:
            f.write(f"\n{'='*70}\n")
            f.write(f"Notebook: {result['notebook']}\n")
            f.write(f"Estado: {result['status']}\n")
            f.write(f"Celdas ejecutadas: {result['cells_executed']}/{result['cells_total']}\n")
            
            if result['execution_time_seconds']:
                f.write(f"Tiempo de ejecución: {result['execution_time_seconds']:.2f} segundos\n")
            
            if result['error']:
                f.write(f"\nError:\n{result['error']}\n")
                if 'error_traceback' in result:
                    f.write(f"\nTraceback:\n{result['error_traceback']}\n")
            
            f.write(f"\n{'='*70}\n")
        
        # Resumen
        total = len(results)
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        
        f.write(f"\n{'='*70}\n")
        f.write("RESUMEN\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total de notebooks: {total}\n")
        f.write(f"Exitosos: {successful}\n")
        f.write(f"Fallidos: {failed}\n")
        f.write(f"{'='*70}\n")
    
    print(f"\n{'='*70}")
    print("REPORTE GENERADO")
    print(f"{'='*70}")
    print(f"JSON: {report_path}")
    print(f"Texto: {text_report_path}")

def main():
    """Función principal."""
    base_path = Path(__file__).parent
    
    print("="*70)
    print("EJECUCIÓN DE NOTEBOOKS UNIFICADOS")
    print("="*70)
    print(f"Directorio base: {base_path}")
    print(f"Notebooks a ejecutar: {len(NOTEBOOKS_TO_TEST)}")
    
    results = []
    
    for notebook_rel_path in NOTEBOOKS_TO_TEST:
        notebook_path = base_path / notebook_rel_path
        
        if not notebook_path.exists():
            print(f"\n[ADVERTENCIA] No se encontró: {notebook_path}")
            results.append({
                "notebook": str(notebook_path),
                "status": "not_found",
                "error": "Archivo no encontrado"
            })
            continue
        
        result = execute_notebook(notebook_path)
        results.append(result)
    
    # Generar reporte
    generate_report(results)
    
    # Resumen final
    successful = sum(1 for r in results if r['status'] == 'success')
    total = len(results)
    
    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"Notebooks ejecutados exitosamente: {successful}/{total}")
    
    if successful == total:
        print("✓ Todos los notebooks se ejecutaron correctamente")
        sys.exit(0)
    else:
        print("✗ Algunos notebooks fallaron. Revisa el reporte para más detalles.")
        sys.exit(1)

if __name__ == "__main__":
    main()
