#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para validar la estructura de los notebooks unificados sin ejecutarlos completamente.
Útil para verificar que los notebooks están bien formados antes de ejecutarlos en Docker.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Configurar codificación UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

NOTEBOOKS_TO_VALIDATE = [
    "Chapter_3_GettingStarted/Chapter_3_Unified.ipynb",
    "Chapter_4_PerformanceMetrics/Chapter_4_Unified.ipynb",
    "Chapter_5_ModelValidationAndSelection/Chapter_5_Unified.ipynb",
    "Chapter_6_ImbalancedLearning/Chapter_6_Unified.ipynb",
    "Chapter_7_DeepLearning/Chapter_7_Unified.ipynb",
]

def validate_notebook_structure(notebook_path: Path) -> dict:
    """
    Valida la estructura básica de un notebook.
    
    Args:
        notebook_path: Ruta al notebook a validar
        
    Returns:
        Diccionario con información de validación
    """
    import nbformat
    
    result = {
        "notebook": str(notebook_path),
        "status": "unknown",
        "errors": [],
        "warnings": [],
        "cells_total": 0,
        "cells_code": 0,
        "cells_markdown": 0,
        "cells_raw": 0,
        "has_imports": False,
        "has_shared_functions": False
    }
    
    try:
        # Cargar notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Validar estructura básica
        if 'cells' not in nb:
            result["errors"].append("El notebook no tiene la estructura 'cells'")
            result["status"] = "error"
            return result
        
        result["cells_total"] = len(nb.cells)
        
        # Analizar celdas
        for i, cell in enumerate(nb.cells):
            cell_type = cell.get('cell_type', 'unknown')
            
            if cell_type == 'code':
                result["cells_code"] += 1
                source = cell.get('source', '')
                
                # Verificar imports
                if any(imp in source for imp in ['import ', 'from ']):
                    result["has_imports"] = True
                
                # Verificar funciones compartidas
                if 'shared_functions' in source.lower() or 'read_from_files' in source:
                    result["has_shared_functions"] = True
                    
            elif cell_type == 'markdown':
                result["cells_markdown"] += 1
            elif cell_type == 'raw':
                result["cells_raw"] += 1
        
        # Validaciones
        if result["cells_total"] == 0:
            result["errors"].append("El notebook no tiene celdas")
        elif result["cells_code"] == 0:
            result["warnings"].append("El notebook no tiene celdas de código")
        
        if not result["has_imports"]:
            result["warnings"].append("No se detectaron imports en el notebook")
        
        if result["errors"]:
            result["status"] = "error"
        elif result["warnings"]:
            result["status"] = "warning"
        else:
            result["status"] = "valid"
            
    except json.JSONDecodeError as e:
        result["status"] = "error"
        result["errors"].append(f"Error de formato JSON: {str(e)}")
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"Error inesperado: {str(e)}")
    
    return result

def main():
    """Función principal."""
    try:
        import nbformat
    except ImportError:
        print("ERROR: nbformat no está instalado.")
        print("Instala con: pip install nbformat")
        sys.exit(1)
    
    base_path = Path(__file__).parent
    
    print("="*70)
    print("VALIDACIÓN DE ESTRUCTURA DE NOTEBOOKS UNIFICADOS")
    print("="*70)
    print(f"Directorio base: {base_path}")
    print(f"Notebooks a validar: {len(NOTEBOOKS_TO_VALIDATE)}\n")
    
    results = []
    
    for notebook_rel_path in NOTEBOOKS_TO_VALIDATE:
        notebook_path = base_path / notebook_rel_path
        
        if not notebook_path.exists():
            print(f"[ERROR] No se encontró: {notebook_path}")
            results.append({
                "notebook": str(notebook_path),
                "status": "not_found",
                "errors": ["Archivo no encontrado"]
            })
            continue
        
        result = validate_notebook_structure(notebook_path)
        results.append(result)
        
        # Mostrar resultado
        print(f"\n{'='*70}")
        print(f"Notebook: {notebook_path.name}")
        print(f"Estado: {result['status'].upper()}")
        print(f"Celdas totales: {result['cells_total']}")
        print(f"  - Código: {result['cells_code']}")
        print(f"  - Markdown: {result['cells_markdown']}")
        print(f"  - Raw: {result['cells_raw']}")
        
        if result['has_imports']:
            print("  ✓ Contiene imports")
        if result['has_shared_functions']:
            print("  ✓ Contiene referencias a funciones compartidas")
        
        if result['errors']:
            print("\nErrores:")
            for error in result['errors']:
                print(f"  ✗ {error}")
        
        if result['warnings']:
            print("\nAdvertencias:")
            for warning in result['warnings']:
                print(f"  ⚠ {warning}")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    
    valid = sum(1 for r in results if r['status'] == 'valid')
    warnings = sum(1 for r in results if r['status'] == 'warning')
    errors = sum(1 for r in results if r['status'] == 'error')
    not_found = sum(1 for r in results if r['status'] == 'not_found')
    
    print(f"Válidos: {valid}")
    print(f"Con advertencias: {warnings}")
    print(f"Con errores: {errors}")
    print(f"No encontrados: {not_found}")
    
    if errors == 0 and not_found == 0:
        print("\n✓ Todos los notebooks tienen una estructura válida")
        print("\nNOTA: Para ejecutar los notebooks completamente, usa Docker:")
        print("  1. Asegúrate de que Docker Desktop esté ejecutándose")
        print("  2. Ejecuta: docker-compose build")
        print("  3. Ejecuta: docker-compose up")
        sys.exit(0)
    else:
        print("\n✗ Algunos notebooks tienen problemas. Revisa los errores arriba.")
        sys.exit(1)

if __name__ == "__main__":
    main()
