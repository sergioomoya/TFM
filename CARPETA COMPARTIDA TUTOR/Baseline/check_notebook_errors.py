#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para verificar errores de sintaxis y problemas comunes en los notebooks unificados.
Este script NO ejecuta los notebooks, solo verifica errores estáticos.
"""

import json
import ast
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Configurar codificación UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

NOTEBOOKS_TO_CHECK = [
    "Chapter_3_GettingStarted/Chapter_3_Unified.ipynb",
    "Chapter_4_PerformanceMetrics/Chapter_4_Unified.ipynb",
    "Chapter_5_ModelValidationAndSelection/Chapter_5_Unified.ipynb",
    "Chapter_6_ImbalancedLearning/Chapter_6_Unified.ipynb",
    "Chapter_7_DeepLearning/Chapter_7_Unified.ipynb",
]

def check_python_syntax(code: str, cell_index: int) -> List[Dict]:
    """
    Verifica errores de sintaxis en código Python.
    Ignora comandos mágicos de Jupyter que son válidos en notebooks.
    
    Args:
        code: Código Python a verificar
        cell_index: Índice de la celda (para reporte)
        
    Returns:
        Lista de errores encontrados
    """
    errors = []
    
    if not code.strip():
        return errors
    
    # Filtrar comandos mágicos de Jupyter (válidos en notebooks)
    lines = code.split('\n')
    python_code_lines = []
    has_magic_commands = False
    
    for line in lines:
        stripped = line.strip()
        # Ignorar comandos mágicos de Jupyter
        if (stripped.startswith('%') or 
            stripped.startswith('!') or 
            stripped.startswith('%%')):
            has_magic_commands = True
            continue
        python_code_lines.append(line)
    
    # Si solo hay comandos mágicos, no hay código Python que verificar
    if not python_code_lines and has_magic_commands:
        return errors
    
    python_code = '\n'.join(python_code_lines)
    
    if not python_code.strip():
        return errors
    
    try:
        ast.parse(python_code)
    except SyntaxError as e:
        errors.append({
            "type": "syntax_error",
            "cell_index": cell_index,
            "message": str(e),
            "line": e.lineno,
            "offset": e.offset,
            "text": e.text
        })
    except Exception as e:
        errors.append({
            "type": "parse_error",
            "cell_index": cell_index,
            "message": str(e)
        })
    
    return errors

def check_common_issues(code: str, cell_index: int) -> List[Dict]:
    """
    Verifica problemas comunes en el código.
    
    Args:
        code: Código Python a verificar
        cell_index: Índice de la celda
        
    Returns:
        Lista de advertencias encontradas
    """
    warnings = []
    
    # Verificar imports problemáticos
    if 'import *' in code:
        warnings.append({
            "type": "wildcard_import",
            "cell_index": cell_index,
            "message": "Uso de 'import *' puede causar conflictos de nombres"
        })
    
    # Verificar posibles problemas con funciones compartidas
    if 'read_from_files' in code and 'shared_functions' not in code.lower():
        warnings.append({
            "type": "missing_shared_functions",
            "cell_index": cell_index,
            "message": "Uso de 'read_from_files' pero no se detecta importación de funciones compartidas"
        })
    
    # Verificar posibles problemas con rutas
    if 'open(' in code and '../' in code:
        warnings.append({
            "type": "relative_path",
            "cell_index": cell_index,
            "message": "Uso de rutas relativas puede causar problemas dependiendo del directorio de trabajo"
        })
    
    return warnings

def analyze_notebook(notebook_path: Path) -> Dict:
    """
    Analiza un notebook en busca de errores y problemas.
    
    Args:
        notebook_path: Ruta al notebook
        
    Returns:
        Diccionario con resultados del análisis
    """
    result = {
        "notebook": str(notebook_path),
        "status": "unknown",
        "syntax_errors": [],
        "warnings": [],
        "cells_analyzed": 0,
        "cells_with_errors": 0,
        "cells_with_warnings": 0
    }
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        if 'cells' not in nb:
            result["status"] = "error"
            result["syntax_errors"].append({
                "type": "structure_error",
                "message": "El notebook no tiene la estructura 'cells'"
            })
            return result
        
        # Analizar cada celda de código
        for i, cell in enumerate(nb['cells']):
            if cell.get('cell_type') != 'code':
                continue
            
            result["cells_analyzed"] += 1
            source = cell.get('source', [])
            
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            
            # Verificar sintaxis
            syntax_errors = check_python_syntax(code, i)
            if syntax_errors:
                result["syntax_errors"].extend(syntax_errors)
                result["cells_with_errors"] += 1
            
            # Verificar problemas comunes
            warnings = check_common_issues(code, i)
            if warnings:
                result["warnings"].extend(warnings)
                result["cells_with_warnings"] += 1
        
        # Determinar estado final
        if result["syntax_errors"]:
            result["status"] = "error"
        elif result["warnings"]:
            result["status"] = "warning"
        else:
            result["status"] = "ok"
            
    except json.JSONDecodeError as e:
        result["status"] = "error"
        result["syntax_errors"].append({
            "type": "json_error",
            "message": f"Error al parsear JSON: {str(e)}"
        })
    except Exception as e:
        result["status"] = "error"
        result["syntax_errors"].append({
            "type": "unexpected_error",
            "message": f"Error inesperado: {str(e)}"
        })
    
    return result

def generate_report(results: List[Dict]) -> None:
    """
    Genera un reporte de los errores encontrados.
    
    Args:
        results: Lista de resultados del análisis
    """
    report_path = Path("execution_results") / f"error_check_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE VERIFICACIÓN DE ERRORES EN NOTEBOOKS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        total_errors = 0
        total_warnings = 0
        
        for result in results:
            f.write(f"\n{'='*70}\n")
            f.write(f"Notebook: {result['notebook']}\n")
            f.write(f"Estado: {result['status'].upper()}\n")
            f.write(f"Celdas analizadas: {result['cells_analyzed']}\n")
            f.write(f"Celdas con errores: {result['cells_with_errors']}\n")
            f.write(f"Celdas con advertencias: {result['cells_with_warnings']}\n")
            
            if result['syntax_errors']:
                f.write(f"\nERRORES DE SINTAXIS ({len(result['syntax_errors'])}):\n")
                for error in result['syntax_errors']:
                    f.write(f"  Celda {error.get('cell_index', '?')}: {error.get('message', 'Error desconocido')}\n")
                    if 'line' in error:
                        f.write(f"    Línea: {error['line']}\n")
                    if 'text' in error:
                        f.write(f"    Código: {error['text']}\n")
                total_errors += len(result['syntax_errors'])
            
            if result['warnings']:
                f.write(f"\nADVERTENCIAS ({len(result['warnings'])}):\n")
                for warning in result['warnings']:
                    f.write(f"  Celda {warning.get('cell_index', '?')}: {warning.get('message', 'Advertencia desconocida')}\n")
                total_warnings += len(result['warnings'])
            
            f.write(f"\n{'='*70}\n")
        
        # Resumen
        f.write(f"\n{'='*70}\n")
        f.write("RESUMEN GENERAL\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total de notebooks analizados: {len(results)}\n")
        f.write(f"Total de errores encontrados: {total_errors}\n")
        f.write(f"Total de advertencias: {total_warnings}\n")
        
        notebooks_with_errors = sum(1 for r in results if r['status'] == 'error')
        notebooks_with_warnings = sum(1 for r in results if r['status'] == 'warning')
        notebooks_ok = sum(1 for r in results if r['status'] == 'ok')
        
        f.write(f"\nNotebooks sin errores: {notebooks_ok}\n")
        f.write(f"Notebooks con advertencias: {notebooks_with_warnings}\n")
        f.write(f"Notebooks con errores: {notebooks_with_errors}\n")
        f.write(f"{'='*70}\n")
    
    print(f"\n{'='*70}")
    print("REPORTE GENERADO")
    print(f"{'='*70}")
    print(f"Ruta: {report_path}")

def main():
    """Función principal."""
    base_path = Path(__file__).parent
    
    print("="*70)
    print("VERIFICACIÓN DE ERRORES EN NOTEBOOKS UNIFICADOS")
    print("="*70)
    print(f"Directorio base: {base_path}")
    print(f"Notebooks a verificar: {len(NOTEBOOKS_TO_CHECK)}\n")
    print("NOTA: Esta verificación solo detecta errores de sintaxis y problemas estáticos.")
    print("Para ejecutar los notebooks completamente, usa Docker.\n")
    
    results = []
    
    for notebook_rel_path in NOTEBOOKS_TO_CHECK:
        notebook_path = base_path / notebook_rel_path
        
        if not notebook_path.exists():
            print(f"\n[ERROR] No se encontró: {notebook_path}")
            results.append({
                "notebook": str(notebook_path),
                "status": "not_found",
                "syntax_errors": [{"type": "file_not_found", "message": "Archivo no encontrado"}],
                "warnings": [],
                "cells_analyzed": 0,
                "cells_with_errors": 0,
                "cells_with_warnings": 0
            })
            continue
        
        print(f"\n{'='*70}")
        print(f"Analizando: {notebook_path.name}")
        print('='*70)
        
        result = analyze_notebook(notebook_path)
        results.append(result)
        
        # Mostrar resumen rápido
        if result['status'] == 'ok':
            print(f"✓ Sin errores detectados ({result['cells_analyzed']} celdas analizadas)")
        elif result['status'] == 'warning':
            print(f"⚠ {len(result['warnings'])} advertencias encontradas ({result['cells_analyzed']} celdas analizadas)")
        else:
            print(f"✗ {len(result['syntax_errors'])} errores encontrados ({result['cells_analyzed']} celdas analizadas)")
    
    # Generar reporte
    generate_report(results)
    
    # Resumen final
    total_errors = sum(len(r['syntax_errors']) for r in results)
    total_warnings = sum(len(r['warnings']) for r in results)
    notebooks_with_errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"Notebooks analizados: {len(results)}")
    print(f"Errores de sintaxis encontrados: {total_errors}")
    print(f"Advertencias encontradas: {total_warnings}")
    print(f"Notebooks con errores: {notebooks_with_errors}")
    
    if notebooks_with_errors == 0:
        print("\n✓ No se encontraron errores de sintaxis en los notebooks")
        print("\nNOTA: Para ejecutar los notebooks completamente y detectar errores en tiempo de ejecución:")
        print("  1. Inicia Docker Desktop")
        print("  2. Ejecuta: docker compose build")
        print("  3. Ejecuta: docker compose up")
        sys.exit(0)
    else:
        print(f"\n✗ Se encontraron errores en {notebooks_with_errors} notebook(s)")
        print("Revisa el reporte generado para más detalles.")
        sys.exit(1)

if __name__ == "__main__":
    main()
