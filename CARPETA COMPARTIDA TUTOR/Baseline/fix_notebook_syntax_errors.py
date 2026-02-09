#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para corregir errores de sintaxis en notebooks unificados.
Corrige bloques if seguidos de comandos de shell.
"""

import json
import sys
from pathlib import Path

# Configurar codificación UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

NOTEBOOKS_TO_FIX = [
    "Chapter_3_GettingStarted/Chapter_3_Unified.ipynb",
    "Chapter_4_PerformanceMetrics/Chapter_4_Unified.ipynb",
    "Chapter_5_ModelValidationAndSelection/Chapter_5_Unified.ipynb",
    "Chapter_6_ImbalancedLearning/Chapter_6_Unified.ipynb",
    "Chapter_7_DeepLearning/Chapter_7_Unified.ipynb",
]

def fix_if_shell_command_issue(cell_source: list) -> list:
    """
    Corrige el problema de bloques if seguidos de comandos de shell.
    Agrega 'pass' después del if si el siguiente comando es un comando de shell.
    """
    if not cell_source:
        return cell_source
    
    source_text = ''.join(cell_source)
    lines = source_text.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        fixed_lines.append(line)
        
        # Detectar si hay un if seguido de un comando de shell
        if line.strip().startswith('if ') and line.strip().endswith(':'):
            # Verificar la siguiente línea
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # Si la siguiente línea es un comando de shell o está vacía seguida de comando shell
                if next_line.strip().startswith('!') or (not next_line.strip() and i + 2 < len(lines) and lines[i + 2].strip().startswith('!')):
                    # Agregar 'pass' después del if
                    fixed_lines.append('    pass')
        
        i += 1
    
    # Convertir de vuelta a lista de strings
    fixed_text = '\n'.join(fixed_lines)
    # Mantener el formato original (con \n al final de cada línea excepto la última)
    if isinstance(cell_source, list):
        return [line + '\n' if i < len(fixed_lines) - 1 else line 
                for i, line in enumerate(fixed_lines)]
    else:
        return fixed_text.split('\n')

def fix_notebook(notebook_path: Path) -> dict:
    """
    Corrige errores de sintaxis en un notebook.
    
    Args:
        notebook_path: Ruta al notebook
        
    Returns:
        Diccionario con información de las correcciones
    """
    result = {
        "notebook": str(notebook_path),
        "cells_fixed": 0,
        "errors": []
    }
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        if 'cells' not in nb:
            result["errors"].append("El notebook no tiene la estructura 'cells'")
            return result
        
        # Analizar y corregir cada celda
        for i, cell in enumerate(nb['cells']):
            if cell.get('cell_type') != 'code':
                continue
            
            source = cell.get('source', [])
            if not source:
                continue
            
            source_text = ''.join(source) if isinstance(source, list) else source
            
            # Verificar si hay un problema de if + comando shell
            if 'if ' in source_text and ':' in source_text and '!' in source_text:
                # Verificar si el if está seguido directamente de un comando shell
                lines = source_text.split('\n')
                for j, line in enumerate(lines):
                    if line.strip().startswith('if ') and line.strip().endswith(':'):
                        # Buscar el siguiente comando no vacío
                        for k in range(j + 1, len(lines)):
                            next_line = lines[k].strip()
                            if next_line.startswith('!'):
                                # Hay un problema: if seguido de comando shell
                                # Corregir agregando pass
                                fixed_source = fix_if_shell_command_issue(source if isinstance(source, list) else source.split('\n'))
                                cell['source'] = fixed_source
                                result["cells_fixed"] += 1
                                print(f"  [FIX] Celda {i}: Corregido if seguido de comando shell")
                                break
        
        # Guardar notebook corregido
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        
    except Exception as e:
        result["errors"].append(f"Error al procesar notebook: {str(e)}")
    
    return result

def main():
    """Función principal."""
    base_path = Path(__file__).parent
    
    print("="*70)
    print("CORRECCIÓN DE ERRORES DE SINTAXIS EN NOTEBOOKS")
    print("="*70)
    print(f"Directorio base: {base_path}")
    print(f"Notebooks a corregir: {len(NOTEBOOKS_TO_FIX)}\n")
    
    results = []
    
    for notebook_rel_path in NOTEBOOKS_TO_FIX:
        notebook_path = base_path / notebook_rel_path
        
        if not notebook_path.exists():
            print(f"\n[ERROR] No se encontró: {notebook_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Corrigiendo: {notebook_path.name}")
        print('='*70)
        
        result = fix_notebook(notebook_path)
        results.append(result)
        
        if result["cells_fixed"] > 0:
            print(f"✓ {result['cells_fixed']} celda(s) corregida(s)")
        else:
            print("✓ No se encontraron problemas a corregir")
        
        if result["errors"]:
            for error in result["errors"]:
                print(f"✗ {error}")
    
    # Resumen
    total_fixed = sum(r["cells_fixed"] for r in results)
    
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    print(f"Total de celdas corregidas: {total_fixed}")
    
    if total_fixed > 0:
        print("\n✓ Correcciones aplicadas. Ejecuta check_notebook_errors.py nuevamente para verificar.")
    else:
        print("\n✓ No se encontraron problemas a corregir.")

if __name__ == "__main__":
    main()
