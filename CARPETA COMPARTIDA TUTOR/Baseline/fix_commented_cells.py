#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para corregir celdas comentadas en notebooks unificados.

Durante la unificación, las celdas con magics de Jupyter (%%capture, %time, etc.)
fueron completamente comentadas, dejando variables sin definir.
Este script las descomenta restaurando su funcionalidad.
También corrige errores de indentación específicos.
"""

import json
import sys
import copy
from pathlib import Path

NOTEBOOKS_TO_FIX = [
    "Chapter_3_GettingStarted/Chapter_3_Unified.ipynb",
    "Chapter_4_PerformanceMetrics/Chapter_4_Unified.ipynb",
    "Chapter_5_ModelValidationAndSelection/Chapter_5_Unified.ipynb",
    "Chapter_6_ImbalancedLearning/Chapter_6_Unified.ipynb",
    "Chapter_7_DeepLearning/Chapter_7_Unified.ipynb",
]


def is_fully_commented_cell(cell):
    """Verifica si una celda de código está completamente comentada."""
    if cell['cell_type'] != 'code':
        return False
    source = cell.get('source', [])
    if not source:
        return False
    return all(
        line.strip() == '' or line.strip().startswith('#')
        for line in source
    )


def contains_magic_or_shell(source_lines):
    """Verifica si las líneas comentadas contienen magics o comandos shell."""
    joined = ''.join(source_lines)
    magic_patterns = [
        '%%capture', '%time ', '%time\n', '%run ', '%matplotlib',
        '!curl', '!git', '!pip'
    ]
    return any(pattern in joined for pattern in magic_patterns)


def uncomment_cell(source_lines):
    """
    Descomenta una celda que fue completamente comentada.
    Maneja diferentes patrones de comentario:
    - '# código'  -> 'código'
    - '#código'    -> 'código'
    - '# '         -> ''  (línea vacía comentada)
    """
    new_lines = []
    for line in source_lines:
        stripped = line.lstrip()
        leading_whitespace = line[:len(line) - len(stripped)]

        if stripped == '':
            # Línea vacía, mantener
            new_lines.append(line)
        elif stripped.startswith('# '):
            # Patrón más común: '# código'
            new_lines.append(leading_whitespace + stripped[2:])
        elif stripped.startswith('#\n') or stripped == '#':
            # Línea con solo '#'
            new_lines.append(leading_whitespace + stripped[1:].lstrip() + '\n' if '\n' in stripped else '')
        elif stripped.startswith('#'):
            # '#código' sin espacio
            new_lines.append(leading_whitespace + stripped[1:])
        else:
            new_lines.append(line)

    return new_lines


def fix_indentation_errors(nb_data, nb_path):
    """
    Corrige errores de indentación específicos:
    - Comentarios con indentación incorrecta antes de 'def'
    """
    fixes = 0
    for i, cell in enumerate(nb_data['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = cell['source']
        if not source or len(source) < 2:
            continue

        # Buscar patrón: comentario indentado seguido de 'def' sin indentar
        new_source = list(source)
        changed = False
        for j in range(len(new_source) - 1):
            line = new_source[j]
            next_line = new_source[j + 1] if j + 1 < len(new_source) else ''

            # Si la línea es un comentario indentado y la siguiente es un 'def' al nivel 0
            if (line.startswith('    #') and
                    not line.startswith('    # %%') and
                    next_line.startswith('def ')):
                # Quitar indentación del comentario
                new_source[j] = line.lstrip()
                changed = True
                fixes += 1

        if changed:
            cell['source'] = new_source
            print(f"  [INDENT FIX] Celda {i}: Corregida indentación de comentario")

    return fixes


def fix_notebook(nb_path):
    """Corrige un notebook completo."""
    print(f"\n{'='*70}")
    print(f"Procesando: {nb_path}")
    print(f"{'='*70}")

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    total_uncommented = 0
    total_indent_fixes = 0

    # Paso 1: Descomentar celdas completamente comentadas con magics
    for i, cell in enumerate(nb['cells']):
        if not is_fully_commented_cell(cell):
            continue

        source = cell['source']
        if not contains_magic_or_shell(source):
            continue

        # Descomentar la celda
        new_source = uncomment_cell(source)

        # Verificar que el descomentado produjo contenido válido
        new_joined = ''.join(new_source).strip()
        if new_joined:
            cell['source'] = new_source
            first_line = new_source[0].strip()[:60] if new_source else '???'
            print(f"  [UNCOMMENT] Celda {i}: {first_line}")
            total_uncommented += 1

    # Paso 2: Corregir errores de indentación
    total_indent_fixes = fix_indentation_errors(nb, nb_path)

    # Guardar notebook corregido
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n  Resumen para {Path(nb_path).name}:")
    print(f"    Celdas descomentadas: {total_uncommented}")
    print(f"    Errores de indentación corregidos: {total_indent_fixes}")

    return total_uncommented, total_indent_fixes


def main():
    """Función principal."""
    print("="*70)
    print("CORRECCIÓN DE CELDAS COMENTADAS EN NOTEBOOKS UNIFICADOS")
    print("="*70)

    grand_total_uncommented = 0
    grand_total_indent = 0

    for nb_path in NOTEBOOKS_TO_FIX:
        if not Path(nb_path).exists():
            print(f"\n[ADVERTENCIA] No encontrado: {nb_path}")
            continue
        uncommented, indented = fix_notebook(nb_path)
        grand_total_uncommented += uncommented
        grand_total_indent += indented

    print(f"\n{'='*70}")
    print("RESUMEN TOTAL")
    print(f"{'='*70}")
    print(f"Total celdas descomentadas: {grand_total_uncommented}")
    print(f"Total errores de indentación corregidos: {grand_total_indent}")
    print(f"Notebooks procesados: {len(NOTEBOOKS_TO_FIX)}")


if __name__ == "__main__":
    main()
