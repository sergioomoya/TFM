#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para unificar los notebooks de cada capítulo en un solo notebook ejecutable.
"""

import json
import os
import sys
from pathlib import Path

# Configurar codificación UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def load_notebook(notebook_path):
    """Carga un notebook desde un archivo."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(notebook, output_path):
    """Guarda un notebook en un archivo."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

def extract_cells_from_notebook(notebook_path, skip_magic_commands=True):
    """Extrae las celdas de un notebook, filtrando comandos mágicos si es necesario."""
    notebook = load_notebook(notebook_path)
    cells = []
    
    for cell in notebook.get('cells', []):
        source = ''.join(cell.get('source', []))
        
        # Saltar celdas con comandos !curl o %run
        if skip_magic_commands:
            if source.strip().startswith('!curl') or source.strip().startswith('%run'):
                continue
        
        # Comentar comandos mágicos %%capture y %time
        if skip_magic_commands:
            if source.strip().startswith('%%capture'):
                cell['source'] = ['# ' + line for line in cell['source']]
            elif source.strip().startswith('%time'):
                cell['source'] = ['# ' + line for line in cell['source']]
        
        cells.append(cell)
    
    return cells

def get_shared_functions_cells():
    """Obtiene las celdas de funciones compartidas."""
    shared_path = Path('Chapter_References/shared_functions.ipynb')
    if shared_path.exists():
        return extract_cells_from_notebook(str(shared_path), skip_magic_commands=False)
    return []

def create_unified_notebook(chapter_name, notebook_files, output_path):
    """Crea un notebook unificado combinando varios notebooks."""
    print(f"\n{'='*70}")
    print(f"Creando {output_path}")
    print(f"{'='*70}")
    
    # Estructura base del notebook
    unified_notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Celda de título
    title_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Capítulo {chapter_name.split('_')[1]}: {chapter_name.split('_', 1)[1].replace('_', ' ')}\n\n",
            "Este cuaderno unificado contiene todos los ejemplos y ejercicios del capítulo.\n",
            "Puede ejecutarse de forma independiente en Google Colab.\n\n",
            "**Nota:** Este cuaderno incluye automáticamente todas las funciones compartidas necesarias."
        ]
    }
    unified_notebook['cells'].append(title_cell)
    
    # Celda de encabezado para funciones compartidas (sección colapsable)
    shared_header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Funciones Compartidas\n\n",
            "Esta sección contiene todas las funciones compartidas necesarias para ejecutar los ejemplos.\n",
            "Puedes colapsar esta sección haciendo clic en el encabezado.\n\n",
            "**Nota:** Estas funciones se cargan automáticamente al inicio del cuaderno."
        ]
    }
    unified_notebook['cells'].append(shared_header)
    
    # Agregar funciones compartidas
    shared_cells = get_shared_functions_cells()
    if shared_cells:
        print(f"  [OK] Agregando {len(shared_cells)} celdas de funciones compartidas")
        unified_notebook['cells'].extend(shared_cells)
    
    # Celda de cierre de sección (opcional, para mejor organización)
    shared_footer = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n\n",
            "**Fin de Funciones Compartidas**"
        ]
    }
    unified_notebook['cells'].append(shared_footer)
    
    # Combinar notebooks del capítulo
    for notebook_file in notebook_files:
        notebook_path = Path(notebook_file)
        if not notebook_path.exists():
            print(f"  [WARN] Advertencia: No se encontró {notebook_file}")
            continue
        
        print(f"  [OK] Procesando {notebook_path.name}")
        cells = extract_cells_from_notebook(str(notebook_path))
        
        # Agregar separador con encabezado de sección
        separator = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"\n---\n\n## {notebook_path.stem}\n"]
        }
        unified_notebook['cells'].append(separator)
        
        # Agregar celdas
        unified_notebook['cells'].extend(cells)
    
    # Guardar notebook unificado
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    save_notebook(unified_notebook, str(output_path_obj))
    
    print(f"  [OK] Notebook unificado creado: {output_path}")
    print(f"  [OK] Total de celdas: {len(unified_notebook['cells'])}")

def main():
    """Función principal."""
    base_path = Path(__file__).parent
    
    # Definir estructura de capítulos y sus notebooks
    chapters = {
        "Chapter_3_GettingStarted": {
            "notebooks": [
                "Chapter_3_GettingStarted/SimulatedDataset.ipynb",
                "Chapter_3_GettingStarted/BaselineFeatureTransformation.ipynb",
                "Chapter_3_GettingStarted/BaselineModeling.ipynb",
                "Chapter_3_GettingStarted/Baseline_RealWorldData.ipynb"
            ],
            "output": "Chapter_3_GettingStarted/Chapter_3_Unified.ipynb"
        },
        "Chapter_4_PerformanceMetrics": {
            "notebooks": [
                "Chapter_4_PerformanceMetrics/ThresholdBased.ipynb",
                "Chapter_4_PerformanceMetrics/ThresholdFree.ipynb",
                "Chapter_4_PerformanceMetrics/TopKBased.ipynb",
                "Chapter_4_PerformanceMetrics/Assessment_RealWorldData.ipynb"
            ],
            "output": "Chapter_4_PerformanceMetrics/Chapter_4_Unified.ipynb"
        },
        "Chapter_5_ModelValidationAndSelection": {
            "notebooks": [
                "Chapter_5_ModelValidationAndSelection/ValidationStrategies.ipynb",
                "Chapter_5_ModelValidationAndSelection/ModelSelection.ipynb",
                "Chapter_5_ModelValidationAndSelection/ModelSelection_RealWorldData.ipynb"
            ],
            "output": "Chapter_5_ModelValidationAndSelection/Chapter_5_Unified.ipynb"
        },
        "Chapter_6_ImbalancedLearning": {
            "notebooks": [
                "Chapter_6_ImbalancedLearning/Resampling.ipynb",
                "Chapter_6_ImbalancedLearning/CostSensitive.ipynb",
                "Chapter_6_ImbalancedLearning/Ensembling.ipynb",
                "Chapter_6_ImbalancedLearning/Imbalanced_RealWorldData.ipynb"
            ],
            "output": "Chapter_6_ImbalancedLearning/Chapter_6_Unified.ipynb"
        },
        "Chapter_7_DeepLearning": {
            "notebooks": [
                "Chapter_7_DeepLearning/FeedForwardNeuralNetworks.ipynb",
                "Chapter_7_DeepLearning/Autoencoders.ipynb",
                "Chapter_7_DeepLearning/SequentialModeling.ipynb",
                "Chapter_7_DeepLearning/RealWorldData.ipynb"
            ],
            "output": "Chapter_7_DeepLearning/Chapter_7_Unified.ipynb"
        }
    }
    
    print("="*70)
    print("UNIFICACIÓN DE NOTEBOOKS")
    print("="*70)
    
    # Crear notebooks unificados
    for chapter_name, config in chapters.items():
        create_unified_notebook(
            chapter_name,
            config['notebooks'],
            config['output']
        )
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO")
    print("="*70)
    print(f"\n[OK] Se han creado {len(chapters)} notebooks unificados")

if __name__ == "__main__":
    main()
