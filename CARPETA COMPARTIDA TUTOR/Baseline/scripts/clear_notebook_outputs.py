#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Limpia las salidas de los notebooks para reducir tamaño y uso de RAM."""
import sys
from pathlib import Path

try:
    import nbformat
except ImportError:
    print("Instala nbformat: pip install nbformat")
    sys.exit(1)

def clear_outputs(notebook_path: Path) -> bool:
    nb = nbformat.read(notebook_path, as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.outputs:
            cell.outputs = []
            cell.execution_count = None
            changed = True
    if changed:
        nbformat.write(nb, notebook_path)
    return changed

if __name__ == "__main__":
    base = Path(__file__).parent.parent
    notebooks = [
        base / "Chapter_7_DeepLearning" / "Chapter_7_Unified.ipynb",
    ]
    for p in notebooks:
        if p.exists():
            if clear_outputs(p):
                print(f"Outputs borrados: {p}")
            else:
                print(f"Sin outputs: {p}")
        else:
            print(f"No encontrado: {p}")
