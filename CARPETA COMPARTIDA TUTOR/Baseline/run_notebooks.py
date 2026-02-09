#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script wrapper para ejecutar notebooks con parche de lxml.html.clean
"""

# PARCHE CRÍTICO: Debe ejecutarse ANTES de cualquier import de nbconvert
import sys
import types

try:
    import lxml_html_clean
    # Crear módulos simulados para lxml.html.clean
    if 'lxml' not in sys.modules:
        lxml_module = types.ModuleType('lxml')
        sys.modules['lxml'] = lxml_module
    
    if 'lxml.html' not in sys.modules:
        lxml_html_module = types.ModuleType('lxml.html')
        sys.modules['lxml.html'] = lxml_html_module
        sys.modules['lxml'].html = lxml_html_module
    
    if 'lxml.html.clean' not in sys.modules:
        lxml_html_clean_module = types.ModuleType('lxml.html.clean')
        lxml_html_clean_module.clean_html = lxml_html_clean.clean_html
        sys.modules['lxml.html.clean'] = lxml_html_clean_module
        sys.modules['lxml.html'].clean = lxml_html_clean_module
    
    # Verificar que funciona
    import nbconvert
    print("✓ Parche de lxml.html.clean aplicado correctamente")
except Exception as e:
    print(f"✗ Error al aplicar parche: {e}")
    sys.exit(1)

# Ahora ejecutar el script principal
# Importar execute_notebooks después del parche
if __name__ == "__main__":
    # Importar el módulo execute_notebooks (el parche ya está aplicado)
    import importlib.util
    spec = importlib.util.spec_from_file_location("execute_notebooks", "execute_notebooks.py")
    execute_notebooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(execute_notebooks)
    execute_notebooks.main()
