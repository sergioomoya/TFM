#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para traducir cuadernos usando OpenAI API para traducciones
más naturales y contextuales.
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("ADVERTENCIA: openai no está instalado. Instala con: pip install openai")

def get_openai_client():
    """Obtiene el cliente de OpenAI."""
    if not OPENAI_AVAILABLE:
        raise ImportError("openai no está instalado")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY no está configurada. "
            "Configúrala como variable de entorno o en el código."
        )
    
    return OpenAI(api_key=api_key)

def translate_text_with_openai(text: str, cell_type: str, client: Optional[OpenAI] = None) -> str:
    """
    Traduce texto usando OpenAI API.
    """
    if not OPENAI_AVAILABLE:
        return text
    
    if client is None:
        try:
            client = get_openai_client()
        except Exception as e:
            print(f"  [ADVERTENCIA] No se pudo inicializar OpenAI: {e}")
            return text
    
    # Preparar el prompt
    if cell_type == 'markdown':
        system_prompt = (
            "Eres un traductor profesional de inglés a español. "
            "Traduce el siguiente texto markdown preservando: "
            "- Todos los bloques de código (entre ```) "
            "- Código inline (entre `) "
            "- Enlaces markdown [texto](url) "
            "- Nombres de variables, funciones y clases "
            "- Estructura y formato markdown. "
            "Solo traduce el texto explicativo, no el código."
        )
    else:  # code
        system_prompt = (
            "Eres un traductor profesional de inglés a español. "
            "Traduce solo los comentarios en el siguiente código Python. "
            "NO traduzcas: "
            "- Nombres de variables, funciones o clases "
            "- Código Python "
            "- Strings que sean parte del código. "
            "Solo traduce los comentarios que empiezan con #."
        )
    
    user_prompt = f"Traduce al español el siguiente texto:\n\n{text}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Usar gpt-4o-mini para costos menores, cambiar a gpt-4 si se prefiere
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Baja temperatura para traducciones más consistentes
        )
        
        translated = response.choices[0].message.content.strip()
        return translated
    
    except Exception as e:
        print(f"  [ERROR en traducción] {e}")
        return text

def should_translate_cell(cell: Dict) -> bool:
    """Determina si una celda debe ser traducida."""
    cell_type = cell.get('cell_type', '')
    source = cell.get('source', [])
    
    if not source:
        return False
    
    if cell_type == 'markdown':
        # Traducir todo markdown
        full_text = ''.join(source) if isinstance(source, list) else source
        # Verificar si hay texto en inglés (simple heurística)
        if any(c.isalpha() for c in full_text):
            return True
    elif cell_type == 'code':
        # Traducir solo comentarios
        full_text = ''.join(source) if isinstance(source, list) else source
        if '#' in full_text:
            lines = full_text.split('\n')
            for line in lines:
                if '#' in line:
                    comment = line.split('#', 1)[1].strip() if '#' in line else ''
                    if comment and any(c.isalpha() for c in comment):
                        # Hay comentario con texto
                        return True
    return False

def extract_text_to_translate(cell: Dict) -> tuple:
    """
    Extrae el texto a traducir de una celda, preservando código y estructuras especiales.
    """
    source = cell.get('source', [])
    if not source:
        return '', []
    
    full_text = ''.join(source) if isinstance(source, list) else source
    cell_type = cell.get('cell_type', '')
    
    preservations = []
    text_to_translate = full_text
    
    if cell_type == 'markdown':
        # Preservar bloques de código
        code_blocks = []
        code_pattern = r'```[\s\S]*?```'
        for i, match in enumerate(re.finditer(code_pattern, text_to_translate)):
            placeholder = f'__CODE_BLOCK_{i}__'
            code_blocks.append((match.group(0), placeholder))
            text_to_translate = text_to_translate.replace(match.group(0), placeholder, 1)
        preservations.extend(code_blocks)
        
        # Preservar código inline
        inline_code = []
        inline_pattern = r'`[^`\n]+`'
        for i, match in enumerate(re.finditer(inline_pattern, text_to_translate)):
            placeholder = f'__INLINE_{i}__'
            inline_code.append((match.group(0), placeholder))
            text_to_translate = text_to_translate.replace(match.group(0), placeholder, 1)
        preservations.extend(inline_code)
        
        # Preservar enlaces (solo la URL, traducir el texto)
        # Los enlaces se traducen normalmente, solo preservamos si es necesario
        
    elif cell_type == 'code':
        # Para código, extraer solo comentarios para traducir
        lines = text_to_translate.split('\n')
        translated_lines = []
        for line in lines:
            if '#' in line and not line.strip().startswith('#'):
                # Comentario al final de línea
                code_part, comment_part = line.split('#', 1)
                if comment_part.strip():
                    # Traducir solo el comentario
                    translated_lines.append(f"{code_part}# {comment_part.strip()}")
                else:
                    translated_lines.append(line)
            elif line.strip().startswith('#'):
                # Línea de comentario completa
                translated_lines.append(line)
            else:
                # Código, no traducir
                translated_lines.append(line)
        text_to_translate = '\n'.join(translated_lines)
    
    return text_to_translate, preservations

def restore_preservations(text: str, preservations: List[tuple]) -> str:
    """Restaura los elementos preservados en el texto traducido."""
    result = text
    for original, placeholder in preservations:
        result = result.replace(placeholder, original, 1)
    return result

def translate_cell_with_llm(cell: Dict, client: Optional[OpenAI] = None) -> Dict:
    """
    Traduce una celda usando LLM, preservando código y estructuras.
    """
    if not should_translate_cell(cell):
        return cell
    
    text_to_translate, preservations = extract_text_to_translate(cell)
    
    if not text_to_translate.strip():
        return cell
    
    # Traducir con LLM
    translated_text = translate_text_with_openai(
        text_to_translate, 
        cell.get('cell_type', ''),
        client
    )
    
    # Restaurar preservaciones
    final_text = restore_preservations(translated_text, preservations)
    
    # Actualizar la celda
    new_cell = cell.copy()
    lines = final_text.split('\n')
    new_cell['source'] = [line + '\n' if i < len(lines) - 1 else line 
                          for i, line in enumerate(lines)]
    
    return new_cell

def translate_notebook_with_llm(notebook_path: str, client: Optional[OpenAI] = None):
    """
    Traduce un cuaderno completo usando LLM.
    """
    print(f"  Leyendo cuaderno: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells = notebook.get('cells', [])
    total_cells = len(cells)
    translated_count = 0
    
    print(f"  Total de celdas: {total_cells}")
    
    translated_cells = []
    for i, cell in enumerate(cells):
        if should_translate_cell(cell):
            try:
                translated_cell = translate_cell_with_llm(cell, client)
                translated_cells.append(translated_cell)
                translated_count += 1
                if translated_count % 5 == 0:
                    print(f"  Procesadas {translated_count} celdas traducibles...")
            except Exception as e:
                print(f"  [ERROR en celda {i}] {e}")
                translated_cells.append(cell)  # Mantener original si falla
        else:
            translated_cells.append(cell)
    
    notebook['cells'] = translated_cells
    
    print(f"  Celdas traducidas: {translated_count}/{total_cells}")
    
    # Guardar
    print(f"  Guardando cuaderno traducido...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print(f"  ✓ Completado: {notebook_path}")

def main():
    """Función principal."""
    base_path = Path(__file__).parent
    
    notebooks = [
        "Chapter_3_GettingStarted/Chapter_3_Unified.ipynb",
        "Chapter_4_PerformanceMetrics/Chapter_4_Unified.ipynb",
        "Chapter_5_ModelValidationAndSelection/Chapter_5_Unified.ipynb",
        "Chapter_6_ImbalancedLearning/Chapter_6_Unified.ipynb",
        "Chapter_7_DeepLearning/Chapter_7_Unified.ipynb",
    ]
    
    print("="*70)
    print("TRADUCCIÓN CON OPENAI API")
    print("="*70)
    
    # Inicializar cliente OpenAI una vez
    client = None
    if OPENAI_AVAILABLE:
        try:
            client = get_openai_client()
            print("✓ Cliente OpenAI inicializado correctamente\n")
        except Exception as e:
            print(f"✗ Error al inicializar OpenAI: {e}\n")
            print("Configura OPENAI_API_KEY como variable de entorno o en el código.\n")
            return
    else:
        print("✗ openai no está instalado. Instala con: pip install openai\n")
        return
    
    for notebook_path in notebooks:
        full_path = base_path / notebook_path
        if not full_path.exists():
            print(f"\n[ADVERTENCIA] No se encontró {notebook_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Procesando: {notebook_path}")
        print('='*70)
        
        try:
            translate_notebook_with_llm(str(full_path), client)
        except Exception as e:
            print(f"[ERROR] Error al procesar {notebook_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO")
    print("="*70)

if __name__ == "__main__":
    main()
