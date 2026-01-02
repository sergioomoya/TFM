import os
import yaml
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def read_file_content(filepath):
    """
    Lee un archivo .md o .ipynb y devuelve una lista de celdas nbformat.
    """
    if not os.path.exists(filepath):
        # Intentamos añadir extensiones si faltan
        if os.path.exists(filepath + '.md'):
            filepath += '.md'
        elif os.path.exists(filepath + '.ipynb'):
            filepath += '.ipynb'
        else:
            print(f"ADVERTENCIA: No se encontró el archivo {filepath}")
            return []

    ext = os.path.splitext(filepath)[1]
    
    if ext == '.ipynb':
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        return nb.cells
    
    elif ext in ['.md', '.txt']:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        # Convertimos el contenido markdown en una celda Markdown
        return [new_markdown_cell(text)]
    
    return []

def main():
    # Nombre del archivo de salida
    OUTPUT_FILENAME = "Fraud_Detection_Handbook_Completo.ipynb"
    
    # Cargar la tabla de contenidos
    if not os.path.exists('_toc.yml'):
        print("Error: No se encontró _toc.yml en el directorio actual.")
        return

    with open('_toc.yml', 'r', encoding='utf-8') as f:
        toc = yaml.safe_load(f)

    # Crear el notebook maestro
    master_nb = new_notebook()
    # Metadatos básicos (opcional, copia del kernel de python)
    master_nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    }

    print("Iniciando unificación...")

    # 1. Procesar el archivo raíz (Foreword)
    if 'root' in toc:
        print(f"Procesando Root: {toc['root']}")
        cells = read_file_content(toc['root'])
        master_nb.cells.extend(cells)

    # 2. Procesar las partes y capítulos
    if 'parts' in toc:
        for part in toc['parts']:
            part_caption = part.get('caption', 'Sin Título')
            print(f"--- Procesando Parte: {part_caption} ---")
            
            # Añadir un encabezado para la Parte
            master_nb.cells.append(new_markdown_cell(f"# {part_caption}"))
            
            if 'chapters' in part:
                for chapter in part['chapters']:
                    file_path = chapter.get('file')
                    if file_path:
                        print(f"   Agregando capítulo: {file_path}")
                        cells = read_file_content(file_path)
                        master_nb.cells.extend(cells)

    # Guardar el archivo final
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        nbformat.write(master_nb, f)

    print(f"\n¡Éxito! El libro completo se ha guardado como: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()