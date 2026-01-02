import json
import os
import glob

def scan_notebook_comments(folder_path):
    notebooks = glob.glob(os.path.join(folder_path, "*.ipynb"))
    
    for nb_path in notebooks:
        print(f"\nScanning: {os.path.basename(nb_path)}")
        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            for cell in nb['cells']:
                if cell['cell_type'] == 'code':
                    for line in cell['source']:
                        if '#' in line:
                            print(f"  {line.strip()}")
        except Exception as e:
            print(f"Error reading {nb_path}: {e}")

if __name__ == "__main__":
    folder = r"c:\Programacion\fraud-detection-handbook-main\Chapter_3_GettingStarted"
    scan_notebook_comments(folder)
