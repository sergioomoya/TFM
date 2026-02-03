import json

with open('Chapter_3_GettingStarted/Chapter_3_Unified.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '## SimulatedDataset' in source:
            print(f"Celda {idx}: SimulatedDataset")
        elif '(Transaction_data_Simulator)=' in source:
            print(f"Celda {idx}: Transaction data simulator")
            print(source[:300])
