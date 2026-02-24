import nbformat
nb = nbformat.read('Chapter_7_DeepLearning/Chapter_7_Unified.ipynb', as_version=4)
injected = False
for c in nb.cells:
    if c.cell_type == 'code':
        if not injected:
            c.source = 'import torch\ndevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n' + c.source
            injected = True
        break
nbformat.write(nb, 'Chapter_7_DeepLearning/Chapter_7_Unified.ipynb')
