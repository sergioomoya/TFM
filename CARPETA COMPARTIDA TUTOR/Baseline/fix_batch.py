import nbformat
import re
nb = nbformat.read('Chapter_7_DeepLearning/Chapter_7_Unified.ipynb', as_version=4)

changed = False
for c in nb.cells:
    if c.cell_type == 'code':
        orig = c.source
        c.source = re.sub(r"'batch_size':\s*64", "'batch_size': 4096", c.source)
        c.source = re.sub(r"'batch_size':\s*128", "'batch_size': 4096", c.source)
        c.source = re.sub(r"'batch_size':\s*256", "'batch_size': 4096", c.source)
        c.source = re.sub(r'batch_size\s*=\s*64', 'batch_size=4096', c.source)
        c.source = re.sub(r'batch_size\s*=\s*128', 'batch_size=4096', c.source)
        
        if c.source != orig:
            changed = True
            print('Changed a batch size!')
             
if changed:
    nbformat.write(nb, 'Chapter_7_DeepLearning/Chapter_7_Unified.ipynb')
    print('Notebook updated!')
else:
    print('No changes needed.')
