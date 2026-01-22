"""Patch notebook to be self-contained for Colab."""
import json

with open('SSZ_Lensing_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find install cell or add one at position 1
install_code = '''#@title 1. Install Dependencies (Run First!)
!pip install gradio matplotlib numpy -q
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
from enum import Enum
import json as json_module
import os
from datetime import datetime
print("Dependencies ready! Run next cells.")
'''

# Check if install cell exists
has_install = False
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if '!pip install gradio' in src:
        has_install = True
        break

if not has_install:
    nb['cells'].insert(1, {
        'cell_type': 'code',
        'source': [install_code],
        'metadata': {},
        'execution_count': None,
        'outputs': []
    })
    print("Added install cell")

# Save
with open('SSZ_Lensing_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook patched for Colab!")
print("Structure: Install -> Modules -> Gradio UI")
