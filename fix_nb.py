import json

with open('SSZ_Lensing_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 3 has class definitions, cell 12 has Gradio UI
# Merge them into cell 12

cell3_src = nb['cells'][3]['source']
if isinstance(cell3_src, list):
    cell3_src = ''.join(cell3_src)

cell12_src = nb['cells'][12]['source']
if isinstance(cell12_src, list):
    cell12_src = ''.join(cell12_src)

# Remove the try/except check from cell 12
lines = cell12_src.split('\n')
new_lines = []
skip = False
for line in lines:
    if 'Run cells above first' in line or 'try:' in line:
        skip = True
    elif skip and ('MorphologyClassifier' in line and 'except' not in line and '=' not in line):
        continue
    elif skip and 'except NameError' in line:
        continue
    elif skip and 'raise RuntimeError' in line:
        skip = False
        continue
    else:
        skip = False
        new_lines.append(line)

cell12_clean = '\n'.join(new_lines)

# Combine
combined = cell3_src + '\n\n# === GRADIO UI ===\n' + cell12_clean

nb['cells'][12]['source'] = combined

with open('SSZ_Lensing_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Fixed notebook - cell 12 now contains all class definitions')
