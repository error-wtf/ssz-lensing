"""Fix Colab: Add missing imports to Gradio cell."""
import json

with open('SSZ_Lensing_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Gradio cell and prepend class definitions
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        if 'classifier = MorphologyClassifier()' in src and 'gr.Blocks' in src:
            # Add "run previous cells" check
            prefix = '''# Run cells above first! Or use: Runtime > Run all
try:
    MorphologyClassifier
except NameError:
    raise RuntimeError("Run all cells above first! Go to Runtime > Run all")

'''
            if isinstance(cell['source'], list):
                cell['source'] = [prefix] + cell['source']
            else:
                cell['source'] = prefix + cell['source']
            print("Fixed Gradio cell")
            break

with open('SSZ_Lensing_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook saved")
