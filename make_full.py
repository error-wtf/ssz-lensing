"""Build comprehensive Gradio UI - Part 1: Write code to file"""
import json

# Read the template and write a complete notebook
code = open('full_ui_code.py').read()

nb = {
    'nbformat': 4, 'nbformat_minor': 0,
    'metadata': {'colab': {'provenance': []}, 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
    'cells': [{'cell_type': 'code', 'source': [code], 'metadata': {}, 'execution_count': None, 'outputs': []}]
}

with open('SSZ_Lensing_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
print('Created notebook')
