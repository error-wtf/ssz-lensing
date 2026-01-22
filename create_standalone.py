"""Create standalone Colab notebook."""
import json

cell_code = r'''!pip install -q gradio numpy matplotlib

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

ARCSEC = np.pi / (180 * 3600)

class MorphologyClassifier:
    def __init__(self, ctr=(0,0)): self.ctr = np.array(ctr)
    def classify(self, pos):
        n, rel = len(pos), pos - self.ctr
        r = np.sqrt(rel[:,0]**2 + rel[:,1]**2)
        r_m = np.mean(r)
        r_s = np.std(r)/r_m if r_m>0 else 1
        if n==4: return 'QUAD', 0.9, r_m, ['m2','m2+shear']
        elif n==2: return 'DOUBLE', 0.9, r_m, ['m2']
        elif r_s<0.15: return 'RING', 0.85, r_m, ['iso','m2']
        else: return 'ARC', 0.7, r_m, ['m2+shear']

class RingAnalyzer:
    def fit(self, pos, ctr=None):
        cx, cy = ctr if ctr else tuple(np.mean(pos, axis=0))
        rel = pos - np.array([cx, cy])
        r = np.sqrt(rel[:,0]**2 + rel[:,1]**2)
        phi = np.arctan2(rel[:,1], rel[:,0])
        R, res = np.mean(r), r - np.mean(r)
        m2 = np.sqrt((2*np.mean(res*np.cos(2*phi)))**2 + (2*np.mean(res*np.sin(2*phi)))**2)
        m4 = np.sqrt((2*np.mean(res*np.cos(4*phi)))**2 + (2*np.mean(res*np.sin(4*phi)))**2)
        return cx, cy, R, np.sqrt(np.mean(res**2)), m2, m4

clf, ring = MorphologyClassifier(), RingAnalyzer()

def parse(txt, unit='arcsec'):
    fac = {'arcsec':ARCSEC, 'mas':ARCSEC/1000, 'rad':1}[unit]
    lines = [l.strip() for l in txt.strip().split('\n') if l.strip()]
    return np.array([[float(x) for x in l.replace(',',' ').split()[:2]] for l in lines]) * fac

def analyze(txt, unit):
    try:
        pos = parse(txt, unit)
        if len(pos) < 2: return 'Need >= 2 points', None
        ctr = np.mean(pos, axis=0)
        clf.ctr = ctr
        morph, conf, r_m, models = clf.classify(pos)
        cx, cy, R, rms, m2, m4 = ring.fit(pos, tuple(ctr))
        
        out = f"""## Results
| | |
|---|---|
| Points | {len(pos)} |
| Type | **{morph}** ({conf:.0%}) |
| theta_E | {R/ARCSEC:.4f} arcsec |
| RMS | {rms/ARCSEC:.4f} arcsec |
| m2 | {m2:.4f} |
| m4 | {m4:.4f} |

**Models:** {", ".join(models)}"""
        
        fig, ax = plt.subplots(figsize=(6,6))
        p, c = pos/ARCSEC, ctr/ARCSEC
        t = np.linspace(0, 2*np.pi, 100)
        ax.scatter(p[:,0], p[:,1], s=100, c='blue', zorder=5)
        ax.plot(c[0]+R/ARCSEC*np.cos(t), c[1]+R/ARCSEC*np.sin(t), 'g--', lw=2)
        ax.scatter([c[0]], [c[1]], s=150, c='red', marker='+', lw=3)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_xlabel('x (arcsec)'); ax.set_ylabel('y (arcsec)')
        ax.set_title(f'{morph}: theta_E = {R/ARCSEC:.3f}"')
        plt.tight_layout()
        return out, fig
    except Exception as e:
        return f'Error: {e}', None

QUAD = """0.740, 0.565
-0.635, 0.470
-0.480, -0.755
0.870, -0.195"""

with gr.Blocks(title='RSG Lensing') as demo:
    gr.Markdown('# RSG Lensing Inversion Framework')
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(value=QUAD, lines=6, label='Positions (x, y)')
            unit = gr.Dropdown(['arcsec','mas','rad'], value='arcsec')
            btn = gr.Button('Analyze', variant='primary')
        out = gr.Markdown()
    plot = gr.Plot()
    btn.click(analyze, [txt, unit], [out, plot])

demo.launch(share=True)
'''

nb = {
    'nbformat': 4, 'nbformat_minor': 0,
    'metadata': {
        'colab': {'provenance': []},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}
    },
    'cells': [
        {
            'cell_type': 'markdown',
            'source': ['# RSG Lensing - Run this ONE cell!\n\nClick the play button below.'],
            'metadata': {}
        },
        {
            'cell_type': 'code',
            'source': [cell_code],
            'metadata': {},
            'execution_count': None,
            'outputs': []
        }
    ]
}

with open('SSZ_Lensing_Standalone.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Created SSZ_Lensing_Standalone.ipynb')
