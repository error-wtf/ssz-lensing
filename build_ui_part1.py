"""Build UI Part 1: Core functions for Gradio tabs."""
import json

GRADIO_PART1 = '''#@title RSG/SSZ Lensing Suite - Core Functions
import gradio as gr
import os
from datetime import datetime

classifier = MorphologyClassifier()
ring_analyzer = RingAnalyzer()

EXAMPLES = {'quad': "0.740, 0.565\\n-0.635, 0.470\\n-0.480, -0.755\\n0.870, -0.195",
            'ring': "0.95, 0.31\\n0.59, 0.81\\n0.00, 1.00\\n-0.59, 0.81\\n-0.95, 0.31\\n-0.95, -0.31\\n-0.59, -0.81\\n0.00, -1.00\\n0.59, -0.81\\n0.95, -0.31"}

def parse_pos(text, unit='arcsec'):
    lines = [l.strip() for l in text.strip().split('\\n') if l.strip()]
    pos = np.array([[float(x) for x in l.replace(',', ' ').split()[:2]] for l in lines])
    factor = {'arcsec': ARCSEC_TO_RAD, 'mas': MAS_TO_RAD, 'uas': MUAS_TO_RAD, 'rad': 1.0}
    return pos * factor.get(unit, ARCSEC_TO_RAD)

def run_quicklook(pos_text, pos_unit, center_known, cx, cy, c_unit):
    try:
        pos = parse_pos(pos_text, pos_unit)
        n = len(pos)
        if n < 2: return "Need >= 2 points", "", "", None, None
        
        if center_known:
            ctr = np.array([cx, cy]) * ANGLE_UNITS.get(c_unit, ARCSEC_TO_RAD)
        else:
            ctr = np.mean(pos, axis=0)
        
        classifier.center = ctr
        morph = classifier.classify(pos)
        ring = ring_analyzer.fit_ring(pos, initial_center=tuple(ctr))
        mode = "QUAD" if n==4 else ("DOUBLE" if n==2 else "RING/ARC")
        
        summary = f"## Summary\\n- Points: {n}\\n- Mode: {mode}\\n- Radius: {format_angle(ring.radius)}\\n- RMS: {format_angle(ring.rms_residual)}"
        morph_txt = f"## Morphology: {morph.primary.value.upper()}\\n- radial_scatter={morph.radial_scatter:.4f}\\n- azimuthal_cov={morph.azimuthal_coverage:.2f}"
        harm_txt = f"## Harmonics (diagnostic)\\n- m2: {ring.m2_component[0]:.6f}\\n- m4: {ring.m4_component[0]:.6f}"
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        p = pos/ARCSEC_TO_RAD
        ax[0].scatter(p[:,0], p[:,1], c='red', s=100)
        t = np.linspace(0, 2*np.pi, 100)
        r = ring.radius/ARCSEC_TO_RAD
        ax[0].plot(ctr[0]/ARCSEC_TO_RAD + r*np.cos(t), ctr[1]/ARCSEC_TO_RAD + r*np.sin(t), 'b--')
        ax[0].set_aspect('equal'); ax[0].set_title(f'Quicklook: {mode}')
        ax[1].scatter(np.degrees(ring.azimuthal_angles), ring.radial_residuals/ARCSEC_TO_RAD*1000)
        ax[1].axhline(0, color='gray', ls='--'); ax[1].set_title('Residuals (diagnostic)')
        plt.tight_layout()
        
        state = {'pos': pos, 'ring': ring, 'morph': morph, 'mode': mode, 'n': n}
        return summary, morph_txt, harm_txt, fig, state
    except Exception as e:
        return str(e), "", "", None, None

def run_inversion(ql_state, m2, shear, m3, m4):
    if ql_state is None: return "Run Quicklook first", "", None, None
    try:
        models = []
        if m2: models.append('m2')
        if shear: models.append('m2_shear')
        if m3: models.append('m2_m3')
        if m4: models.append('m2_m4')
        if not models: return "Select models", "", None, None
        
        results = run_model_zoo(ql_state['pos'], models)
        if not results: return "No results", "", None, None
        best = results[0]
        
        lb = "## Leaderboard\\n| Model | Residual | Exact |\\n|---|---|---|\\n"
        for r in results:
            lb += f"| {r.model_name} | {format_angle(r.max_residual)} | {'Y' if r.is_exact else 'N'} |\\n"
        
        det = f"## Best: {best.model_name}\\n"
        for k,v in best.params.items():
            det += f"- {k}: {v:.6f}\\n"
        det += f"\\nβ scatter: {format_angle(best.source_consistency.beta_scatter)}"
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        p = ql_state['pos']/ARCSEC_TO_RAD
        ax[0].scatter(p[:,0], p[:,1], c='red', s=100); ax[0].set_title('Image Plane')
        b = best.source_consistency.beta_positions/ARCSEC_TO_RAD
        bm = best.source_consistency.beta_mean/ARCSEC_TO_RAD
        ax[1].scatter(b[:,0], b[:,1], c='blue', s=100)
        ax[1].scatter([bm[0]], [bm[1]], c='red', s=150, marker='*')
        ax[1].set_title('Source Plane (β consistency)'); ax[1].set_aspect('equal')
        plt.tight_layout()
        
        return lb, det, fig, {'results': results, 'best': best, 'pos': ql_state['pos']}
    except Exception as e:
        return str(e), "", None, None

print("Core functions loaded!")
'''

with open('SSZ_Lensing_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find last code cell before Gradio and insert
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'demo.launch' in src:
        nb['cells'].insert(i, {'cell_type': 'code', 'source': [GRADIO_PART1], 
                               'metadata': {}, 'execution_count': None, 'outputs': []})
        print(f"Inserted core functions at {i}")
        break

with open('SSZ_Lensing_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
print("Part 1 done!")
