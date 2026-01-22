"""Rebuild Colab with complete 4-tab UI per specification."""
import json

# Load notebook
with open('SSZ_Lensing_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and replace the Gradio cell
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'demo.launch' in src and 'gr.Blocks' in src:
        # Build new UI code
        new_code = '''#@title RSG/SSZ Lensing Suite - 4 Tab UI
import gradio as gr
import os
from datetime import datetime
import json as json_module

classifier = MorphologyClassifier()
ring_analyzer = RingAnalyzer()

QUAD_EX = """0.740, 0.565
-0.635, 0.470
-0.480, -0.755
0.870, -0.195"""

def parse_pos(text, unit='arcsec'):
    lines = [l.strip() for l in text.strip().split('\\n') if l.strip()]
    pos = np.array([[float(x) for x in l.replace(',', ' ').split()[:2]] for l in lines])
    fac = {'arcsec': ARCSEC_TO_RAD, 'mas': MAS_TO_RAD, 'uas': MUAS_TO_RAD, 'rad': 1.0}
    return pos * fac.get(unit, ARCSEC_TO_RAD)

def quicklook_fn(pos_text, pos_unit, center_known, cx, cy, c_unit):
    try:
        pos = parse_pos(pos_text, pos_unit)
        n = len(pos)
        if n < 2: return "Need >= 2 pts", "", "", None, None
        ctr = np.array([cx, cy]) * ANGLE_UNITS.get(c_unit, ARCSEC_TO_RAD) if center_known else np.mean(pos, axis=0)
        classifier.center = ctr
        morph = classifier.classify(pos)
        ring = ring_analyzer.fit_ring(pos, initial_center=tuple(ctr))
        mode = "QUAD" if n==4 else ("DOUBLE" if n==2 else "RING/ARC")
        
        summary = f"## Summary\\n| Metric | Value |\\n|---|---|\\n| Points | {n} |\\n| Mode | {mode} |\\n| Radius | {format_angle(ring.radius)} |\\n| RMS | {format_angle(ring.rms_residual)} |"
        morph_txt = f"## Morphology: {morph.primary.value.upper()}\\n- radial_scatter={morph.radial_scatter:.4f}\\n- azimuthal_cov={morph.azimuthal_coverage:.2f}\\n\\n" + "\\n".join(f"- {n}" for n in morph.notes)
        harm_txt = f"## Harmonics (DIAGNOSTIC)\\n⚠️ Pattern descriptors, not lens params\\n- m2: {ring.m2_component[0]:.6f}\\n- m4: {ring.m4_component[0]:.6f}"
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        p = pos/ARCSEC_TO_RAD
        t = np.linspace(0, 2*np.pi, 100)
        r = ring.radius/ARCSEC_TO_RAD
        c = ctr/ARCSEC_TO_RAD
        ax[0].plot(c[0]+r*np.cos(t), c[1]+r*np.sin(t), 'b--', lw=2, label=f'r={r:.3f}"')
        ax[0].scatter(p[:,0], p[:,1], c='red', s=100, zorder=5)
        for j, pt in enumerate(p): ax[0].annotate(chr(65+j), (pt[0]+0.02, pt[1]+0.02), fontweight='bold')
        ax[0].set_aspect('equal'); ax[0].set_title(f'Quicklook: {mode}'); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].scatter(np.degrees(ring.azimuthal_angles), ring.radial_residuals/ARCSEC_TO_RAD*1000)
        ax[1].axhline(0, color='gray', ls='--'); ax[1].set_title('Residuals (DIAGNOSTIC)'); ax[1].grid(alpha=0.3)
        plt.tight_layout()
        return summary, morph_txt, harm_txt, fig, {'pos': pos, 'ring': ring, 'morph': morph, 'mode': mode, 'n': n}
    except Exception as e:
        import traceback
        return str(e), traceback.format_exc(), "", None, None

def inversion_fn(ql_state, m2, shear, m3, m4):
    if ql_state is None: return "Run Quicklook first", "", None, None
    try:
        pos = ql_state['pos']
        models = []
        if m2: models.append('m2')
        if shear: models.append('m2_shear')
        if m3: models.append('m2_m3')
        if m4: models.append('m2_m4')
        if not models: return "Select models", "", None, None
        
        results = run_model_zoo(pos, models)
        if not results: return "No results", "", None, None
        best = results[0]
        
        # Regime gate
        A, b, names = build_system(pos, best.model_name)
        try:
            _, s, _ = np.linalg.svd(A, full_matrices=False)
            rank = int(np.sum(s > max(A.shape)*np.finfo(float).eps*s[0]))
            cond = s[0]/s[-1] if s[-1]>1e-15 else float('inf')
        except: rank, cond = A.shape[1], 1.0
        nullspace = A.shape[1] - rank
        
        lb = "## Leaderboard\\n| Model | Residual | Exact |\\n|---|---|---|\\n"
        for r in results:
            lb += f"| {r.model_name} | {format_angle(r.max_residual)} | {'Y' if r.is_exact else 'N'} |\\n"
        lb += f"\\n### Regime\\n- Constraints: {A.shape[0]}, Params: {A.shape[1]}, Rank: {rank}, Nullspace: {nullspace}"
        if nullspace > 0: lb += "\\n⚠️ **Underdetermined** - add flux/time-delay"
        
        det = f"## Best: {best.model_name}\\n| Param | Value |\\n|---|---|\\n"
        for k,v in best.params.items():
            det += f"| {k} | {format_angle(v) if k in ['theta_E','beta_x','beta_y'] else f'{v:.6f}'} |\\n"
        cons = best.source_consistency
        det += f"\\n### β Consistency\\n- Scatter: {format_angle(cons.beta_scatter)}\\n- Consistent: {'✓' if cons.is_consistent else '✗'}"
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        p = pos/ARCSEC_TO_RAD
        ax[0].scatter(p[:,0], p[:,1], c='red', s=100); ax[0].set_aspect('equal'); ax[0].set_title('Image Plane')
        t = np.linspace(0, 2*np.pi, 100)
        theta_E = best.params.get('theta_E', 0.1)/ARCSEC_TO_RAD
        ax[0].plot(theta_E*np.cos(t), theta_E*np.sin(t), 'b--', label=f'θ_E')
        ax[0].legend(); ax[0].grid(alpha=0.3)
        
        beta = cons.beta_positions/ARCSEC_TO_RAD
        bm = cons.beta_mean/ARCSEC_TO_RAD
        ax[1].scatter(beta[:,0], beta[:,1], c='blue', s=100)
        ax[1].scatter([bm[0]], [bm[1]], c='red', s=150, marker='*', label='Mean β')
        ax[1].set_aspect('equal'); ax[1].set_title(f'Source: scatter={format_angle(cons.beta_scatter)}')
        ax[1].legend(); ax[1].grid(alpha=0.3)
        plt.tight_layout()
        
        return lb, det, fig, {'results': results, 'best': best, 'pos': pos}
    except Exception as e:
        import traceback
        return str(e), traceback.format_exc(), None, None

def scene_fn(dist_mode, d_L, d_L_u, d_S, d_S_u, z_L, z_S, cosmo_name, mass, ql_state, inv_state):
    try:
        # Get angles
        if inv_state:
            theta_E = inv_state['best'].params.get('theta_E', 1.0*ARCSEC_TO_RAD)
            beta_x = inv_state['best'].params.get('beta_x', 0.0)
            beta_y = inv_state['best'].params.get('beta_y', 0.0)
            positions = inv_state['pos']
        elif ql_state:
            theta_E = ql_state['ring'].radius
            beta_x, beta_y = 0.0, 0.0
            positions = ql_state['pos']
        else:
            theta_E = 1.0*ARCSEC_TO_RAD
            beta_x, beta_y = 0.1*ARCSEC_TO_RAD, 0.05*ARCSEC_TO_RAD
            positions = None
        beta_mag = np.sqrt(beta_x**2 + beta_y**2)
        
        # Distances
        normalized = dist_mode == "Normalized"
        if normalized:
            D_L_m, D_S_m = 1.0, 2.0
            mode_str = "**NORMALIZED** (sizes not physical)"
        elif dist_mode == "Direct distances":
            D_L_m = d_L * DISTANCE_UNITS[d_L_u]
            D_S_m = d_S * DISTANCE_UNITS[d_S_u]
            mode_str = f"Direct: D_L={d_L} {d_L_u}, D_S={d_S} {d_S_u}"
        else:
            cosmo = COSMOLOGIES[cosmo_name]
            D_L_m, D_S_m, _ = lensing_distances(z_L, z_S, cosmo)
            mode_str = f"z_L={z_L}, z_S={z_S} ({cosmo_name})"
        
        D_LS_m = D_S_m - D_L_m
        R_E = D_L_m * theta_E
        R_beta = D_S_m * beta_mag
        
        # Format
        if normalized:
            fmt = lambda x, n: f"{x:.4g} (norm)"
        else:
            fmt = lambda x, n: format_distance(x).display_string if n=='d' else format_radius(x).display_string
        
        units = f"""## Units & Scales
**Mode:** {mode_str}

### Distances
| Qty | Value | [m] |
|---|---|---|
| D_L | {fmt(D_L_m, 'd') if not normalized else f'{D_L_m} (norm)'} | {D_L_m:.4e} |
| D_S | {fmt(D_S_m, 'd') if not normalized else f'{D_S_m} (norm)'} | {D_S_m:.4e} |
| D_LS | {fmt(D_LS_m, 'd') if not normalized else f'{D_LS_m} (norm)'} | {D_LS_m:.4e} |

### Angles
| Qty | Value | [rad] |
|---|---|---|
| θ_E | {format_angle(theta_E)} | {theta_E:.4e} |
| |β| | {format_angle(beta_mag)} | {beta_mag:.4e} |

### Radii
| Qty | Formula | Value |
|---|---|---|
| R_E | D_L×θ_E | {fmt(R_E, 'r') if not normalized else f'{R_E:.4g} (norm)'} |
| R_β | D_S×|β| | {fmt(R_beta, 'r') if not normalized else f'{R_beta:.4g} (norm)'} |
"""
        if mass and mass > 0:
            mass_kg = mass * MSUN_TO_KG
            r_s = schwarzschild_radius(mass_kg)
            r_s_s = f"{r_s/1e3:.4g} km" if r_s < 1e9 else f"{r_s/AU_TO_M:.4g} AU"
            ratio = R_E / r_s if r_s > 0 else float('inf')
            units += f"\\n### Lens\\n| M | {mass:.4g} M_sun |\\n| r_s | {r_s_s} |\\n| R_E/r_s | {ratio:.4g} |"
        
        scene = Scene3D(D_L=D_L_m, D_S=D_S_m, theta_E=theta_E, beta=(beta_x, beta_y), units='norm' if normalized else 'm')
        fig_3d = plot_scene_3d(scene, positions)
        fig_side = plot_scene_side(scene, positions)
        
        return units, fig_3d, fig_side, {'scene': scene, 'D_L': D_L_m, 'D_S': D_S_m, 'theta_E': theta_E, 'normalized': normalized}
    except Exception as e:
        import traceback
        return str(e) + traceback.format_exc(), None, None, None

def save_run(name, ql_state, inv_state, scene_state, pos_text, pos_unit, dist_mode):
    if ql_state is None: return "Run Quicklook first"
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = name.replace(' ', '_') if name else 'unnamed'
        run_dir = f"runs/{ts}_{safe}"
        os.makedirs(f"{run_dir}/solutions", exist_ok=True)
        os.makedirs(f"{run_dir}/figures", exist_ok=True)
        
        # input_snapshot.json
        snap = {'timestamp': ts, 'name': name, 'raw_positions': pos_text, 'position_unit': pos_unit,
                'distance_mode': dist_mode, 'n_points': ql_state['n'], 'mode': ql_state['mode']}
        with open(f"{run_dir}/input_snapshot.json", 'w') as f: json_module.dump(snap, f, indent=2)
        
        # quicklook.json
        ql = {'radius_rad': float(ql_state['ring'].radius), 'rms_rad': float(ql_state['ring'].rms_residual),
              'morphology': ql_state['morph'].primary.value, 'mode': ql_state['mode']}
        with open(f"{run_dir}/quicklook.json", 'w') as f: json_module.dump(ql, f, indent=2)
        
        # solutions/<model>.json
        if inv_state:
            for r in inv_state['results']:
                sol = {'model': r.model_name, 'params': {k: float(v) for k,v in r.params.items()},
                       'max_residual': float(r.max_residual), 'is_exact': r.is_exact, 'regime': r.regime}
                with open(f"{run_dir}/solutions/{r.model_name}.json", 'w') as f: json_module.dump(sol, f, indent=2)
        
        # scene3d.json
        if scene_state:
            sc = {'D_L_m': scene_state['D_L'], 'D_S_m': scene_state['D_S'], 
                  'theta_E_rad': scene_state['theta_E'], 'normalized': scene_state['normalized'],
                  'internal_units': {'angle': 'rad', 'distance': 'm', 'time': 's', 'mass': 'kg'}}
            with open(f"{run_dir}/scene3d.json", 'w') as f: json_module.dump(sc, f, indent=2)
        
        # report.md
        with open(f"{run_dir}/report.md", 'w') as f:
            f.write(f"# Run: {name}\\n\\nTimestamp: {ts}\\nMode: {ql_state['mode']}\\n")
            f.write(f"\\n## Input\\n- {ql_state['n']} points ({pos_unit})\\n- Distance: {dist_mode}\\n")
            if inv_state:
                f.write(f"\\n## Best Model: {inv_state['best'].model_name}\\n")
        
        return f"✓ Saved to: {run_dir}\\n- input_snapshot.json\\n- quicklook.json\\n- solutions/*.json\\n- scene3d.json\\n- report.md"
    except Exception as e:
        return f"Error: {e}"

def list_runs():
    runs = []
    if os.path.exists('runs'):
        for d in sorted(os.listdir('runs'), reverse=True)[:20]:
            path = f"runs/{d}/input_snapshot.json"
            if os.path.exists(path):
                with open(path) as f:
                    snap = json_module.load(f)
                runs.append([snap.get('timestamp',''), snap.get('name',''), snap.get('mode','')])
    return runs

# ============== GRADIO UI ==============
with gr.Blocks(title="RSG/SSZ Lensing Suite", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RSG / SSZ Lensing Suite")
    
    ql_state = gr.State(None)
    inv_state = gr.State(None)
    scene_state = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Input")
            with gr.Accordion("Observations", open=True):
                pos_text = gr.Textbox(value=QUAD_EX, lines=5, label="Positions (x, y)")
                pos_unit = gr.Dropdown(["arcsec", "mas", "uas", "rad"], value="arcsec", label="Unit")
                with gr.Row():
                    center_known = gr.Checkbox(False, label="Center known?")
                    cx = gr.Number(0.0, label="x0", scale=1)
                    cy = gr.Number(0.0, label="y0", scale=1)
                    c_unit = gr.Dropdown(["arcsec", "mas"], value="arcsec", scale=1)
            
            with gr.Accordion("Distances (3D)", open=False):
                dist_mode = gr.Radio(["Normalized", "Direct distances", "Redshifts"], value="Normalized")
                gr.Markdown("*⚠️ Normalized: sizes not physical*")
                with gr.Group(visible=False) as direct_grp:
                    d_L = gr.Number(1.3, label="D_L"); d_L_u = gr.Dropdown(["Gpc","Mpc","kpc"], value="Gpc")
                    d_S = gr.Number(2.1, label="D_S"); d_S_u = gr.Dropdown(["Gpc","Mpc","kpc"], value="Gpc")
                with gr.Group(visible=False) as z_grp:
                    z_L = gr.Number(0.5, label="z_L"); z_S = gr.Number(2.0, label="z_S")
                    cosmo = gr.Dropdown(["Planck18","Planck15","WMAP9"], value="Planck18")
                lens_mass = gr.Number(None, label="Lens mass (M_sun)")
            
            with gr.Accordion("Model Zoo", open=False):
                m2 = gr.Checkbox(True, label="m=2")
                shear = gr.Checkbox(True, label="+shear")
                m3 = gr.Checkbox(True, label="+m3")
                m4 = gr.Checkbox(True, label="+m4")
        
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Quicklook"):
                    btn_ql = gr.Button("Run Quicklook", variant="primary")
                    with gr.Row():
                        ql_summary = gr.Markdown()
                        ql_morph = gr.Markdown()
                    ql_harm = gr.Markdown()
                    ql_plot = gr.Plot()
                
                with gr.Tab("Inversion"):
                    btn_inv = gr.Button("Run Inversion", variant="primary")
                    with gr.Row():
                        inv_lb = gr.Markdown()
                        inv_det = gr.Markdown()
                    inv_plot = gr.Plot()
                
                with gr.Tab("3D Scene"):
                    btn_scene = gr.Button("Generate Scene", variant="primary")
                    scene_units = gr.Markdown()
                    with gr.Row():
                        scene_3d = gr.Plot()
                        scene_side = gr.Plot()
                
                with gr.Tab("Runs"):
                    run_name = gr.Textbox(label="Run name", value="my_run")
                    btn_save = gr.Button("Save Run")
                    save_status = gr.Markdown()
                    btn_refresh = gr.Button("Refresh")
                    runs_table = gr.Dataframe(headers=["Timestamp","Name","Mode"])
    
    dist_mode.change(lambda m: (gr.update(visible=m=="Direct distances"), gr.update(visible=m=="Redshifts")), 
                    [dist_mode], [direct_grp, z_grp])
    btn_ql.click(quicklook_fn, [pos_text, pos_unit, center_known, cx, cy, c_unit], 
                [ql_summary, ql_morph, ql_harm, ql_plot, ql_state])
    btn_inv.click(inversion_fn, [ql_state, m2, shear, m3, m4], [inv_lb, inv_det, inv_plot, inv_state])
    btn_scene.click(scene_fn, [dist_mode, d_L, d_L_u, d_S, d_S_u, z_L, z_S, cosmo, lens_mass, ql_state, inv_state],
                   [scene_units, scene_3d, scene_side, scene_state])
    btn_save.click(save_run, [run_name, ql_state, inv_state, scene_state, pos_text, pos_unit, dist_mode], [save_status])
    btn_refresh.click(list_runs, [], [runs_table])

demo.launch(share=True)
'''
        nb['cells'][i]['source'] = [new_code]
        print(f"Replaced Gradio cell at position {i}")
        break

with open('SSZ_Lensing_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Done! Notebook rebuilt with complete 4-tab UI.")
