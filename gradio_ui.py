"""Complete Gradio UI for RSG/SSZ Lensing Suite."""

GRADIO_UI_CODE = '''
# ============== GRADIO UI ==============
with gr.Blocks(title="RSG/SSZ Lensing Suite", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RSG / SSZ Lensing Suite")
    gr.Markdown("4 Tabs: Quicklook | Inversion | 3D Scene | Runs")
    
    # Shared states
    ql_state = gr.State(None)
    inv_state = gr.State(None)
    scene_state = gr.State(None)
    
    with gr.Row():
        # ========== LEFT: INPUT PANEL ==========
        with gr.Column(scale=1):
            gr.Markdown("## Input Panel")
            
            # Observations
            with gr.Accordion("1) Observations", open=True):
                obs_mode = gr.Radio(["QUAD (4 images)", "RING/ARC (many)"], 
                                   value="QUAD (4 images)", label="Mode")
                pos_text = gr.Textbox(value=QUAD_EXAMPLE, lines=6, 
                                     label="Image positions (x, y per line)")
                pos_unit = gr.Dropdown(["arcsec", "mas", "uas", "rad"], 
                                      value="arcsec", label="Position unit")
                with gr.Row():
                    center_known = gr.Checkbox(False, label="Center known?")
                    cx = gr.Number(0.0, label="x0")
                    cy = gr.Number(0.0, label="y0")
                    c_unit = gr.Dropdown(["arcsec", "mas"], value="arcsec", label="unit")
            
            # Distance mode
            with gr.Accordion("2) Distances (for 3D Scene)", open=False):
                dist_mode = gr.Radio(["Normalized", "Direct distances", "Redshifts"], 
                                    value="Normalized", label="Mode")
                gr.Markdown("⚠️ *Normalized: physical sizes not meaningful*")
                with gr.Group(visible=False) as direct_grp:
                    d_L = gr.Number(1.3, label="D_L")
                    d_L_u = gr.Dropdown(["Gpc","Mpc","kpc","pc"], value="Gpc")
                    d_S = gr.Number(2.1, label="D_S")
                    d_S_u = gr.Dropdown(["Gpc","Mpc","kpc","pc"], value="Gpc")
                with gr.Group(visible=False) as z_grp:
                    z_L = gr.Number(0.5, label="z_L")
                    z_S = gr.Number(2.0, label="z_S")
                    cosmo = gr.Dropdown(["Planck18","Planck15","WMAP9"], value="Planck18")
                lens_mass = gr.Number(None, label="Lens mass (M_sun, optional)")
            
            # Model zoo
            with gr.Accordion("3) Model Zoo", open=False):
                m2 = gr.Checkbox(True, label="m=2")
                shear = gr.Checkbox(True, label="+ shear")
                m3 = gr.Checkbox(True, label="+ m=3")
                m4 = gr.Checkbox(True, label="+ m=4")
        
        # ========== RIGHT: OUTPUT TABS ==========
        with gr.Column(scale=3):
            with gr.Tabs():
                # TAB 1: QUICKLOOK
                with gr.Tab("Quicklook (Geometry)"):
                    btn_ql = gr.Button("Run Quicklook", variant="primary")
                    with gr.Row():
                        ql_summary = gr.Markdown(label="Summary")
                        ql_morph = gr.Markdown(label="Morphology")
                    ql_harm = gr.Markdown(label="Harmonics (diagnostic)")
                    ql_plot = gr.Plot(label="Quicklook Plot")
                
                # TAB 2: INVERSION
                with gr.Tab("Inversion (Model Zoo)"):
                    btn_inv = gr.Button("Run Inversion", variant="primary")
                    with gr.Row():
                        inv_lb = gr.Markdown(label="Leaderboard")
                        inv_det = gr.Markdown(label="Best Model")
                    inv_plot = gr.Plot(label="Inversion Plot")
                
                # TAB 3: 3D SCENE
                with gr.Tab("3D Scene (Final)"):
                    btn_scene = gr.Button("Generate 3D Scene", variant="primary")
                    scene_units = gr.Markdown(label="Units & Scales")
                    with gr.Row():
                        scene_3d = gr.Plot(label="3D View")
                        scene_side = gr.Plot(label="Side View")
                
                # TAB 4: RUNS/STORAGE
                with gr.Tab("Runs / Storage"):
                    run_name = gr.Textbox(label="Run name", value="my_run")
                    btn_save = gr.Button("Save Run", variant="secondary")
                    save_status = gr.Markdown()
                    gr.Markdown("### Saved Runs")
                    runs_list = gr.Dataframe(headers=["Timestamp","Name","Mode"])
                    btn_refresh = gr.Button("Refresh List")
    
    # Distance mode toggle
    def toggle_dist(mode):
        return gr.update(visible=mode=="Direct distances"), gr.update(visible=mode=="Redshifts")
    dist_mode.change(toggle_dist, [dist_mode], [direct_grp, z_grp])
    
    # Quicklook
    btn_ql.click(quicklook_analysis, 
                [pos_text, pos_unit, center_known, cx, cy, c_unit],
                [ql_summary, ql_morph, ql_harm, ql_plot, ql_state])
    
    # Inversion
    btn_inv.click(inversion_analysis,
                 [ql_state, m2, shear, m3, m4],
                 [inv_lb, inv_det, inv_plot, inv_state])
    
    # 3D Scene
    btn_scene.click(scene_3d_analysis,
                   [dist_mode, d_L, d_L_u, d_S, d_S_u, z_L, z_S, cosmo, lens_mass, ql_state, inv_state],
                   [scene_units, scene_3d, scene_side, scene_state])
    
    # Save run
    btn_save.click(save_run_bundle,
                  [run_name, ql_state, inv_state, scene_state, pos_text, pos_unit, dist_mode],
                  [save_status])
    
    # Refresh runs
    btn_refresh.click(list_saved_runs, [], [runs_list])

demo.launch(share=True)
'''

print("Gradio UI code ready in GRADIO_UI_CODE variable")
