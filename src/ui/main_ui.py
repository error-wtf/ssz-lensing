"""Main Gradio UI with Data Tab as single source of truth."""

UI_CODE = '''
import gradio as gr
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").absolute()))

from src.ui.state import empty_dataset_state, default_run_state, DatasetState
from src.ui.data_tab import (
    QUAD_EXAMPLE, RING_EXAMPLE, get_fallback_choices,
    load_fallback_btn, build_user_btn, activate_btn
)
from src.ui.consumer_tabs import run_quicklook, run_inversion, render_scene3d

def toggle_source(source):
    return gr.update(visible=source=="User input"), gr.update(visible=source!="User input")

def toggle_mode(mode):
    return QUAD_EXAMPLE if "QUAD" in mode else RING_EXAMPLE

with gr.Blocks(title="RSG/SSZ Lensing Suite", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RSG / SSZ Lensing Suite")
    gr.Markdown("**Data Tab = Single Source of Truth**")
    
    dataset_state = gr.State(empty_dataset_state())
    dataset_tmp = gr.State(empty_dataset_state())
    run_state = gr.State(default_run_state())
    
    with gr.Tabs():
        # ===== DATA TAB =====
        with gr.Tab("📊 Data"):
            gr.Markdown("## Load or Enter Dataset")
            with gr.Row():
                with gr.Column(scale=1):
                    data_source = gr.Radio(
                        ["User input", "Fallback (real)"],
                        value="Fallback (real)", label="Source"
                    )
                    with gr.Group(visible=False) as user_grp:
                        user_mode = gr.Dropdown(
                            ["QUAD (4 images)", "RING/ARC"],
                            value="QUAD (4 images)", label="Mode"
                        )
                        user_text = gr.Textbox(value=QUAD_EXAMPLE, lines=6, label="Points")
                        user_unit = gr.Dropdown(["arcsec","mas","rad"], value="arcsec")
                        with gr.Row():
                            user_center = gr.Checkbox(False, label="Center known?")
                            user_cx = gr.Number(0.0, label="x0")
                            user_cy = gr.Number(0.0, label="y0")
                        with gr.Row():
                            user_zl = gr.Number(None, label="z_lens")
                            user_zs = gr.Number(None, label="z_source")
                        btn_build = gr.Button("Build Dataset")
                    
                    with gr.Group(visible=True) as fb_grp:
                        fb_id = gr.Dropdown(choices=get_fallback_choices(), 
                                           value="Q2237+0305 (QUAD)", label="Dataset")
                        btn_load_fb = gr.Button("Load Snapshot")
                        fb_preview = gr.Textbox(label="Preview", lines=4, interactive=False)
                
                with gr.Column(scale=1):
                    report = gr.Markdown("*Load a dataset*")
                    btn_activate = gr.Button("✅ Activate Dataset", variant="primary")
                    status = gr.Markdown("*No dataset active*")
            
            data_source.change(toggle_source, [data_source], [user_grp, fb_grp])
            user_mode.change(toggle_mode, [user_mode], [user_text])
            btn_load_fb.click(load_fallback_btn, [fb_id], [fb_preview, report, dataset_tmp])
            btn_build.click(build_user_btn, 
                [user_text, user_mode, user_unit, user_center, user_cx, user_cy, user_zl, user_zs],
                [report, dataset_tmp])
        
        # ===== QUICKLOOK TAB =====
        with gr.Tab("🔍 Quicklook"):
            btn_ql = gr.Button("Run Quicklook", variant="primary", interactive=False)
            with gr.Row():
                ql_summary = gr.Markdown()
                ql_metrics = gr.Markdown()
            ql_plot = gr.Plot()
        
        # ===== INVERSION TAB =====
        with gr.Tab("🎯 Inversion"):
            with gr.Row():
                inv_m2 = gr.Checkbox(True, label="m=2")
                inv_shear = gr.Checkbox(True, label="+shear")
                inv_m3 = gr.Checkbox(True, label="+m=3")
                inv_m4 = gr.Checkbox(True, label="+m=4")
            btn_inv = gr.Button("Run Inversion", variant="primary", interactive=False)
            with gr.Row():
                inv_lb = gr.Markdown()
                inv_det = gr.Markdown()
            inv_plot = gr.Plot()
        
        # ===== 3D SCENE TAB =====
        with gr.Tab("🌌 3D Scene"):
            with gr.Row():
                scene_mode = gr.Radio(["Normalized","Direct"], value="Normalized", label="Distances")
                scene_dl = gr.Number(1.0, label="D_L")
                scene_ds = gr.Number(2.0, label="D_S")
                scene_unit = gr.Dropdown(["Gpc","Mpc"], value="Gpc")
            btn_3d = gr.Button("Render 3D", variant="primary", interactive=False)
            scene_info = gr.Markdown()
            with gr.Row():
                scene_3d = gr.Plot()
                scene_side = gr.Plot()
    
    # Activate wiring
    def do_activate(ds_tmp):
        ds, msg, ok = activate_btn(ds_tmp)
        return ds, msg, gr.update(interactive=ok), gr.update(interactive=ok), gr.update(interactive=ok)
    
    btn_activate.click(do_activate, [dataset_tmp], 
                       [dataset_state, status, btn_ql, btn_inv, btn_3d])
    
    # Consumer wiring
    btn_ql.click(run_quicklook, [dataset_state], [ql_summary, ql_metrics, ql_plot])
    btn_inv.click(run_inversion, [dataset_state, inv_m2, inv_shear, inv_m3, inv_m4],
                  [inv_lb, inv_det, inv_plot])
    btn_3d.click(render_scene3d, [dataset_state, scene_mode, scene_dl, scene_ds, scene_unit],
                 [scene_info, scene_3d, scene_side])

demo.launch(share=True)
'''

if __name__ == "__main__":
    print(UI_CODE)
