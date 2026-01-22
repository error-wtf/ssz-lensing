"""Update Colab notebook with complete unit system and 3 input modes."""
import json

with open('SSZ_Lensing_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# =============================================================================
# CELL: Complete Unit System (embedded for Colab)
# =============================================================================
units_code = '''#@title Unit System: Auto-Scaling + Cosmology
# Internal units: rad, m, s, kg
# External: auto-scaled to human-readable

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

# === CONSTANTS ===
ARCSEC_TO_RAD = np.pi / (180 * 3600)
MAS_TO_RAD = ARCSEC_TO_RAD / 1000
MUAS_TO_RAD = MAS_TO_RAD / 1000
DEG_TO_RAD = np.pi / 180

AU_TO_M = 1.495978707e11
LY_TO_M = 9.4607304725808e15
PC_TO_M = 3.0856775814913673e16
KPC_TO_M = PC_TO_M * 1e3
MPC_TO_M = PC_TO_M * 1e6
GPC_TO_M = PC_TO_M * 1e9

C_M_S = 299792458.0
G_SI = 6.67430e-11
MSUN_TO_KG = 1.98892e30

DISTANCE_UNITS = {'m':1, 'km':1e3, 'AU':AU_TO_M, 'ly':LY_TO_M, 'pc':PC_TO_M, 'kpc':KPC_TO_M, 'Mpc':MPC_TO_M, 'Gpc':GPC_TO_M}
ANGLE_UNITS = {'rad':1, 'deg':DEG_TO_RAD, 'arcsec':ARCSEC_TO_RAD, 'mas':MAS_TO_RAD, 'uas':MUAS_TO_RAD}

@dataclass
class FormattedValue:
    internal_value: float
    display_value: float
    display_unit: str
    display_string: str
    internal_unit: str
    alternatives: Dict[str, float] = None
    def __str__(self): return self.display_string
    def to_dict(self):
        d = {'value': self.internal_value, 'unit': self.internal_unit, 'display': self.display_string}
        if self.alternatives: d['alternatives'] = self.alternatives
        return d

def format_angle(rad, precision=4):
    """Auto-scale angle: < 1e-9 rad -> uas, < 1e-6 -> mas, else arcsec"""
    abs_r = abs(rad)
    if abs_r < 1e-9: unit, factor = 'uas', MUAS_TO_RAD
    elif abs_r < 1e-6: unit, factor = 'mas', MAS_TO_RAD
    else: unit, factor = 'arcsec', ARCSEC_TO_RAD
    val = rad / factor
    alts = {'rad': rad, 'arcsec': rad/ARCSEC_TO_RAD, 'mas': rad/MAS_TO_RAD, 'uas': rad/MUAS_TO_RAD}
    return FormattedValue(rad, val, unit, f"{val:.{precision}g} {unit}", 'rad', alts)

def format_distance(meters, precision=4):
    """Auto-scale distance: AU/ly/pc/kpc/Mpc/Gpc"""
    abs_m = abs(meters)
    if abs_m < 1e8: unit, factor = 'km', 1e3
    elif abs_m < 1e11: unit, factor = 'AU', AU_TO_M
    elif abs_m < 1e16: unit, factor = 'ly', LY_TO_M
    elif abs_m < 1e19: unit, factor = 'pc', PC_TO_M
    elif abs_m < 1e22: unit, factor = 'kpc', KPC_TO_M
    elif abs_m < 1e25: unit, factor = 'Mpc', MPC_TO_M
    else: unit, factor = 'Gpc', GPC_TO_M
    val = meters / factor
    alts = {u: meters/f for u, f in DISTANCE_UNITS.items()}
    return FormattedValue(meters, val, unit, f"{val:.{precision}g} {unit}", 'm', alts)

def format_radius(meters, precision=4):
    """Auto-scale radius (R_E): AU/pc/kpc"""
    abs_m = abs(meters)
    if abs_m < 1e6: unit, factor = 'km', 1e3
    elif abs_m < 1e14: unit, factor = 'AU', AU_TO_M
    elif abs_m < 1e19: unit, factor = 'pc', PC_TO_M
    else: unit, factor = 'kpc', KPC_TO_M
    val = meters / factor
    return FormattedValue(meters, val, unit, f"{val:.{precision}g} {unit}", 'm')

def schwarzschild_radius(mass_kg):
    return 2 * G_SI * mass_kg / (C_M_S ** 2)

# === COSMOLOGY ===
@dataclass
class Cosmology:
    name: str; H0: float; Omega_m: float; Omega_L: float
    @property
    def Omega_k(self): return 1.0 - self.Omega_m - self.Omega_L

PLANCK18 = Cosmology('Planck18', 67.4, 0.315, 0.685)
PLANCK15 = Cosmology('Planck15', 67.74, 0.3089, 0.6911)
WMAP9 = Cosmology('WMAP9', 69.32, 0.2865, 0.7135)
COSMOLOGIES = {'Planck18': PLANCK18, 'Planck15': PLANCK15, 'WMAP9': WMAP9}

def E_z(z, cosmo):
    return np.sqrt(cosmo.Omega_m*(1+z)**3 + cosmo.Omega_k*(1+z)**2 + cosmo.Omega_L)

def comoving_distance(z, cosmo, n=1000):
    if z <= 0: return 0.0
    H0_1s = cosmo.H0 * 1e3 / (PC_TO_M * 1e6)
    D_H = C_M_S / H0_1s
    z_arr = np.linspace(0, z, n+1)
    dz = z / n
    integrand = 1.0 / E_z(z_arr, cosmo)
    integral = (dz/3) * (integrand[0] + 4*np.sum(integrand[1:-1:2]) + 2*np.sum(integrand[2:-1:2]) + integrand[-1])
    return D_H * integral

def angular_diameter_distance(z, cosmo):
    return comoving_distance(z, cosmo) / (1 + z)

def angular_diameter_distance_z1_z2(z1, z2, cosmo):
    return (comoving_distance(z2, cosmo) - comoving_distance(z1, cosmo)) / (1 + z2)

def lensing_distances(z_L, z_S, cosmo=PLANCK18):
    """Returns D_L, D_S, D_LS in meters."""
    D_L = angular_diameter_distance(z_L, cosmo)
    D_S = angular_diameter_distance(z_S, cosmo)
    D_LS = angular_diameter_distance_z1_z2(z_L, z_S, cosmo)
    return D_L, D_S, D_LS

def einstein_radius_from_mass(mass_kg, D_L, D_S, D_LS):
    """theta_E in radians from mass."""
    return np.sqrt(4 * G_SI * mass_kg / C_M_S**2 * D_LS / (D_L * D_S))

def mass_from_einstein_radius(theta_E_rad, D_L, D_S, D_LS):
    """Mass in kg from theta_E."""
    return C_M_S**2 / (4 * G_SI) * theta_E_rad**2 * D_L * D_S / D_LS

def parse_distance_input(value, unit):
    """Parse distance input to meters."""
    if unit not in DISTANCE_UNITS:
        raise ValueError(f"Unknown unit: {unit}")
    return value * DISTANCE_UNITS[unit]

print("Unit System loaded!")
print("  format_angle(), format_distance(), format_radius()")
print("  lensing_distances(z_L, z_S, cosmo)")
print("  Cosmologies: Planck18, Planck15, WMAP9")
'''

# Find the 3D Scene Module cell and insert unit system before it
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if '3D Lensing Scene Module' in src or 'Scene3D' in src:
        nb['cells'].insert(i, {
            'cell_type': 'code',
            'source': [units_code],
            'metadata': {},
            'execution_count': None,
            'outputs': []
        })
        print(f"Inserted Unit System cell at position {i}")
        break

# =============================================================================
# UPDATE 3D Scene Tab in Gradio UI with 3 input modes
# =============================================================================
scene3d_tab_new = '''
    with gr.Tab("3D Scene (Final)"):
        gr.Markdown("""### 3D Lensing Geometry: Observer - Lens - Source

**Internal Units:** rad, m, s, kg (strict)  
**Display Units:** Auto-scaled to human-readable (arcsec/mas, pc/kpc/Mpc/Gpc)

Choose input mode:
- **Direct Distances:** Enter D_L, D_S with units
- **Redshifts:** Enter z_L, z_S + cosmology → computes distances
- **Normalized:** Dimensionless (D_L=1, D_S=2) - clearly marked
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_mode = gr.Radio(
                    ["Direct Distances", "Redshifts + Cosmology", "Normalized"],
                    value="Normalized",
                    label="Input Mode"
                )
                
                # Direct Distance Inputs
                with gr.Group(visible=True) as direct_group:
                    gr.Markdown("### Direct Distance Input")
                    direct_D_L = gr.Number(value=1.3, label="D_L value")
                    direct_D_L_unit = gr.Dropdown(["Gpc", "Mpc", "kpc", "pc", "ly", "AU"], value="Gpc", label="D_L unit")
                    direct_D_S = gr.Number(value=2.1, label="D_S value")
                    direct_D_S_unit = gr.Dropdown(["Gpc", "Mpc", "kpc", "pc", "ly", "AU"], value="Gpc", label="D_S unit")
                
                # Redshift Inputs
                with gr.Group(visible=False) as redshift_group:
                    gr.Markdown("### Redshift Input")
                    z_L_input = gr.Number(value=0.5, label="z_L (Lens redshift)")
                    z_S_input = gr.Number(value=2.0, label="z_S (Source redshift)")
                    cosmo_choice = gr.Dropdown(["Planck18", "Planck15", "WMAP9"], value="Planck18", label="Cosmology")
                
                # Optional Parameters
                gr.Markdown("### From Inversion (or manual)")
                theta_E_input = gr.Number(value=1.0, label="theta_E (arcsec)")
                beta_x_input = gr.Number(value=0.1, label="beta_x (arcsec)")
                beta_y_input = gr.Number(value=0.05, label="beta_y (arcsec)")
                
                gr.Markdown("### Optional: Lens Mass")
                lens_mass = gr.Number(value=None, label="M_lens (M_sun, optional)")
                
                btn_scene = gr.Button("Generate 3D Scene", variant="primary")
                btn_save_scene = gr.Button("Save Scene to Run")
        
        scene_info = gr.Markdown(label="Scene Information")
        
        with gr.Row():
            out_3d = gr.Plot(label="3D Perspective View")
            out_side = gr.Plot(label="Side View (Distances)")
        
        scene_state = gr.State(None)
        
        def toggle_input_groups(mode):
            return (
                gr.update(visible=(mode == "Direct Distances")),
                gr.update(visible=(mode == "Redshifts + Cosmology"))
            )
        
        input_mode.change(toggle_input_groups, [input_mode], [direct_group, redshift_group])
        
        def generate_3d_scene_full(mode, d_L_val, d_L_unit, d_S_val, d_S_unit, 
                                   z_L, z_S, cosmo_name, theta_E_arcsec, 
                                   beta_x_arcsec, beta_y_arcsec, mass_msun, inv_state):
            try:
                # === DETERMINE DISTANCES ===
                if mode == "Direct Distances":
                    D_L_m = parse_distance_input(d_L_val, d_L_unit)
                    D_S_m = parse_distance_input(d_S_val, d_S_unit)
                    input_desc = f"Direct: D_L={d_L_val} {d_L_unit}, D_S={d_S_val} {d_S_unit}"
                    normalized = False
                elif mode == "Redshifts + Cosmology":
                    cosmo = COSMOLOGIES[cosmo_name]
                    D_L_m, D_S_m, _ = lensing_distances(z_L, z_S, cosmo)
                    input_desc = f"z_L={z_L}, z_S={z_S}, {cosmo_name}"
                    normalized = False
                else:  # Normalized
                    D_L_m = 1.0
                    D_S_m = 2.0
                    input_desc = "**NORMALIZED (dimensionless)**"
                    normalized = True
                
                D_LS_m = D_S_m - D_L_m
                
                # === CONVERT ANGLES TO RADIANS ===
                theta_E_rad = theta_E_arcsec * ARCSEC_TO_RAD
                beta_x_rad = beta_x_arcsec * ARCSEC_TO_RAD
                beta_y_rad = beta_y_arcsec * ARCSEC_TO_RAD
                beta_mag_rad = np.sqrt(beta_x_rad**2 + beta_y_rad**2)
                
                # === COMPUTE PHYSICAL RADII ===
                R_E_m = D_L_m * theta_E_rad
                R_beta_m = D_S_m * beta_mag_rad
                
                # === FORMAT FOR DISPLAY ===
                if normalized:
                    D_L_fmt = f"{D_L_m:.4g} (normalized)"
                    D_S_fmt = f"{D_S_m:.4g} (normalized)"
                    D_LS_fmt = f"{D_LS_m:.4g} (normalized)"
                    R_E_fmt = f"{R_E_m:.6g} (normalized)"
                    R_beta_fmt = f"{R_beta_m:.6g} (normalized)"
                else:
                    D_L_fmt = format_distance(D_L_m).display_string
                    D_S_fmt = format_distance(D_S_m).display_string
                    D_LS_fmt = format_distance(D_LS_m).display_string
                    R_E_fmt = format_radius(R_E_m).display_string
                    R_beta_fmt = format_radius(R_beta_m).display_string
                
                theta_E_fmt = format_angle(theta_E_rad).display_string
                beta_fmt = format_angle(beta_mag_rad).display_string
                
                # === OPTIONAL: MASS & r_s ===
                mass_info = ""
                if mass_msun and mass_msun > 0:
                    mass_kg = mass_msun * MSUN_TO_KG
                    r_s_m = schwarzschild_radius(mass_kg)
                    if r_s_m < 1e9:
                        r_s_fmt = f"{r_s_m/1e3:.4g} km"
                    else:
                        r_s_fmt = f"{r_s_m/AU_TO_M:.4g} AU"
                    ratio = R_E_m / r_s_m if r_s_m > 0 else float('inf')
                    mass_info = f"""
### Lens Mass
| Quantity | Value |
|----------|-------|
| M | {mass_msun:.4g} M_sun |
| r_s | {r_s_fmt} |
| R_E / r_s | {ratio:.4g} |
"""
                
                # === BUILD INFO PANEL ===
                info = f"""## 3D Scene Summary
**Input:** {input_desc}

### Distances (internal: meters)
| Quantity | Display | Internal [m] |
|----------|---------|--------------|
| D_L (O→L) | {D_L_fmt} | {D_L_m:.6e} |
| D_S (O→S) | {D_S_fmt} | {D_S_m:.6e} |
| D_LS (L→S) | {D_LS_fmt} | {D_LS_m:.6e} |

### Angular Scales (internal: radians)
| Quantity | Display | Internal [rad] |
|----------|---------|----------------|
| θ_E | {theta_E_fmt} | {theta_E_rad:.6e} |
| |β| | {beta_fmt} | {beta_mag_rad:.6e} |

### Physical Radii (internal: meters)
| Quantity | Formula | Display | Internal [m] |
|----------|---------|---------|--------------|
| R_E | D_L × θ_E | {R_E_fmt} | {R_E_m:.6e} |
| R_β | D_S × |β| | {R_beta_fmt} | {R_beta_m:.6e} |
{mass_info}
"""
                
                # === CREATE SCENE ===
                scene = Scene3D(
                    D_L=D_L_m, D_S=D_S_m, theta_E=theta_E_rad,
                    beta=(beta_x_rad, beta_y_rad),
                    units='normalized' if normalized else 'm'
                )
                
                # Get image positions from inversion if available
                positions = None
                if inv_state is not None:
                    pos, _, _ = inv_state
                    positions = pos
                
                fig_3d = plot_scene_3d(scene, positions)
                fig_side = plot_scene_side(scene, positions)
                
                # Build state for saving
                save_data = {
                    'input_mode': mode,
                    'inputs': {'D_L_val': d_L_val, 'D_L_unit': d_L_unit, 'D_S_val': d_S_val, 'D_S_unit': d_S_unit,
                               'z_L': z_L, 'z_S': z_S, 'cosmology': cosmo_name},
                    'internal_units': {'angle': 'rad', 'distance': 'm', 'time': 's', 'mass': 'kg'},
                    'computed': {'D_L_m': D_L_m, 'D_S_m': D_S_m, 'D_LS_m': D_LS_m,
                                 'theta_E_rad': theta_E_rad, 'R_E_m': R_E_m,
                                 'beta_rad': beta_mag_rad, 'R_beta_m': R_beta_m},
                    'display': {'D_L': D_L_fmt, 'D_S': D_S_fmt, 'D_LS': D_LS_fmt,
                                'theta_E': theta_E_fmt, 'R_E': R_E_fmt, 'beta': beta_fmt, 'R_beta': R_beta_fmt}
                }
                
                return info, fig_3d, fig_side, (scene, positions, save_data)
            
            except Exception as e:
                import traceback
                return f"Error: {e}\\n{traceback.format_exc()}", None, None, None
        
        def save_scene_run(scene_state):
            if scene_state is None:
                return "No scene to save"
            scene, positions, save_data = scene_state
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = f"runs/{timestamp}_scene3d"
            os.makedirs(f"{run_dir}/figures", exist_ok=True)
            
            # Save scene3d.json with all unit info
            with open(f"{run_dir}/scene3d.json", 'w') as f:
                json_module.dump(save_data, f, indent=2)
            
            # Save figures
            fig_3d = plot_scene_3d(scene, positions)
            fig_3d.savefig(f"{run_dir}/figures/scene3d_perspective.png", dpi=150, bbox_inches='tight')
            plt.close(fig_3d)
            
            fig_side = plot_scene_side(scene, positions)
            fig_side.savefig(f"{run_dir}/figures/scene3d_sideview.png", dpi=150, bbox_inches='tight')
            plt.close(fig_side)
            
            return f"Saved to: {run_dir}\\n- scene3d.json (with unit metadata)\\n- figures/scene3d_*.png"
        
        btn_scene.click(
            generate_3d_scene_full,
            [input_mode, direct_D_L, direct_D_L_unit, direct_D_S, direct_D_S_unit,
             z_L_input, z_S_input, cosmo_choice, theta_E_input,
             beta_x_input, beta_y_input, lens_mass, inv_state],
            [scene_info, out_3d, out_side, scene_state]
        )
        btn_save_scene.click(save_scene_run, [scene_state], scene_info)
'''

# Find and replace the 3D Scene tab in the Gradio cell
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'demo.launch(share=True)' in src and 'with gr.Tab' in src:
        # Find and replace the 3D Scene tab
        if '3D Scene (Final)' in src:
            # Remove old 3D Scene tab
            start_marker = "with gr.Tab(\"3D Scene (Final)\")"
            end_marker = "demo.launch(share=True)"
            
            start_idx = src.find(start_marker)
            end_idx = src.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                # Replace the old tab with new one
                new_src = src[:start_idx] + scene3d_tab_new + "\\n\\n" + src[end_idx:]
                nb['cells'][i]['source'] = [new_src]
                print(f"Updated 3D Scene tab in Gradio cell at position {i}")
        break

# Save
with open('SSZ_Lensing_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\\nColab notebook updated with:")
print("  - Unit System module (auto-scaling)")
print("  - Cosmology (Planck18, WMAP9)")
print("  - 3 Input modes (Direct/Redshift/Normalized)")
print("  - Full unit metadata in scene3d.json")
