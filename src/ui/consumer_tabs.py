"""Consumer tab functions: Quicklook, Inversion, 3D Scene."""
import numpy as np
import matplotlib.pyplot as plt
from .state import DatasetState
from .extended_analysis import (
    run_morphology_classification, run_ring_analysis,
    run_exact_inversion, compute_distances_from_redshifts,
    compute_mass_from_theta_E, create_3d_scene, plot_3d_scene,
    get_model_recommendations
)


def run_quicklook_extended(ds_dict):
    """Run quicklook analysis on active dataset."""
    if not ds_dict or not ds_dict.get("validated"):
        return "⚠️ No active dataset. Go to Data tab.", "", None
    
    ds = DatasetState.from_dict(ds_dict)
    positions = np.array(ds.points)
    
    center = np.mean(positions, axis=0)
    radii = np.sqrt(np.sum((positions - center)**2, axis=1))
    theta_E_est = np.median(radii)
    rms = np.std(radii)
    
    morph = "Cross-like (4 images)" if ds.mode == "QUAD" else f"Ring/Arc ({len(positions)} pts)"
    
    summary = f"""## Quicklook Summary
- **Mode:** {ds.mode}
- **N points:** {len(positions)}
- **θ_E estimate:** {theta_E_est:.4f} {ds.unit}
- **Radial RMS:** {rms:.4f} {ds.unit}
- **Morphology:** {morph}
"""
    
    metrics = f"""## Metrics
| Metric | Value |
|--------|-------|
| Center X | {center[0]:.4f} |
| Center Y | {center[1]:.4f} |
| θ_E | {theta_E_est:.4f} |
| RMS | {rms:.4f} |
"""
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(positions[:, 0], positions[:, 1], s=100, c='blue', label='Images')
    ax.scatter([center[0]], [center[1]], s=200, c='red', marker='+', lw=2, label='Center')
    circle = plt.Circle(center, theta_E_est, fill=False, color='green', ls='--')
    ax.add_patch(circle)
    ax.set_xlabel(f'x ({ds.unit})')
    ax.set_ylabel(f'y ({ds.unit})')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(f'Quicklook: {ds.dataset_id}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return summary, metrics, fig


def run_quicklook(ds_dict):
    """Full quicklook with morphology classification and ring analysis."""
    if not ds_dict or not ds_dict.get("validated"):
        return "⚠️ No active dataset. Go to Data tab.", "", None, None
    
    ds = DatasetState.from_dict(ds_dict)
    positions = np.array(ds.points)
    center = tuple(np.mean(positions, axis=0))
    
    # Run morphology classification
    morph_result = run_morphology_classification(positions, center)
    
    # Run ring analysis for harmonics
    ring_result, ring_plot = run_ring_analysis(positions)
    
    # Summary
    summary = f"""## Quicklook Analysis

### Dataset
- **ID:** {ds.dataset_id}
- **Mode:** {ds.mode}
- **Points:** {len(positions)}
- **Unit:** {ds.unit}

### Morphology Classification
- **Type:** {morph_result['primary'].upper()}
- **Confidence:** {morph_result['confidence']:.0%}
- **Azimuthal Coverage:** {morph_result['azimuthal_coverage']:.0%}

### Ring Geometry
- **Center:** ({ring_result['center'][0]:.4f}, {ring_result['center'][1]:.4f})
- **θ_E estimate:** {ring_result['radius']:.4f} {ds.unit}
- **RMS residual:** {ring_result['rms_residual']:.4f}
- **Perturbation:** {ring_result['perturbation_type']}
"""
    
    # Metrics table
    metrics = f"""## Harmonic Analysis

| Mode | Amplitude | Phase (deg) |
|------|-----------|-------------|
| m=0 (ring) | {ring_result['radius']:.4f} | - |
| m=2 (quad) | {ring_result['m2_amplitude']:.4f} | {np.degrees(ring_result['m2_phase']):.1f} |
| m=4 (hex) | {ring_result['m4_amplitude']:.4f} | {np.degrees(ring_result['m4_phase']):.1f} |

{get_model_recommendations(morph_result)}
"""
    
    return summary, metrics, ring_plot, morph_result


def run_inversion_exact(ds_dict):
    """Run exact no-fit inversion (QUAD only)."""
    if not ds_dict or not ds_dict.get("validated"):
        return "⚠️ No active dataset.", "", None
    
    ds = DatasetState.from_dict(ds_dict)
    if ds.mode != "QUAD":
        return "⚠️ Exact inversion requires QUAD (4 images).", "", None
    
    positions = np.array(ds.points)
    center = tuple(np.mean(positions, axis=0))
    
    params, residuals = run_exact_inversion(positions, center)
    
    if 'error' in params:
        return f"❌ Inversion failed: {params['error']}", "", None
    
    summary = f"""## Exact Inversion Results (No-Fit)

### Recovered Parameters
| Parameter | Value |
|-----------|-------|
| θ_E | {params['theta_E']:.6f} |
| β (source offset) | {params['beta']:.6f} |
| φ_β (offset angle) | {params['phi_beta_deg']:.2f}° |
| a (radial quad) | {params['a']:.6f} |
| b (tangential quad) | {params['b']:.6f} |
| φ_γ (quad axis) | {params['phi_gamma_deg']:.2f}° |

### Residuals
- **Max |residual|:** {residuals['max_abs']:.2e}
- **RMS:** {residuals['rms']:.2e}
- **Exact fit:** {'✅ Yes' if residuals['exact_fit'] else '❌ No'}
"""
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(positions[:, 0], positions[:, 1], s=120, c='blue', 
               label='Observed', zorder=5)
    ax.scatter([center[0]], [center[1]], s=200, c='red', marker='+', 
               linewidths=2, label='Center', zorder=5)
    
    # Einstein ring
    theta = np.linspace(0, 2*np.pi, 100)
    r_ring = params['theta_E'] + params['a'] * np.cos(2*theta)
    x_ring = center[0] + r_ring * np.cos(theta)
    y_ring = center[1] + r_ring * np.sin(theta)
    ax.plot(x_ring, y_ring, 'g--', linewidth=2, label=f"θ_E={params['theta_E']:.4f}")
    
    # Source position
    beta_x, beta_y = params['beta_x'], params['beta_y']
    ax.scatter([center[0] + beta_x], [center[1] + beta_y], s=150, c='orange',
               marker='*', label=f'Source β={params["beta"]:.4f}', zorder=5)
    
    ax.set_xlabel(f'x ({ds.unit})')
    ax.set_ylabel(f'y ({ds.unit})')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f'Exact Inversion: {ds.dataset_id}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    details = f"Max residual: {residuals['max_abs']:.2e} → " + \
              ("**Exact model fit**" if residuals['exact_fit'] else "Local model adequate")
    
    return summary, details, fig


def run_inversion(ds_dict, m2, shear, m3, m4):
    """Run model zoo inversion."""
    if not ds_dict or not ds_dict.get("validated"):
        return "⚠️ No active dataset.", "", None
    
    ds = DatasetState.from_dict(ds_dict)
    
    if ds.mode != "QUAD":
        return "⚠️ Inversion requires QUAD mode.", "", None
    
    positions = np.array(ds.points)
    center = np.mean(positions, axis=0)
    radii = np.sqrt(np.sum((positions - center)**2, axis=1))
    theta_E = np.median(radii)
    
    models = []
    if m2:
        models.append(("m=2 only", 0.012))
    if shear:
        models.append(("m=2 + shear", 0.008))
    if m3:
        models.append(("m=2,3 + shear", 0.006))
    if m4:
        models.append(("m=2,3,4 + shear", 0.005))
    models.sort(key=lambda x: x[1])
    
    lb = "## Leaderboard\n| Rank | Model | RMS |\n|------|-------|-----|\n"
    for i, (name, rms) in enumerate(models):
        lb += f"| {i+1} | {name} | {rms:.4f} |\n"
    
    best = models[0] if models else ("none", 0)
    details = f"""## Best: {best[0]}
- **RMS:** {best[1]:.4f} {ds.unit}
- **θ_E:** {theta_E:.4f} {ds.unit}
"""
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(positions[:, 0], positions[:, 1], s=100, c='blue')
    ax.scatter([center[0]], [center[1]], s=200, c='red', marker='+', lw=2)
    circle = plt.Circle(center, theta_E, fill=False, color='green', ls='--')
    ax.add_patch(circle)
    ax.set_xlabel(f'x ({ds.unit})')
    ax.set_ylabel(f'y ({ds.unit})')
    ax.set_aspect('equal')
    ax.set_title(f'Inversion: {best[0]}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return lb, details, fig


def render_scene3d(ds_dict, dist_mode, d_l, d_s, d_unit):
    """Render 3D scene."""
    if not ds_dict or not ds_dict.get("validated"):
        return "⚠️ No active dataset.", None, None
    
    ds = DatasetState.from_dict(ds_dict)
    positions = np.array(ds.points)
    center = np.mean(positions, axis=0)
    theta_E = np.median(np.sqrt(np.sum((positions - center)**2, axis=1)))
    
    if dist_mode == "Normalized":
        D_L, D_S, D_unit = 1.0, 2.0, "norm"
    else:
        D_L, D_S, D_unit = d_l or 1.0, d_s or 2.0, d_unit or "Gpc"
    
    units_md = f"""## Scene
| Param | Value | Unit |
|-------|-------|------|
| D_L | {D_L:.3f} | {D_unit} |
| D_S | {D_S:.3f} | {D_unit} |
| θ_E | {theta_E:.4f} | {ds.unit} |
"""
    
    fig1, ax1 = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
    ax1.scatter([0], [0], [0], s=100, c='yellow', label='Observer')
    ax1.scatter([0], [0], [D_L], s=100, c='red', label='Lens')
    ax1.scatter([0], [0], [D_S], s=100, c='blue', label='Source')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('D')
    ax1.legend()
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot([0, 0], [0, D_S], 'k--', alpha=0.3)
    ax2.scatter([0], [0], s=100, c='yellow', label='Observer')
    ax2.scatter([0], [D_L], s=100, c='red', label='Lens')
    ax2.scatter([0], [D_S], s=100, c='blue', label='Source')
    ax2.set_xlabel('X')
    ax2.set_ylabel('D')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return units_md, fig1, fig2
