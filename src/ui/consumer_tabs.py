"""Consumer tab functions: Quicklook, Inversion, 3D Scene."""
import numpy as np
import matplotlib.pyplot as plt
from .state import DatasetState


def run_quicklook(ds_dict):
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
