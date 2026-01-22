"""
3D Scene Visualization for Gravitational Lensing

Two views:
A) 3D Perspective Plot - shows O, L, S with planes and rays
B) Orthographic Side View (xz-plane) - shows distances as ruler

Authors: Carmen N. Wrede, Lino P. Casu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, List, Tuple
import sys
sys.path.insert(0, 'src')

try:
    from geometry.scene3d import Scene3D, Vec3, image_positions_to_rays
except ImportError:
    from src.geometry.scene3d import Scene3D, Vec3, image_positions_to_rays


def plot_scene_3d_perspective(scene: Scene3D, 
                               image_positions: Optional[np.ndarray] = None,
                               ax: Optional[plt.Axes] = None,
                               show_planes: bool = True,
                               show_einstein_ring: bool = True,
                               show_rays: bool = True) -> plt.Figure:
    """
    3D perspective view of lensing geometry.
    
    Shows:
    - Observer (O), Lens (L), Source (S) as labeled points
    - Optical axis O-L-S
    - Lens plane (z=D_L) and Source plane (z=D_S) as transparent rectangles
    - Einstein ring on lens plane
    - Rays from observer through image positions to lens plane
    """
    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Extract coordinates
    O = np.array(scene.observer.to_list())
    L = np.array(scene.lens.to_list())
    S = np.array(scene.source.to_list())
    
    # Plot main points
    ax.scatter(*O, c='blue', s=200, marker='o', label='Observer (O)', zorder=10)
    ax.scatter(*L, c='red', s=300, marker='*', label='Lens (L)', zorder=10)
    ax.scatter(*S, c='green', s=200, marker='s', label='Source (S)', zorder=10)
    
    # Optical axis
    ax.plot([O[0], S[0]], [O[1], S[1]], [O[2], S[2]], 
            'k--', lw=1, alpha=0.5, label='Optical axis')
    
    # Distance lines with labels
    ax.plot([O[0], L[0]], [O[1], L[1]], [O[2], L[2]], 
            'b-', lw=2, alpha=0.7)
    ax.plot([L[0], S[0]], [L[1], S[1]], [L[2], S[2]], 
            'g-', lw=2, alpha=0.7)
    
    # Add distance text
    mid_OL = (O + L) / 2
    mid_LS = (L + S) / 2
    ax.text(mid_OL[0], mid_OL[1] - 0.1, mid_OL[2], 
            f'D_L={scene.D_L:.2f}', fontsize=10, color='blue')
    ax.text(mid_LS[0], mid_LS[1] - 0.1, mid_LS[2], 
            f'D_LS={scene.D_LS:.2f}', fontsize=10, color='green')
    
    # Scale for planes
    plane_size = max(scene.R_E * 3, 0.5)
    
    if show_planes:
        # Lens plane (z = D_L)
        xx, yy = np.meshgrid(
            np.linspace(-plane_size, plane_size, 2),
            np.linspace(-plane_size, plane_size, 2)
        )
        zz = np.full_like(xx, scene.D_L)
        ax.plot_surface(xx, yy, zz, alpha=0.15, color='red', 
                        label='Lens plane')
        
        # Source plane (z = D_S)
        zz_s = np.full_like(xx, scene.D_S)
        ax.plot_surface(xx, yy, zz_s, alpha=0.15, color='green')
    
    if show_einstein_ring and scene.R_E > 0:
        # Einstein ring on lens plane
        theta = np.linspace(0, 2*np.pi, 100)
        ring_x = scene.R_E * np.cos(theta)
        ring_y = scene.R_E * np.sin(theta)
        ring_z = np.full_like(ring_x, scene.D_L)
        ax.plot(ring_x, ring_y, ring_z, 'r-', lw=2, 
                label=f'Einstein ring (R_E={scene.R_E:.4f})')
    
    if show_rays and image_positions is not None:
        rays = image_positions_to_rays(scene, image_positions)
        colors = plt.cm.Set1(np.linspace(0, 1, len(rays)))
        for i, ((start, end), color) in enumerate(zip(rays, colors)):
            ax.plot([start.x, end.x], [start.y, end.y], [start.z, end.z],
                    color=color, lw=2, alpha=0.8, 
                    label=f'Ray {i+1}' if i < 4 else '')
            ax.scatter([end.x], [end.y], [end.z], c=[color], s=100, 
                       marker='x', zorder=5)
    
    # Source offset visualization
    if scene.beta_magnitude > 0:
        ax.scatter([S[0]], [S[1]], [S[2]], c='lime', s=150, marker='o',
                   edgecolors='green', linewidths=2, zorder=9)
        # Line from optical axis to source
        ax.plot([0, S[0]], [0, S[1]], [scene.D_S, scene.D_S], 
                'g:', lw=2, alpha=0.7)
        ax.text(S[0]/2, S[1]/2, scene.D_S, 
                f'R_beta={scene.R_beta:.4f}', fontsize=9, color='green')
    
    # Labels and formatting
    ax.set_xlabel(f'X [{scene.units}]', fontsize=11)
    ax.set_ylabel(f'Y [{scene.units}]', fontsize=11)
    ax.set_zlabel(f'Z (distance) [{scene.units}]', fontsize=11)
    ax.set_title('3D Lensing Geometry: Observer - Lens - Source', 
                 fontsize=13, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=9)
    
    # Set equal aspect ratio approximately
    max_range = max(scene.D_S, plane_size) * 1.1
    ax.set_xlim(-max_range/2, max_range/2)
    ax.set_ylim(-max_range/2, max_range/2)
    ax.set_zlim(0, scene.D_S * 1.1)
    
    return fig


def plot_scene_side_view(scene: Scene3D,
                         image_positions: Optional[np.ndarray] = None,
                         ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Orthographic side view (xz-plane) showing distances as ruler.
    
    This view clearly shows:
    - D_L, D_S, D_LS as horizontal distances
    - R_E as vertical extent at lens position
    - Source offset beta as vertical displacement
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig = ax.figure
    
    # Observer at origin
    ax.scatter([0], [0], c='blue', s=200, marker='o', zorder=10)
    ax.annotate('Observer (O)', (0, 0), xytext=(0, -0.15), 
                ha='center', fontsize=11, fontweight='bold', color='blue')
    
    # Lens at D_L
    ax.scatter([scene.D_L], [0], c='red', s=300, marker='*', zorder=10)
    ax.annotate('Lens (L)', (scene.D_L, 0), xytext=(scene.D_L, -0.15),
                ha='center', fontsize=11, fontweight='bold', color='red')
    
    # Source at D_S with y-offset
    source_y = scene.source.x  # x-component becomes y in side view
    ax.scatter([scene.D_S], [source_y], c='green', s=200, marker='s', zorder=10)
    ax.annotate('Source (S)', (scene.D_S, source_y), 
                xytext=(scene.D_S + 0.05, source_y + 0.1),
                fontsize=11, fontweight='bold', color='green')
    
    # Optical axis
    ax.axhline(y=0, color='gray', linestyle='--', lw=1, alpha=0.5)
    
    # Distance markers (ruler style)
    y_ruler = -0.3 * max(scene.R_E, 0.1)
    
    # D_L bracket
    ax.annotate('', xy=(scene.D_L, y_ruler), xytext=(0, y_ruler),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(scene.D_L/2, y_ruler - 0.05, f'D_L = {scene.D_L:.3f} {scene.units}',
            ha='center', fontsize=10, color='blue')
    
    # D_S bracket
    y_ruler2 = y_ruler - 0.15
    ax.annotate('', xy=(scene.D_S, y_ruler2), xytext=(0, y_ruler2),
                arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2))
    ax.text(scene.D_S/2, y_ruler2 - 0.05, f'D_S = {scene.D_S:.3f} {scene.units}',
            ha='center', fontsize=10, color='darkgreen')
    
    # D_LS bracket
    y_ruler3 = y_ruler - 0.30
    ax.annotate('', xy=(scene.D_S, y_ruler3), xytext=(scene.D_L, y_ruler3),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text((scene.D_L + scene.D_S)/2, y_ruler3 - 0.05, 
            f'D_LS = {scene.D_LS:.3f} {scene.units}',
            ha='center', fontsize=10, color='purple')
    
    # Einstein ring (shown as vertical bar at lens position)
    if scene.R_E > 0:
        ax.plot([scene.D_L, scene.D_L], [-scene.R_E, scene.R_E], 
                'r-', lw=3, alpha=0.8)
        ax.annotate(f'R_E = {scene.R_E:.4f}', 
                    (scene.D_L, scene.R_E), xytext=(scene.D_L + 0.05, scene.R_E),
                    fontsize=9, color='red')
    
    # Source offset
    if scene.beta_magnitude > 0:
        ax.plot([scene.D_S, scene.D_S], [0, source_y], 'g:', lw=2)
        ax.annotate(f'R_beta = {scene.R_beta:.4f}',
                    (scene.D_S, source_y/2), xytext=(scene.D_S + 0.05, source_y/2),
                    fontsize=9, color='green')
    
    # Rays from observer
    if image_positions is not None:
        colors = plt.cm.Set1(np.linspace(0, 1, len(image_positions)))
        for i, (theta, color) in enumerate(zip(image_positions, colors)):
            # In side view, show x-component of angle
            theta_x = theta[0]
            # Ray goes from (0,0) to (D_L, D_L*theta_x)
            x_at_lens = scene.D_L * theta_x
            ax.plot([0, scene.D_L], [0, x_at_lens], color=color, lw=2, alpha=0.7)
            ax.scatter([scene.D_L], [x_at_lens], c=[color], s=80, marker='x')
    
    # Formatting
    ax.set_xlabel(f'Distance along optical axis [{scene.units}]', fontsize=11)
    ax.set_ylabel(f'Transverse distance [{scene.units}]', fontsize=11)
    ax.set_title('Side View: Lensing Geometry with Distances', 
                 fontsize=13, fontweight='bold')
    
    # Set limits
    y_max = max(scene.R_E * 2, abs(source_y) * 1.5, 0.2)
    ax.set_xlim(-0.1 * scene.D_S, scene.D_S * 1.1)
    ax.set_ylim(y_ruler3 - 0.2, y_max)
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('auto')
    
    return fig


def plot_scene_info_panel(scene: Scene3D, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Information panel showing all computed values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    else:
        fig = ax.figure
    
    ax.axis('off')
    
    # Build info text
    info = f"""
3D LENSING SCENE
================

POSITIONS
---------
Observer (O):  (0, 0, 0)
Lens (L):      (0, 0, {scene.D_L:.4f})
Source (S):    ({scene.source.x:.4f}, {scene.source.y:.4f}, {scene.D_S:.4f})

DISTANCES [{scene.units}]
---------
D_L  (Observer-Lens):   {scene.D_L:.4f}
D_S  (Observer-Source): {scene.D_S:.4f}
D_LS (Lens-Source):     {scene.D_LS:.4f}

ANGLES [radians]
------
theta_E (Einstein):     {scene.theta_E:.6f}
                        = {np.degrees(scene.theta_E)*3600:.4f} arcsec
|beta| (Source offset): {scene.beta_magnitude:.6f}
                        = {np.degrees(scene.beta_magnitude)*3600:.4f} arcsec

PHYSICAL RADII [{scene.units}]
--------------
R_E = D_L * theta_E:    {scene.R_E:.6f}
R_beta = D_S * |beta|:  {scene.R_beta:.6f}
"""
    
    if scene.z_L is not None or scene.z_S is not None:
        info += f"""
REDSHIFTS
---------
z_L (Lens):    {scene.z_L if scene.z_L else 'N/A'}
z_S (Source):  {scene.z_S if scene.z_S else 'N/A'}
"""
    
    if scene.lens_mass is not None and scene.R_s is not None:
        info += f"""
LENS PROPERTIES
---------------
Mass:          {scene.lens_mass:.2e} M_sun
R_s:           {scene.R_s:.2e} {scene.units}
R_E / R_s:     {scene.R_E / scene.R_s:.2e}
"""
    
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return fig


def plot_complete_scene(scene: Scene3D,
                        image_positions: Optional[np.ndarray] = None,
                        save_dir: Optional[str] = None) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Generate all three scene visualizations.
    
    Returns (fig_3d, fig_side, fig_info)
    """
    fig_3d = plot_scene_3d_perspective(scene, image_positions)
    fig_side = plot_scene_side_view(scene, image_positions)
    fig_info = plot_scene_info_panel(scene)
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fig_3d.savefig(f"{save_dir}/scene3d_perspective.png", dpi=150, bbox_inches='tight')
        fig_side.savefig(f"{save_dir}/scene3d_sideview.png", dpi=150, bbox_inches='tight')
        fig_info.savefig(f"{save_dir}/scene3d_info.png", dpi=150, bbox_inches='tight')
        scene.save(f"{save_dir}/scene3d.json")
    
    return fig_3d, fig_side, fig_info


# =============================================================================
# CONVENIENCE FUNCTIONS FOR UI
# =============================================================================

def scene_from_inversion(theta_E: float, beta: Tuple[float, float],
                         D_L: float = 1.0, D_S: float = 2.0,
                         units: str = 'normalized',
                         z_L: Optional[float] = None,
                         z_S: Optional[float] = None,
                         lens_mass: Optional[float] = None) -> Scene3D:
    """Create Scene3D from inversion results."""
    return Scene3D(
        D_L=D_L, D_S=D_S, theta_E=theta_E, beta=beta,
        units=units, z_L=z_L, z_S=z_S, lens_mass=lens_mass
    )
