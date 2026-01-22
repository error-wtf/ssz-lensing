"""
Geometry Plots: Matplotlib visualizations for 3D scene and projections.

Outputs:
1. 3D Scene Plot (O, L, S + optical axis + rays)
2. Lens Plane Plot (θ observed vs predicted + Einstein ring)
3. Source Plane Plot (β cluster from each image)
4. Ray Bundle Diagram (O → Lens → Source with deflection)
5. Ring Overlay Plot (best-fit circle + residuals)

Authors: Carmen N. Wrede, Lino P. Casu
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .triad_scene import TriadScene, LensPlaneSetup
from .projection import project_to_lens_plane, backproject_to_source


class GeometryPlotter:
    """Generate all geometry visualizations."""
    
    def __init__(self, scene: TriadScene, figsize: Tuple[int, int] = (10, 8)):
        self.scene = scene
        self.figsize = figsize
        self.setups = project_to_lens_plane(scene)
    
    def generate_all(self, output_dir: str,
                     images: Optional[np.ndarray] = None,
                     params: Optional[Dict[str, float]] = None,
                     predicted: Optional[np.ndarray] = None):
        """Generate all plots and save to output directory."""
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping plots")
            return
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 3D Scene
        self.plot_3d_scene(images, str(out_path / "scene_3d.png"))
        
        # 2. Lens Plane
        if images is not None:
            self.plot_lens_plane(images, predicted, 
                                 str(out_path / "lens_plane.png"))
        
        # 3. Source Plane
        if images is not None and params is not None:
            self.plot_source_plane(images, params,
                                   str(out_path / "source_plane.png"))
        
        # 4. Ray Bundle
        if images is not None:
            self.plot_ray_bundle(images, str(out_path / "ray_bundle.png"))
        
        # 5. Overview (2x2)
        if images is not None:
            self.plot_overview(images, params, predicted,
                               str(out_path / "overview.png"))
    
    def plot_3d_scene(self, images: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None):
        """
        Plot 3D scene: Observer, Lens, Source(s), optical axis.
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Observer at origin
        ax.scatter([0], [0], [0], c='blue', s=100, marker='o', label='Observer')
        
        # Lens
        L = self.scene.lens.position
        ax.scatter([L.x], [L.y], [L.z], c='red', s=150, marker='s', label='Lens')
        
        # Sources
        for src in self.scene.sources:
            S = src.position
            ax.scatter([S.x], [S.y], [S.z], c='gold', s=100, marker='*',
                       label=f'Source {src.source_id}')
        
        # Optical axis
        if self.scene.sources:
            S = self.scene.sources[0].position
            ax.plot([0, L.x, S.x], [0, L.y, S.y], [0, L.z, S.z],
                    'k--', alpha=0.5, label='Optical axis')
        
        # Image rays (if provided)
        if images is not None:
            D_L = L.z
            D_S = self.scene.sources[0].position.z if self.scene.sources else 2*D_L
            for i, img in enumerate(images):
                # Ray from observer to lens plane
                x_L = img[0] * D_L
                y_L = img[1] * D_L
                ax.plot([0, x_L], [0, y_L], [0, D_L], 
                        'g-', alpha=0.3, linewidth=1)
                # Continue to source plane (simplified)
                x_S = img[0] * D_S
                y_S = img[1] * D_S
                ax.plot([x_L, x_S], [y_L, y_S], [D_L, D_S],
                        'g--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (distance)')
        ax.set_title(f'3D Lensing Geometry: {self.scene.name}')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_lens_plane(self, images: np.ndarray,
                        predicted: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None):
        """
        Plot lens plane: observed images, predicted, Einstein ring.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Einstein ring
        theta_E = self.scene.lens.einstein_radius
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta_E * np.cos(theta), theta_E * np.sin(theta),
                'b--', alpha=0.5, label=f'Einstein ring (θ_E={theta_E:.2f})')
        
        # Observed images
        ax.scatter(images[:, 0], images[:, 1], c='red', s=100, 
                   marker='o', label='Observed', zorder=5)
        for i, img in enumerate(images):
            ax.annotate(f'{i+1}', (img[0]+0.05, img[1]+0.05), fontsize=10)
        
        # Predicted images
        if predicted is not None:
            ax.scatter(predicted[:, 0], predicted[:, 1], c='green', s=80,
                       marker='x', label='Predicted', zorder=4)
            # Residual arrows
            for obs, pred in zip(images, predicted):
                ax.annotate('', xy=pred, xytext=obs,
                            arrowprops=dict(arrowstyle='->', color='orange',
                                            alpha=0.7))
        
        # Lens center
        ax.scatter([0], [0], c='black', s=50, marker='+', label='Lens center')
        
        ax.set_xlabel('θ_x (arcsec)')
        ax.set_ylabel('θ_y (arcsec)')
        ax.set_title(f'Lens Plane: {self.scene.name}')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_source_plane(self, images: np.ndarray, params: Dict[str, float],
                          save_path: Optional[str] = None):
        """
        Plot source plane: back-projected β from each image.
        Quality indicator: tight cluster = good model.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Back-project each image to source plane
        if self.setups:
            betas = backproject_to_source(images, self.setups[0], params)
            
            ax.scatter(betas[:, 0], betas[:, 1], c='purple', s=100,
                       marker='o', label='Back-projected β')
            for i, beta in enumerate(betas):
                ax.annotate(f'{i+1}', (beta[0]+0.01, beta[1]+0.01), fontsize=10)
            
            # Mean β
            mean_beta = np.mean(betas, axis=0)
            ax.scatter([mean_beta[0]], [mean_beta[1]], c='red', s=150,
                       marker='x', label=f'Mean β ({mean_beta[0]:.3f}, {mean_beta[1]:.3f})')
            
            # Spread indicator
            spread = np.std(betas, axis=0)
            ax.add_patch(plt.Circle(mean_beta, np.mean(spread)*3, 
                                    fill=False, color='red', linestyle='--',
                                    alpha=0.5, label=f'3σ spread'))
        
        # True β if available
        if self.setups:
            true_beta = self.setups[0].beta
            ax.scatter([true_beta[0]], [true_beta[1]], c='gold', s=150,
                       marker='*', label=f'True β', zorder=10)
        
        ax.set_xlabel('β_x (arcsec)')
        ax.set_ylabel('β_y (arcsec)')
        ax.set_title(f'Source Plane Consistency: {self.scene.name}')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_ray_bundle(self, images: np.ndarray,
                        save_path: Optional[str] = None):
        """
        Plot ray bundle diagram: O → Lens → Source with deflection.
        Didactic visualization of the lensing geometry.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        D_L = self.scene.lens.position.z
        D_S = self.scene.sources[0].position.z if self.scene.sources else 2*D_L
        theta_E = self.scene.lens.einstein_radius
        
        # Planes
        ax.axvline(0, color='blue', linestyle='-', linewidth=2, 
                   label='Observer plane')
        ax.axvline(D_L, color='red', linestyle='-', linewidth=2,
                   label='Lens plane')
        ax.axvline(D_S, color='gold', linestyle='-', linewidth=2,
                   label='Source plane')
        
        # Lens indicator
        ax.fill_betweenx([-theta_E, theta_E], D_L-0.05, D_L+0.05,
                         color='red', alpha=0.3)
        
        # Rays for each image
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(images)))
        for i, (img, c) in enumerate(zip(images, colors)):
            r = np.sqrt(img[0]**2 + img[1]**2)
            
            # Incoming ray (observed angle)
            y_at_L = r * D_L
            ax.plot([0, D_L], [0, y_at_L], color=c, linewidth=2, 
                    label=f'Image {i+1}' if i < 4 else None)
            
            # Deflected ray to source
            beta_r = r - theta_E / r if r > 0.1 else 0
            y_at_S = beta_r * D_S
            ax.plot([D_L, D_S], [y_at_L, y_at_S], color=c, linewidth=2,
                    linestyle='--')
            
            # Deflection indicator
            ax.annotate('', xy=(D_L+0.1, y_at_S*D_L/D_S), xytext=(D_L, y_at_L),
                        arrowprops=dict(arrowstyle='->', color=c, alpha=0.5))
        
        # Source position
        if self.setups:
            beta = self.setups[0].beta
            r_beta = np.sqrt(beta[0]**2 + beta[1]**2)
            ax.scatter([D_S], [r_beta], c='gold', s=100, marker='*', zorder=10)
        
        ax.set_xlabel('Distance z')
        ax.set_ylabel('Radial position r')
        ax.set_title(f'Ray Bundle Diagram: {self.scene.name}')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_overview(self, images: np.ndarray,
                      params: Optional[Dict[str, float]] = None,
                      predicted: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None):
        """
        Generate 2x2 overview plot with all visualizations.
        """
        fig = plt.figure(figsize=(14, 12))
        
        # 1. 3D Scene (top-left)
        ax1 = fig.add_subplot(221, projection='3d')
        self._plot_3d_on_ax(ax1, images)
        
        # 2. Lens Plane (top-right)
        ax2 = fig.add_subplot(222)
        self._plot_lens_on_ax(ax2, images, predicted)
        
        # 3. Source Plane (bottom-left)
        ax3 = fig.add_subplot(223)
        if params:
            self._plot_source_on_ax(ax3, images, params)
        else:
            ax3.text(0.5, 0.5, 'No params', ha='center', va='center')
        
        # 4. Ray Bundle (bottom-right)
        ax4 = fig.add_subplot(224)
        self._plot_rays_on_ax(ax4, images)
        
        plt.suptitle(f'Lensing Analysis: {self.scene.name}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_3d_on_ax(self, ax, images):
        """Helper for 3D plot on given axis."""
        L = self.scene.lens.position
        ax.scatter([0], [0], [0], c='blue', s=50, marker='o')
        ax.scatter([L.x], [L.y], [L.z], c='red', s=75, marker='s')
        for src in self.scene.sources:
            S = src.position
            ax.scatter([S.x], [S.y], [S.z], c='gold', s=50, marker='*')
        ax.set_title('3D Scene')
    
    def _plot_lens_on_ax(self, ax, images, predicted):
        """Helper for lens plane plot."""
        theta_E = self.scene.lens.einstein_radius
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta_E*np.cos(theta), theta_E*np.sin(theta), 'b--', alpha=0.5)
        ax.scatter(images[:, 0], images[:, 1], c='red', s=50, marker='o')
        if predicted is not None:
            ax.scatter(predicted[:, 0], predicted[:, 1], c='green', s=40, marker='x')
        ax.set_aspect('equal')
        ax.set_title('Lens Plane')
        ax.grid(True, alpha=0.3)
    
    def _plot_source_on_ax(self, ax, images, params):
        """Helper for source plane plot."""
        if self.setups:
            betas = backproject_to_source(images, self.setups[0], params)
            ax.scatter(betas[:, 0], betas[:, 1], c='purple', s=50)
            mean_beta = np.mean(betas, axis=0)
            ax.scatter([mean_beta[0]], [mean_beta[1]], c='red', s=100, marker='x')
        ax.set_aspect('equal')
        ax.set_title('Source Plane')
        ax.grid(True, alpha=0.3)
    
    def _plot_rays_on_ax(self, ax, images):
        """Helper for ray bundle plot."""
        D_L = self.scene.lens.position.z
        D_S = self.scene.sources[0].position.z if self.scene.sources else 2*D_L
        ax.axvline(0, color='blue', linewidth=1)
        ax.axvline(D_L, color='red', linewidth=1)
        ax.axvline(D_S, color='gold', linewidth=1)
        for img in images[:4]:
            r = np.sqrt(img[0]**2 + img[1]**2)
            ax.plot([0, D_L, D_S], [0, r*D_L, r*D_S*0.5], alpha=0.5)
        ax.set_title('Ray Bundle')
        ax.grid(True, alpha=0.3)


def plot_ring_analysis(positions: np.ndarray, ring_result,
                       save_path: Optional[str] = None):
    """
    Plot ring analysis: circle overlay + radial residual vs angle.
    """
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Ring overlay
    ax1 = axes[0]
    ax1.scatter(positions[:, 0], positions[:, 1], c='blue', s=30, alpha=0.7,
                label='Arc points')
    theta = np.linspace(0, 2*np.pi, 100)
    cx, cy = ring_result.center_x, ring_result.center_y
    R = ring_result.radius
    ax1.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), 'r-', linewidth=2,
             label=f'Best-fit (R={R:.3f})')
    ax1.scatter([cx], [cy], c='red', s=100, marker='+', label='Center')
    ax1.set_aspect('equal')
    ax1.set_title('Ring Overlay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Radial residual vs angle
    ax2 = axes[1]
    phi = ring_result.azimuthal_angles
    dr = ring_result.radial_residuals
    idx = np.argsort(phi)
    ax2.scatter(np.degrees(phi[idx]), dr[idx], c='blue', s=30)
    ax2.axhline(0, color='gray', linestyle='--')
    
    # Model curves
    phi_model = np.linspace(-np.pi, np.pi, 100)
    m2_model = ring_result.m2_component[0] * np.cos(
        2*phi_model - ring_result.m2_component[1])
    m4_model = ring_result.m4_component[0] * np.cos(
        4*phi_model - ring_result.m4_component[1])
    ax2.plot(np.degrees(phi_model), m2_model, 'g-', alpha=0.7,
             label=f'm=2: {ring_result.m2_component[0]:.4f}')
    ax2.plot(np.degrees(phi_model), m4_model, 'orange', alpha=0.7,
             label=f'm=4: {ring_result.m4_component[0]:.4f}')
    
    ax2.set_xlabel('Angle φ (degrees)')
    ax2.set_ylabel('Δr (residual)')
    ax2.set_title('Radial Residual vs Angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Harmonic signature
    ax3 = axes[2]
    harmonics = ['m=2', 'm=4']
    amplitudes = [ring_result.m2_component[0], ring_result.m4_component[0]]
    colors = ['green', 'orange']
    ax3.bar(harmonics, amplitudes, color=colors, alpha=0.7)
    ax3.axhline(0.02 * R, color='red', linestyle='--', 
                label=f'2% threshold')
    ax3.set_ylabel('Amplitude')
    ax3.set_title(f'Harmonic Signature: {ring_result.perturbation_type}')
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
