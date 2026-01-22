"""Extended analysis functions: real implementations from existing code."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.morphology import MorphologyClassifier, Morphology
from src.models.ring_analysis import RingAnalyzer, RingFitResult
from src.gauge_lens_inversion import invert_no_fit, moment_estimate
from src.units.cosmology import (
    lensing_distances, COSMOLOGIES, einstein_radius_from_mass,
    mass_from_einstein_radius
)
from src.geometry.scene3d import Scene3D, compute_scene_summary


def run_morphology_classification(positions: np.ndarray, 
                                   center: Tuple[float, float] = (0, 0)
                                   ) -> Dict:
    """
    Run full morphology classification.
    
    Returns dict with morphology type, confidence, metrics, recommendations.
    """
    classifier = MorphologyClassifier(center=center)
    result = classifier.classify(positions)
    return result.to_dict()


def run_ring_analysis(positions: np.ndarray) -> Tuple[Dict, plt.Figure]:
    """
    Run ring fit analysis with harmonic decomposition.
    
    Returns (result_dict, diagnostic_plot).
    """
    analyzer = RingAnalyzer()
    result = analyzer.fit_ring(positions)
    diag_data = analyzer.generate_diagnostic_data(result)
    
    # Create diagnostic plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: Ring fit
    ax1 = axes[0]
    ax1.scatter(positions[:, 0], positions[:, 1], s=30, c='blue', label='Data')
    ax1.scatter([result.center_x], [result.center_y], s=100, c='red', 
                marker='+', linewidths=2, label='Center')
    circle = plt.Circle((result.center_x, result.center_y), result.radius, 
                         fill=False, color='green', linestyle='--', label='Fit')
    ax1.add_patch(circle)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.set_title(f'Ring Fit: R={result.radius:.4f}')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Radial residuals vs azimuth
    ax2 = axes[1]
    phi = np.array(diag_data['phi_data'])
    dr = np.array(diag_data['dr_data'])
    ax2.scatter(np.degrees(phi), dr, s=20, c='blue', label='Residuals')
    phi_m = np.array(diag_data['phi_model'])
    m2_m = np.array(diag_data['m2_model'])
    ax2.plot(np.degrees(phi_m), m2_m, 'r-', label=f'm=2: {result.m2_component[0]:.4f}')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('φ (deg)')
    ax2.set_ylabel('Δr')
    ax2.legend()
    ax2.set_title(f'Residuals: RMS={result.rms_residual:.4f}')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Harmonic amplitudes
    ax3 = axes[2]
    modes = ['m=0\n(ring)', 'm=2\n(quad)', 'm=4\n(hex)']
    amps = [result.radius, result.m2_component[0], result.m4_component[0]]
    colors = ['green', 'orange', 'purple']
    bars = ax3.bar(modes, amps, color=colors)
    ax3.set_ylabel('Amplitude')
    ax3.set_title(f'Harmonics: {result.perturbation_type}')
    for bar, amp in zip(bars, amps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{amp:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return result.to_dict(), fig


def run_exact_inversion(positions: np.ndarray, 
                        center: Tuple[float, float] = (0, 0)) -> Tuple[Dict, Dict]:
    """
    Run no-fit exact inversion (bisection rootfinding).
    
    Returns (params, residuals).
    """
    params, residuals, diagnostics = invert_no_fit(positions, center)
    
    if params is None:
        return {'error': diagnostics.get('error', 'Inversion failed')}, {}
    
    return params, {
        'max_abs': residuals['max_abs'],
        'rms': residuals['rms'],
        'exact_fit': residuals['max_abs'] < 1e-10
    }


def compute_distances_from_redshifts(z_L: float, z_S: float, 
                                      cosmo_name: str = 'Planck18') -> Dict:
    """
    Compute lensing distances from redshifts.
    
    Returns distances in Mpc and meters.
    """
    cosmo = COSMOLOGIES.get(cosmo_name, COSMOLOGIES['Planck18'])
    D_L, D_S, D_LS = lensing_distances(z_L, z_S, cosmo)
    
    # Convert to Mpc
    pc_to_m = 3.0856775814913673e16
    Mpc = 1e6 * pc_to_m
    
    return {
        'D_L_m': D_L,
        'D_S_m': D_S,
        'D_LS_m': D_LS,
        'D_L_Mpc': D_L / Mpc,
        'D_S_Mpc': D_S / Mpc,
        'D_LS_Mpc': D_LS / Mpc,
        'z_L': z_L,
        'z_S': z_S,
        'cosmology': cosmo_name
    }


def compute_mass_from_theta_E(theta_E_arcsec: float, D_L_Mpc: float, 
                               D_S_Mpc: float, D_LS_Mpc: float) -> Dict:
    """
    Compute lens mass from Einstein radius.
    
    Returns mass in solar masses and kg.
    """
    # Convert units
    theta_E_rad = theta_E_arcsec * np.pi / (180 * 3600)
    pc_to_m = 3.0856775814913673e16
    Mpc = 1e6 * pc_to_m
    
    D_L = D_L_Mpc * Mpc
    D_S = D_S_Mpc * Mpc
    D_LS = D_LS_Mpc * Mpc
    
    mass_kg = mass_from_einstein_radius(theta_E_rad, D_L, D_S, D_LS)
    M_sun = 1.989e30
    mass_Msun = mass_kg / M_sun
    
    return {
        'mass_kg': mass_kg,
        'mass_Msun': mass_Msun,
        'mass_1e10_Msun': mass_Msun / 1e10,
        'theta_E_arcsec': theta_E_arcsec,
        'theta_E_rad': theta_E_rad
    }


def create_3d_scene(theta_E_arcsec: float, beta_arcsec: Tuple[float, float],
                    D_L_Mpc: float, D_S_Mpc: float,
                    z_L: Optional[float] = None, z_S: Optional[float] = None,
                    mass_Msun: Optional[float] = None) -> Tuple[Dict, str]:
    """
    Create full 3D scene with all derived quantities.
    
    Returns (scene_dict, summary_text).
    """
    # Convert to radians
    arcsec_to_rad = np.pi / (180 * 3600)
    theta_E = theta_E_arcsec * arcsec_to_rad
    beta = (beta_arcsec[0] * arcsec_to_rad, beta_arcsec[1] * arcsec_to_rad)
    
    scene = Scene3D(
        D_L=D_L_Mpc,
        D_S=D_S_Mpc,
        theta_E=theta_E,
        beta=beta,
        units='Mpc',
        z_L=z_L,
        z_S=z_S,
        lens_mass=mass_Msun
    )
    
    summary = compute_scene_summary(scene)
    return scene.to_dict(), summary


def plot_3d_scene(scene_dict: Dict) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create 3D and side-view plots of the lensing scene.
    """
    D_L = scene_dict['D_L']
    D_S = scene_dict['D_S']
    R_E = scene_dict['R_E']
    source = scene_dict['source']
    
    # 3D plot
    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Observer
    ax1.scatter([0], [0], [0], s=200, c='yellow', marker='o', label='Observer')
    
    # Lens plane (Einstein ring)
    theta = np.linspace(0, 2*np.pi, 100)
    ring_x = R_E * np.cos(theta)
    ring_y = R_E * np.sin(theta)
    ring_z = np.full_like(ring_x, D_L)
    ax1.plot(ring_x, ring_y, ring_z, 'g-', linewidth=2, label=f'Einstein Ring (R_E)')
    ax1.scatter([0], [0], [D_L], s=150, c='red', marker='s', label='Lens')
    
    # Source
    ax1.scatter([source[0]], [source[1]], [D_S], s=150, c='blue', 
                marker='*', label='Source')
    
    # Optical axis
    ax1.plot([0, 0], [0, 0], [0, D_S*1.1], 'k--', alpha=0.3, label='Optical axis')
    
    # Ray to source
    ax1.plot([0, source[0]], [0, source[1]], [0, D_S], 'b--', alpha=0.5)
    
    ax1.set_xlabel('X (Mpc)')
    ax1.set_ylabel('Y (Mpc)')
    ax1.set_zlabel('D (Mpc)')
    ax1.legend(loc='upper left')
    ax1.set_title('3D Lensing Geometry')
    
    # Side view
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Observer
    ax2.scatter([0], [0], s=200, c='yellow', edgecolors='black', 
                marker='o', label='Observer', zorder=5)
    
    # Lens plane
    ax2.axhline(D_L, color='red', linestyle='-', alpha=0.3)
    ax2.scatter([0], [D_L], s=150, c='red', marker='s', label='Lens', zorder=5)
    ax2.scatter([-R_E], [D_L], s=50, c='green', marker='|', zorder=5)
    ax2.scatter([R_E], [D_L], s=50, c='green', marker='|', zorder=5)
    ax2.annotate('', xy=(R_E, D_L), xytext=(-R_E, D_L),
                 arrowprops=dict(arrowstyle='<->', color='green'))
    ax2.text(0, D_L*1.02, f'R_E = {R_E:.4f}', ha='center', fontsize=9, color='green')
    
    # Source plane
    ax2.axhline(D_S, color='blue', linestyle='-', alpha=0.3)
    r_source = np.sqrt(source[0]**2 + source[1]**2)
    ax2.scatter([r_source], [D_S], s=150, c='blue', marker='*', 
                label='Source', zorder=5)
    
    # Distances
    ax2.annotate('', xy=(D_S*0.15, D_L), xytext=(D_S*0.15, 0),
                 arrowprops=dict(arrowstyle='<->', color='gray'))
    ax2.text(D_S*0.17, D_L/2, f'D_L = {D_L:.2f}', fontsize=9, color='gray')
    
    ax2.annotate('', xy=(D_S*0.25, D_S), xytext=(D_S*0.25, 0),
                 arrowprops=dict(arrowstyle='<->', color='gray'))
    ax2.text(D_S*0.27, D_S/2, f'D_S = {D_S:.2f}', fontsize=9, color='gray')
    
    ax2.set_xlabel('Transverse (Mpc)')
    ax2.set_ylabel('Distance (Mpc)')
    ax2.legend(loc='upper right')
    ax2.set_title('Side View: Observer - Lens - Source')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-D_S*0.3, D_S*0.4)
    ax2.set_ylim(-D_S*0.05, D_S*1.1)
    
    plt.tight_layout()
    return fig1, fig2


def get_model_recommendations(morphology_result: Dict) -> str:
    """Generate model recommendations based on morphology."""
    morph = morphology_result.get('primary', 'unknown')
    models = morphology_result.get('recommended_models', [])
    m2 = morphology_result.get('m2_amplitude', 0)
    m4 = morphology_result.get('m4_amplitude', 0)
    
    lines = ["## Model Recommendations\n"]
    lines.append(f"**Morphology:** {morph.upper()}")
    lines.append(f"**m=2 amplitude:** {m2:.4f}")
    lines.append(f"**m=4 amplitude:** {m4:.4f}\n")
    
    lines.append("**Suggested models (in order):**")
    for i, m in enumerate(models, 1):
        lines.append(f"{i}. {m}")
    
    if morph == 'quad':
        lines.append("\n*QUAD regime: Use exact inversion for no-fit solution*")
    elif morph == 'ring':
        lines.append("\n*RING regime: Start with isotropic, add multipoles if needed*")
    
    return "\n".join(lines)
