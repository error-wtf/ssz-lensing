"""
Ring Analysis: Specialized analysis for Einstein Ring morphology.

Ring-fit as geometry inversion:
1. Radius estimation: θ_E ≈ median(r_i)
2. Center estimation: algebraic circle fit
3. Residual checks: radial + azimuthal systematics

Authors: Carmen N. Wrede, Lino P. Casu
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List


@dataclass
class RingFitResult:
    """Result of ring geometry fit."""
    center_x: float
    center_y: float
    radius: float
    radial_residuals: np.ndarray
    azimuthal_angles: np.ndarray
    rms_residual: float
    m2_component: Tuple[float, float]  # (amplitude, phase)
    m4_component: Tuple[float, float]  # (amplitude, phase)
    is_perturbed: bool
    perturbation_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'center': [self.center_x, self.center_y],
            'radius': self.radius,
            'rms_residual': self.rms_residual,
            'm2_amplitude': self.m2_component[0],
            'm2_phase': self.m2_component[1],
            'm4_amplitude': self.m4_component[0],
            'm4_phase': self.m4_component[1],
            'is_perturbed': self.is_perturbed,
            'perturbation_type': self.perturbation_type
        }


class RingAnalyzer:
    """
    Analyze Einstein Ring geometry.
    
    Philosophy: start simple, only add complexity if needed.
    """
    
    PERTURBATION_THRESHOLD = 0.02  # 2% of radius
    
    def __init__(self):
        pass
    
    def fit_ring(self, positions: np.ndarray, 
                 initial_center: Optional[Tuple[float, float]] = None) -> RingFitResult:
        """
        Fit ring geometry to positions.
        
        Args:
            positions: Nx2 array of (x, y) arc/image positions
            initial_center: Optional starting center
        """
        # Step 1: Estimate center
        if initial_center is None:
            cx, cy = self._estimate_center(positions)
        else:
            cx, cy = initial_center
        
        # Step 2: Compute radii and angles
        rel = positions - np.array([cx, cy])
        r = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)
        phi = np.arctan2(rel[:, 1], rel[:, 0])
        
        # Step 3: Estimate radius
        radius = np.median(r)
        
        # Step 4: Compute residuals
        dr = r - radius
        rms = np.sqrt(np.mean(dr**2))
        
        # Step 5: Harmonic analysis
        m2_amp, m2_phase = self._fit_harmonic(dr, phi, 2)
        m4_amp, m4_phase = self._fit_harmonic(dr, phi, 4)
        
        # Step 6: Classify perturbation
        is_perturbed = (m2_amp > self.PERTURBATION_THRESHOLD * radius or
                        m4_amp > self.PERTURBATION_THRESHOLD * radius)
        
        if m2_amp > m4_amp and m2_amp > self.PERTURBATION_THRESHOLD * radius:
            ptype = "quadrupole (m=2)"
        elif m4_amp > m2_amp and m4_amp > self.PERTURBATION_THRESHOLD * radius:
            ptype = "hexadecapole (m=4)"
        elif is_perturbed:
            ptype = "mixed"
        else:
            ptype = "isotropic"
        
        return RingFitResult(
            center_x=cx,
            center_y=cy,
            radius=radius,
            radial_residuals=dr,
            azimuthal_angles=phi,
            rms_residual=rms,
            m2_component=(m2_amp, m2_phase),
            m4_component=(m4_amp, m4_phase),
            is_perturbed=is_perturbed,
            perturbation_type=ptype
        )
    
    def _estimate_center(self, positions: np.ndarray) -> Tuple[float, float]:
        """Algebraic circle center estimation."""
        n = len(positions)
        if n < 3:
            return (np.mean(positions[:, 0]), np.mean(positions[:, 1]))
        
        x = positions[:, 0]
        y = positions[:, 1]
        
        A = np.column_stack([x, y, np.ones(n)])
        b = x**2 + y**2
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return (coeffs[0] / 2, coeffs[1] / 2)
        except Exception:
            return (np.mean(x), np.mean(y))
    
    def _fit_harmonic(self, dr: np.ndarray, phi: np.ndarray, 
                      m: int) -> Tuple[float, float]:
        """Fit harmonic component: dr ≈ A*cos(m*phi - phase)."""
        c = np.mean(dr * np.cos(m * phi))
        s = np.mean(dr * np.sin(m * phi))
        amp = 2 * np.sqrt(c**2 + s**2)  # Factor 2 for amplitude
        phase = np.arctan2(s, c)
        return (amp, phase)
    
    def generate_diagnostic_data(self, result: RingFitResult) -> Dict[str, Any]:
        """Generate data for ring diagnostic plots."""
        phi = result.azimuthal_angles
        dr = result.radial_residuals
        
        # Sort by angle for plotting
        idx = np.argsort(phi)
        phi_sorted = phi[idx]
        dr_sorted = dr[idx]
        
        # Model curves
        phi_model = np.linspace(-np.pi, np.pi, 100)
        m2_model = result.m2_component[0] * np.cos(
            2 * phi_model - result.m2_component[1]
        )
        m4_model = result.m4_component[0] * np.cos(
            4 * phi_model - result.m4_component[1]
        )
        
        return {
            'phi_data': phi_sorted.tolist(),
            'dr_data': dr_sorted.tolist(),
            'phi_model': phi_model.tolist(),
            'm2_model': m2_model.tolist(),
            'm4_model': m4_model.tolist(),
            'ring_center': [result.center_x, result.center_y],
            'ring_radius': result.radius,
            'rms': result.rms_residual
        }


def generate_ring_points(theta_E: float = 1.0, n_points: int = 50,
                         center: Tuple[float, float] = (0.0, 0.0),
                         c2: float = 0.0, s2: float = 0.0,
                         c4: float = 0.0, s4: float = 0.0,
                         noise: float = 0.0) -> np.ndarray:
    """
    Generate synthetic ring/arc points.
    
    Args:
        theta_E: Einstein radius
        n_points: Number of points
        center: Ring center
        c2, s2: m=2 perturbation (cos/sin)
        c4, s4: m=4 perturbation
        noise: Gaussian noise level
    """
    phi = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    r = theta_E + c2*np.cos(2*phi) + s2*np.sin(2*phi)
    r += c4*np.cos(4*phi) + s4*np.sin(4*phi)
    
    x = center[0] + r * np.cos(phi)
    y = center[1] + r * np.sin(phi)
    
    if noise > 0:
        x += np.random.normal(0, noise, n_points)
        y += np.random.normal(0, noise, n_points)
    
    return np.column_stack([x, y])
