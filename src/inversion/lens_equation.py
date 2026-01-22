"""
Lens Equation: β = θ - α(θ; p)

The fundamental equation of gravitational lensing.
This module provides the REAL inversion framework, not heuristics.

Key insight: For a given model with parameters p, each image θ_i 
maps back to a source position β_i. If the model is correct,
ALL β_i must coincide (source consistency).

Authors: Carmen N. Wrede, Lino P. Casu
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class InputMode(Enum):
    """Distinguish between input types - critical for correct analysis."""
    QUAD = "quad"      # 4 discrete image positions (Einstein Cross)
    RING = "ring"      # Many arc/contour points (Einstein Ring)
    DOUBLE = "double"  # 2 image positions
    UNKNOWN = "unknown"


@dataclass
class MorphologyResult:
    """
    Deterministic morphology classification with explicit criteria.
    
    NO fake confidence percentages - only traceable criteria.
    """
    mode: InputMode
    n_points: int
    
    # Ring vs Quad discriminators (deterministic)
    radial_scatter: float       # σ(r) / mean(r) - low for rings
    azimuthal_coverage: float   # fraction of 2π covered
    max_azimuthal_gap: float    # largest gap in φ (radians)
    n_clusters: int             # number of azimuthal clusters
    
    # Criteria used for classification
    criteria: Dict[str, bool] = field(default_factory=dict)
    explanation: str = ""


def classify_morphology(positions: np.ndarray, center: Tuple[float, float] = (0., 0.)) -> MorphologyResult:
    """
    Deterministic morphology classification with explicit, traceable criteria.
    
    Criteria for QUAD:
    - n_points == 4
    - Large azimuthal gaps (4 isolated clusters)
    - Radial scatter can be moderate
    
    Criteria for RING:
    - n_points > 10 (extended emission)
    - Low radial scatter (σ(r)/r̄ < 0.1)
    - High azimuthal coverage (> 70% of 2π)
    - Small maximum gap (< 60°)
    
    Parameters
    ----------
    positions : ndarray, shape (n, 2)
        Image positions (x, y)
    center : tuple
        Assumed lens center for polar conversion
        
    Returns
    -------
    MorphologyResult with explicit criteria
    """
    n = len(positions)
    
    # Convert to polar
    rel = positions - np.array(center)
    r = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)
    phi = np.arctan2(rel[:, 1], rel[:, 0])
    
    # Radial statistics
    r_mean = np.mean(r)
    r_std = np.std(r)
    radial_scatter = r_std / r_mean if r_mean > 0 else float('inf')
    
    # Azimuthal statistics
    phi_sorted = np.sort(phi)
    gaps = np.diff(phi_sorted)
    wrap_gap = 2*np.pi + phi_sorted[0] - phi_sorted[-1]
    gaps = np.append(gaps, wrap_gap)
    max_gap = np.max(gaps)
    azimuthal_coverage = 1.0 - max_gap / (2*np.pi)
    
    # Count clusters (gaps > 30° indicate cluster boundaries)
    cluster_threshold = np.radians(30)
    n_clusters = np.sum(gaps > cluster_threshold)
    if n_clusters == 0:
        n_clusters = 1  # All points in one cluster
    
    # Deterministic criteria
    criteria = {
        'is_4_points': n == 4,
        'is_2_points': n == 2,
        'is_many_points': n > 10,
        'low_radial_scatter': radial_scatter < 0.1,
        'high_azimuthal_coverage': azimuthal_coverage > 0.7,
        'small_max_gap': max_gap < np.radians(60),
        'has_4_clusters': n_clusters == 4,
        'has_2_clusters': n_clusters == 2,
    }
    
    # Classification logic (deterministic)
    if n == 2:
        mode = InputMode.DOUBLE
        explanation = "2 points: Double-image system (source outside caustic)"
    elif n == 4 and criteria['has_4_clusters']:
        mode = InputMode.QUAD
        explanation = "4 points with 4 azimuthal clusters: Einstein Cross configuration"
    elif n == 4:
        mode = InputMode.QUAD
        explanation = "4 points: Likely Einstein Cross (verify clustering)"
    elif criteria['is_many_points'] and criteria['low_radial_scatter'] and criteria['high_azimuthal_coverage']:
        mode = InputMode.RING
        explanation = f"Extended emission (n={n}), low radial scatter ({radial_scatter:.3f}), high coverage ({azimuthal_coverage:.1%}): Ring"
    elif criteria['is_many_points'] and not criteria['high_azimuthal_coverage']:
        mode = InputMode.RING  # Partial ring / arc
        explanation = f"Extended emission (n={n}), partial coverage ({azimuthal_coverage:.1%}): Arc/partial ring"
    else:
        mode = InputMode.UNKNOWN
        explanation = f"Ambiguous: n={n}, scatter={radial_scatter:.3f}, coverage={azimuthal_coverage:.1%}"
    
    return MorphologyResult(
        mode=mode,
        n_points=n,
        radial_scatter=radial_scatter,
        azimuthal_coverage=azimuthal_coverage,
        max_azimuthal_gap=max_gap,
        n_clusters=n_clusters,
        criteria=criteria,
        explanation=explanation
    )


@dataclass
class SourceConsistency:
    """
    Result of source position consistency check.
    
    For correct lens model: all β_i should coincide.
    """
    beta_positions: np.ndarray  # (n_images, 2) - source position from each image
    beta_mean: np.ndarray       # (2,) - mean source position
    beta_scatter: float         # RMS deviation from mean
    max_deviation: float        # Maximum deviation from mean
    is_consistent: bool         # True if scatter < tolerance
    per_image_residuals: np.ndarray  # Distance of each β_i from mean


def compute_source_positions(
    theta: np.ndarray,
    deflection_func,
    params: Dict
) -> np.ndarray:
    """
    Compute source positions β_i = θ_i - α(θ_i; p) for each image.
    
    Parameters
    ----------
    theta : ndarray, shape (n, 2)
        Image positions
    deflection_func : callable
        α(θ, params) -> deflection angle
    params : dict
        Lens model parameters
        
    Returns
    -------
    beta : ndarray, shape (n, 2)
        Source positions for each image
    """
    n = len(theta)
    beta = np.zeros_like(theta)
    
    for i in range(n):
        alpha = deflection_func(theta[i], params)
        beta[i] = theta[i] - alpha
    
    return beta


def check_source_consistency(
    beta: np.ndarray,
    tolerance: float = 1e-6
) -> SourceConsistency:
    """
    Check if all source positions are consistent (coincide).
    
    This is the KEY diagnostic for model validity:
    - If model is correct: all β_i ≈ same point
    - If model is wrong: β_i scatter
    
    Parameters
    ----------
    beta : ndarray, shape (n, 2)
        Source positions from each image
    tolerance : float
        Threshold for "consistent" classification
        
    Returns
    -------
    SourceConsistency with full diagnostics
    """
    beta_mean = np.mean(beta, axis=0)
    
    # Per-image residuals (distance from mean)
    residuals = np.sqrt(np.sum((beta - beta_mean)**2, axis=1))
    
    beta_scatter = np.sqrt(np.mean(residuals**2))  # RMS
    max_deviation = np.max(residuals)
    
    return SourceConsistency(
        beta_positions=beta,
        beta_mean=beta_mean,
        beta_scatter=beta_scatter,
        max_deviation=max_deviation,
        is_consistent=max_deviation < tolerance,
        per_image_residuals=residuals
    )


# ============================================================================
# DEFLECTION MODELS (LINEAR COMPONENT FORM: c_m, s_m, γ1, γ2)
# ============================================================================

def deflection_m2(theta: np.ndarray, params: Dict) -> np.ndarray:
    """
    Quadrupole (m=2) deflection in linear component form.
    
    α(θ) = θ_E² / |θ|² * θ̂ + quadrupole terms
    
    Parameters (linear):
    - theta_E: Einstein radius
    - c2, s2: quadrupole cos/sin components (NOT amplitude+phase!)
    
    The quadrupole contribution:
    α_m2 = (c2 * cos(2φ) + s2 * sin(2φ)) * r̂
    """
    x, y = theta
    r = np.sqrt(x**2 + y**2)
    if r < 1e-10:
        return np.zeros(2)
    
    phi = np.arctan2(y, x)
    theta_E = params.get('theta_E', 1.0)
    c2 = params.get('c2', 0.0)
    s2 = params.get('s2', 0.0)
    
    # Monopole (Einstein ring)
    alpha_r = theta_E**2 / r
    
    # Quadrupole (m=2) in component form
    alpha_r += c2 * np.cos(2*phi) + s2 * np.sin(2*phi)
    
    # Convert to Cartesian
    alpha_x = alpha_r * np.cos(phi)
    alpha_y = alpha_r * np.sin(phi)
    
    return np.array([alpha_x, alpha_y])


def deflection_m2_shear(theta: np.ndarray, params: Dict) -> np.ndarray:
    """
    Quadrupole + external shear deflection.
    
    Parameters (linear):
    - theta_E: Einstein radius
    - c2, s2: internal quadrupole
    - gamma1, gamma2: external shear components
    
    Shear contribution:
    α_shear = γ₁(x, -y) + γ₂(y, x)
    """
    x, y = theta
    r = np.sqrt(x**2 + y**2)
    if r < 1e-10:
        return np.zeros(2)
    
    phi = np.arctan2(y, x)
    theta_E = params.get('theta_E', 1.0)
    c2 = params.get('c2', 0.0)
    s2 = params.get('s2', 0.0)
    gamma1 = params.get('gamma1', 0.0)
    gamma2 = params.get('gamma2', 0.0)
    
    # Monopole + quadrupole
    alpha_r = theta_E**2 / r + c2 * np.cos(2*phi) + s2 * np.sin(2*phi)
    alpha_x = alpha_r * np.cos(phi)
    alpha_y = alpha_r * np.sin(phi)
    
    # Add external shear
    alpha_x += gamma1 * x + gamma2 * y
    alpha_y += -gamma1 * y + gamma2 * x
    
    return np.array([alpha_x, alpha_y])


def deflection_m2_m3(theta: np.ndarray, params: Dict) -> np.ndarray:
    """
    Quadrupole + octupole (m=3) deflection.
    
    Parameters (linear):
    - theta_E, c2, s2: as before
    - c3, s3: octupole components
    """
    x, y = theta
    r = np.sqrt(x**2 + y**2)
    if r < 1e-10:
        return np.zeros(2)
    
    phi = np.arctan2(y, x)
    theta_E = params.get('theta_E', 1.0)
    c2 = params.get('c2', 0.0)
    s2 = params.get('s2', 0.0)
    c3 = params.get('c3', 0.0)
    s3 = params.get('s3', 0.0)
    
    alpha_r = theta_E**2 / r
    alpha_r += c2 * np.cos(2*phi) + s2 * np.sin(2*phi)
    alpha_r += c3 * np.cos(3*phi) + s3 * np.sin(3*phi)
    
    alpha_x = alpha_r * np.cos(phi)
    alpha_y = alpha_r * np.sin(phi)
    
    return np.array([alpha_x, alpha_y])


def deflection_m2_m4(theta: np.ndarray, params: Dict) -> np.ndarray:
    """
    Quadrupole + hexadecapole (m=4) deflection.
    
    Parameters (linear):
    - theta_E, c2, s2: as before
    - c4, s4: hexadecapole components
    """
    x, y = theta
    r = np.sqrt(x**2 + y**2)
    if r < 1e-10:
        return np.zeros(2)
    
    phi = np.arctan2(y, x)
    theta_E = params.get('theta_E', 1.0)
    c2 = params.get('c2', 0.0)
    s2 = params.get('s2', 0.0)
    c4 = params.get('c4', 0.0)
    s4 = params.get('s4', 0.0)
    
    alpha_r = theta_E**2 / r
    alpha_r += c2 * np.cos(2*phi) + s2 * np.sin(2*phi)
    alpha_r += c4 * np.cos(4*phi) + s4 * np.sin(4*phi)
    
    alpha_x = alpha_r * np.cos(phi)
    alpha_y = alpha_r * np.sin(phi)
    
    return np.array([alpha_x, alpha_y])


def deflection_full(theta: np.ndarray, params: Dict) -> np.ndarray:
    """
    Full model: m=2 + shear + m=3 + m=4.
    
    Parameters (all linear):
    - theta_E: Einstein radius
    - c2, s2: quadrupole
    - gamma1, gamma2: external shear
    - c3, s3: octupole
    - c4, s4: hexadecapole
    """
    x, y = theta
    r = np.sqrt(x**2 + y**2)
    if r < 1e-10:
        return np.zeros(2)
    
    phi = np.arctan2(y, x)
    theta_E = params.get('theta_E', 1.0)
    c2 = params.get('c2', 0.0)
    s2 = params.get('s2', 0.0)
    gamma1 = params.get('gamma1', 0.0)
    gamma2 = params.get('gamma2', 0.0)
    c3 = params.get('c3', 0.0)
    s3 = params.get('s3', 0.0)
    c4 = params.get('c4', 0.0)
    s4 = params.get('s4', 0.0)
    
    alpha_r = theta_E**2 / r
    alpha_r += c2 * np.cos(2*phi) + s2 * np.sin(2*phi)
    alpha_r += c3 * np.cos(3*phi) + s3 * np.sin(3*phi)
    alpha_r += c4 * np.cos(4*phi) + s4 * np.sin(4*phi)
    
    alpha_x = alpha_r * np.cos(phi)
    alpha_y = alpha_r * np.sin(phi)
    
    # Add external shear
    alpha_x += gamma1 * x + gamma2 * y
    alpha_y += -gamma1 * y + gamma2 * x
    
    return np.array([alpha_x, alpha_y])


# Model registry
DEFLECTION_MODELS = {
    'm2': (deflection_m2, ['theta_E', 'c2', 's2']),
    'm2_shear': (deflection_m2_shear, ['theta_E', 'c2', 's2', 'gamma1', 'gamma2']),
    'm2_m3': (deflection_m2_m3, ['theta_E', 'c2', 's2', 'c3', 's3']),
    'm2_m4': (deflection_m2_m4, ['theta_E', 'c2', 's2', 'c4', 's4']),
    'm2_shear_m3': (deflection_full, ['theta_E', 'c2', 's2', 'gamma1', 'gamma2', 'c3', 's3']),
    'm2_shear_m4': (deflection_full, ['theta_E', 'c2', 's2', 'gamma1', 'gamma2', 'c4', 's4']),
    'm2_m3_m4': (deflection_full, ['theta_E', 'c2', 's2', 'c3', 's3', 'c4', 's4']),
    'full': (deflection_full, ['theta_E', 'c2', 's2', 'gamma1', 'gamma2', 'c3', 's3', 'c4', 's4']),
}
