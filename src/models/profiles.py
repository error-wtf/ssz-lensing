"""
Radial Mass Profiles for Gravitational Lensing

Implements various mass density profiles and their lensing properties.
Ported from SSZ framework with power-law validation (R² = 0.997).

Key profiles:
- SIS (Singular Isothermal Sphere): ρ ∝ r^(-2)
- Power-Law: ρ ∝ r^(-η) with variable η
- Cored profiles: Finite central density
- NFW (Navarro-Frenk-White): Dark matter halos

Physical Foundation:
- Power-law scaling validated across 6 orders of magnitude in SSZ
- β ≈ 1 exponent indicates near-linear geometric scaling
- Hermite C² blending for smooth regime transitions

Authors: Carmen N. Wrede, Lino P. Casu
License: ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

# Golden ratio (fundamental in SSZ)
PHI = (1.0 + np.sqrt(5.0)) / 2.0


@dataclass
class ProfileParams:
    """Parameters for a radial mass profile."""
    theta_E: float  # Einstein radius (mass scale)
    eta: float = 2.0  # Power-law slope (η=2 is SIS)
    r_core: float = 0.0  # Core radius (0 = singular)
    r_s: float = 1.0  # Scale radius (for NFW)
    
    def __post_init__(self):
        if self.theta_E <= 0:
            raise ValueError("theta_E must be positive")
        if self.eta < 0:
            raise ValueError("eta must be non-negative")
        if self.r_core < 0:
            raise ValueError("r_core must be non-negative")


def kappa_sis(theta: float, theta_E: float) -> float:
    """
    Convergence for Singular Isothermal Sphere.
    
    κ(θ) = θ_E / (2|θ|)
    
    This is the η=2 case of power-law.
    """
    if abs(theta) < 1e-15:
        return np.inf
    return theta_E / (2.0 * abs(theta))


def kappa_power_law(theta: float, theta_E: float, eta: float) -> float:
    """
    Convergence for power-law profile ρ ∝ r^(-η).
    
    κ(θ) = (3-η)/2 × (θ_E/θ)^(η-1)
    
    Valid for 1 < η < 3.
    
    Parameters
    ----------
    theta : float
        Angular position (arcsec)
    theta_E : float
        Einstein radius
    eta : float
        Power-law slope (η=2 is SIS)
    
    Returns
    -------
    kappa : float
        Convergence (dimensionless surface density)
    
    Notes
    -----
    From SSZ POWER_LAW_FINDINGS.md:
    - β ≈ 0.98 ≈ 1 for energy scaling
    - This suggests η ≈ 2 (SIS) is a good approximation
    - Deviations from η=2 encode galaxy structure
    """
    if eta <= 1 or eta >= 3:
        raise ValueError(f"eta must be in (1, 3), got {eta}")
    
    if abs(theta) < 1e-15:
        if eta > 1:
            return np.inf
        return 0.0
    
    prefactor = (3.0 - eta) / 2.0
    ratio = theta_E / abs(theta)
    return prefactor * (ratio ** (eta - 1))


def kappa_cored(theta: float, theta_E: float, r_core: float, eta: float = 2.0) -> float:
    """
    Convergence for cored power-law profile.
    
    κ(θ) = κ_0 × (1 + (θ/r_core)²)^(-(η-1)/2)
    
    This avoids the central singularity of pure power-law.
    Inspired by SSZ: "NO SINGULARITY at natural boundary"
    
    Parameters
    ----------
    theta : float
        Angular position
    theta_E : float
        Einstein radius
    r_core : float
        Core radius (softening scale)
    eta : float
        Asymptotic power-law slope
    """
    if r_core <= 0:
        return kappa_power_law(theta, theta_E, eta)
    
    # Normalization: κ_0 chosen so that Einstein radius is preserved
    # For SIS-like behavior at large radii
    kappa_0 = theta_E / (2.0 * r_core)
    
    u_sq = (theta / r_core) ** 2
    exponent = -(eta - 1) / 2.0
    
    return kappa_0 * (1.0 + u_sq) ** exponent


def alpha_sis(theta: float, theta_E: float) -> float:
    """
    Deflection angle for SIS.
    
    α(θ) = θ_E × sign(θ)
    
    Constant magnitude, direction toward center.
    """
    if abs(theta) < 1e-15:
        return 0.0
    return theta_E * np.sign(theta)


def alpha_power_law(theta: float, theta_E: float, eta: float) -> float:
    """
    Deflection angle for power-law profile.
    
    α(θ) = θ_E × (θ/θ_E)^(2-η) × sign(θ)  for η ≠ 2
    α(θ) = θ_E × sign(θ)                   for η = 2 (SIS)
    
    Parameters
    ----------
    theta : float
        Angular position
    theta_E : float
        Einstein radius
    eta : float
        Power-law slope
    
    Returns
    -------
    alpha : float
        Deflection angle (same units as theta)
    """
    if abs(eta - 2.0) < 1e-10:
        return alpha_sis(theta, theta_E)
    
    if abs(theta) < 1e-15:
        if eta < 2:
            return 0.0
        return np.inf * np.sign(theta) if theta != 0 else 0.0
    
    sign = np.sign(theta)
    ratio = abs(theta) / theta_E
    exponent = 2.0 - eta
    
    return theta_E * (ratio ** exponent) * sign


def alpha_cored(theta: float, theta_E: float, r_core: float, eta: float = 2.0) -> float:
    """
    Deflection angle for cored profile.
    
    Numerical integration of κ for general case.
    For η=2 (cored isothermal):
        α(θ) = θ_E × θ / √(r_core² + θ²)
    """
    if r_core <= 0:
        return alpha_power_law(theta, theta_E, eta)
    
    if abs(eta - 2.0) < 1e-10:
        # Analytic for cored isothermal
        return theta_E * theta / np.sqrt(r_core**2 + theta**2)
    
    # Numerical integration for general η
    # α = (2/θ) ∫₀^θ κ(θ') θ' dθ'
    from scipy.integrate import quad
    
    def integrand(t):
        return kappa_cored(t, theta_E, r_core, eta) * t
    
    if abs(theta) < 1e-15:
        return 0.0
    
    result, _ = quad(integrand, 0, abs(theta))
    return 2.0 * result / theta * np.sign(theta)


# =============================================================================
# HERMITE C² BLENDING (from SSZ)
# =============================================================================

def hermite_blend(x: float, x0: float, x1: float) -> float:
    """
    Hermite C² smooth blending function.
    
    Returns 0 for x ≤ x0, 1 for x ≥ x1, smooth transition between.
    
    h(t) = 3t² - 2t³  (C¹ continuous)
    
    For C² continuity, we use:
    h(t) = 6t⁵ - 15t⁴ + 10t³
    
    From SSZ: Used for regime transitions (weak ↔ strong field)
    """
    if x <= x0:
        return 0.0
    if x >= x1:
        return 1.0
    
    t = (x - x0) / (x1 - x0)
    # C² smooth (smootherstep)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def blend_profiles(
    theta: float,
    profile1: Callable[[float], float],
    profile2: Callable[[float], float],
    theta_blend_start: float,
    theta_blend_end: float
) -> float:
    """
    Smoothly blend two profiles using Hermite interpolation.
    
    Parameters
    ----------
    theta : float
        Angular position
    profile1 : callable
        Inner profile (used for θ < θ_blend_start)
    profile2 : callable
        Outer profile (used for θ > θ_blend_end)
    theta_blend_start : float
        Start of blending zone
    theta_blend_end : float
        End of blending zone
    
    Returns
    -------
    value : float
        Blended profile value
    """
    w = hermite_blend(abs(theta), theta_blend_start, theta_blend_end)
    return (1.0 - w) * profile1(theta) + w * profile2(theta)


# =============================================================================
# DEFLECTION IN 2D (for lensing)
# =============================================================================

def deflection_2d_power_law(
    x: float, 
    y: float, 
    theta_E: float, 
    eta: float = 2.0,
    r_core: float = 0.0
) -> Tuple[float, float]:
    """
    2D deflection angle for (cored) power-law profile.
    
    Parameters
    ----------
    x, y : float
        Image position
    theta_E : float
        Einstein radius
    eta : float
        Power-law slope
    r_core : float
        Core radius (0 = singular)
    
    Returns
    -------
    alpha_x, alpha_y : float
        Deflection components
    """
    r = np.sqrt(x**2 + y**2)
    
    if r < 1e-15:
        return 0.0, 0.0
    
    # Magnitude of deflection
    if r_core > 0:
        alpha_mag = alpha_cored(r, theta_E, r_core, eta)
    else:
        alpha_mag = alpha_power_law(r, theta_E, eta)
    
    # Direction: toward center (radial)
    alpha_x = alpha_mag * x / r
    alpha_y = alpha_mag * y / r
    
    return alpha_x, alpha_y


def deflection_2d_elliptical(
    x: float,
    y: float,
    theta_E: float,
    q: float,
    phi_q: float,
    eta: float = 2.0
) -> Tuple[float, float]:
    """
    2D deflection for elliptical power-law (SPEMD-like).
    
    Parameters
    ----------
    x, y : float
        Image position
    theta_E : float
        Einstein radius
    q : float
        Axis ratio (0 < q ≤ 1)
    phi_q : float
        Position angle of major axis (radians)
    eta : float
        Power-law slope
    
    Returns
    -------
    alpha_x, alpha_y : float
        Deflection components
    
    Notes
    -----
    For q=1 (circular), reduces to spherical power-law.
    Uses the approximation valid for moderate ellipticity.
    """
    if q <= 0 or q > 1:
        raise ValueError(f"Axis ratio q must be in (0, 1], got {q}")
    
    # Rotate to align with ellipse
    cos_phi = np.cos(phi_q)
    sin_phi = np.sin(phi_q)
    
    x_rot = x * cos_phi + y * sin_phi
    y_rot = -x * sin_phi + y * cos_phi
    
    # Elliptical radius
    r_ell = np.sqrt(x_rot**2 + (y_rot / q)**2)
    
    if r_ell < 1e-15:
        return 0.0, 0.0
    
    # Deflection magnitude (spherical approximation scaled)
    alpha_mag = alpha_power_law(r_ell, theta_E, eta)
    
    # Components in rotated frame
    # For elliptical: different scaling in x and y
    f = np.sqrt(1.0 - q**2) if q < 1 else 0.0
    
    if f > 1e-10:
        # Full elliptical formula
        psi = np.sqrt(q**2 * x_rot**2 + y_rot**2)
        alpha_x_rot = theta_E * q / f * np.arctan(f * x_rot / psi) if psi > 1e-15 else 0.0
        alpha_y_rot = theta_E * q / f * np.arctanh(f * y_rot / psi) if psi > 1e-15 else 0.0
    else:
        # Circular limit
        alpha_x_rot = alpha_mag * x_rot / r_ell
        alpha_y_rot = alpha_mag * y_rot / r_ell
    
    # Rotate back
    alpha_x = alpha_x_rot * cos_phi - alpha_y_rot * sin_phi
    alpha_y = alpha_x_rot * sin_phi + alpha_y_rot * cos_phi
    
    return alpha_x, alpha_y


# =============================================================================
# MAGNIFICATION
# =============================================================================

def magnification_power_law(theta: float, theta_E: float, eta: float) -> float:
    """
    Magnification for power-law profile.
    
    μ = 1 / ((1 - κ)² - γ²)
    
    For spherical: γ = κ, so μ = 1 / (1 - 2κ)
    """
    kappa = kappa_power_law(theta, theta_E, eta)
    
    denom = 1.0 - 2.0 * kappa
    if abs(denom) < 1e-15:
        return np.inf  # Critical curve
    
    return 1.0 / denom


def critical_curve_power_law(theta_E: float, eta: float) -> float:
    """
    Critical curve radius for power-law.
    
    Solve κ(θ_crit) = 1/2
    
    θ_crit = θ_E × ((3-η))^(1/(η-1))
    """
    if eta <= 1 or eta >= 3:
        raise ValueError(f"eta must be in (1, 3), got {eta}")
    
    # κ = (3-η)/2 × (θ_E/θ)^(η-1) = 1/2
    # (θ_E/θ)^(η-1) = 1/(3-η)
    # θ/θ_E = (3-η)^(1/(η-1))
    
    return theta_E * ((3.0 - eta) ** (1.0 / (eta - 1)))


# =============================================================================
# SSZ-INSPIRED POWER LAW VALIDATION
# =============================================================================

def ssz_power_law_energy_ratio(r_s: float, R: float, alpha: float = 0.3187, beta: float = 0.9821) -> float:
    """
    Universal power-law energy ratio from SSZ.
    
    E_obs/E_rest = 1 + α × (r_s/R)^β
    
    Validated with R² = 0.997 across 6 orders of magnitude.
    
    Parameters
    ----------
    r_s : float
        Schwarzschild radius
    R : float
        Object radius
    alpha : float
        Amplitude (0.3187 ± 0.0023)
    beta : float
        Exponent (0.9821 ± 0.0089)
    
    Returns
    -------
    ratio : float
        E_obs / E_rest
    """
    if R <= 0:
        raise ValueError("R must be positive")
    if r_s < 0:
        raise ValueError("r_s must be non-negative")
    
    compactness = r_s / R
    return 1.0 + alpha * (compactness ** beta)


def infer_eta_from_ssz_beta(ssz_beta: float = 0.9821) -> float:
    """
    Infer lens profile slope η from SSZ energy scaling exponent β.
    
    The near-unity β ≈ 1 in SSZ suggests geometric scaling.
    For lensing, this maps approximately to η ≈ 2 (isothermal).
    
    The relationship is:
        η ≈ 2 + (1 - β)
    
    For β = 0.98: η ≈ 2.02 (slightly steeper than SIS)
    """
    return 2.0 + (1.0 - ssz_beta)
