"""
Cosmology module for redshift → distance conversion.

Supports Planck18 cosmology by default.
All outputs in internal units (meters).
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# Speed of light in m/s
C_M_S = 299792458.0

# Hubble constant conversion: km/s/Mpc → 1/s
KM_S_MPC_TO_1_S = 1e3 / 3.0856775814913673e22


@dataclass
class CosmologyParams:
    """Cosmological parameters."""
    name: str
    H0: float          # Hubble constant in km/s/Mpc
    Omega_m: float     # Matter density
    Omega_L: float     # Dark energy density
    Omega_k: float = 0.0  # Curvature (computed in __post_init__)

    def __post_init__(self):
        self.Omega_k = 1.0 - self.Omega_m - self.Omega_L

    def to_dict(self):
        return {
            'name': self.name,
            'H0': self.H0,
            'Omega_m': self.Omega_m,
            'Omega_L': self.Omega_L,
            'Omega_k': self.Omega_k,
        }


# Pre-defined cosmologies
PLANCK18 = CosmologyParams(
    name='Planck18',
    H0=67.4,
    Omega_m=0.315,
    Omega_L=0.685,
)

PLANCK15 = CosmologyParams(
    name='Planck15',
    H0=67.74,
    Omega_m=0.3089,
    Omega_L=0.6911,
)

WMAP9 = CosmologyParams(
    name='WMAP9',
    H0=69.32,
    Omega_m=0.2865,
    Omega_L=0.7135,
)

COSMOLOGIES = {
    'Planck18': PLANCK18,
    'Planck15': PLANCK15,
    'WMAP9': WMAP9,
}


def E(z: float, cosmo: CosmologyParams) -> float:
    """Dimensionless Hubble parameter E(z) = H(z)/H0."""
    return np.sqrt(
        cosmo.Omega_m * (1 + z)**3 +
        cosmo.Omega_k * (1 + z)**2 +
        cosmo.Omega_L
    )


def comoving_distance(z: float, cosmo: CosmologyParams, n_steps: int = 1000) -> float:
    """
    Compute comoving distance D_c(z) in meters.

    Uses numerical integration: D_c = (c/H0) * integral_0^z dz'/E(z')
    """
    if z <= 0:
        return 0.0

    # Hubble distance in meters
    H0_1_s = cosmo.H0 * KM_S_MPC_TO_1_S
    D_H = C_M_S / H0_1_s

    # Simpson integration
    z_arr = np.linspace(0, z, n_steps + 1)
    dz = z / n_steps
    integrand = 1.0 / E(z_arr, cosmo)

    # Simpson's rule
    integral = (dz / 3) * (
        integrand[0] +
        4 * np.sum(integrand[1:-1:2]) +
        2 * np.sum(integrand[2:-1:2]) +
        integrand[-1]
    )

    return D_H * integral


def angular_diameter_distance(z: float, cosmo: CosmologyParams) -> float:
    """
    Compute angular diameter distance D_A(z) in meters.

    D_A = D_c / (1 + z) for flat universe.
    """
    D_c = comoving_distance(z, cosmo)
    return D_c / (1 + z)


def angular_diameter_distance_z1_z2(
    z1: float,
    z2: float,
    cosmo: CosmologyParams
) -> float:
    """
    Compute angular diameter distance between two redshifts D_A(z1, z2).

    For flat universe: D_A(z1, z2) = [D_c(z2) - D_c(z1)] / (1 + z2)
    """
    if z2 <= z1:
        raise ValueError(f"z2 ({z2}) must be greater than z1 ({z1})")

    D_c1 = comoving_distance(z1, cosmo)
    D_c2 = comoving_distance(z2, cosmo)

    return (D_c2 - D_c1) / (1 + z2)


def luminosity_distance(z: float, cosmo: CosmologyParams) -> float:
    """
    Compute luminosity distance D_L(z) in meters.

    D_L = D_A * (1 + z)^2
    """
    D_A = angular_diameter_distance(z, cosmo)
    return D_A * (1 + z)**2


def lensing_distances(
    z_L: float,
    z_S: float,
    cosmo: CosmologyParams = PLANCK18
) -> Tuple[float, float, float]:
    """
    Compute the three lensing distances in meters.

    Parameters
    ----------
    z_L : float
        Lens redshift
    z_S : float
        Source redshift
    cosmo : CosmologyParams
        Cosmology to use (default: Planck18)

    Returns
    -------
    D_L, D_S, D_LS : tuple of float
        Angular diameter distances in meters:
        - D_L: Observer to Lens
        - D_S: Observer to Source
        - D_LS: Lens to Source
    """
    if z_S <= z_L:
        raise ValueError(f"z_S ({z_S}) must be > z_L ({z_L})")

    D_L = angular_diameter_distance(z_L, cosmo)
    D_S = angular_diameter_distance(z_S, cosmo)
    D_LS = angular_diameter_distance_z1_z2(z_L, z_S, cosmo)

    return D_L, D_S, D_LS


def einstein_radius_from_mass(
    mass_kg: float,
    D_L: float,
    D_S: float,
    D_LS: float
) -> float:
    """
    Compute Einstein radius (in radians) from lens mass.

    theta_E = sqrt(4 G M / c^2 * D_LS / (D_L * D_S))
    """
    G_SI = 6.67430e-11
    factor = 4 * G_SI * mass_kg / (C_M_S**2)
    theta_E_sq = factor * D_LS / (D_L * D_S)
    return np.sqrt(theta_E_sq)


def mass_from_einstein_radius(
    theta_E_rad: float,
    D_L: float,
    D_S: float,
    D_LS: float
) -> float:
    """
    Compute lens mass (in kg) from Einstein radius.

    M = c^2 / (4 G) * theta_E^2 * D_L * D_S / D_LS
    """
    G_SI = 6.67430e-11
    factor = C_M_S**2 / (4 * G_SI)
    return factor * theta_E_rad**2 * D_L * D_S / D_LS


@dataclass
class LensingDistanceResult:
    """Result from lensing distance calculation."""
    D_L_m: float
    D_S_m: float
    D_LS_m: float
    z_L: Optional[float]
    z_S: Optional[float]
    cosmology: Optional[str]
    input_mode: str  # 'redshift', 'direct', 'normalized'

    def to_dict(self):
        return {
            'D_L_m': self.D_L_m,
            'D_S_m': self.D_S_m,
            'D_LS_m': self.D_LS_m,
            'z_L': self.z_L,
            'z_S': self.z_S,
            'cosmology': self.cosmology,
            'input_mode': self.input_mode,
        }
