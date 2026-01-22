"""
Lens Models for Gauge Gravitational Lensing

Available models:
- LensModel: Abstract base class
- RingQuadrupoleOffsetModel: Minimal model (m=2)
- MultipoleModel: General multipole expansion
- ExtendedMultipoleModel: With shear, higher multipoles, power-law profile

Extended with:
- Power-law radial profiles (η variable)
- External shear (γ, φ_γ)
- Higher multipoles (m=3, m=4)
- Hermite C² blending

Authors: Carmen N. Wrede, Lino P. Casu
"""

from .base_model import LensModel
from .ring_quadrupole_offset import RingQuadrupoleOffsetModel
from .multipole_model import MultipoleModel
from .extended_model import ExtendedMultipoleModel, ExtendedParams
from .linear_model import LinearMultipoleModel, LinearParams
from .profiles import (
    kappa_power_law, kappa_cored, kappa_sis,
    alpha_power_law, alpha_cored, alpha_sis,
    deflection_2d_power_law, deflection_2d_elliptical,
    hermite_blend, blend_profiles,
    ProfileParams, PHI
)
from .root_finders import (
    bisection, find_all_roots, find_all_roots_safe,
    newton_raphson, brent, find_minimum_bracketed
)

__all__ = [
    'LensModel',
    'MultipoleModel',
    'RingQuadrupoleOffsetModel',
    'ExtendedMultipoleModel',
    'ExtendedParams',
    'LinearMultipoleModel',
    'LinearParams',
    'ProfileParams',
    'kappa_power_law', 'kappa_cored', 'kappa_sis',
    'alpha_power_law', 'alpha_cored', 'alpha_sis',
    'deflection_2d_power_law', 'deflection_2d_elliptical',
    'hermite_blend', 'blend_profiles',
    'bisection', 'find_all_roots', 'find_all_roots_safe',
    'newton_raphson', 'brent', 'find_minimum_bracketed',
    'PHI',
]
