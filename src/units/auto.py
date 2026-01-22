"""
Auto-scaling unit system for RSG Lensing Framework.

Internal units (strict):
- Angles: radians
- Lengths/Distances: meters
- Times: seconds
- Masses: kilograms

External display: auto-scaled to human-readable units.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

# =============================================================================
# CONSTANTS
# =============================================================================

# Angle conversions (to radians)
ARCSEC_TO_RAD = np.pi / (180 * 3600)
MAS_TO_RAD = ARCSEC_TO_RAD / 1000
MUAS_TO_RAD = MAS_TO_RAD / 1000
DEG_TO_RAD = np.pi / 180

# Distance conversions (to meters)
AU_TO_M = 1.495978707e11          # Astronomical Unit
LY_TO_M = 9.4607304725808e15      # Light-year
PC_TO_M = 3.0856775814913673e16   # Parsec
KPC_TO_M = PC_TO_M * 1e3          # Kiloparsec
MPC_TO_M = PC_TO_M * 1e6          # Megaparsec
GPC_TO_M = PC_TO_M * 1e9          # Gigaparsec

# Time conversions (to seconds)
HOUR_TO_S = 3600
DAY_TO_S = 86400
YEAR_TO_S = 365.25 * DAY_TO_S

# Mass conversions (to kg)
MSUN_TO_KG = 1.98892e30           # Solar mass

# Speed of light
C_M_S = 299792458.0               # m/s

# Gravitational constant
G_SI = 6.67430e-11                # m^3 kg^-1 s^-2

# =============================================================================
# UNIT DEFINITIONS
# =============================================================================

ANGLE_UNITS = {
    'rad': 1.0,
    'deg': DEG_TO_RAD,
    'arcsec': ARCSEC_TO_RAD,
    'mas': MAS_TO_RAD,
    'µas': MUAS_TO_RAD,
    'uas': MUAS_TO_RAD,  # ASCII alias
}

DISTANCE_UNITS = {
    'm': 1.0,
    'km': 1e3,
    'AU': AU_TO_M,
    'ly': LY_TO_M,
    'pc': PC_TO_M,
    'kpc': KPC_TO_M,
    'Mpc': MPC_TO_M,
    'Gpc': GPC_TO_M,
}

TIME_UNITS = {
    's': 1.0,
    'min': 60.0,
    'h': HOUR_TO_S,
    'd': DAY_TO_S,
    'yr': YEAR_TO_S,
}

MASS_UNITS = {
    'kg': 1.0,
    'M_sun': MSUN_TO_KG,
    'Msun': MSUN_TO_KG,
}

# =============================================================================
# FORMATTING DATACLASS
# =============================================================================

@dataclass
class FormattedValue:
    """A value with its display string and unit metadata."""
    internal_value: float      # In base units (rad/m/s/kg)
    display_value: float       # In display units
    display_unit: str          # Unit name for display
    display_string: str        # Complete formatted string
    internal_unit: str         # Base unit name
    alternatives: Dict[str, float] = None  # Other unit representations

    def __str__(self):
        return self.display_string

    def to_dict(self):
        d = {
            'internal_value': self.internal_value,
            'internal_unit': self.internal_unit,
            'display_value': self.display_value,
            'display_unit': self.display_unit,
            'display_string': self.display_string,
        }
        if self.alternatives:
            d['alternatives'] = self.alternatives
        return d

# =============================================================================
# AUTO-SCALING FUNCTIONS
# =============================================================================

def format_angle(rad: float, precision: int = 4, include_alternatives: bool = True) -> FormattedValue:
    """
    Format angle from radians to best human-readable unit.
    
    Auto-scaling rules:
    - < 1e-9 rad → µas
    - < 1e-6 rad → mas  
    - < 1e-3 rad → arcsec
    - else → deg (or rad if very large)
    """
    abs_rad = abs(rad)
    
    if abs_rad < 1e-9:
        unit = 'µas'
        factor = MUAS_TO_RAD
    elif abs_rad < 1e-6:
        unit = 'mas'
        factor = MAS_TO_RAD
    elif abs_rad < 1e-3:
        unit = 'arcsec'
        factor = ARCSEC_TO_RAD
    elif abs_rad < 0.1:
        unit = 'arcsec'
        factor = ARCSEC_TO_RAD
    else:
        unit = 'deg'
        factor = DEG_TO_RAD
    
    display_val = rad / factor
    display_str = f"{display_val:.{precision}g} {unit}"
    
    alternatives = None
    if include_alternatives:
        alternatives = {
            'rad': rad,
            'deg': rad / DEG_TO_RAD,
            'arcsec': rad / ARCSEC_TO_RAD,
            'mas': rad / MAS_TO_RAD,
            'µas': rad / MUAS_TO_RAD,
        }
    
    return FormattedValue(
        internal_value=rad,
        display_value=display_val,
        display_unit=unit,
        display_string=display_str,
        internal_unit='rad',
        alternatives=alternatives
    )


def format_distance(meters: float, precision: int = 4, include_alternatives: bool = True) -> FormattedValue:
    """
    Format distance from meters to best human-readable unit.
    
    Auto-scaling rules:
    - < 1e11 m → AU
    - < 1e16 m → ly
    - < 1e19 m → pc
    - < 1e22 m → kpc
    - < 1e25 m → Mpc
    - else → Gpc
    """
    abs_m = abs(meters)
    
    if abs_m < 1e8:
        unit = 'km'
        factor = 1e3
    elif abs_m < 1e11:
        unit = 'AU'
        factor = AU_TO_M
    elif abs_m < 1e16:
        unit = 'ly'
        factor = LY_TO_M
    elif abs_m < 1e19:
        unit = 'pc'
        factor = PC_TO_M
    elif abs_m < 1e22:
        unit = 'kpc'
        factor = KPC_TO_M
    elif abs_m < 1e25:
        unit = 'Mpc'
        factor = MPC_TO_M
    else:
        unit = 'Gpc'
        factor = GPC_TO_M
    
    display_val = meters / factor
    display_str = f"{display_val:.{precision}g} {unit}"
    
    alternatives = None
    if include_alternatives:
        alternatives = {
            'm': meters,
            'km': meters / 1e3,
            'AU': meters / AU_TO_M,
            'ly': meters / LY_TO_M,
            'pc': meters / PC_TO_M,
            'kpc': meters / KPC_TO_M,
            'Mpc': meters / MPC_TO_M,
            'Gpc': meters / GPC_TO_M,
        }
    
    return FormattedValue(
        internal_value=meters,
        display_value=display_val,
        display_unit=unit,
        display_string=display_str,
        internal_unit='m',
        alternatives=alternatives
    )


def format_radius(meters: float, precision: int = 4, include_alternatives: bool = True) -> FormattedValue:
    """
    Format physical radius (like Einstein radius R_E) from meters.
    
    Similar to distance but favors AU/pc/kpc for typical lens scales.
    """
    abs_m = abs(meters)
    
    if abs_m < 1e6:
        unit = 'km'
        factor = 1e3
    elif abs_m < 1e14:
        unit = 'AU'
        factor = AU_TO_M
    elif abs_m < 1e19:
        unit = 'pc'
        factor = PC_TO_M
    elif abs_m < 1e22:
        unit = 'kpc'
        factor = KPC_TO_M
    else:
        unit = 'Mpc'
        factor = MPC_TO_M
    
    display_val = meters / factor
    display_str = f"{display_val:.{precision}g} {unit}"
    
    alternatives = None
    if include_alternatives:
        alternatives = {
            'm': meters,
            'km': meters / 1e3,
            'AU': meters / AU_TO_M,
            'pc': meters / PC_TO_M,
            'kpc': meters / KPC_TO_M,
        }
    
    return FormattedValue(
        internal_value=meters,
        display_value=display_val,
        display_unit=unit,
        display_string=display_str,
        internal_unit='m',
        alternatives=alternatives
    )


def format_time(seconds: float, precision: int = 4, include_alternatives: bool = True) -> FormattedValue:
    """
    Format time from seconds to best human-readable unit.
    
    Auto-scaling:
    - < 60 s → s
    - < 3600 s → min
    - < 86400 s → h
    - < 365.25 d → d
    - else → yr
    """
    abs_s = abs(seconds)
    
    if abs_s < 60:
        unit = 's'
        factor = 1.0
    elif abs_s < 3600:
        unit = 'min'
        factor = 60.0
    elif abs_s < DAY_TO_S:
        unit = 'h'
        factor = HOUR_TO_S
    elif abs_s < YEAR_TO_S:
        unit = 'd'
        factor = DAY_TO_S
    else:
        unit = 'yr'
        factor = YEAR_TO_S
    
    display_val = seconds / factor
    display_str = f"{display_val:.{precision}g} {unit}"
    
    alternatives = None
    if include_alternatives:
        alternatives = {
            's': seconds,
            'min': seconds / 60,
            'h': seconds / HOUR_TO_S,
            'd': seconds / DAY_TO_S,
            'yr': seconds / YEAR_TO_S,
        }
    
    return FormattedValue(
        internal_value=seconds,
        display_value=display_val,
        display_unit=unit,
        display_string=display_str,
        internal_unit='s',
        alternatives=alternatives
    )


def format_mass(kg: float, precision: int = 4, include_alternatives: bool = True) -> FormattedValue:
    """Format mass from kg, preferring solar masses for astrophysics."""
    if abs(kg) > 1e20:
        unit = 'M_sun'
        factor = MSUN_TO_KG
    else:
        unit = 'kg'
        factor = 1.0
    
    display_val = kg / factor
    display_str = f"{display_val:.{precision}g} {unit}"
    
    alternatives = None
    if include_alternatives:
        alternatives = {
            'kg': kg,
            'M_sun': kg / MSUN_TO_KG,
        }
    
    return FormattedValue(
        internal_value=kg,
        display_value=display_val,
        display_unit=unit,
        display_string=display_str,
        internal_unit='kg',
        alternatives=alternatives
    )


# =============================================================================
# PARSING & CONVERSION
# =============================================================================

def parse_value_with_unit(text: str, unit_dict: Dict[str, float]) -> Tuple[float, str]:
    """
    Parse a value with unit from text like "1.3 Gpc" or "2.5arcsec".
    
    Returns (value_in_base_units, original_unit_name).
    """
    text = text.strip()
    
    # Try each unit (longest first to avoid partial matches)
    for unit in sorted(unit_dict.keys(), key=len, reverse=True):
        if text.endswith(unit):
            value_str = text[:-len(unit)].strip()
            try:
                value = float(value_str)
                return value * unit_dict[unit], unit
            except ValueError:
                continue
    
    # No unit found, try parsing as pure number
    try:
        return float(text), None
    except ValueError:
        raise ValueError(f"Cannot parse '{text}' as value with unit")


def convert_to_internal(value: float, unit: str, quantity_type: str) -> float:
    """
    Convert a value with given unit to internal units.
    
    quantity_type: 'angle', 'distance', 'time', 'mass'
    """
    unit_dicts = {
        'angle': ANGLE_UNITS,
        'distance': DISTANCE_UNITS,
        'time': TIME_UNITS,
        'mass': MASS_UNITS,
    }
    
    if quantity_type not in unit_dicts:
        raise ValueError(f"Unknown quantity type: {quantity_type}")
    
    unit_dict = unit_dicts[quantity_type]
    
    if unit not in unit_dict:
        raise ValueError(f"Unknown {quantity_type} unit: {unit}")
    
    return value * unit_dict[unit]


def convert_from_internal(value: float, unit: str, quantity_type: str) -> float:
    """
    Convert a value from internal units to given unit.
    
    quantity_type: 'angle', 'distance', 'time', 'mass'
    """
    unit_dicts = {
        'angle': ANGLE_UNITS,
        'distance': DISTANCE_UNITS,
        'time': TIME_UNITS,
        'mass': MASS_UNITS,
    }
    
    if quantity_type not in unit_dicts:
        raise ValueError(f"Unknown quantity type: {quantity_type}")
    
    unit_dict = unit_dicts[quantity_type]
    
    if unit not in unit_dict:
        raise ValueError(f"Unknown {quantity_type} unit: {unit}")
    
    return value / unit_dict[unit]


# =============================================================================
# SCHWARZSCHILD RADIUS
# =============================================================================

def schwarzschild_radius(mass_kg: float) -> float:
    """Compute Schwarzschild radius in meters: r_s = 2GM/c^2"""
    return 2 * G_SI * mass_kg / (C_M_S ** 2)


def format_schwarzschild_radius(mass_kg: float, precision: int = 4) -> FormattedValue:
    """Format Schwarzschild radius with auto-scaling (km for stars, AU for SMBH)."""
    r_s = schwarzschild_radius(mass_kg)
    
    if r_s < 1e9:
        unit = 'km'
        factor = 1e3
    else:
        unit = 'AU'
        factor = AU_TO_M
    
    display_val = r_s / factor
    display_str = f"{display_val:.{precision}g} {unit}"
    
    return FormattedValue(
        internal_value=r_s,
        display_value=display_val,
        display_unit=unit,
        display_string=display_str,
        internal_unit='m',
        alternatives={'m': r_s, 'km': r_s / 1e3, 'AU': r_s / AU_TO_M}
    )


# =============================================================================
# CONVENIENCE: ALL SCENE QUANTITIES
# =============================================================================

def format_scene_quantities(
    D_L_m: float,
    D_S_m: float,
    theta_E_rad: float,
    beta_rad: float,
    mass_kg: Optional[float] = None,
    precision: int = 4
) -> Dict[str, FormattedValue]:
    """
    Format all quantities for a lensing scene.
    
    All inputs in internal units (m, rad, kg).
    Returns dict of FormattedValue objects.
    """
    D_LS_m = D_S_m - D_L_m
    R_E_m = D_L_m * theta_E_rad
    R_beta_m = D_S_m * beta_rad
    
    result = {
        'D_L': format_distance(D_L_m, precision),
        'D_S': format_distance(D_S_m, precision),
        'D_LS': format_distance(D_LS_m, precision),
        'theta_E': format_angle(theta_E_rad, precision),
        'R_E': format_radius(R_E_m, precision),
        'beta': format_angle(beta_rad, precision),
        'R_beta': format_radius(R_beta_m, precision),
    }
    
    if mass_kg is not None:
        result['M'] = format_mass(mass_kg, precision)
        r_s = schwarzschild_radius(mass_kg)
        result['r_s'] = format_schwarzschild_radius(mass_kg, precision)
        result['R_E_over_r_s'] = FormattedValue(
            internal_value=R_E_m / r_s,
            display_value=R_E_m / r_s,
            display_unit='',
            display_string=f"{R_E_m / r_s:.{precision}g}",
            internal_unit='dimensionless'
        )
    
    return result
