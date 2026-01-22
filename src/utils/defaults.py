"""Deterministic default generators for No-Null/No-NaN contract.

All functions return numeric values (never null/NaN) plus provenance info.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# Default parameters (tuned for arcsec-scale lensing)
EPS_SCALE = 1e-3          # 0.1% of characteristic scale
SX_MIN_ARCSEC = 1e-6      # Minimum positional uncertainty
NN_FRACTION = 0.1         # Fraction of nearest-neighbor distance for RING

# Normalized distance defaults
D_L_NORMALIZED = 1.0
D_S_NORMALIZED = 2.0
D_LS_NORMALIZED = 1.0


@dataclass
class ValueWithProvenance:
    """A numeric value with provenance tracking."""
    value: float
    is_measured: bool
    source: str
    
    def to_dict(self):
        return {
            'value': self.value,
            'is_measured': self.is_measured,
            'source': self.source
        }


def compute_default_sigma(
    positions: np.ndarray,
    mode: str = "QUAD"
) -> Tuple[float, str]:
    """
    Compute default positional uncertainty from point distribution.
    
    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        Point positions
    mode : str
        "QUAD" or "RING"
        
    Returns
    -------
    sigma : float
        Default uncertainty (always > 0)
    source : str
        Provenance description
    """
    if len(positions) < 2:
        return SX_MIN_ARCSEC, "fallback_minimum_v1"
    
    if mode == "QUAD":
        # Use median radius from centroid
        center = np.mean(positions, axis=0)
        radii = np.sqrt(np.sum((positions - center)**2, axis=1))
        r_median = np.median(radii)
        sigma = max(r_median * EPS_SCALE, SX_MIN_ARCSEC)
        source = "auto_from_scale_v1"
    else:
        # RING/ARC: use nearest-neighbor distance
        from scipy.spatial.distance import cdist
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        nn_dists = np.min(dists, axis=1)
        median_nn = np.median(nn_dists)
        sigma = max(median_nn * NN_FRACTION, SX_MIN_ARCSEC)
        source = "auto_from_nn_distance_v1"
    
    return float(sigma), source


def fill_position_uncertainties(
    positions: np.ndarray,
    sx_input: List[float] = None,
    sy_input: List[float] = None,
    mode: str = "QUAD"
) -> Tuple[List[ValueWithProvenance], List[ValueWithProvenance]]:
    """
    Fill position uncertainties with defaults where missing.
    
    Returns lists of ValueWithProvenance for sx and sy.
    """
    n = len(positions)
    default_sigma, default_source = compute_default_sigma(positions, mode)
    
    sx_out = []
    sy_out = []
    
    for i in range(n):
        # sx
        if sx_input and i < len(sx_input) and sx_input[i] is not None and sx_input[i] > 0:
            sx_out.append(ValueWithProvenance(
                value=float(sx_input[i]),
                is_measured=True,
                source="user_input"
            ))
        else:
            sx_out.append(ValueWithProvenance(
                value=default_sigma,
                is_measured=False,
                source=default_source
            ))
        
        # sy
        if sy_input and i < len(sy_input) and sy_input[i] is not None and sy_input[i] > 0:
            sy_out.append(ValueWithProvenance(
                value=float(sy_input[i]),
                is_measured=True,
                source="user_input"
            ))
        else:
            sy_out.append(ValueWithProvenance(
                value=default_sigma,
                is_measured=False,
                source=default_source
            ))
    
    return sx_out, sy_out


def get_normalized_distances() -> dict:
    """Get normalized distance defaults with provenance."""
    return {
        'D_L': ValueWithProvenance(D_L_NORMALIZED, False, "normalized_defaults_v1"),
        'D_S': ValueWithProvenance(D_S_NORMALIZED, False, "normalized_defaults_v1"),
        'D_LS': ValueWithProvenance(D_LS_NORMALIZED, False, "normalized_defaults_v1"),
    }


def estimate_center(positions: np.ndarray) -> Tuple[ValueWithProvenance, ValueWithProvenance]:
    """
    Estimate lens center from positions.
    
    Returns (cx, cy) as ValueWithProvenance.
    """
    center = np.mean(positions, axis=0)
    return (
        ValueWithProvenance(float(center[0]), False, "estimated_from_points_v1"),
        ValueWithProvenance(float(center[1]), False, "estimated_from_points_v1")
    )


def estimate_beta_from_residuals(
    positions: np.ndarray,
    theta_E: float
) -> Tuple[ValueWithProvenance, ValueWithProvenance]:
    """
    Estimate source offset β from symmetry analysis.
    
    Simple heuristic: centroid offset from origin scaled by theta_E.
    """
    center = np.mean(positions, axis=0)
    # Rough estimate: centroid offset is related to beta
    beta_x = center[0] * 0.1  # Heuristic scaling
    beta_y = center[1] * 0.1
    
    return (
        ValueWithProvenance(float(beta_x), False, "estimated_from_symmetry_v1"),
        ValueWithProvenance(float(beta_y), False, "estimated_from_symmetry_v1")
    )


def estimate_theta_E(positions: np.ndarray) -> ValueWithProvenance:
    """Estimate Einstein radius from positions."""
    center = np.mean(positions, axis=0)
    radii = np.sqrt(np.sum((positions - center)**2, axis=1))
    theta_E = float(np.median(radii))
    return ValueWithProvenance(theta_E, False, "estimated_from_median_radius_v1")


@dataclass
class FullNumericPoint:
    """A point with all fields numeric (no nulls)."""
    id: str
    x: float
    y: float
    sx: float
    sy: float
    x_is_measured: bool
    y_is_measured: bool
    sx_is_measured: bool
    sy_is_measured: bool
    sx_source: str
    sy_source: str
    
    def to_dict(self):
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'sx': self.sx,
            'sy': self.sy,
            'x_is_measured': self.x_is_measured,
            'y_is_measured': self.y_is_measured,
            'sx_is_measured': self.sx_is_measured,
            'sy_is_measured': self.sy_is_measured,
            'sx_source': self.sx_source,
            'sy_source': self.sy_source
        }


def create_full_numeric_points(
    positions: np.ndarray,
    ids: List[str] = None,
    sx_input: List[float] = None,
    sy_input: List[float] = None,
    mode: str = "QUAD"
) -> List[FullNumericPoint]:
    """
    Create fully numeric point list with provenance.
    
    All fields will have values (no nulls).
    """
    n = len(positions)
    if ids is None:
        ids = [chr(65 + i) for i in range(n)]
    
    sx_prov, sy_prov = fill_position_uncertainties(positions, sx_input, sy_input, mode)
    
    points = []
    for i in range(n):
        points.append(FullNumericPoint(
            id=ids[i],
            x=float(positions[i, 0]),
            y=float(positions[i, 1]),
            sx=sx_prov[i].value,
            sy=sy_prov[i].value,
            x_is_measured=True,
            y_is_measured=True,
            sx_is_measured=sx_prov[i].is_measured,
            sy_is_measured=sy_prov[i].is_measured,
            sx_source=sx_prov[i].source,
            sy_source=sy_prov[i].source
        ))
    
    return points


def count_assumptions(points: List[FullNumericPoint]) -> dict:
    """Count how many values are assumed vs measured."""
    total = len(points)
    sx_assumed = sum(1 for p in points if not p.sx_is_measured)
    sy_assumed = sum(1 for p in points if not p.sy_is_measured)
    
    return {
        'total_points': total,
        'sx_assumed': sx_assumed,
        'sy_assumed': sy_assumed,
        'all_measured': sx_assumed == 0 and sy_assumed == 0
    }
