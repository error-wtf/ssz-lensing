"""
Morphology Classifier: Ring vs Cross (and everything in between).

Classifies lensing observations into:
- RING: Rotationally symmetric (Einstein Ring)
- QUAD: Broken symmetry (Einstein Cross)
- ARC: Partial ring / giant arcs
- DOUBLE: Two-image configuration

Authors: Carmen N. Wrede, Lino P. Casu
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


class Morphology(Enum):
    """Primary morphology classification."""
    RING = "ring"           # Full Einstein ring
    QUAD = "quad"           # Einstein cross (4 images)
    ARC = "arc"             # Partial ring / giant arcs
    DOUBLE = "double"       # Two-image system
    UNKNOWN = "unknown"


@dataclass
class MorphologyAnalysis:
    """Result of morphology classification."""
    primary: Morphology
    confidence: float
    
    # Geometric metrics
    mean_radius: float
    radial_scatter: float       # σ(r) / r_mean
    azimuthal_coverage: float   # Fraction of 2π covered
    azimuthal_uniformity: float # How evenly distributed
    
    # Harmonic analysis
    m2_amplitude: float         # cos(2φ) component
    m4_amplitude: float         # cos(4φ) component
    
    # Recommendations
    recommended_models: List[str]
    notes: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary': self.primary.value,
            'confidence': self.confidence,
            'mean_radius': self.mean_radius,
            'radial_scatter': self.radial_scatter,
            'azimuthal_coverage': self.azimuthal_coverage,
            'azimuthal_uniformity': self.azimuthal_uniformity,
            'm2_amplitude': self.m2_amplitude,
            'm4_amplitude': self.m4_amplitude,
            'recommended_models': self.recommended_models,
            'notes': self.notes
        }


class MorphologyClassifier:
    """
    Classify lensing morphology from observed positions.
    
    Ring indicators:
    - Many points with small radial scatter
    - Good azimuthal coverage
    
    Cross indicators:
    - 4 discrete images
    - Moderate radial scatter
    - Azimuthal clustering
    """
    
    # Thresholds
    RING_RADIAL_SCATTER = 0.05    # σ(r)/r < 5% => ring-like
    RING_AZIMUTHAL_COV = 0.7      # > 70% coverage => ring
    QUAD_AZIMUTHAL_CLUSTER = 0.3  # < 30% coverage => discrete images
    
    def __init__(self, center: Tuple[float, float] = (0.0, 0.0)):
        self.center = np.array(center)
    
    def classify(self, positions: np.ndarray) -> MorphologyAnalysis:
        """
        Classify morphology from image/arc positions.
        
        Args:
            positions: Nx2 array of (x, y) positions
        """
        n_points = len(positions)
        
        # Convert to polar relative to center
        rel = positions - self.center
        r = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)
        phi = np.arctan2(rel[:, 1], rel[:, 0])
        
        # Radial statistics
        r_mean = np.mean(r)
        r_std = np.std(r)
        radial_scatter = r_std / r_mean if r_mean > 0 else 1.0
        
        # Azimuthal coverage
        phi_sorted = np.sort(phi)
        gaps = np.diff(phi_sorted)
        gaps = np.append(gaps, 2*np.pi + phi_sorted[0] - phi_sorted[-1])
        max_gap = np.max(gaps)
        azimuthal_coverage = 1.0 - max_gap / (2*np.pi)
        
        # Azimuthal uniformity (for n points, ideal gap = 2π/n)
        ideal_gap = 2*np.pi / n_points
        gap_variance = np.var(gaps)
        azimuthal_uniformity = 1.0 / (1.0 + gap_variance / ideal_gap**2)
        
        # Harmonic analysis: fit r(φ) = r0 + a2*cos(2φ) + b2*sin(2φ) + ...
        m2_cos = np.mean((r - r_mean) * np.cos(2*phi))
        m2_sin = np.mean((r - r_mean) * np.sin(2*phi))
        m2_amplitude = np.sqrt(m2_cos**2 + m2_sin**2) / r_mean
        
        m4_cos = np.mean((r - r_mean) * np.cos(4*phi))
        m4_sin = np.mean((r - r_mean) * np.sin(4*phi))
        m4_amplitude = np.sqrt(m4_cos**2 + m4_sin**2) / r_mean
        
        # Classification logic
        primary, confidence, notes = self._classify_primary(
            n_points, radial_scatter, azimuthal_coverage, 
            azimuthal_uniformity, m2_amplitude, m4_amplitude
        )
        
        # Recommend models based on morphology
        recommended = self._recommend_models(
            primary, radial_scatter, m2_amplitude, m4_amplitude
        )
        
        return MorphologyAnalysis(
            primary=primary,
            confidence=confidence,
            mean_radius=r_mean,
            radial_scatter=radial_scatter,
            azimuthal_coverage=azimuthal_coverage,
            azimuthal_uniformity=azimuthal_uniformity,
            m2_amplitude=m2_amplitude,
            m4_amplitude=m4_amplitude,
            recommended_models=recommended,
            notes=notes
        )
    
    def _classify_primary(self, n_points, radial_scatter, azimuthal_coverage,
                          azimuthal_uniformity, m2_amp, m4_amp):
        """Determine primary morphology."""
        notes = []
        
        # Quad: exactly 4 points is always QUAD (discrete images)
        if n_points == 4:
            conf = 0.9
            notes.append("Quad: 4 discrete images")
            return Morphology.QUAD, conf, notes
        
        # Double: 2 points
        if n_points == 2:
            notes.append("Double: two-image system")
            return Morphology.DOUBLE, 0.9, notes
        
        # Ring: many points (>4), small radial scatter, good coverage
        if (n_points > 4 and radial_scatter < self.RING_RADIAL_SCATTER and 
            azimuthal_coverage > self.RING_AZIMUTHAL_COV):
            conf = 1.0 - radial_scatter / self.RING_RADIAL_SCATTER
            notes.append("Ring-like: small radial scatter, good coverage")
            if m2_amp > 0.01:
                notes.append(f"m=2 perturbation detected: {m2_amp:.3f}")
            return Morphology.RING, min(conf, 0.95), notes
        
        # Arc: partial coverage with many points
        if n_points > 4 and azimuthal_coverage < self.RING_AZIMUTHAL_COV:
            conf = azimuthal_uniformity
            notes.append("Quad-like: 4 discrete images")
            return Morphology.QUAD, min(conf, 0.95), notes
        
        # Arc: partial coverage
        if (radial_scatter < 0.15 and 
            self.QUAD_AZIMUTHAL_CLUSTER < azimuthal_coverage < self.RING_AZIMUTHAL_COV):
            conf = 0.7
            notes.append("Arc-like: partial ring structure")
            return Morphology.ARC, conf, notes
        
        # Double: 2 points
        if n_points == 2:
            notes.append("Double: two-image system")
            return Morphology.DOUBLE, 0.9, notes
        
        # Default to quad for 4 points, unknown otherwise
        if n_points == 4:
            notes.append("Assuming quad from 4 points")
            return Morphology.QUAD, 0.5, notes
        
        notes.append("Morphology unclear")
        return Morphology.UNKNOWN, 0.3, notes
    
    def _recommend_models(self, morphology, radial_scatter, m2_amp, m4_amp):
        """Recommend models based on morphology."""
        models = []
        
        if morphology == Morphology.RING:
            models.append("isotropic")
            if m2_amp > 0.005:
                models.append("isotropic+shear")
                models.append("isotropic+m2")
            if m4_amp > 0.005:
                models.append("isotropic+m4")
        
        elif morphology == Morphology.QUAD:
            models.append("m2")
            models.append("m2+shear")
            if m2_amp > 0.02 or radial_scatter > 0.05:
                models.append("m2+m3")
                models.append("m2+shear+m3")
            if m4_amp > 0.01:
                models.append("m2+m4")
        
        elif morphology == Morphology.ARC:
            models.append("isotropic")
            models.append("m2")
            models.append("m2+shear")
        
        else:
            models.append("m2")
            models.append("m2+shear")
        
        return models


def estimate_ring_center(positions: np.ndarray) -> Tuple[float, float]:
    """
    Algebraic circle center estimation from points.
    
    Uses least-squares fit to find (xc, yc) minimizing
    sum of (r_i - R)^2.
    """
    n = len(positions)
    if n < 3:
        return (np.mean(positions[:, 0]), np.mean(positions[:, 1]))
    
    # Algebraic circle fit (Kasa method)
    x = positions[:, 0]
    y = positions[:, 1]
    
    A = np.column_stack([x, y, np.ones(n)])
    b = x**2 + y**2
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        xc = coeffs[0] / 2
        yc = coeffs[1] / 2
        return (xc, yc)
    except:
        return (np.mean(x), np.mean(y))


def estimate_ring_radius(positions: np.ndarray, 
                         center: Optional[Tuple[float, float]] = None) -> float:
    """Estimate Einstein ring radius from positions."""
    if center is None:
        center = estimate_ring_center(positions)
    
    rel = positions - np.array(center)
    r = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)
    return np.median(r)
