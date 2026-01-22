"""
ObservablesBundle: Container for all observables (existing + extended).

Add-only design: This WRAPS existing image_positions without changing
how the core solvers work. Extended observables are optional add-ons.

Authors: Carmen N. Wrede, Lino P. Casu
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ImageSet:
    """Image positions for one background source."""
    positions: np.ndarray  # Shape (N, 2) for N images
    source_id: int = 0
    label: str = "source_0"
    
    @property
    def n_images(self) -> int:
        return len(self.positions)
    
    @property
    def n_constraints(self) -> int:
        return 2 * self.n_images


@dataclass
class FluxRatios:
    """Relative flux ratios between images (optional)."""
    ratios: np.ndarray  # N-1 independent ratios for N images
    uncertainties: Optional[np.ndarray] = None
    reference_image: int = 0
    
    @property
    def n_constraints(self) -> int:
        return len(self.ratios)


@dataclass
class TimeDelays:
    """Relative time delays between images (optional)."""
    delays: np.ndarray  # N-1 independent delays for N images (days)
    uncertainties: Optional[np.ndarray] = None
    reference_image: int = 0
    
    @property
    def n_constraints(self) -> int:
        return len(self.delays)


@dataclass
class ArcPoints:
    """Extended arc/ring points (optional)."""
    positions: np.ndarray  # Shape (M, 2) for M arc points
    source_id: int = 0  # Which source the arc belongs to
    
    @property
    def n_points(self) -> int:
        return len(self.positions)
    
    @property
    def n_constraints(self) -> int:
        return 2 * self.n_points


@dataclass
class ObservablesBundle:
    """
    Complete bundle of observables for a lens system.
    
    Design principle: image_positions (existing) + optional extras.
    The core solvers only need image_positions.
    Extended observables enable more complex models.
    """
    name: str
    image_sets: List[ImageSet] = field(default_factory=list)
    flux_ratios: Optional[FluxRatios] = None
    time_delays: Optional[TimeDelays] = None
    arc_points: Optional[ArcPoints] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_sources(self) -> int:
        return len(self.image_sets)
    
    @property
    def total_images(self) -> int:
        return sum(s.n_images for s in self.image_sets)
    
    def count_constraints(self) -> int:
        """Total constraints from all observables."""
        n = sum(s.n_constraints for s in self.image_sets)
        if self.flux_ratios:
            n += self.flux_ratios.n_constraints
        if self.time_delays:
            n += self.time_delays.n_constraints
        if self.arc_points:
            n += self.arc_points.n_constraints
        return n
    
    def get_primary_images(self) -> np.ndarray:
        """Get images from first source (for existing solvers)."""
        if self.image_sets:
            return self.image_sets[0].positions
        return np.array([])
    
    def has_extended_observables(self) -> bool:
        """Check if any extended observables are present."""
        return (self.flux_ratios is not None or
                self.time_delays is not None or
                self.arc_points is not None or
                len(self.image_sets) > 1)
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"ObservablesBundle: {self.name}"]
        lines.append(f"  Sources: {self.n_sources}")
        lines.append(f"  Total images: {self.total_images}")
        lines.append(f"  Total constraints: {self.count_constraints()}")
        if self.flux_ratios:
            lines.append(f"  Flux ratios: {self.flux_ratios.n_constraints}")
        if self.time_delays:
            lines.append(f"  Time delays: {self.time_delays.n_constraints}")
        if self.arc_points:
            lines.append(f"  Arc points: {self.arc_points.n_points}")
        return "\n".join(lines)
    
    @classmethod
    def from_positions(cls, name: str, positions: np.ndarray) -> 'ObservablesBundle':
        """Create bundle from simple position array (backward compatible)."""
        return cls(
            name=name,
            image_sets=[ImageSet(positions=positions, source_id=0)]
        )
