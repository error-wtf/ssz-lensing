"""
TriadScene: 3D representation of Observer-Lens-Source geometry.

This makes the derivation visible:
    3D Scene -> Projection -> Lens Equation -> Inversion

Convention:
    - Observer at origin: O = (0, 0, 0)
    - Lens on optical axis: L = (0, 0, D_L)
    - Source behind lens: S = (x_S, y_S, D_S)

Authors: Carmen N. Wrede, Lino P. Casu
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class Position3D:
    """3D position in the scene."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    label: str = ""
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def distance_from_origin(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {'x': self.x, 'y': self.y, 'z': self.z, 'label': self.label}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Position3D':
        return cls(x=d['x'], y=d['y'], z=d['z'], label=d.get('label', ''))


@dataclass
class LensProperties:
    """Properties of the gravitational lens."""
    position: Position3D
    einstein_radius: float = 1.0  # In angular units (arcsec)
    position_angle: float = 0.0   # PA in radians
    ellipticity: float = 0.0
    redshift: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position': self.position.to_dict(),
            'einstein_radius': self.einstein_radius,
            'position_angle': self.position_angle,
            'ellipticity': self.ellipticity,
            'redshift': self.redshift
        }


@dataclass
class SourceProperties:
    """Properties of a background source."""
    position: Position3D
    source_id: int = 0
    redshift: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position': self.position.to_dict(),
            'source_id': self.source_id,
            'redshift': self.redshift
        }


@dataclass
class LensPlaneSetup:
    """Projected 2D lensing setup derived from 3D scene."""
    beta: np.ndarray          # Source position angle (x, y) in arcsec
    D_L: float                # Angular diameter distance to lens
    D_S: float                # Angular diameter distance to source
    D_LS: float               # Angular diameter distance lens-source
    theta_E: float            # Einstein radius (angular)
    source_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'beta': self.beta.tolist(),
            'D_L': self.D_L,
            'D_S': self.D_S,
            'D_LS': self.D_LS,
            'theta_E': self.theta_E,
            'source_id': self.source_id
        }


@dataclass 
class TriadScene:
    """
    Complete 3D scene: Observer + Lens + Source(s).
    
    This is the geometry layer that makes derivation visible:
    - Where is the observer?
    - Where is the lens?
    - Where are the sources?
    - How do these project to the lens equation coordinates?
    """
    name: str
    observer: Position3D = field(default_factory=lambda: Position3D(0, 0, 0, "Observer"))
    lens: LensProperties = None
    sources: List[SourceProperties] = field(default_factory=list)
    
    # Cosmology (optional, for future use)
    H0: float = 70.0  # km/s/Mpc
    Omega_m: float = 0.3
    Omega_Lambda: float = 0.7
    
    def __post_init__(self):
        if self.lens is None:
            self.lens = LensProperties(
                position=Position3D(0, 0, 1.0, "Lens")
            )
    
    def add_source(self, x: float, y: float, z: float, 
                   source_id: int = None, redshift: float = None):
        """Add a source to the scene."""
        if source_id is None:
            source_id = len(self.sources)
        pos = Position3D(x, y, z, f"Source_{source_id}")
        self.sources.append(SourceProperties(
            position=pos, source_id=source_id, redshift=redshift
        ))
    
    def get_optical_axis(self) -> np.ndarray:
        """Get the optical axis direction (Observer -> Lens)."""
        return self.lens.position.to_array() - self.observer.to_array()
    
    def get_distances(self) -> Dict[str, float]:
        """Get all relevant distances (euclidean for now)."""
        D_L = self.lens.position.distance_from_origin()
        distances = {'D_L': D_L}
        
        for src in self.sources:
            D_S = src.position.distance_from_origin()
            D_LS = D_S - D_L  # Simplified euclidean
            distances[f'D_S_{src.source_id}'] = D_S
            distances[f'D_LS_{src.source_id}'] = D_LS
        
        return distances
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize scene to dictionary."""
        return {
            'name': self.name,
            'observer': self.observer.to_dict(),
            'lens': self.lens.to_dict(),
            'sources': [s.to_dict() for s in self.sources],
            'cosmology': {
                'H0': self.H0,
                'Omega_m': self.Omega_m,
                'Omega_Lambda': self.Omega_Lambda
            }
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, filepath: str):
        """Save scene to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TriadScene':
        """Load scene from dictionary."""
        scene = cls(name=d['name'])
        scene.observer = Position3D.from_dict(d['observer'])
        
        lens_d = d['lens']
        scene.lens = LensProperties(
            position=Position3D.from_dict(lens_d['position']),
            einstein_radius=lens_d.get('einstein_radius', 1.0),
            position_angle=lens_d.get('position_angle', 0.0),
            ellipticity=lens_d.get('ellipticity', 0.0),
            redshift=lens_d.get('redshift')
        )
        
        for src_d in d.get('sources', []):
            scene.sources.append(SourceProperties(
                position=Position3D.from_dict(src_d['position']),
                source_id=src_d.get('source_id', 0),
                redshift=src_d.get('redshift')
            ))
        
        if 'cosmology' in d:
            scene.H0 = d['cosmology'].get('H0', 70.0)
            scene.Omega_m = d['cosmology'].get('Omega_m', 0.3)
            scene.Omega_Lambda = d['cosmology'].get('Omega_Lambda', 0.7)
        
        return scene
    
    @classmethod
    def load(cls, filepath: str) -> 'TriadScene':
        """Load scene from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))
    
    @classmethod
    def create_standard(cls, name: str, D_L: float = 1.0, D_S: float = 2.0,
                        beta_x: float = 0.1, beta_y: float = -0.05,
                        theta_E: float = 1.0) -> 'TriadScene':
        """
        Create standard scene with one source.
        
        Args:
            D_L: Distance to lens (normalized)
            D_S: Distance to source (normalized)
            beta_x, beta_y: Source offset from optical axis
            theta_E: Einstein radius
        """
        scene = cls(name=name)
        scene.lens = LensProperties(
            position=Position3D(0, 0, D_L, "Lens"),
            einstein_radius=theta_E
        )
        scene.add_source(beta_x * D_S, beta_y * D_S, D_S)
        return scene
