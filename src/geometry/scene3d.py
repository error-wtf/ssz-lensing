"""
3D Lensing Scene: Observer - Lens - Source Geometry

Shows the physical setup with:
- Distances: D_L, D_S, D_LS
- Radii/Scales: theta_E, R_E = D_L * theta_E, beta, R_beta = D_S * |beta|
- Ray geometry from observer through lens plane to source

Convention (stable & simple):
- Observer at origin (0, 0, 0)
- Lens on z-axis at (0, 0, D_L)
- Source at (x_S, y_S, D_S)

Authors: Carmen N. Wrede, Lino P. Casu
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import json


@dataclass
class Vec3:
    """3D vector."""
    x: float
    y: float
    z: float
    
    def __array__(self):
        return np.array([self.x, self.y, self.z])
    
    def to_list(self):
        return [self.x, self.y, self.z]
    
    @classmethod
    def from_array(cls, arr):
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def norm(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalized(self):
        n = self.norm()
        if n < 1e-15:
            return Vec3(0, 0, 1)
        return Vec3(self.x/n, self.y/n, self.z/n)


@dataclass
class Scene3D:
    """
    3D Lensing Scene with Observer, Lens, Source.
    
    Convention:
    - Observer at origin (0, 0, 0)
    - Lens at (0, 0, D_L) - on optical axis
    - Source at (beta_x * D_S, beta_y * D_S, D_S) for small angles
    
    Parameters
    ----------
    D_L : float
        Distance to lens (observer-lens)
    D_S : float
        Distance to source (observer-source)
    theta_E : float
        Einstein angle (radians)
    beta : tuple
        Source offset angle (beta_x, beta_y) in radians
    units : str
        Distance units for display ('Mpc', 'pc', 'kpc', 'normalized')
    z_L : float, optional
        Lens redshift (for cosmological calculations)
    z_S : float, optional
        Source redshift
    lens_mass : float, optional
        Lens mass in solar masses (for R_s calculation)
    """
    D_L: float
    D_S: float
    theta_E: float
    beta: Tuple[float, float] = (0.0, 0.0)
    units: str = 'normalized'
    z_L: Optional[float] = None
    z_S: Optional[float] = None
    lens_mass: Optional[float] = None
    
    # Computed properties
    observer: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    lens: Vec3 = field(init=False)
    source: Vec3 = field(init=False)
    
    def __post_init__(self):
        # Lens on z-axis
        self.lens = Vec3(0, 0, self.D_L)
        
        # Source position from beta angles (small angle approximation)
        beta_x, beta_y = self.beta
        self.source = Vec3(beta_x * self.D_S, beta_y * self.D_S, self.D_S)
    
    @property
    def D_LS(self) -> float:
        """Lens-Source distance."""
        return self.D_S - self.D_L
    
    @property
    def R_E(self) -> float:
        """Einstein radius on lens plane: R_E = D_L * theta_E"""
        return self.D_L * self.theta_E
    
    @property
    def beta_magnitude(self) -> float:
        """Source offset angle magnitude."""
        return np.sqrt(self.beta[0]**2 + self.beta[1]**2)
    
    @property
    def R_beta(self) -> float:
        """Source offset radius on source plane: R_beta = D_S * |beta|"""
        return self.D_S * self.beta_magnitude
    
    @property
    def R_s(self) -> Optional[float]:
        """Schwarzschild radius if mass is given (in same units as D_L)."""
        if self.lens_mass is None:
            return None
        # R_s = 2GM/c^2
        # For M in solar masses, R_s in pc: R_s ≈ 9.57e-14 * M pc
        # For M in solar masses, R_s in Mpc: R_s ≈ 9.57e-20 * M Mpc
        G = 6.674e-11  # m^3 kg^-1 s^-2
        c = 3e8  # m/s
        M_sun = 1.989e30  # kg
        pc_to_m = 3.086e16
        
        R_s_m = 2 * G * self.lens_mass * M_sun / c**2
        
        if self.units == 'pc':
            return R_s_m / pc_to_m
        elif self.units == 'kpc':
            return R_s_m / (pc_to_m * 1e3)
        elif self.units == 'Mpc':
            return R_s_m / (pc_to_m * 1e6)
        else:
            return R_s_m  # meters or normalized
    
    def to_dict(self) -> Dict:
        """Export scene to dictionary for JSON saving."""
        return {
            'observer': self.observer.to_list(),
            'lens': self.lens.to_list(),
            'source': self.source.to_list(),
            'D_L': self.D_L,
            'D_S': self.D_S,
            'D_LS': self.D_LS,
            'theta_E': self.theta_E,
            'R_E': self.R_E,
            'beta': list(self.beta),
            'beta_magnitude': self.beta_magnitude,
            'R_beta': self.R_beta,
            'units': self.units,
            'z_L': self.z_L,
            'z_S': self.z_S,
            'lens_mass': self.lens_mass,
            'R_s': self.R_s
        }
    
    def save(self, path: str):
        """Save scene to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Scene3D':
        """Load scene from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            D_L=data['D_L'],
            D_S=data['D_S'],
            theta_E=data['theta_E'],
            beta=tuple(data['beta']),
            units=data.get('units', 'normalized'),
            z_L=data.get('z_L'),
            z_S=data.get('z_S'),
            lens_mass=data.get('lens_mass')
        )
    
    @classmethod
    def from_quicklook(cls, theta_E: float, beta: Tuple[float, float],
                       D_L: float = 1.0, D_S: float = 2.0,
                       units: str = 'normalized') -> 'Scene3D':
        """Create scene from Quicklook/Inversion results."""
        return cls(D_L=D_L, D_S=D_S, theta_E=theta_E, beta=beta, units=units)
    
    @classmethod
    def normalized(cls, theta_E: float = 1.0, 
                   beta: Tuple[float, float] = (0.1, 0.0)) -> 'Scene3D':
        """Create scene in normalized units (D_L=1, D_S=2)."""
        return cls(D_L=1.0, D_S=2.0, theta_E=theta_E, beta=beta, 
                   units='normalized')


# =============================================================================
# PROJECTION HELPERS: 3D <-> 2D Lensing
# =============================================================================

def angles_from_scene(scene: Scene3D) -> Tuple[float, float]:
    """
    Compute source offset angle beta from 3D scene.
    
    beta ≈ (x_S/D_S, y_S/D_S) in small angle approximation.
    """
    return (scene.source.x / scene.D_S, scene.source.y / scene.D_S)


def lens_plane_scale(scene: Scene3D, theta: float) -> float:
    """
    Convert angle to physical radius on lens plane.
    
    R = D_L * theta
    """
    return scene.D_L * theta


def source_plane_scale(scene: Scene3D, beta: float) -> float:
    """
    Convert angle to physical radius on source plane.
    
    R = D_S * beta
    """
    return scene.D_S * beta


def ray_to_lens_plane(scene: Scene3D, theta_x: float, theta_y: float) -> Vec3:
    """
    Compute where ray from observer at angle (theta_x, theta_y) 
    intersects the lens plane (z = D_L).
    
    Returns intersection point.
    """
    # Ray direction (small angle: tan(theta) ≈ theta)
    direction = Vec3(theta_x, theta_y, 1.0).normalized()
    
    # Parametric ray: P = O + t * direction
    # Intersection with z = D_L: t = D_L / direction.z
    t = scene.D_L / direction.z
    
    return Vec3(
        scene.observer.x + t * direction.x,
        scene.observer.y + t * direction.y,
        scene.D_L
    )


def image_positions_to_rays(scene: Scene3D, 
                            theta_positions: np.ndarray) -> List[Tuple[Vec3, Vec3]]:
    """
    Convert 2D image positions (angles) to 3D ray segments.
    
    Returns list of (start, end) points for each ray.
    """
    rays = []
    for theta in theta_positions:
        theta_x, theta_y = theta[0], theta[1]
        intersection = ray_to_lens_plane(scene, theta_x, theta_y)
        rays.append((scene.observer, intersection))
    return rays


def compute_scene_summary(scene: Scene3D) -> str:
    """Generate human-readable summary of scene geometry."""
    lines = [
        "=" * 50,
        "3D LENSING SCENE GEOMETRY",
        "=" * 50,
        "",
        f"Units: {scene.units}",
        "",
        "POSITIONS:",
        f"  Observer (O): ({scene.observer.x:.4f}, {scene.observer.y:.4f}, {scene.observer.z:.4f})",
        f"  Lens (L):     ({scene.lens.x:.4f}, {scene.lens.y:.4f}, {scene.lens.z:.4f})",
        f"  Source (S):   ({scene.source.x:.4f}, {scene.source.y:.4f}, {scene.source.z:.4f})",
        "",
        "DISTANCES:",
        f"  D_L  (O-L):   {scene.D_L:.4f} {scene.units}",
        f"  D_S  (O-S):   {scene.D_S:.4f} {scene.units}",
        f"  D_LS (L-S):   {scene.D_LS:.4f} {scene.units}",
        "",
        "ANGLES:",
        f"  theta_E:      {scene.theta_E:.6f} rad = {np.degrees(scene.theta_E)*3600:.2f} arcsec",
        f"  |beta|:       {scene.beta_magnitude:.6f} rad = {np.degrees(scene.beta_magnitude)*3600:.2f} arcsec",
        "",
        "PHYSICAL RADII:",
        f"  R_E = D_L * theta_E:  {scene.R_E:.6f} {scene.units}",
        f"  R_beta = D_S * |beta|: {scene.R_beta:.6f} {scene.units}",
    ]
    
    if scene.R_s is not None:
        lines.extend([
            "",
            "SCHWARZSCHILD RADIUS:",
            f"  R_s:          {scene.R_s:.2e} {scene.units}",
            f"  R_E / R_s:    {scene.R_E / scene.R_s:.2e}",
        ])
    
    if scene.z_L is not None or scene.z_S is not None:
        lines.extend([
            "",
            "REDSHIFTS:",
            f"  z_L: {scene.z_L if scene.z_L else 'N/A'}",
            f"  z_S: {scene.z_S if scene.z_S else 'N/A'}",
        ])
    
    lines.append("=" * 50)
    return "\n".join(lines)
