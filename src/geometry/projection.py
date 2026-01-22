"""
Projection: 3D Scene -> 2D Lensing Coordinates.

This is the bridge that makes derivation visible:
    TriadScene (3D) -> LensPlaneSetup (theta, beta, distances)

Authors: Carmen N. Wrede, Lino P. Casu
"""

import numpy as np
from typing import List, Dict, Any
from .triad_scene import TriadScene, LensPlaneSetup


def project_to_lens_plane(scene: TriadScene) -> List[LensPlaneSetup]:
    """
    Project 3D scene to 2D lensing coordinates.
    
    Takes the full 3D geometry and produces the exact (theta, beta, D)
    inputs used by the inversion solvers.
    
    Returns one LensPlaneSetup per source.
    """
    setups = []
    
    D_L = scene.lens.position.z
    theta_E = scene.lens.einstein_radius
    
    for src in scene.sources:
        D_S = src.position.z
        D_LS = D_S - D_L
        
        # Source angle as seen from observer (small angle approx)
        beta_x = src.position.x / D_S
        beta_y = src.position.y / D_S
        
        setup = LensPlaneSetup(
            beta=np.array([beta_x, beta_y]),
            D_L=D_L,
            D_S=D_S,
            D_LS=D_LS,
            theta_E=theta_E,
            source_id=src.source_id
        )
        setups.append(setup)
    
    return setups


def projection_to_dict(setups: List[LensPlaneSetup]) -> Dict[str, Any]:
    """Serialize projection results."""
    return {
        'n_sources': len(setups),
        'setups': [s.to_dict() for s in setups]
    }


def forward_lens_images(setup: LensPlaneSetup, 
                        multipole_params: Dict[str, float]) -> np.ndarray:
    """
    Forward model: given beta and lens params, compute image positions.
    
    This uses the reduced deflection model:
    theta = beta + alpha(theta)
    
    For quad lens, returns 4 image positions.
    """
    beta = setup.beta
    theta_E = setup.theta_E
    
    # Simple SIS + perturbation model
    # For exact images, solve lens equation
    # Here we use approximate positions for visualization
    
    c_2 = multipole_params.get('c_2', 0.0)
    s_2 = multipole_params.get('s_2', 0.0)
    
    # Approximate quad positions (for visualization)
    angles = np.array([0.3, 1.8, 3.5, 5.2])
    images = []
    
    for phi in angles:
        r = theta_E + c_2*np.cos(2*phi) + s_2*np.sin(2*phi)
        x = r * np.cos(phi) + beta[0]
        y = r * np.sin(phi) + beta[1]
        images.append([x, y])
    
    return np.array(images)


def backproject_to_source(theta: np.ndarray, setup: LensPlaneSetup,
                          multipole_params: Dict[str, float]) -> np.ndarray:
    """
    Back-project image position to source plane.
    
    beta = theta - alpha(theta)
    
    Used to check source-plane consistency.
    """
    theta_E = setup.theta_E
    c_2 = multipole_params.get('c_2', 0.0)
    s_2 = multipole_params.get('s_2', 0.0)
    
    betas = []
    for t in theta:
        x, y = t[0], t[1]
        phi = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        
        # Deflection
        alpha_r = theta_E + c_2*np.cos(2*phi) + s_2*np.sin(2*phi)
        alpha_x = alpha_r * np.cos(phi)
        alpha_y = alpha_r * np.sin(phi)
        
        beta_x = x - alpha_x
        beta_y = y - alpha_y
        betas.append([beta_x, beta_y])
    
    return np.array(betas)


class ProjectionTracer:
    """
    Traces the full projection chain for documentation.
    
    Records every step: 3D -> angles -> lens equation -> solution
    """
    
    def __init__(self, scene: TriadScene):
        self.scene = scene
        self.steps = []
    
    def trace(self) -> Dict[str, Any]:
        """Run full projection trace."""
        self.steps = []
        
        # Step 1: 3D positions
        self.steps.append({
            'step': '3D_positions',
            'observer': self.scene.observer.to_dict(),
            'lens': self.scene.lens.position.to_dict(),
            'sources': [s.position.to_dict() for s in self.scene.sources]
        })
        
        # Step 2: Distances
        distances = self.scene.get_distances()
        self.steps.append({
            'step': 'distances',
            'values': distances
        })
        
        # Step 3: Projection to angles
        setups = project_to_lens_plane(self.scene)
        self.steps.append({
            'step': 'projection',
            'setups': [s.to_dict() for s in setups]
        })
        
        return {
            'scene_name': self.scene.name,
            'n_steps': len(self.steps),
            'trace': self.steps
        }
