"""
Dataset Generation and Standard Test Cases

Provides:
- Synthetic data generation with known ground truth
- Standard test cases (Einstein Cross Q2237+0305)
- Noise injection for robustness testing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def generate_cross_images(
    theta_E: float = 1.0,
    beta: float = 0.1,
    phi_beta: float = 0.0,
    a: float = 0.0,
    b: float = 0.15,
    phi_gamma: float = 0.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate synthetic Einstein Cross image positions.
    
    Uses the local quadrupole model to create 4 images
    from a source at offset (beta, phi_beta) from the lens center.
    
    Parameters
    ----------
    theta_E : float
        Einstein radius
    beta : float
        Source offset magnitude
    phi_beta : float
        Source offset angle (radians)
    a : float
        Radial quadrupole coefficient
    b : float
        Tangential quadrupole coefficient
    phi_gamma : float
        Quadrupole axis angle (radians)
    
    Returns
    -------
    images : ndarray, shape (4, 2)
        Image positions
    params : dict
        True parameters used for generation
    """
    beta_x = beta * np.cos(phi_beta)
    beta_y = beta * np.sin(phi_beta)
    
    params = {
        'theta_E': theta_E,
        'beta_x': beta_x,
        'beta_y': beta_y,
        'beta': beta,
        'phi_beta': phi_beta,
        'a': a,
        'b': b,
        'phi_gamma': phi_gamma
    }
    
    images = []
    
    for k in range(4):
        phi_k = phi_gamma + (k * np.pi / 2) + np.pi / 4
        
        r_k = theta_E * (1 + a + b * np.cos(2 * (phi_k - phi_gamma)))
        
        delta_r = beta * np.cos(phi_k - phi_beta) / (1 - a)
        r_k += delta_r
        
        x_k = r_k * np.cos(phi_k)
        y_k = r_k * np.sin(phi_k)
        
        images.append([x_k, y_k])
    
    return np.array(images), params


def einstein_cross_q2237() -> Tuple[np.ndarray, Dict]:
    """
    Return observed image positions for Q2237+0305 (Einstein Cross).
    
    This is real observational data normalized to Einstein radius ~ 0.9".
    
    Returns
    -------
    images : ndarray
        Image positions in arcseconds
    info : dict
        Metadata about the system
    """
    images = np.array([
        [+0.673, +0.619],
        [-0.673, +0.619],
        [-0.619, -0.673],
        [+0.619, -0.673]
    ])
    
    info = {
        'name': 'Q2237+0305',
        'alias': 'Einstein Cross',
        'source_redshift': 1.695,
        'lens_redshift': 0.039,
        'reference': 'Huchra et al. 1985',
        'notes': 'Positions approximate, normalized coordinates'
    }
    
    return images, info


def generate_ring_images(
    theta_E: float = 1.0,
    n_points: int = 36,
    noise: float = 0.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate points on an Einstein ring.
    
    Useful for testing ring-only models or as initialization.
    
    Parameters
    ----------
    theta_E : float
        Einstein radius
    n_points : int
        Number of points around the ring
    noise : float
        Standard deviation of position noise
    
    Returns
    -------
    images : ndarray
        Points on ring
    params : dict
        Parameters (just theta_E for ring)
    """
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    r = theta_E * np.ones(n_points)
    if noise > 0:
        r += np.random.normal(0, noise, n_points)
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    if noise > 0:
        x += np.random.normal(0, noise, n_points)
        y += np.random.normal(0, noise, n_points)
    
    images = np.column_stack([x, y])
    params = {'theta_E': theta_E, 'beta_x': 0.0, 'beta_y': 0.0}
    
    return images, params


def generate_multipole_images(
    theta_E: float = 1.0,
    beta_x: float = 0.05,
    beta_y: float = 0.05,
    multipoles: Dict[int, Tuple[float, float]] = None,
    n_images: int = 4
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate images from general multipole model.
    
    Parameters
    ----------
    theta_E : float
        Einstein radius
    beta_x, beta_y : float
        Source position
    multipoles : dict
        {m: (amplitude, phase)} for each multipole order
    n_images : int
        Number of images to generate
    
    Returns
    -------
    images : ndarray
        Image positions
    params : dict
        Full parameter set
    """
    if multipoles is None:
        multipoles = {2: (0.1, 0.0)}
    
    params = {
        'theta_E': theta_E,
        'beta_x': beta_x,
        'beta_y': beta_y
    }
    
    for m, (amp, phase) in multipoles.items():
        params[f'amp_{m}'] = amp
        params[f'phi_{m}'] = phase
    
    images = []
    
    for k in range(n_images):
        phi_k = 2 * np.pi * k / n_images
        
        r_k = theta_E
        for m, (amp, phase) in multipoles.items():
            r_k += theta_E * amp * np.cos(m * (phi_k - phase))
        
        beta_r = np.sqrt(beta_x**2 + beta_y**2)
        beta_phi = np.arctan2(beta_y, beta_x)
        r_k += beta_r * np.cos(phi_k - beta_phi)
        
        x_k = r_k * np.cos(phi_k)
        y_k = r_k * np.sin(phi_k)
        images.append([x_k, y_k])
    
    return np.array(images), params


def add_noise(
    images: np.ndarray,
    sigma: float = 0.01,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian noise to image positions.
    
    Parameters
    ----------
    images : ndarray
        Clean image positions
    sigma : float
        Standard deviation of noise (same units as images)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    noisy : ndarray
        Noisy image positions
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(0, sigma, images.shape)
    return images + noise


def perturb_params(
    params: Dict[str, float],
    perturbations: Dict[str, float]
) -> Dict[str, float]:
    """
    Add perturbations to parameters.
    
    Useful for testing sensitivity or generating nearby test cases.
    
    Parameters
    ----------
    params : dict
        Original parameters
    perturbations : dict
        {param_name: delta} to add to each parameter
    
    Returns
    -------
    perturbed : dict
        Modified parameters
    """
    result = params.copy()
    for key, delta in perturbations.items():
        if key in result:
            result[key] += delta
    return result


def standard_test_cases() -> List[Dict]:
    """
    Return list of standard test configurations.
    
    Each test case includes:
    - name: identifier
    - images: image positions
    - true_params: known parameters (for synthetic)
    - description: what the test checks
    """
    cases = []
    
    images, params = generate_cross_images(
        theta_E=1.0, beta=0.1, phi_beta=0.3,
        a=0.0, b=0.15, phi_gamma=0.5
    )
    cases.append({
        'name': 'synthetic_standard',
        'images': images,
        'true_params': params,
        'description': 'Standard 4-image configuration for exact recovery test'
    })
    
    images, params = generate_cross_images(
        theta_E=1.0, beta=0.05, phi_beta=0.0,
        a=0.0, b=0.1, phi_gamma=0.0
    )
    cases.append({
        'name': 'symmetric_cross',
        'images': images,
        'true_params': params,
        'description': 'Nearly symmetric configuration (aligned axes)'
    })
    
    images, params = generate_cross_images(
        theta_E=1.0, beta=0.2, phi_beta=0.7,
        a=0.1, b=0.2, phi_gamma=0.8
    )
    cases.append({
        'name': 'asymmetric_cross',
        'images': images,
        'true_params': params,
        'description': 'Asymmetric configuration with radial term'
    })
    
    images, info = einstein_cross_q2237()
    cases.append({
        'name': 'Q2237+0305',
        'images': images,
        'true_params': None,
        'description': 'Real data from Einstein Cross quasar',
        'info': info
    })
    
    return cases


def validate_images(images: np.ndarray) -> Tuple[bool, str]:
    """
    Validate image array format and basic properties.
    
    Returns
    -------
    valid : bool
        Whether images pass validation
    message : str
        Description of any issues
    """
    if not isinstance(images, np.ndarray):
        return False, "Images must be numpy array"
    
    if images.ndim != 2:
        return False, f"Images must be 2D, got {images.ndim}D"
    
    if images.shape[1] != 2:
        return False, f"Images must have shape (n, 2), got {images.shape}"
    
    if len(images) < 1:
        return False, "Need at least 1 image"
    
    if np.any(np.isnan(images)):
        return False, "Images contain NaN values"
    
    if np.any(np.isinf(images)):
        return False, "Images contain infinite values"
    
    radii = np.sqrt(images[:, 0]**2 + images[:, 1]**2)
    if np.std(radii) / np.mean(radii) > 0.5:
        return True, "Warning: large radial variation (may indicate non-lensing)"
    
    return True, "Images valid"
