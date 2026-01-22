"""
Multi-Source / Same-Lens No-Fit Inversion Framework

Multiple background sources lensed by the SAME foreground lens share:
- θ_E (Einstein radius)
- Multipole components (a_m, b_m) for m = 2, 3, ...
- External shear (γ_1, γ_2)

Each source k has its own:
- β^(k) = (β_x^(k), β_y^(k)) source position

Key insight: K sources with 4 images each give 8K constraints.
With shared lens params, we add only 2 new params per source (β_x, β_y).

DOF scaling:
- 1 source:  8 constraints, 5 params (m=2) → +3 redundancy
- 2 sources: 16 constraints, 7 params (m=2) → +9 redundancy → can add shear + m=3!
- 3 sources: 24 constraints, 9 params (m=2) → +15 redundancy → full model possible

This is the mathematically clean way to extend the model without violating No-Fit.

Authors: Carmen N. Wrede, Lino P. Casu
License: Anti-Capitalist Software License v1.4
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MultiSourceParams:
    """
    Parameters for multi-source lensing.
    
    Shared lens parameters + per-source β.
    Phase is NEVER a free parameter - only (a_m, b_m) components.
    """
    # Shared lens parameters
    theta_E: float = 1.0
    
    # Shear components (NOT gamma, phi_gamma!)
    gamma_1: float = 0.0  # γ·cos(2φ_γ)
    gamma_2: float = 0.0  # γ·sin(2φ_γ)
    
    # Multipole components: a_m·cos(m·θ) + b_m·sin(m·θ)
    # No phase parameter - that's the whole point!
    multipoles: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    # multipoles[m] = (a_m, b_m)
    
    # Per-source positions
    betas: List[Tuple[float, float]] = field(default_factory=list)
    # betas[k] = (β_x^(k), β_y^(k))
    
    def get_amplitude_phase(self, m: int) -> Tuple[float, float]:
        """
        Derive amplitude and phase from components (OUTPUT, not input!).
        
        A_m = sqrt(a_m² + b_m²)
        φ_m = arctan2(b_m, a_m) / m
        """
        if m not in self.multipoles:
            return 0.0, 0.0
        a_m, b_m = self.multipoles[m]
        A_m = np.sqrt(a_m**2 + b_m**2)
        phi_m = np.arctan2(b_m, a_m) / m
        return A_m, phi_m
    
    def get_shear_amplitude_phase(self) -> Tuple[float, float]:
        """Derive shear amplitude and phase from components."""
        gamma = np.sqrt(self.gamma_1**2 + self.gamma_2**2)
        phi_gamma = np.arctan2(self.gamma_2, self.gamma_1) / 2
        return gamma, phi_gamma


class DOFGatekeeper:
    """
    Degrees of Freedom Gatekeeper - enforces No-Fit policy.
    
    Rule: n_params > n_constraints - 1 → FORBIDDEN
    
    This is not negotiable. Underdetermined systems are rejected.
    """
    
    @staticmethod
    def check(n_params: int, n_constraints: int) -> Tuple[bool, str]:
        """
        Check if parameter count is allowed.
        
        Returns
        -------
        allowed : bool
            True if system is solvable (over- or exactly-determined)
        message : str
            Diagnostic message
        """
        if n_params > n_constraints:
            return False, (
                f"FORBIDDEN: {n_params} params > {n_constraints} constraints. "
                f"Need more observables or fewer params."
            )
        elif n_params == n_constraints:
            return True, f"EXACT: {n_params} params = {n_constraints} constraints. Unique solution."
        else:
            redundancy = n_constraints - n_params
            return True, f"OVERDETERMINED: +{redundancy} redundant equations for consistency check."
    
    @staticmethod
    def max_params_for_sources(n_sources: int, n_images_per_source: int = 4) -> int:
        """
        Maximum allowed parameters for given number of sources.
        
        Each source with 4 images gives 8 constraints.
        We allow up to n_constraints parameters (exactly determined).
        """
        n_constraints = 2 * n_images_per_source * n_sources
        return n_constraints


class MultiSourceLinearSystemBuilder:
    """
    Builds the stacked linear system for multi-source lensing.
    
    For K sources with 4 images each:
    - 8K total constraints (2 per image)
    - Unknowns: [θ_E, a_2, b_2, ..., γ_1, γ_2, β_x^(1), β_y^(1), ..., β_x^(K), β_y^(K)]
    
    All parameters are LINEAR - no phase grid search!
    """
    
    def __init__(
        self,
        m_max: int = 2,
        include_shear: bool = False
    ):
        self.m_max = m_max
        self.include_shear = include_shear
    
    def unknowns(self, n_sources: int) -> List[str]:
        """List of all unknown parameters."""
        params = ['theta_E']
        
        # Multipole components (all linear!)
        for m in range(2, self.m_max + 1):
            params.extend([f'a_{m}', f'b_{m}'])
        
        # Shear components (linear!)
        if self.include_shear:
            params.extend(['gamma_1', 'gamma_2'])
        
        # Per-source beta (2 per source)
        for k in range(n_sources):
            params.extend([f'beta_x_{k}', f'beta_y_{k}'])
        
        return params
    
    def n_params(self, n_sources: int) -> int:
        """Total number of parameters."""
        return len(self.unknowns(n_sources))
    
    def n_constraints(self, sources: List[np.ndarray]) -> int:
        """Total number of constraints from all images."""
        return sum(2 * len(images) for images in sources)
    
    def build_system(
        self,
        sources: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build the full linear system Ax = b for all sources.
        
        Parameters
        ----------
        sources : list of ndarray
            sources[k] = array of shape (n_images_k, 2) with image positions
        
        Returns
        -------
        A : ndarray, shape (n_constraints, n_params)
            Coefficient matrix
        b : ndarray, shape (n_constraints,)
            Right-hand side (image positions)
        unknowns : list of str
            Parameter names in order
        """
        n_sources = len(sources)
        n_total_images = sum(len(s) for s in sources)
        n_constraints = 2 * n_total_images
        unknowns = self.unknowns(n_sources)
        n_params = len(unknowns)
        
        # DOF check
        allowed, msg = DOFGatekeeper.check(n_params, n_constraints)
        if not allowed:
            raise ValueError(msg)
        
        # Build coefficient matrix
        A = np.zeros((n_constraints, n_params))
        b = np.zeros(n_constraints)
        
        row = 0
        for k, images in enumerate(sources):
            for i, (x, y) in enumerate(images):
                # Image position in polar
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                
                # Lens equation: β = θ - α(θ)
                # In components: β_x = x - α_x, β_y = y - α_y
                # Rearranged: x = β_x + α_x, y = β_y + α_y
                
                # Row for x-component
                col = 0
                
                # θ_E contribution: α_x contains θ_E·(r/|r|)_x = θ_E·cos(φ)
                A[row, col] = cos_phi  # theta_E
                col += 1
                
                # Multipole contributions
                for m in range(2, self.m_max + 1):
                    # α_r = θ_E·(a_m·cos(mφ) + b_m·sin(mφ))
                    # α_x = α_r·cos(φ) - α_φ·sin(φ)
                    # For radial multipole: α_φ = 0
                    cos_m = np.cos(m * phi)
                    sin_m = np.sin(m * phi)
                    A[row, col] = cos_m * cos_phi      # a_m
                    A[row, col + 1] = sin_m * cos_phi  # b_m
                    col += 2
                
                # Shear contribution
                if self.include_shear:
                    # α_x = γ_1·x + γ_2·y
                    A[row, col] = x      # gamma_1
                    A[row, col + 1] = y  # gamma_2
                    col += 2
                
                # Beta for this source
                beta_col = col + 2 * k
                A[row, beta_col] = 1.0  # beta_x coefficient
                
                b[row] = x
                row += 1
                
                # Row for y-component
                col = 0
                
                # θ_E contribution
                A[row, col] = sin_phi  # theta_E
                col += 1
                
                # Multipole contributions
                for m in range(2, self.m_max + 1):
                    cos_m = np.cos(m * phi)
                    sin_m = np.sin(m * phi)
                    A[row, col] = cos_m * sin_phi      # a_m
                    A[row, col + 1] = sin_m * sin_phi  # b_m
                    col += 2
                
                # Shear contribution
                if self.include_shear:
                    # α_y = γ_2·x - γ_1·y
                    A[row, col] = -y     # gamma_1
                    A[row, col + 1] = x  # gamma_2
                    col += 2
                
                # Beta for this source
                A[row, beta_col + 1] = 1.0  # beta_y coefficient
                
                b[row] = y
                row += 1
        
        return A, b, unknowns
    
    def solve(
        self,
        sources: List[np.ndarray]
    ) -> Dict:
        """
        Solve the multi-source lensing problem exactly.
        
        Strategy:
        - If overdetermined: solve subset, use rest as consistency check
        - If exactly determined: unique solution
        - If underdetermined: REJECT (DOFGatekeeper)
        
        Returns
        -------
        result : dict
            'params': parameter values (dict)
            'residuals': residual vector
            'max_residual': maximum absolute residual
            'consistency': "PASS" if max_residual < tolerance
            'dof_status': DOF diagnostic message
        """
        A, b, unknowns = self.build_system(sources)
        n_params = len(unknowns)
        n_constraints = len(b)
        
        _, dof_msg = DOFGatekeeper.check(n_params, n_constraints)
        
        if n_constraints > n_params:
            # Overdetermined: solve subset exactly, check consistency
            A_solve = A[:n_params, :]
            b_solve = b[:n_params]
            
            try:
                p = np.linalg.solve(A_solve, b_solve)
            except np.linalg.LinAlgError:
                return {'error': 'Singular matrix in subset solve'}
            
            # Check ALL equations
            residuals = A @ p - b
            max_res = np.max(np.abs(residuals))
            
        else:
            # Exactly determined
            try:
                p = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return {'error': 'Singular matrix'}
            
            residuals = A @ p - b
            max_res = np.max(np.abs(residuals))
        
        # Build parameter dictionary
        params = {name: val for name, val in zip(unknowns, p)}
        
        # Derive phases as OUTPUT (not input!)
        derived = {}
        for m in range(2, self.m_max + 1):
            a_m = params.get(f'a_{m}', 0.0)
            b_m = params.get(f'b_{m}', 0.0)
            A_m = np.sqrt(a_m**2 + b_m**2)
            phi_m = np.arctan2(b_m, a_m) / m
            derived[f'A_{m}'] = A_m
            derived[f'phi_{m}'] = phi_m
        
        if self.include_shear:
            g1 = params.get('gamma_1', 0.0)
            g2 = params.get('gamma_2', 0.0)
            derived['gamma'] = np.sqrt(g1**2 + g2**2)
            derived['phi_gamma'] = np.arctan2(g2, g1) / 2
        
        return {
            'params': params,
            'derived': derived,
            'residuals': residuals,
            'max_residual': max_res,
            'consistency': 'PASS' if max_res < 1e-8 else f'FAIL (max_res={max_res:.2e})',
            'dof_status': dof_msg,
            'n_params': n_params,
            'n_constraints': n_constraints
        }


def generate_multi_source_synthetic(
    theta_E: float = 1.0,
    multipoles: Dict[int, Tuple[float, float]] = None,
    gamma_1: float = 0.0,
    gamma_2: float = 0.0,
    betas: List[Tuple[float, float]] = None,
    n_images: int = 4
) -> List[np.ndarray]:
    """
    Generate synthetic multi-source lensed images.
    
    All sources are lensed by the SAME lens with shared parameters.
    
    Parameters
    ----------
    theta_E : float
        Einstein radius (shared)
    multipoles : dict
        {m: (a_m, b_m)} multipole components (shared)
    gamma_1, gamma_2 : float
        Shear components (shared)
    betas : list of (float, float)
        Source positions for each source
    n_images : int
        Number of images per source (typically 4 for quad lens)
    
    Returns
    -------
    sources : list of ndarray
        sources[k] = image positions for source k
    """
    if multipoles is None:
        multipoles = {2: (0.1, 0.05)}
    if betas is None:
        betas = [(0.05, 0.03), (0.08, -0.02)]
    
    sources = []
    
    for beta_x, beta_y in betas:
        # Find image positions by solving lens equation
        # β = θ - α(θ)
        # This is a simplified version - real implementation would use rootfinding
        
        images = []
        for k in range(n_images):
            # Approximate image angles (for quad lens, ~90° apart)
            phi_k = 2 * np.pi * k / n_images + np.pi / 4
            
            # Approximate radius from Einstein ring + perturbations
            r_k = theta_E
            for m, (a_m, b_m) in multipoles.items():
                r_k += a_m * np.cos(m * phi_k) + b_m * np.sin(m * phi_k)
            
            # Add source offset effect
            r_k += beta_x * np.cos(phi_k) + beta_y * np.sin(phi_k)
            
            # Shear effect
            x_temp = r_k * np.cos(phi_k)
            y_temp = r_k * np.sin(phi_k)
            x_k = x_temp + gamma_1 * x_temp + gamma_2 * y_temp
            y_k = y_temp + gamma_2 * x_temp - gamma_1 * y_temp
            
            images.append([x_k, y_k])
        
        sources.append(np.array(images))
    
    return sources


# Convenience functions
def analyze_dof(n_sources: int, m_max: int = 2, include_shear: bool = False) -> str:
    """
    Analyze DOF for a given configuration.
    
    Returns human-readable summary.
    """
    n_images = 4 * n_sources
    n_constraints = 2 * n_images
    
    # Count params
    n_params = 1  # theta_E
    n_params += 2 * (m_max - 1)  # multipoles m=2 to m_max
    if include_shear:
        n_params += 2
    n_params += 2 * n_sources  # betas
    
    allowed, msg = DOFGatekeeper.check(n_params, n_constraints)
    
    status = "[OK]" if allowed else "[FORBIDDEN]"
    return f"""
DOF Analysis: {n_sources} source(s), m_max={m_max}, shear={include_shear}
------------------------------------------------------------
  Images:      {n_images} ({n_sources} sources x 4 images)
  Constraints: {n_constraints} (2 per image)
  Parameters:  {n_params}
  Status:      {status} {msg}
------------------------------------------------------------
"""
