"""
Fully Linear Multipole Lens Model - True No-Fit Inversion

This model uses (c_m, s_m) components instead of (A_m, phi_m),
making the ENTIRE problem linear-algebraic with NO grid search.

Key insight: The deflection from multipole m is:
    alpha_r = theta_E * (c_m * cos(m*phi) + s_m * sin(m*phi))
    
where (c_m, s_m) absorb the phase. No phi_m parameter needed!

Similarly for shear: (gamma_1, gamma_2) instead of (gamma, phi_gamma).

DOF Analysis for Quad Lens (8 constraints):
    - m=2 only:           5 params -> 3 redundant equations (consistency check)
    - m=2 + shear:        7 params -> 1 redundant equation
    - m=2 + m=3:          7 params -> 1 redundant equation  
    - m=2 + shear + m=3:  9 params -> UNDERDETERMINED (needs more data!)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .base_model import LensModel


@dataclass
class LinearParams:
    """All parameters are directly solvable - no phases!"""
    beta_x: float = 0.0
    beta_y: float = 0.0
    theta_E: float = 1.0
    
    # Shear components (NOT gamma, phi_gamma!)
    gamma_1: float = 0.0  # gamma * cos(2*phi_gamma)
    gamma_2: float = 0.0  # gamma * sin(2*phi_gamma)
    
    # Multipole components for m=2,3,4
    # c_m, s_m such that alpha_r = theta_E * (c_m*cos(m*phi) + s_m*sin(m*phi))
    c_2: float = 0.0
    s_2: float = 0.0
    c_3: float = 0.0
    s_3: float = 0.0
    c_4: float = 0.0
    s_4: float = 0.0


class LinearMultipoleModel(LensModel):
    """
    Fully linear multipole lens model.
    
    No grid search, no rootfinding on phases - just linear algebra.
    
    The trade-off: We get (c_m, s_m) instead of (amplitude, phase).
    To recover physical parameters:
        amplitude = sqrt(c_m^2 + s_m^2)
        phase = arctan2(s_m, c_m) / m
    """
    
    def __init__(
        self,
        m_max: int = 2,
        include_shear: bool = False,
        eta: float = 2.0  # Power-law slope (fixed, not fitted)
    ):
        self._m_max = m_max
        self._include_shear = include_shear
        self._eta = eta
    
    @property
    def name(self) -> str:
        shear_str = "+shear" if self._include_shear else ""
        return f"Linear Model (m_max={self._m_max}{shear_str})"
    
    @property
    def m_max(self) -> int:
        return self._m_max
    
    def unknowns(self) -> List[str]:
        """All unknowns - ALL are linear!"""
        params = ['beta_x', 'beta_y', 'theta_E']
        
        if self._include_shear:
            params.extend(['gamma_1', 'gamma_2'])
        
        for m in range(2, self._m_max + 1):
            params.extend([f'c_{m}', f's_{m}'])
        
        return params
    
    def nonlinear_unknowns(self) -> List[str]:
        """No nonlinear unknowns - that's the whole point!"""
        return []
    
    def linear_unknowns(self) -> List[str]:
        """All unknowns are linear."""
        return self.unknowns()
    
    def observables(self) -> List[str]:
        return ['image_positions']
    
    def n_constraints(self, n_images: int) -> int:
        """Number of scalar constraints from images."""
        return 2 * n_images
    
    def n_parameters(self) -> int:
        """Number of parameters."""
        return len(self.unknowns())
    
    def dof_status(self, n_images: int) -> str:
        """Degrees of freedom analysis."""
        n_con = self.n_constraints(n_images)
        n_par = self.n_parameters()
        diff = n_con - n_par
        
        if diff > 0:
            return f"OVERDETERMINED (+{diff} redundant equations)"
        elif diff == 0:
            return "EXACTLY DETERMINED"
        else:
            return f"UNDERDETERMINED ({-diff} more data needed!)"
    
    def deflection_monopole(self, x: float, y: float, theta_E: float) -> Tuple[float, float]:
        """SIS monopole deflection."""
        r = np.sqrt(x**2 + y**2)
        if r < 1e-15:
            return 0.0, 0.0
        return theta_E * x / r, theta_E * y / r
    
    def deflection_shear(
        self, x: float, y: float, gamma_1: float, gamma_2: float
    ) -> Tuple[float, float]:
        """
        Shear deflection using (gamma_1, gamma_2) components.
        
        gamma_1 = gamma * cos(2*phi_gamma)
        gamma_2 = gamma * sin(2*phi_gamma)
        
        alpha_x = gamma_1 * x + gamma_2 * y
        alpha_y = gamma_2 * x - gamma_1 * y
        """
        return gamma_1 * x + gamma_2 * y, gamma_2 * x - gamma_1 * y
    
    def deflection_multipole(
        self, x: float, y: float, theta_E: float, m: int, c_m: float, s_m: float
    ) -> Tuple[float, float]:
        """
        Multipole deflection using (c_m, s_m) components.
        
        alpha_r = theta_E * (c_m * cos(m*phi) + s_m * sin(m*phi))
        alpha_phi = theta_E * m * (-c_m * sin(m*phi) + s_m * cos(m*phi))
        """
        r = np.sqrt(x**2 + y**2)
        if r < 1e-15:
            return 0.0, 0.0
        
        phi = np.arctan2(y, x)
        cos_m_phi = np.cos(m * phi)
        sin_m_phi = np.sin(m * phi)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # Radial and tangential components
        alpha_r = theta_E * (c_m * cos_m_phi + s_m * sin_m_phi)
        alpha_phi = theta_E * m * (-c_m * sin_m_phi + s_m * cos_m_phi)
        
        # Convert to Cartesian
        alpha_x = alpha_r * cos_phi - alpha_phi * sin_phi
        alpha_y = alpha_r * sin_phi + alpha_phi * cos_phi
        
        return alpha_x, alpha_y
    
    def deflection_total(self, x: float, y: float, params: LinearParams) -> Tuple[float, float]:
        """Total deflection from all components."""
        # Monopole
        alpha_x, alpha_y = self.deflection_monopole(x, y, params.theta_E)
        
        # Shear
        if self._include_shear:
            sx, sy = self.deflection_shear(x, y, params.gamma_1, params.gamma_2)
            alpha_x += sx
            alpha_y += sy
        
        # Multipoles
        for m in range(2, self._m_max + 1):
            c_m = getattr(params, f'c_{m}')
            s_m = getattr(params, f's_{m}')
            if abs(c_m) > 1e-15 or abs(s_m) > 1e-15:
                mx, my = self.deflection_multipole(x, y, params.theta_E, m, c_m, s_m)
                alpha_x += mx
                alpha_y += my
        
        return alpha_x, alpha_y
    
    def predict_images(
        self,
        source_params: Dict[str, float],
        lens_params: Dict[str, float]
    ) -> np.ndarray:
        """Forward model: find images from source + lens parameters."""
        params = LinearParams(
            beta_x=source_params.get('beta_x', 0.0),
            beta_y=source_params.get('beta_y', 0.0),
            theta_E=lens_params.get('theta_E', 1.0),
            gamma_1=lens_params.get('gamma_1', 0.0),
            gamma_2=lens_params.get('gamma_2', 0.0),
        )
        for m in range(2, self._m_max + 1):
            setattr(params, f'c_{m}', lens_params.get(f'c_{m}', 0.0))
            setattr(params, f's_{m}', lens_params.get(f's_{m}', 0.0))
        
        # Grid search for image positions
        theta_E = params.theta_E
        n_r, n_phi = 50, 72
        r_vals = np.linspace(0.3 * theta_E, 2.5 * theta_E, n_r)
        phi_vals = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        
        images = []
        for r_init in r_vals:
            for phi_init in phi_vals:
                x, y = r_init * np.cos(phi_init), r_init * np.sin(phi_init)
                
                # Newton-Raphson refinement
                for _ in range(20):
                    alpha_x, alpha_y = self.deflection_total(x, y, params)
                    res_x = x - alpha_x - params.beta_x
                    res_y = y - alpha_y - params.beta_y
                    
                    if res_x**2 + res_y**2 < 1e-20:
                        break
                    
                    # Numerical Jacobian
                    eps = 1e-7
                    ax1, ay1 = self.deflection_total(x + eps, y, params)
                    ax2, ay2 = self.deflection_total(x, y + eps, params)
                    
                    J = np.array([
                        [1 - (ax1 - alpha_x)/eps, -(ax2 - alpha_x)/eps],
                        [-(ay1 - alpha_y)/eps, 1 - (ay2 - alpha_y)/eps]
                    ])
                    
                    try:
                        delta = np.linalg.solve(J, [res_x, res_y])
                        x -= delta[0]
                        y -= delta[1]
                    except np.linalg.LinAlgError:
                        break
                
                # Check if converged and unique
                alpha_x, alpha_y = self.deflection_total(x, y, params)
                res = np.sqrt((x - alpha_x - params.beta_x)**2 + 
                             (y - alpha_y - params.beta_y)**2)
                
                if res < 1e-10:
                    is_new = True
                    for img in images:
                        if np.sqrt((x - img[0])**2 + (y - img[1])**2) < 0.01:
                            is_new = False
                            break
                    if is_new:
                        images.append([x, y])
        
        return np.array(images) if images else np.array([]).reshape(0, 2)
    
    def equations(
        self,
        images: np.ndarray,
        source_params: Dict[str, float],
        lens_params: Dict[str, float]
    ) -> np.ndarray:
        """Compute residuals of lens equation."""
        params = LinearParams(
            beta_x=source_params.get('beta_x', 0.0),
            beta_y=source_params.get('beta_y', 0.0),
            theta_E=lens_params.get('theta_E', 1.0),
            gamma_1=lens_params.get('gamma_1', 0.0),
            gamma_2=lens_params.get('gamma_2', 0.0),
        )
        for m in range(2, self._m_max + 1):
            setattr(params, f'c_{m}', lens_params.get(f'c_{m}', 0.0))
            setattr(params, f's_{m}', lens_params.get(f's_{m}', 0.0))
        
        residuals = []
        for x, y in images:
            alpha_x, alpha_y = self.deflection_total(x, y, params)
            residuals.append(x - alpha_x - params.beta_x)
            residuals.append(y - alpha_y - params.beta_y)
        
        return np.array(residuals)
    
    def build_linear_system(
        self,
        images: np.ndarray,
        fixed_nonlinear: Dict[str, float] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build FULLY LINEAR system - no fixed phases needed!
        
        Lens equation: x - alpha_x(theta_E, c_m, s_m, gamma_1, gamma_2) = beta_x
        
        Rearranged: beta_x + alpha_x = x
        """
        n = len(images)
        param_names = self.unknowns()
        n_params = len(param_names)
        
        A = np.zeros((2*n, n_params))
        b_vec = np.zeros(2*n)
        
        for i, (x, y) in enumerate(images):
            r = np.sqrt(x**2 + y**2)
            if r < 1e-15:
                r = 1e-15
            phi = np.arctan2(y, x)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            row_x = 2 * i
            row_y = 2 * i + 1
            
            # RHS: observed positions
            b_vec[row_x] = x
            b_vec[row_y] = y
            
            col = 0
            
            # beta_x, beta_y
            A[row_x, col] = 1.0
            A[row_y, col + 1] = 1.0
            col += 2
            
            # theta_E (monopole): alpha = theta_E * (x/r, y/r)
            A[row_x, col] = x / r
            A[row_y, col] = y / r
            col += 1
            
            # Shear: alpha_x = gamma_1*x + gamma_2*y, alpha_y = gamma_2*x - gamma_1*y
            if self._include_shear:
                A[row_x, col] = x      # gamma_1
                A[row_y, col] = -y     # gamma_1
                A[row_x, col + 1] = y  # gamma_2
                A[row_y, col + 1] = x  # gamma_2
                col += 2
            
            # Multipoles: alpha from (c_m, s_m)
            for m in range(2, self._m_max + 1):
                cos_m_phi = np.cos(m * phi)
                sin_m_phi = np.sin(m * phi)
                
                # Coefficients for c_m contribution to alpha_x, alpha_y
                # alpha_r = c_m * cos(m*phi), alpha_phi = -m * c_m * sin(m*phi)
                # alpha_x = alpha_r * cos_phi - alpha_phi * sin_phi
                #         = c_m * (cos(m*phi)*cos_phi + m*sin(m*phi)*sin_phi)
                coeff_c_x = cos_m_phi * cos_phi + m * sin_m_phi * sin_phi
                coeff_c_y = cos_m_phi * sin_phi - m * sin_m_phi * cos_phi
                
                # Coefficients for s_m contribution
                # alpha_r = s_m * sin(m*phi), alpha_phi = m * s_m * cos(m*phi)
                coeff_s_x = sin_m_phi * cos_phi - m * cos_m_phi * sin_phi
                coeff_s_y = sin_m_phi * sin_phi + m * cos_m_phi * cos_phi
                
                # Note: these are already "times theta_E" implicitly through 
                # the definition c_m = theta_E * amplitude * cos(...)
                # For proper scaling, we include theta_E dependency
                # Actually, simpler: treat c_m, s_m as "effective" components
                A[row_x, col] = coeff_c_x
                A[row_y, col] = coeff_c_y
                A[row_x, col + 1] = coeff_s_x
                A[row_y, col + 1] = coeff_s_y
                col += 2
        
        return A, b_vec, param_names
    
    def consistency_function(
        self,
        images: np.ndarray,
        nonlinear_value: float,
        nonlinear_name: str
    ) -> float:
        """Not needed - no nonlinear unknowns!"""
        return 0.0
    
    def initial_guess(self, images: np.ndarray) -> Dict[str, float]:
        """Simple initial guess based on image geometry."""
        n = len(images)
        if n == 0:
            return {'theta_E': 1.0, 'beta_x': 0.0, 'beta_y': 0.0}
        
        # Centroid for source guess
        centroid = np.mean(images, axis=0)
        
        # Mean radius for theta_E
        radii = np.sqrt(images[:, 0]**2 + images[:, 1]**2)
        theta_E = np.mean(radii)
        
        return {
            'beta_x': centroid[0] * 0.1,
            'beta_y': centroid[1] * 0.1,
            'theta_E': theta_E
        }
    
    def invert(self, images: np.ndarray, tol: float = 1e-10) -> List[Dict]:
        """
        TRUE No-Fit Inversion: Direct linear solve.
        
        No grid search, no rootfinding - just linear algebra!
        
        Returns list with single solution (or empty if underdetermined).
        """
        n_images = len(images)
        n_con = self.n_constraints(n_images)
        n_par = self.n_parameters()
        
        A, b_vec, param_names = self.build_linear_system(images)
        
        if n_con < n_par:
            # Underdetermined - need more data
            return []
        
        if n_con == n_par:
            # Exactly determined
            try:
                p = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                return []
            
            params = dict(zip(param_names, p))
            residuals = A @ p - b_vec
            
        else:
            # Overdetermined - solve exactly using first n_par equations,
            # use remaining equations as consistency check
            try:
                p = np.linalg.solve(A[:n_par], b_vec[:n_par])
            except np.linalg.LinAlgError:
                return []
            
            params = dict(zip(param_names, p))
            
            # Full residuals (including consistency equations)
            residuals = A @ p - b_vec
        
        # Check for physical validity
        if params.get('theta_E', 0) <= 0:
            return []
        
        max_res = np.max(np.abs(residuals))
        rms_res = np.sqrt(np.mean(residuals**2))
        
        # Compute consistency measure (redundant equations only)
        if n_con > n_par:
            consistency_res = residuals[n_par:]
            consistency = np.max(np.abs(consistency_res))
        else:
            consistency = 0.0
        
        return [{
            'params': params,
            'residuals': residuals,
            'report': {
                'max_abs': max_res,
                'rms': rms_res,
                'consistency': consistency,
                'dof_status': self.dof_status(n_images),
                'n_constraints': n_con,
                'n_parameters': n_par
            }
        }]
    
    def convert_to_physical(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Convert (c_m, s_m) back to (amplitude_m, phase_m).
        
        amplitude_m = sqrt(c_m^2 + s_m^2)
        phase_m = arctan2(s_m, c_m) / m
        """
        physical = {
            'beta_x': params.get('beta_x', 0.0),
            'beta_y': params.get('beta_y', 0.0),
            'theta_E': params.get('theta_E', 1.0)
        }
        
        if self._include_shear:
            g1 = params.get('gamma_1', 0.0)
            g2 = params.get('gamma_2', 0.0)
            physical['gamma'] = np.sqrt(g1**2 + g2**2)
            physical['phi_gamma'] = 0.5 * np.arctan2(g2, g1)
        
        for m in range(2, self._m_max + 1):
            c_m = params.get(f'c_{m}', 0.0)
            s_m = params.get(f's_{m}', 0.0)
            physical[f'amplitude_{m}'] = np.sqrt(c_m**2 + s_m**2)
            physical[f'phase_{m}'] = np.arctan2(s_m, c_m) / m
        
        return physical
