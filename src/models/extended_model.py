"""
Extended Lens Model with External Shear and Higher Multipoles

This module extends the basic quadrupole model to include:
1. External shear (γ_ext, φ_γ) from neighboring galaxies/clusters
2. Higher-order multipoles (m=3 octupole, m=4 hexadecapole)
3. Power-law radial profile (η variable)
4. Hermite C² blending for smooth transitions

Key insight: The system remains conditionally linear!
- Phases (φ_m) are nonlinear → rootfinding
- Amplitudes, source position, shear strength → exact linear solve

Physical motivation:
- External shear: Essential for B1608+656 (two-galaxy system)
- m=3 (octupole): Bar structures like Q2237+0305
- m=4 (hexadecapole): Isophotal deviations (boxy/disky)

Authors: Carmen N. Wrede, Lino P. Casu
License: ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from .base_model import LensModel
from .profiles import (
    hermite_blend, alpha_power_law, deflection_2d_power_law,
    PHI
)


@dataclass
class ExtendedParams:
    """Parameters for extended lens model."""
    # Source
    beta_x: float = 0.0
    beta_y: float = 0.0
    
    # Monopole (mass scale)
    theta_E: float = 1.0
    
    # Power-law slope (η=2 is SIS)
    eta: float = 2.0
    
    # Core radius (0 = singular)
    r_core: float = 0.0
    
    # External shear
    gamma_ext: float = 0.0  # Shear strength
    phi_gamma: float = 0.0  # Shear angle (radians)
    
    # Multipole amplitudes and phases
    # m=1: dipole (usually absorbed into source offset)
    a_1: float = 0.0
    b_1: float = 0.0
    phi_1: float = 0.0
    
    # m=2: quadrupole (ellipticity)
    a_2: float = 0.0
    b_2: float = 0.0
    phi_2: float = 0.0
    
    # m=3: octupole (triangular/bar distortion)
    a_3: float = 0.0
    b_3: float = 0.0
    phi_3: float = 0.0
    
    # m=4: hexadecapole (boxy/disky)
    a_4: float = 0.0
    b_4: float = 0.0
    phi_4: float = 0.0
    
    def validate(self) -> Tuple[bool, str]:
        """Check physical validity."""
        if self.theta_E <= 0:
            return False, "theta_E must be positive"
        if self.eta <= 1 or self.eta >= 3:
            return False, f"eta must be in (1, 3), got {self.eta}"
        if self.r_core < 0:
            return False, "r_core must be non-negative"
        if self.gamma_ext < 0:
            return False, "gamma_ext must be non-negative"
        return True, "OK"


class ExtendedMultipoleModel(LensModel):
    """
    Extended multipole lens model with external shear.
    
    Potential:
        ψ(θ, φ) = ψ_monopole + ψ_shear + ψ_multipoles
        
    where:
        ψ_monopole: Power-law radial profile
        ψ_shear: -γ/2 × θ² × cos(2(φ - φ_γ))
        ψ_multipoles: Σ_m [a_m cos(m(φ-φ_m)) + b_m sin(m(φ-φ_m))] × f_m(θ)
    
    The deflection α = ∇ψ is linear in (β, θ_E, γ, a_m, b_m) when 
    phases (φ_m, φ_γ) are fixed → exact linear solve after rootfinding.
    """
    
    def __init__(self, m_max: int = 4, include_shear: bool = True, eta: float = 2.0):
        """
        Initialize extended model.
        
        Parameters
        ----------
        m_max : int
            Maximum multipole order (2, 3, or 4)
        include_shear : bool
            Include external shear term
        eta : float
            Power-law slope (2.0 = isothermal)
        """
        self._m_max = min(max(m_max, 2), 4)  # Clamp to [2, 4]
        self._include_shear = include_shear
        self._eta = eta
    
    @property
    def name(self) -> str:
        shear_str = "+shear" if self._include_shear else ""
        return f"Extended Model (m_max={self._m_max}, eta={self._eta:.2f}{shear_str})"
    
    @property
    def m_max(self) -> int:
        return self._m_max
    
    @property
    def eta(self) -> float:
        return self._eta
    
    @property
    def include_shear(self) -> bool:
        return self._include_shear
    
    def unknowns(self) -> List[str]:
        """All unknown parameters."""
        params = ['beta_x', 'beta_y', 'theta_E']
        
        if self._include_shear:
            params.extend(['gamma_ext', 'phi_gamma_ext'])
        
        for m in range(2, self._m_max + 1):
            params.extend([f'a_{m}', f'b_{m}', f'phi_{m}'])
        
        return params
    
    def nonlinear_unknowns(self) -> List[str]:
        """Phases determined by rootfinding."""
        params = []
        if self._include_shear:
            params.append('phi_gamma_ext')
        
        # Phases for m ≥ 2
        for m in range(2, self._m_max + 1):
            params.append(f'phi_{m}')
        
        return params
    
    def linear_unknowns(self) -> List[str]:
        """Parameters solvable linearly when phases fixed."""
        params = ['beta_x', 'beta_y', 'theta_E']
        
        if self._include_shear:
            params.append('gamma_ext')
        
        for m in range(2, self._m_max + 1):
            params.extend([f'a_{m}', f'b_{m}'])
        
        return params
    
    def observables(self) -> List[str]:
        return ['image_positions']
    
    def deflection_shear(
        self, 
        x: float, 
        y: float, 
        gamma: float, 
        phi_gamma: float
    ) -> Tuple[float, float]:
        """
        Deflection from external shear.
        
        ψ_shear = -γ/2 × (x² - y²) × cos(2φ_γ) - γ × x × y × sin(2φ_γ)
        
        (in aligned coordinates, simpler form)
        
        α_x = γ × (x × cos(2φ_γ) + y × sin(2φ_γ))
        α_y = γ × (x × sin(2φ_γ) - y × cos(2φ_γ))
        """
        cos_2phi = np.cos(2 * phi_gamma)
        sin_2phi = np.sin(2 * phi_gamma)
        
        alpha_x = gamma * (x * cos_2phi + y * sin_2phi)
        alpha_y = gamma * (x * sin_2phi - y * cos_2phi)
        
        return alpha_x, alpha_y
    
    def deflection_multipole(
        self,
        r: float,
        phi: float,
        theta_E: float,
        m: int,
        a_m: float,
        b_m: float,
        phi_m: float
    ) -> Tuple[float, float]:
        """
        Deflection from m-th multipole term.
        
        For near-Einstein-ring images:
        α_r = θ_E × (a_m × cos(m×Δ) + b_m × sin(m×Δ))
        α_φ = θ_E × m × (-a_m × sin(m×Δ) + b_m × cos(m×Δ))
        
        where Δ = φ - φ_m
        """
        delta = phi - phi_m
        cos_m_delta = np.cos(m * delta)
        sin_m_delta = np.sin(m * delta)
        
        # Radial component
        alpha_r = theta_E * (a_m * cos_m_delta + b_m * sin_m_delta)
        
        # Tangential component
        alpha_phi = theta_E * m * (-a_m * sin_m_delta + b_m * cos_m_delta)
        
        # Convert to Cartesian
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        alpha_x = alpha_r * cos_phi - alpha_phi * sin_phi
        alpha_y = alpha_r * sin_phi + alpha_phi * cos_phi
        
        return alpha_x, alpha_y
    
    def deflection_total(
        self,
        x: float,
        y: float,
        params: ExtendedParams
    ) -> Tuple[float, float]:
        """
        Total deflection from all components.
        
        α_total = α_monopole + α_shear + Σ_m α_multipole_m
        """
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        
        # Monopole (power-law profile)
        if r > 1e-15:
            alpha_mono = alpha_power_law(r, params.theta_E, params.eta)
            alpha_x = alpha_mono * x / r
            alpha_y = alpha_mono * y / r
        else:
            alpha_x, alpha_y = 0.0, 0.0
        
        # External shear
        if params.gamma_ext > 0:
            shear_x, shear_y = self.deflection_shear(
                x, y, params.gamma_ext, params.phi_gamma
            )
            alpha_x += shear_x
            alpha_y += shear_y
        
        # Multipoles (m ≥ 2)
        for m in range(2, self._m_max + 1):
            a_m = getattr(params, f'a_{m}')
            b_m = getattr(params, f'b_{m}')
            phi_m = getattr(params, f'phi_{m}')
            
            if abs(a_m) > 1e-15 or abs(b_m) > 1e-15:
                mult_x, mult_y = self.deflection_multipole(
                    r, phi, params.theta_E, m, a_m, b_m, phi_m
                )
                alpha_x += mult_x
                alpha_y += mult_y
        
        return alpha_x, alpha_y
    
    def predict_images(
        self,
        source_params: Dict[str, float],
        lens_params: Dict[str, float]
    ) -> np.ndarray:
        """Forward model: find images from source + lens parameters.
        
        Uses 2D grid search + Newton-Raphson refinement.
        """
        # Build params object
        params = ExtendedParams(
            beta_x=source_params.get('beta_x', 0.0),
            beta_y=source_params.get('beta_y', 0.0),
            theta_E=lens_params.get('theta_E', 1.0),
            eta=lens_params.get('eta', self._eta),
            gamma_ext=lens_params.get('gamma_ext', 0.0),
            phi_gamma=lens_params.get('phi_gamma_ext', 0.0),
        )
        
        for m in range(2, self._m_max + 1):
            setattr(params, f'a_{m}', lens_params.get(f'a_{m}', 0.0))
            setattr(params, f'b_{m}', lens_params.get(f'b_{m}', 0.0))
            setattr(params, f'phi_{m}', lens_params.get(f'phi_{m}', 0.0))
        
        beta_x = params.beta_x
        beta_y = params.beta_y
        theta_E = params.theta_E
        
        # Lens equation residual: f(θ) = θ - α(θ) - β = 0
        def lens_residual(x, y):
            ax, ay = self.deflection_total(x, y, params)
            return x - ax - beta_x, y - ay - beta_y
        
        # 2D grid search over image plane
        n_grid = 100
        r_max = 3.0 * theta_E
        candidates = []
        
        # Grid in polar coordinates for better coverage near Einstein ring
        for i_r in range(1, n_grid + 1):
            r = r_max * i_r / n_grid
            n_phi = max(8, int(2 * np.pi * r / (0.1 * theta_E)))
            
            for i_phi in range(n_phi):
                phi = 2 * np.pi * i_phi / n_phi
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                
                fx, fy = lens_residual(x, y)
                res = np.sqrt(fx**2 + fy**2)
                
                if res < 0.5 * theta_E:  # Candidate region
                    candidates.append([x, y, res])
        
        # Refine candidates with Newton-Raphson
        images = []
        eps = 1e-8
        
        for x0, y0, _ in candidates:
            x, y = x0, y0
            
            for _ in range(50):
                fx, fy = lens_residual(x, y)
                res = np.sqrt(fx**2 + fy**2)
                
                if res < 1e-12:
                    break
                
                # Jacobian via finite differences
                fx_px, fy_px = lens_residual(x + eps, y)
                fx_py, fy_py = lens_residual(x, y + eps)
                
                J11 = (fx_px - fx) / eps
                J12 = (fx_py - fx) / eps
                J21 = (fy_px - fy) / eps
                J22 = (fy_py - fy) / eps
                
                det = J11 * J22 - J12 * J21
                if abs(det) < 1e-14:
                    break
                
                # Newton step
                dx = (J22 * fx - J12 * fy) / det
                dy = (-J21 * fx + J11 * fy) / det
                
                x -= dx
                y -= dy
                
                if np.sqrt(dx**2 + dy**2) < 1e-14:
                    break
            
            # Final check
            fx, fy = lens_residual(x, y)
            res = np.sqrt(fx**2 + fy**2)
            
            if res < 1e-10 and np.sqrt(x**2 + y**2) > 0.01 * theta_E:
                # Check for duplicates
                is_new = True
                for img in images:
                    if np.sqrt((img[0] - x)**2 + (img[1] - y)**2) < 0.01 * theta_E:
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
        """Compute residuals of lens equation for each image."""
        params = ExtendedParams(
            beta_x=source_params.get('beta_x', 0.0),
            beta_y=source_params.get('beta_y', 0.0),
            theta_E=lens_params.get('theta_E', 1.0),
            eta=lens_params.get('eta', self._eta),
            gamma_ext=lens_params.get('gamma_ext', 0.0),
            phi_gamma=lens_params.get('phi_gamma_ext', 0.0),
        )
        
        for m in range(2, self._m_max + 1):
            setattr(params, f'a_{m}', lens_params.get(f'a_{m}', 0.0))
            setattr(params, f'b_{m}', lens_params.get(f'b_{m}', 0.0))
            setattr(params, f'phi_{m}', lens_params.get(f'phi_{m}', 0.0))
        
        residuals = []
        for x, y in images:
            alpha_x, alpha_y = self.deflection_total(x, y, params)
            
            pred_beta_x = x - alpha_x
            pred_beta_y = y - alpha_y
            
            residuals.append(pred_beta_x - params.beta_x)
            residuals.append(pred_beta_y - params.beta_y)
        
        return np.array(residuals)
    
    def build_linear_system(
        self,
        images: np.ndarray,
        fixed_nonlinear: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build linear system for linear unknowns given fixed phases.
        
        Lens equation: θ - α(θ) = β
        Rearranged: β + α(θ) = θ
        
        For SIS-like monopole: α_r = theta_E (radial direction)
        For m-multipole: α depends on a_m, b_m with phase phi_m
        """
        n = len(images)
        
        # Extract fixed phases
        phi_gamma = fixed_nonlinear.get('phi_gamma_ext', 0.0)
        phases = {}
        for m in range(2, self._m_max + 1):
            phases[m] = fixed_nonlinear.get(f'phi_{m}', 0.0)
        
        # Parameter list: beta_x, beta_y, theta_E, [gamma_ext], A_2, B_2, ...
        param_names = ['beta_x', 'beta_y', 'theta_E']
        if self._include_shear:
            param_names.append('gamma_ext')
        for m in range(2, self._m_max + 1):
            param_names.extend([f'A_{m}', f'B_{m}'])
        
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
            
            # Equation: beta_x + alpha_x(theta_E, a_m, b_m, gamma) = x
            # Equation: beta_y + alpha_y(theta_E, a_m, b_m, gamma) = y
            
            # beta_x, beta_y coefficients
            A[row_x, 0] = 1.0  # beta_x
            A[row_y, 1] = 1.0  # beta_y
            
            # theta_E (monopole): α = theta_E * (x/r, y/r)
            A[row_x, 2] = x / r
            A[row_y, 2] = y / r
            
            col = 3
            
            # External shear: α = γ × (x*cos(2φ_γ) + y*sin(2φ_γ), ...)
            if self._include_shear:
                cos_2pg = np.cos(2 * phi_gamma)
                sin_2pg = np.sin(2 * phi_gamma)
                A[row_x, col] = x * cos_2pg + y * sin_2pg
                A[row_y, col] = x * sin_2pg - y * cos_2pg
                col += 1
            
            # Multipoles: A_m = theta_E * a_m, B_m = theta_E * b_m
            for m in range(2, self._m_max + 1):
                phi_m = phases[m]
                delta = phi - phi_m
                cos_md = np.cos(m * delta)
                sin_md = np.sin(m * delta)
                
                # Radial: A_m*cos(m*delta) + B_m*sin(m*delta)
                # Tangential: m*(-A_m*sin(m*delta) + B_m*cos(m*delta))
                # Convert to Cartesian
                
                # alpha_x = alpha_r * cos_phi - alpha_phi/r * (-sin_phi)
                #         = alpha_r * cos_phi + alpha_phi * sin_phi / r
                # For unit coefficients (alpha_r = 1, alpha_phi = m):
                coeff_A_x = cos_md * cos_phi + m * sin_md * sin_phi
                coeff_B_x = sin_md * cos_phi - m * cos_md * sin_phi
                coeff_A_y = cos_md * sin_phi - m * sin_md * cos_phi
                coeff_B_y = sin_md * sin_phi + m * cos_md * cos_phi
                
                A[row_x, col] = coeff_A_x
                A[row_x, col+1] = coeff_B_x
                A[row_y, col] = coeff_A_y
                A[row_y, col+1] = coeff_B_y
                col += 2
            
            b_vec[row_x] = x
            b_vec[row_y] = y
        
        return A, b_vec, param_names
    
    def convert_scaled_params(
        self,
        solved: Dict[str, float],
        fixed_phases: Dict[str, float]
    ) -> Dict[str, float]:
        """Convert scaled parameters back to physical."""
        theta_E = solved['theta_E']
        
        result = {
            'beta_x': solved['beta_x'],
            'beta_y': solved['beta_y'],
            'theta_E': theta_E,
            'eta': self._eta,
        }
        
        if self._include_shear:
            result['gamma_ext'] = solved.get('gamma_ext', 0.0)
            result['phi_gamma_ext'] = fixed_phases.get('phi_gamma_ext', 0.0)
        
        for m in range(2, self._m_max + 1):
            A_m = solved.get(f'A_{m}', 0.0)
            B_m = solved.get(f'B_{m}', 0.0)
            
            if abs(theta_E) > 1e-15:
                a_m = A_m / theta_E
                b_m = B_m / theta_E
            else:
                a_m = b_m = 0.0
            
            result[f'a_{m}'] = a_m
            result[f'b_{m}'] = b_m
            result[f'phi_{m}'] = fixed_phases.get(f'phi_{m}', 0.0)
        
        return result
    
    def consistency_function(
        self,
        images: np.ndarray,
        nonlinear_value: float,
        nonlinear_name: str
    ) -> float:
        """Consistency function for rootfinding on single phase."""
        fixed = {nonlinear_name: nonlinear_value}
        
        # Fill other phases with 0 for now (or could iterate)
        for name in self.nonlinear_unknowns():
            if name not in fixed:
                fixed[name] = 0.0
        
        A, b_vec, param_names = self.build_linear_system(images, fixed)
        
        n_eq = A.shape[0]
        n_params = A.shape[1]
        
        if n_eq <= n_params:
            return 0.0
        
        # Solve using first n_params equations
        try:
            A_sub = A[:n_params, :]
            b_sub = b_vec[:n_params]
            
            if abs(np.linalg.det(A_sub)) < 1e-14:
                return float('nan')
            
            p = np.linalg.solve(A_sub, b_sub)
            
            # Check residual on next equation
            return np.dot(A[n_params], p) - b_vec[n_params]
        except Exception:
            return float('nan')
    
    def initial_guess(self, images: np.ndarray) -> Dict[str, float]:
        """Heuristic initial guess using moments."""
        n = len(images)
        z = images[:, 0] + 1j * images[:, 1]
        
        z_mean = np.mean(z)
        theta_E_est = np.mean(np.abs(z))
        
        guess = {
            'beta_x': z_mean.real,
            'beta_y': z_mean.imag,
            'theta_E': theta_E_est,
            'eta': self._eta,
        }
        
        if self._include_shear:
            guess['gamma_ext'] = 0.01
            guess['phi_gamma_ext'] = 0.0
        
        # Estimate multipoles from moments
        z_centered = z - z_mean
        for m in range(2, self._m_max + 1):
            moment = np.mean(z_centered ** m)
            amp = np.abs(moment) / theta_E_est if theta_E_est > 0 else 0
            phase = np.angle(moment) / m
            
            guess[f'a_{m}'] = amp * 0.5
            guess[f'b_{m}'] = amp * 0.5
            guess[f'phi_{m}'] = phase % (np.pi / m)
        
        return guess
    
    def invert(
        self,
        images: np.ndarray,
        tol: float = 1e-10
    ) -> List[Dict]:
        """
        Full inversion with grid search over phases.
        
        Strategy:
        1. Grid search over all nonlinear phases
        2. Solve linear system at each grid point
        3. Keep solutions with small residuals
        4. Refine best solutions
        """
        n_nonlinear = len(self.nonlinear_unknowns())
        
        if n_nonlinear == 0:
            # Pure linear solve
            return self._invert_linear(images, tol)
        elif n_nonlinear == 1:
            # Single rootfind
            return self._invert_single_phase(images, tol)
        else:
            # Grid search
            return self._invert_grid(images, tol)
    
    def _invert_linear(self, images: np.ndarray, tol: float) -> List[Dict]:
        """Inversion when all parameters are linear."""
        A, b_vec, param_names = self.build_linear_system(images, {})
        
        n_params = len(param_names)
        if A.shape[0] < n_params:
            return []
        
        try:
            p, residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
            solved = dict(zip(param_names, p))
            physical = self.convert_scaled_params(solved, {})
            
            # Compute actual residuals
            res = self.equations(
                images,
                {'beta_x': physical['beta_x'], 'beta_y': physical['beta_y']},
                {k: v for k, v in physical.items() if k not in ['beta_x', 'beta_y']}
            )
            
            return [{
                'params': physical,
                'residuals': res,
                'report': {
                    'max_abs': np.max(np.abs(res)),
                    'rms': np.sqrt(np.mean(res**2))
                }
            }]
        except Exception:
            return []
    
    def _invert_single_phase(self, images: np.ndarray, tol: float) -> List[Dict]:
        """Inversion with single nonlinear phase via grid search."""
        phase_name = self.nonlinear_unknowns()[0]
        
        # Determine search range based on symmetry
        if 'phi_2' in phase_name:
            search_range = (0, np.pi)
            n_grid = 90
        elif 'phi_3' in phase_name:
            search_range = (0, 2*np.pi/3)
            n_grid = 60
        elif 'phi_4' in phase_name:
            search_range = (0, np.pi/2)
            n_grid = 45
        else:
            search_range = (0, np.pi)
            n_grid = 90
        
        # Grid search: find phase that minimizes residual
        phase_values = np.linspace(search_range[0], search_range[1], n_grid)
        candidates = []
        
        for phase_val in phase_values:
            fixed = {phase_name: phase_val}
            
            A, b_vec, param_names = self.build_linear_system(images, fixed)
            n_params = len(param_names)
            
            if A.shape[0] < n_params:
                continue
            
            try:
                # Use least squares for overdetermined system
                p, residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
                
                if rank < n_params:
                    continue
                
                solved = dict(zip(param_names, p))
                physical = self.convert_scaled_params(solved, fixed)
                
                # Validate physical constraints
                if physical['theta_E'] <= 0:
                    continue
                
                # Compute actual lens equation residuals
                res = self.equations(
                    images,
                    {'beta_x': physical['beta_x'], 'beta_y': physical['beta_y']},
                    {k: v for k, v in physical.items() if k not in ['beta_x', 'beta_y']}
                )
                
                max_res = np.max(np.abs(res))
                rms_res = np.sqrt(np.mean(res**2))
                
                candidates.append({
                    'params': physical,
                    'residuals': res,
                    'report': {
                        'max_abs': max_res,
                        'rms': rms_res,
                        'phase_val': phase_val
                    }
                })
            except Exception:
                continue
        
        if not candidates:
            return []
        
        # Sort by RMS residual and keep best solutions
        candidates.sort(key=lambda s: s['report']['rms'])
        
        # Find local minima in residual vs phase
        solutions = []
        best_rms = candidates[0]['report']['rms']
        
        for c in candidates:
            # Accept solutions within 2x of best
            if c['report']['rms'] < best_rms * 2:
                # Check for duplicates
                is_dup = False
                for sol in solutions:
                    phase_diff = abs(c['report']['phase_val'] - sol['report']['phase_val'])
                    if phase_diff < 0.1:  # ~6 degrees
                        is_dup = True
                        break
                if not is_dup:
                    solutions.append(c)
        
        return solutions[:5]  # Return up to 5 best solutions
    
    def _invert_grid(self, images: np.ndarray, tol: float) -> List[Dict]:
        """Grid search inversion for multiple phases using least squares."""
        nonlinear_names = self.nonlinear_unknowns()
        n_phases = len(nonlinear_names)
        
        # Adaptive grid resolution based on number of phases
        n_grid = 20 if n_phases == 1 else (12 if n_phases == 2 else 8)
        
        # Generate grid for each phase
        grids = []
        for name in nonlinear_names:
            if 'phi_2' in name:
                grids.append(np.linspace(0, np.pi, n_grid))
            elif 'phi_3' in name:
                grids.append(np.linspace(0, 2*np.pi/3, n_grid))
            elif 'phi_4' in name:
                grids.append(np.linspace(0, np.pi/2, n_grid))
            elif 'gamma' in name:
                grids.append(np.linspace(0, np.pi, n_grid))
            else:
                grids.append(np.linspace(0, np.pi, n_grid))
        
        # Create meshgrid of all phase combinations
        mesh = np.meshgrid(*grids)
        grid_points = np.column_stack([m.ravel() for m in mesh])
        
        candidates = []
        
        for point in grid_points:
            fixed = dict(zip(nonlinear_names, point))
            
            A, b_vec, param_names = self.build_linear_system(images, fixed)
            n_params = len(param_names)
            
            if A.shape[0] < n_params:
                continue
            
            try:
                # Use least squares for robustness
                p, residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
                
                if rank < n_params:
                    continue
                
                solved = dict(zip(param_names, p))
                physical = self.convert_scaled_params(solved, fixed)
                
                if physical['theta_E'] <= 0:
                    continue
                
                res = self.equations(
                    images,
                    {'beta_x': physical['beta_x'], 'beta_y': physical['beta_y']},
                    {k: v for k, v in physical.items() 
                     if k not in ['beta_x', 'beta_y']}
                )
                
                max_res = np.max(np.abs(res))
                rms_res = np.sqrt(np.mean(res**2))
                
                candidates.append({
                    'params': physical,
                    'residuals': res,
                    'report': {
                        'max_abs': max_res,
                        'rms': rms_res,
                        'phases': point.copy()
                    }
                })
            except Exception:
                continue
        
        if not candidates:
            return []
        
        # Sort by residual quality and keep best
        candidates.sort(key=lambda s: s['report']['max_abs'])
        
        # Remove near-duplicates
        unique = []
        for sol in candidates:
            is_dup = False
            for u in unique:
                if abs(sol['params']['theta_E'] - u['params']['theta_E']) < 0.01:
                    diff_sum = sum(
                        abs(sol['params'].get(k, 0) - u['params'].get(k, 0))
                        for k in ['a_2', 'b_2', 'a_3', 'b_3']
                    )
                    if diff_sum < 0.05:
                        is_dup = True
                        break
            if not is_dup:
                unique.append(sol)
        
        return unique[:10]


# =============================================================================
# HELPER: Root finding
# =============================================================================

class RootFinderModule:
    """Safe root finding utilities."""
    
    @staticmethod
    def find_all_roots_safe(f, x_min, x_max, n_samples=200, tol=1e-10):
        """Find all roots by sign change + bisection."""
        x_test = np.linspace(x_min, x_max, n_samples)
        f_vals = []
        
        for x in x_test:
            try:
                val = f(x)
                if np.isnan(val) or np.isinf(val):
                    f_vals.append(None)
                else:
                    f_vals.append(val)
            except Exception:
                f_vals.append(None)
        
        roots = []
        for i in range(len(x_test) - 1):
            if f_vals[i] is None or f_vals[i+1] is None:
                continue
            if f_vals[i] * f_vals[i+1] < 0:
                root = RootFinderModule._bisection(f, x_test[i], x_test[i+1], tol)
                if root is not None:
                    roots.append(root)
        
        return roots
    
    @staticmethod
    def _bisection(f, a, b, tol, max_iter=100):
        """Simple bisection."""
        try:
            fa, fb = f(a), f(b)
            if fa * fb > 0:
                return None
            
            for _ in range(max_iter):
                mid = 0.5 * (a + b)
                fm = f(mid)
                
                if abs(fm) < tol or (b - a) < tol:
                    return mid
                
                if fa * fm < 0:
                    b = mid
                else:
                    a = mid
                    fa = fm
            
            return 0.5 * (a + b)
        except Exception:
            return None


# Convenience alias
def find_all_roots_safe(f, x_min, x_max, n_samples=200, tol=1e-10):
    return RootFinderModule.find_all_roots_safe(f, x_min, x_max, n_samples, tol)
