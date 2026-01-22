"""
General Multipole Model for Gauge Gravitational Lensing

Implements Ξ as a Fourier expansion up to m_max:
Ξ(r, φ) = Ξ_0(r) + Σ_{m=1}^{m_max} [a_m * cos(m(φ - φ_m)) + b_m * sin(m(φ - φ_m))]

Key insight: The system is LINEAR in amplitudes when phases are fixed.
This enables exact solution via rootfinding (for phases) + linear solve (for amplitudes).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_model import LensModel


class MultipoleModel(LensModel):
    """
    General multipole expansion lens model.
    
    Parameters:
    - theta_E: Einstein radius (mass scale)
    - For each m from 1 to m_max:
      - amplitude_m: strength of m-th multipole
      - phase_m: orientation of m-th multipole
    - beta_x, beta_y: source position
    
    The model uses conditional linearity:
    - Phases (φ_m) are nonlinear → determined by rootfinding
    - Amplitudes and source position → determined by exact linear solve
    """
    
    def __init__(self, m_max: int = 2):
        """
        Initialize multipole model.
        
        Parameters
        ----------
        m_max : int
            Maximum multipole order (default 2 = quadrupole only)
        """
        self._m_max = m_max
    
    @property
    def name(self) -> str:
        return f"Multipole Model (m_max={self._m_max})"
    
    @property
    def m_max(self) -> int:
        return self._m_max
    
    def unknowns(self) -> List[str]:
        """All unknown parameters."""
        params = ['beta_x', 'beta_y', 'theta_E']
        for m in range(1, self._m_max + 1):
            params.extend([f'a_{m}', f'b_{m}', f'phi_{m}'])
        return params
    
    def nonlinear_unknowns(self) -> List[str]:
        """Phase angles determined by rootfinding."""
        return [f'phi_{m}' for m in range(2, self._m_max + 1)]  # phi_1 is just center offset
    
    def linear_unknowns(self) -> List[str]:
        """Parameters solvable by linear system when phases fixed."""
        params = ['beta_x', 'beta_y', 'theta_E']
        for m in range(1, self._m_max + 1):
            params.extend([f'a_{m}', f'b_{m}'])
        return params
    
    def observables(self) -> List[str]:
        return ['image_positions']
    
    def deflection(
        self,
        r: float,
        phi: float,
        theta_E: float,
        multipoles: Dict[int, Tuple[float, float, float]]
    ) -> Tuple[float, float]:
        """
        Compute deflection angle at position (r, φ).
        
        Parameters
        ----------
        r, phi : float
            Position in polar coordinates
        theta_E : float
            Einstein radius
        multipoles : dict
            {m: (a_m, b_m, phi_m)} for each multipole order
            
        Returns
        -------
        alpha_x, alpha_y : float
            Deflection in Cartesian coordinates
        """
        # Monopole contribution (Einstein ring)
        alpha_r = theta_E
        alpha_phi = 0.0
        
        # Add multipole contributions
        for m, (a_m, b_m, phi_m) in multipoles.items():
            if m == 0:
                continue  # Monopole handled separately
            
            # Radial and tangential components
            # α_r contribution from m-th multipole
            # α_φ contribution from m-th multipole
            
            delta_phi = phi - phi_m
            cos_m = np.cos(m * delta_phi)
            sin_m = np.sin(m * delta_phi)
            
            # For local model near Einstein ring:
            # Radial: proportional to cos(m(φ - φ_m))
            # Tangential: proportional to sin(m(φ - φ_m))
            
            alpha_r += theta_E * (a_m * cos_m + b_m * sin_m)
            alpha_phi += theta_E * m * (-a_m * sin_m + b_m * cos_m)
        
        # Convert to Cartesian
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        alpha_x = alpha_r * cos_phi - alpha_phi * sin_phi
        alpha_y = alpha_r * sin_phi + alpha_phi * cos_phi
        
        return alpha_x, alpha_y
    
    def predict_images(
        self,
        source_params: Dict[str, float],
        lens_params: Dict[str, float]
    ) -> np.ndarray:
        """
        Forward model: find image positions from source and lens parameters.
        
        Uses rootfinding to solve the lens equation.
        """
        beta_x = source_params['beta_x']
        beta_y = source_params['beta_y']
        theta_E = lens_params['theta_E']
        
        # Extract multipoles
        multipoles = {}
        for m in range(1, self._m_max + 1):
            a_m = lens_params.get(f'a_{m}', 0.0)
            b_m = lens_params.get(f'b_{m}', 0.0)
            phi_m = lens_params.get(f'phi_{m}', 0.0)
            multipoles[m] = (a_m, b_m, phi_m)
        
        # Find images by solving lens equation
        # β = θ - α(θ)
        # For each angle φ, find r such that lens equation is satisfied
        
        from ..inversion.root_solvers import find_all_roots
        
        def lens_eq_residual(phi):
            """Find angles where images can exist."""
            # At angle φ, try to find r such that:
            # (beta_x, beta_y) = (x, y) - (alpha_x, alpha_y)
            
            # Simplified: look for angles consistent with source direction
            r_est = theta_E  # Start near Einstein ring
            
            for _ in range(10):  # Newton iteration for r
                alpha_x, alpha_y = self.deflection(r_est, phi, theta_E, multipoles)
                x = r_est * np.cos(phi)
                y = r_est * np.sin(phi)
                
                # Residual
                res_x = x - alpha_x - beta_x
                res_y = y - alpha_y - beta_y
                
                # Update r (simple iteration)
                # θ = β + α, so r ≈ |β + α|
                target_x = beta_x + alpha_x
                target_y = beta_y + alpha_y
                r_new = np.sqrt(target_x**2 + target_y**2)
                
                if abs(r_new - r_est) < 1e-12:
                    break
                r_est = r_new
            
            # Return residual in angle
            x = r_est * np.cos(phi)
            y = r_est * np.sin(phi)
            alpha_x, alpha_y = self.deflection(r_est, phi, theta_E, multipoles)
            
            # Check if this produces the correct source angle
            pred_beta_x = x - alpha_x
            pred_beta_y = y - alpha_y
            
            # Angular residual
            beta_angle = np.arctan2(beta_y, beta_x)
            pred_angle = np.arctan2(pred_beta_y, pred_beta_x)
            
            return np.sin(pred_angle - beta_angle)
        
        # Find roots
        phi_candidates = find_all_roots(lens_eq_residual, 0, 2*np.pi, n_samples=500)
        
        images = []
        for phi in phi_candidates:
            # Refine r at this angle
            r_est = theta_E
            for _ in range(20):
                alpha_x, alpha_y = self.deflection(r_est, phi, theta_E, multipoles)
                target_x = beta_x + alpha_x
                target_y = beta_y + alpha_y
                r_new = np.sqrt(target_x**2 + target_y**2)
                
                if abs(r_new - r_est) < 1e-14:
                    break
                r_est = 0.5 * (r_est + r_new)
            
            x = r_est * np.cos(phi)
            y = r_est * np.sin(phi)
            images.append([x, y])
        
        return np.array(images) if images else np.array([]).reshape(0, 2)
    
    def equations(
        self,
        images: np.ndarray,
        source_params: Dict[str, float],
        lens_params: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute residuals of lens equation for each image.
        """
        beta_x = source_params['beta_x']
        beta_y = source_params['beta_y']
        theta_E = lens_params['theta_E']
        
        # Extract multipoles
        multipoles = {}
        for m in range(1, self._m_max + 1):
            a_m = lens_params.get(f'a_{m}', 0.0)
            b_m = lens_params.get(f'b_{m}', 0.0)
            phi_m = lens_params.get(f'phi_{m}', 0.0)
            multipoles[m] = (a_m, b_m, phi_m)
        
        residuals = []
        for img in images:
            x, y = img
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            
            alpha_x, alpha_y = self.deflection(r, phi, theta_E, multipoles)
            
            # Predicted source position
            pred_beta_x = x - alpha_x
            pred_beta_y = y - alpha_y
            
            residuals.append(pred_beta_x - beta_x)
            residuals.append(pred_beta_y - beta_y)
        
        return np.array(residuals)
    
    def build_linear_system(
        self,
        images: np.ndarray,
        fixed_nonlinear: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build linear system for amplitudes given fixed phases.
        
        The lens equation β = θ - α is linear in (β, θ_E, a_m, b_m) when φ_m are fixed.
        """
        n = len(images)
        
        # Extract fixed phases
        phases = {}
        for m in range(2, self._m_max + 1):
            key = f'phi_{m}'
            phases[m] = fixed_nonlinear.get(key, 0.0)
        phases[1] = 0.0  # m=1 phase is arbitrary (can absorb into β)
        
        # Parameters: [beta_x, beta_y, theta_E, a_1, b_1, a_2, b_2, ...]
        param_names = ['beta_x', 'beta_y', 'theta_E']
        for m in range(1, self._m_max + 1):
            param_names.extend([f'a_{m}', f'b_{m}'])
        
        n_params = len(param_names)
        A = np.zeros((2*n, n_params))
        b_vec = np.zeros(2*n)
        
        for i, (x, y) in enumerate(images):
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            # Lens equation: β = θ - α
            # x = beta_x + alpha_x
            # y = beta_y + alpha_y
            #
            # alpha = theta_E * [unit_r + sum_m (a_m cos(m(phi-phi_m)) + b_m sin(m(phi-phi_m))) * radial
            #                        + m * (-a_m sin(...) + b_m cos(...)) * tangential]
            
            # x-equation coefficients
            A[2*i, 0] = 1  # beta_x
            A[2*i, 1] = 0  # beta_y
            A[2*i, 2] = cos_phi  # theta_E (monopole contributes theta_E * cos(phi))
            
            col = 3
            for m in range(1, self._m_max + 1):
                phi_m = phases[m]
                delta = phi - phi_m
                cos_m = np.cos(m * delta)
                sin_m = np.sin(m * delta)
                
                # Radial contribution to alpha
                # alpha_r from a_m: theta_E * a_m * cos(m*delta)
                # alpha_r from b_m: theta_E * b_m * sin(m*delta)
                # Tangential contribution
                # alpha_phi from a_m: theta_E * m * (-a_m) * sin(m*delta)
                # alpha_phi from b_m: theta_E * m * b_m * cos(m*delta)
                
                # alpha_x = alpha_r * cos(phi) - alpha_phi * sin(phi)
                # Coefficient of a_m in alpha_x:
                #   theta_E * [cos_m * cos_phi - m * (-sin_m) * sin_phi]
                #   = theta_E * [cos_m * cos_phi + m * sin_m * sin_phi]
                # But we want coefficient assuming theta_E = 1 for now, then multiply by theta_E
                
                # Actually, for linear system, we need to be careful about theta_E dependence.
                # Since all deflection terms are proportional to theta_E, let's absorb it:
                # Define: A_m = theta_E * a_m, B_m = theta_E * b_m
                # Then the system is linear in [beta_x, beta_y, theta_E, A_1, B_1, A_2, B_2, ...]
                
                # alpha_x from m-th multipole:
                # = (A_m * cos_m + B_m * sin_m) * cos_phi + m*(A_m * sin_m - B_m * cos_m) * sin_phi
                
                coeff_A = cos_m * cos_phi + m * sin_m * sin_phi
                coeff_B = sin_m * cos_phi - m * cos_m * sin_phi
                
                A[2*i, col] = coeff_A      # A_m = theta_E * a_m
                A[2*i, col+1] = coeff_B    # B_m = theta_E * b_m
                col += 2
            
            b_vec[2*i] = x
            
            # y-equation coefficients
            A[2*i+1, 0] = 0  # beta_x
            A[2*i+1, 1] = 1  # beta_y
            A[2*i+1, 2] = sin_phi  # theta_E
            
            col = 3
            for m in range(1, self._m_max + 1):
                phi_m = phases[m]
                delta = phi - phi_m
                cos_m = np.cos(m * delta)
                sin_m = np.sin(m * delta)
                
                # alpha_y from m-th multipole:
                # = (A_m * cos_m + B_m * sin_m) * sin_phi - m*(A_m * sin_m - B_m * cos_m) * cos_phi
                
                coeff_A = cos_m * sin_phi - m * sin_m * cos_phi
                coeff_B = sin_m * sin_phi + m * cos_m * cos_phi
                
                A[2*i+1, col] = coeff_A
                A[2*i+1, col+1] = coeff_B
                col += 2
            
            b_vec[2*i+1] = y
        
        # Rename parameters to indicate scaling
        scaled_names = ['beta_x', 'beta_y', 'theta_E']
        for m in range(1, self._m_max + 1):
            scaled_names.extend([f'A_{m}', f'B_{m}'])
        
        return A, b_vec, scaled_names
    
    def convert_scaled_params(
        self,
        solved: Dict[str, float],
        phases: Dict[int, float]
    ) -> Dict[str, float]:
        """
        Convert scaled parameters back to physical.
        
        A_m = theta_E * a_m → a_m = A_m / theta_E
        B_m = theta_E * b_m → b_m = B_m / theta_E
        """
        theta_E = solved['theta_E']
        
        result = {
            'beta_x': solved['beta_x'],
            'beta_y': solved['beta_y'],
            'theta_E': theta_E
        }
        
        for m in range(1, self._m_max + 1):
            A_m = solved.get(f'A_{m}', 0.0)
            B_m = solved.get(f'B_{m}', 0.0)
            
            if abs(theta_E) > 1e-15:
                a_m = A_m / theta_E
                b_m = B_m / theta_E
            else:
                a_m = 0.0
                b_m = 0.0
            
            result[f'a_{m}'] = a_m
            result[f'b_{m}'] = b_m
            result[f'phi_{m}'] = phases.get(m, 0.0)
        
        return result
    
    def consistency_function(
        self,
        images: np.ndarray,
        nonlinear_value: float,
        nonlinear_name: str
    ) -> float:
        """
        Consistency function for rootfinding.
        
        For single phase (m=2 model), find where extra equation is satisfied.
        """
        # Extract which phase this is
        if not nonlinear_name.startswith('phi_'):
            raise ValueError(f"Unknown nonlinear parameter: {nonlinear_name}")
        
        m = int(nonlinear_name.split('_')[1])
        
        # Build fixed phases dict
        fixed = {nonlinear_name: nonlinear_value}
        
        # Build linear system
        A, b_vec, param_names = self.build_linear_system(images, fixed)
        
        n_eq = A.shape[0]
        n_params = A.shape[1]
        
        if n_eq <= n_params:
            return 0.0  # Underdetermined
        
        # Solve using first n_params equations
        from ..inversion.exact_solvers import solve_linear_subset
        
        rows = list(range(n_params))
        try:
            p, success = solve_linear_subset(A, b_vec, rows)
            if not success:
                return float('nan')
        except Exception:
            return float('nan')
        
        # Evaluate residual on next equation
        h = np.dot(A[n_params], p) - b_vec[n_params]
        return h
    
    def initial_guess(self, images: np.ndarray) -> Dict[str, float]:
        """
        Heuristic initial guess using moment methods.
        """
        n = len(images)
        z = images[:, 0] + 1j * images[:, 1]
        
        # Mean position
        z_mean = np.mean(z)
        beta_x_est = z_mean.real
        beta_y_est = z_mean.imag
        
        # Mean radius
        radii = np.abs(z)
        theta_E_est = np.mean(radii)
        
        guess = {
            'beta_x': beta_x_est,
            'beta_y': beta_y_est,
            'theta_E': theta_E_est
        }
        
        # Estimate multipole amplitudes and phases from moments
        for m in range(1, self._m_max + 1):
            # m-th moment
            z_centered = z - z_mean
            moment_m = np.mean(z_centered ** m)
            
            amp = np.abs(moment_m) / theta_E_est if theta_E_est > 0 else 0
            phase = np.angle(moment_m) / m
            
            guess[f'a_{m}'] = amp * 0.5  # Split between a and b
            guess[f'b_{m}'] = amp * 0.5
            guess[f'phi_{m}'] = phase
        
        return guess
    
    def invert(
        self,
        images: np.ndarray,
        tol: float = 1e-12
    ) -> List[Dict]:
        """
        Full inversion for general multipole model.
        
        Strategy:
        1. For m=2 only: single rootfind over phi_2
        2. For higher m: nested rootfind or grid search
        """
        from ..inversion.root_solvers import find_all_roots
        from ..inversion.exact_solvers import solve_linear_subset, choose_invertible_subset
        from ..inversion.diagnostics import residual_report
        
        if self._m_max == 2:
            # Single phase to find
            return self._invert_m2(images, tol)
        else:
            # Multiple phases - use nested or grid approach
            return self._invert_general(images, tol)
    
    def _invert_m2(self, images: np.ndarray, tol: float) -> List[Dict]:
        """Inversion for m_max = 2 (single nonlinear phase)."""
        from ..inversion.root_solvers import find_all_roots
        from ..inversion.exact_solvers import solve_linear_subset, choose_invertible_subset
        from ..inversion.diagnostics import residual_report
        
        def h(phi):
            return self.consistency_function(images, phi, 'phi_2')
        
        # Search in [0, π) due to m=2 symmetry
        roots = find_all_roots(h, 0, np.pi, n_samples=500, tol=tol)
        
        solutions = []
        for phi_2 in roots:
            fixed = {'phi_2': phi_2}
            A, b_vec, param_names = self.build_linear_system(images, fixed)
            
            n_params = len(param_names)
            rows, success = choose_invertible_subset(A, b_vec, n_params)
            if not success:
                continue
            
            p, ok = solve_linear_subset(A, b_vec, rows)
            if not ok:
                continue
            
            solved = dict(zip(param_names, p))
            physical = self.convert_scaled_params(solved, {1: 0.0, 2: phi_2})
            
            valid, msg = self.validate_params(physical)
            if not valid:
                continue
            
            # Compute residuals
            source_params = {'beta_x': physical['beta_x'], 'beta_y': physical['beta_y']}
            lens_params = {k: v for k, v in physical.items() if k not in ['beta_x', 'beta_y']}
            residuals = self.equations(images, source_params, lens_params)
            
            report = residual_report(residuals)
            
            solutions.append({
                'params': physical,
                'residuals': residuals,
                'report': report
            })
        
        return solutions
    
    def _invert_general(self, images: np.ndarray, tol: float) -> List[Dict]:
        """Inversion for m_max > 2 (multiple nonlinear phases)."""
        # Grid search over phases, then refine
        # This is expensive but deterministic
        
        from ..inversion.exact_solvers import solve_linear_subset, choose_invertible_subset
        from ..inversion.diagnostics import residual_report
        
        n_phases = self._m_max - 1  # phi_2, phi_3, ...
        n_grid = 20  # Points per phase
        
        solutions = []
        best_residual = float('inf')
        
        # Generate grid
        phase_values = np.linspace(0, np.pi, n_grid)
        
        # For simplicity, do exhaustive grid for m_max <= 4
        if n_phases == 1:
            grid = [(p,) for p in phase_values]
        elif n_phases == 2:
            grid = [(p2, p3) for p2 in phase_values for p3 in phase_values]
        else:
            # Too expensive - use random sampling
            grid = [tuple(np.random.uniform(0, np.pi, n_phases)) for _ in range(1000)]
        
        for phases_tuple in grid:
            fixed = {}
            for i, m in enumerate(range(2, self._m_max + 1)):
                fixed[f'phi_{m}'] = phases_tuple[i]
            
            A, b_vec, param_names = self.build_linear_system(images, fixed)
            
            n_params = len(param_names)
            if A.shape[0] < n_params:
                continue
            
            rows, success = choose_invertible_subset(A, b_vec, n_params)
            if not success:
                continue
            
            p, ok = solve_linear_subset(A, b_vec, rows)
            if not ok:
                continue
            
            solved = dict(zip(param_names, p))
            phases_dict = {1: 0.0}
            for i, m in enumerate(range(2, self._m_max + 1)):
                phases_dict[m] = phases_tuple[i]
            
            physical = self.convert_scaled_params(solved, phases_dict)
            
            valid, msg = self.validate_params(physical)
            if not valid:
                continue
            
            # Compute residuals
            source_params = {'beta_x': physical['beta_x'], 'beta_y': physical['beta_y']}
            lens_params = {k: v for k, v in physical.items() if k not in ['beta_x', 'beta_y']}
            residuals = self.equations(images, source_params, lens_params)
            
            max_res = np.max(np.abs(residuals))
            
            # Keep only solutions with good residuals
            if max_res < tol * 1000:  # Relaxed threshold for grid
                report = residual_report(residuals)
                solutions.append({
                    'params': physical,
                    'residuals': residuals,
                    'report': report
                })
        
        # Sort by residual quality
        solutions.sort(key=lambda s: s['report']['max_abs'])
        
        return solutions[:10]  # Return top 10
