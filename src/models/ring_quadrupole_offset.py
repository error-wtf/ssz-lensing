"""
Ring + Quadrupole + Offset Model (Minimal m=2 Model)

This is the existing validated minimal model, now as a plugin for the framework.
Implements exact inversion via rootfinding + linear solve.

Parameters:
- theta_E: Einstein radius
- a: Radial quadrupole coefficient
- b: Tangential quadrupole coefficient  
- phi_gamma: Quadrupole axis orientation
- beta_x, beta_y: Source position

For 4 images, this model is exactly solvable (no fitting).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_model import LensModel


class RingQuadrupoleOffsetModel(LensModel):
    """
    Minimal lens model: Ring + Quadrupole (m=2) + Source Offset.
    
    Local approximation valid near the Einstein radius.
    Implements the no-fit inversion strategy.
    """
    
    @property
    def name(self) -> str:
        return "Ring + Quadrupole + Offset (m=2)"
    
    @property
    def m_max(self) -> int:
        return 2
    
    def unknowns(self) -> List[str]:
        return ['beta_x', 'beta_y', 'theta_E', 'a', 'b', 'phi_gamma']
    
    def nonlinear_unknowns(self) -> List[str]:
        return ['phi_gamma']
    
    def linear_unknowns(self) -> List[str]:
        return ['beta_x', 'beta_y', 'theta_E', 'a', 'b']
    
    def observables(self) -> List[str]:
        return ['image_positions']
    
    def predict_images(
        self,
        source_params: Dict[str, float],
        lens_params: Dict[str, float]
    ) -> np.ndarray:
        """
        Generate image positions from source and lens parameters.
        
        This solves the lens equation for image positions given the source.
        """
        beta_x = source_params['beta_x']
        beta_y = source_params['beta_y']
        theta_E = lens_params['theta_E']
        a = lens_params['a']
        b = lens_params['b']
        phi_gamma = lens_params['phi_gamma']
        
        # Find image angles by solving angular equation
        def angular_equation(phi):
            """Equation for image angle given source direction."""
            beta_angle = np.arctan2(beta_y, beta_x)
            beta_mag = np.sqrt(beta_x**2 + beta_y**2)
            
            # Angular deviation from source direction
            delta_phi = phi - beta_angle
            
            # From lens equation in polar form
            lhs = beta_mag * np.sin(delta_phi)
            rhs = b * theta_E * np.sin(2 * (phi - phi_gamma))
            
            return lhs - rhs
        
        # Find roots (image angles)
        from ..inversion.root_solvers import find_all_roots
        phi_images = find_all_roots(angular_equation, 0, 2*np.pi, n_samples=500)
        
        if len(phi_images) == 0:
            return np.array([])
        
        # Compute radii for each image angle
        images = []
        for phi in phi_images:
            # Radial equation
            cos_term = np.cos(phi - np.arctan2(beta_y, beta_x))
            beta_mag = np.sqrt(beta_x**2 + beta_y**2)
            
            # r from lens equation
            r = theta_E * (1 + a) + beta_mag * cos_term / (1 - a)
            r = max(r, 0.1 * theta_E)  # Prevent negative radii
            
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            images.append([x, y])
        
        return np.array(images)
    
    def equations(
        self,
        images: np.ndarray,
        source_params: Dict[str, float],
        lens_params: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute residuals: predicted β - input β for each image.
        """
        beta_x = source_params['beta_x']
        beta_y = source_params['beta_y']
        theta_E = lens_params['theta_E']
        a = lens_params['a']
        b = lens_params['b']
        phi_gamma = lens_params['phi_gamma']
        
        residuals = []
        for img in images:
            x, y = img
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            
            # Deflection in local model
            alpha_r = theta_E * (1 + a * (r/theta_E - 1) + b * np.cos(2*(phi - phi_gamma)))
            alpha_phi = -theta_E * b * np.sin(2*(phi - phi_gamma))
            
            # Convert to Cartesian
            alpha_x = alpha_r * np.cos(phi) - alpha_phi * np.sin(phi)
            alpha_y = alpha_r * np.sin(phi) + alpha_phi * np.cos(phi)
            
            # Predicted source position
            beta_x_pred = x - alpha_x
            beta_y_pred = y - alpha_y
            
            # Residuals
            residuals.append(beta_x_pred - beta_x)
            residuals.append(beta_y_pred - beta_y)
        
        return np.array(residuals)
    
    def build_linear_system(
        self,
        images: np.ndarray,
        fixed_nonlinear: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build Ax = b for linear unknowns [beta_x, beta_y, theta_E, a, b].
        
        The system is linear in these when phi_gamma is fixed.
        """
        phi_gamma = fixed_nonlinear['phi_gamma']
        n_images = len(images)
        
        # Parameters: [beta_x, beta_y, theta_E, a, b]
        param_names = ['beta_x', 'beta_y', 'theta_E', 'a', 'b']
        n_params = 5
        
        A = np.zeros((2 * n_images, n_params))
        b = np.zeros(2 * n_images)
        
        for i, img in enumerate(images):
            x, y = img
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            cos_2gamma = np.cos(2 * (phi - phi_gamma))
            sin_2gamma = np.sin(2 * (phi - phi_gamma))
            
            # Lens equation: beta = theta - alpha
            # beta_x = x - alpha_x
            # beta_y = y - alpha_y
            #
            # alpha_r = theta_E * (1 + a*(r/theta_E - 1) + b*cos(2(phi-phi_gamma)))
            # alpha_phi = -theta_E * b * sin(2(phi-phi_gamma))
            #
            # alpha_x = alpha_r * cos(phi) - alpha_phi * sin(phi)
            # alpha_y = alpha_r * sin(phi) + alpha_phi * cos(phi)
            
            # Coefficients for beta_x equation (row 2*i)
            A[2*i, 0] = 1  # beta_x
            A[2*i, 1] = 0  # beta_y
            A[2*i, 2] = -(cos_phi * (1 - (r/1.0 - 1)*0) + cos_phi*cos_2gamma + sin_phi*sin_2gamma*0)
            # Simplified: need to be more careful
            
            # Let's redo this more carefully
            # alpha_x = theta_E * [cos(phi) + a*(r/theta_E - 1)*cos(phi) + b*cos_2gamma*cos(phi) + b*sin_2gamma*sin(phi)]
            # alpha_y = theta_E * [sin(phi) + a*(r/theta_E - 1)*sin(phi) + b*cos_2gamma*sin(phi) - b*sin_2gamma*cos(phi)]
            
            # beta_x = x - alpha_x
            # beta_y = y - alpha_y
            
            # Rewrite:
            # beta_x + theta_E*cos(phi) + theta_E*a*(r/theta_E - 1)*cos(phi) + theta_E*b*(cos_2gamma*cos(phi) + sin_2gamma*sin(phi)) = x
            # beta_y + theta_E*sin(phi) + theta_E*a*(r/theta_E - 1)*sin(phi) + theta_E*b*(cos_2gamma*sin(phi) - sin_2gamma*cos(phi)) = y
            
            # Note: cos_2gamma*cos(phi) + sin_2gamma*sin(phi) = cos(2(phi-phi_gamma) - phi) = cos(phi - 2*phi_gamma)
            # Let's keep it explicit
            
            coeff_theta_E_x = cos_phi
            coeff_a_x = (r - 1) * cos_phi  # Note: we absorb theta_E into a rescaling
            coeff_b_x = cos_2gamma * cos_phi + sin_2gamma * sin_phi
            
            coeff_theta_E_y = sin_phi
            coeff_a_y = (r - 1) * sin_phi
            coeff_b_y = cos_2gamma * sin_phi - sin_2gamma * cos_phi
            
            # Equation: beta_x + theta_E*coeff + a*theta_E*coeff_a + b*theta_E*coeff_b = x
            # Rearrange: beta_x + theta_E*(coeff + ...) = x
            # But we want theta_E as separate unknown, so:
            # beta_x = x - theta_E*coeff_theta_E - a*coeff_a - b*coeff_b (approximately, if a,b are normalized)
            
            # Actually, let's use the formulation where a, b are relative to theta_E
            # alpha_r/theta_E = 1 + a*(r/theta_E - 1) + b*cos(2(phi - phi_gamma))
            # 
            # Let rho = r (we measure r in some units)
            # Let's assume theta_E ~ 1 for now and solve for relative quantities
            
            # Simpler approach: use the standard form
            # The lens equation at image i is:
            # beta = theta_i - alpha(theta_i)
            # We can write this as a linear system in the unknowns
            
            # For the local model:
            # x - beta_x = theta_E * [1 + a*(r_i/theta_E - 1) + b*cos(2(phi_i - phi_gamma))] * cos(phi_i)
            #            + theta_E * b * sin(2(phi_i - phi_gamma)) * sin(phi_i)
            
            # Let's define:
            # c_i = cos(phi_i), s_i = sin(phi_i)
            # C_i = cos(2(phi_i - phi_gamma)), S_i = sin(2(phi_i - phi_gamma))
            
            c = cos_phi
            s = sin_phi
            C = cos_2gamma
            S = sin_2gamma
            
            # x - beta_x = theta_E * c + a*(r - theta_E)*c + b*theta_E*(C*c + S*s)
            # y - beta_y = theta_E * s + a*(r - theta_E)*s + b*theta_E*(C*s - S*c)
            
            # Rearrange for linear system [beta_x, beta_y, theta_E, a, b]:
            # -beta_x + theta_E*(c - a + b*(C*c + S*s)) + a*r*c = x
            # Wait, this mixes a with theta_E. Let's be more careful.
            
            # x - beta_x = theta_E*c + a*r*c - a*theta_E*c + b*theta_E*C*c + b*theta_E*S*s
            # x = beta_x + theta_E*c*(1 - a) + a*r*c + b*theta_E*(C*c + S*s)
            
            # If we treat (theta_E, a, b) as separate unknowns, we have:
            # x = beta_x + theta_E*(c - a*c + b*C*c + b*S*s) + a*r*c
            # This is NOT linear because of a*theta_E cross term.
            
            # Solution: Redefine parameters
            # Let: A = theta_E * (1 - a), B = theta_E * b
            # Then: x = beta_x + A*c + a*r*c + B*(C*c + S*s)
            #       x = beta_x + (A + a*r)*c + B*(C*c + S*s)
            # Still not clean because A contains a.
            
            # Better: Let's use a different parametrization
            # Define: p1 = theta_E, p2 = theta_E * a, p3 = theta_E * b
            # Then:
            # x - beta_x = p1*c + p2*(r/p1 - 1)*c + p3*(C*c + S*s)
            # This is still nonlinear in p1, p2.
            
            # BEST APPROACH: Iterate. Start with theta_E estimate from mean radius.
            # Then solve linear system for (beta_x, beta_y, a, b) at fixed theta_E, phi_gamma.
            # Then update theta_E.
            
            # For now, let's use the simpler approach from the original code:
            # Assume theta_E ≈ mean(r_i), then:
            # x_i ≈ beta_x + theta_E * [c_i + a*(r_i/theta_E - 1)*c_i + b*(C_i*c_i + S_i*s_i)]
            
            # Actually, the original code uses a clever trick:
            # Define delta_r_i = r_i - theta_E
            # Then: r_i/theta_E - 1 = delta_r_i / theta_E ≈ delta_r_i (if theta_E ~ 1)
            
            # For exact solution, use:
            # x_i - theta_E*c_i - delta_r_i*a*c_i - theta_E*b*(C_i*c_i + S_i*s_i) = beta_x
            # 
            # Parameters: [beta_x, beta_y, theta_E, a, b]
            # where a, b are scaled appropriately
            
            # Let's use dimensionless form with r_mean = mean radius
            pass
        
        # Use the approach from the original code
        return self._build_linear_system_original(images, phi_gamma)
    
    def _build_linear_system_original(
        self,
        images: np.ndarray,
        phi_gamma: float
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build linear system using the original validated approach.
        
        Parameters: p = [beta_x, beta_y, theta_E, a, b]
        
        Equations derived from lens equation in local approximation.
        """
        n = len(images)
        param_names = ['beta_x', 'beta_y', 'theta_E', 'a', 'b']
        
        A = np.zeros((2*n, 5))
        b_vec = np.zeros(2*n)
        
        for i, (x, y) in enumerate(images):
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            
            c = np.cos(phi)
            s = np.sin(phi)
            C = np.cos(2*(phi - phi_gamma))
            S = np.sin(2*(phi - phi_gamma))
            
            # x-equation (row 2*i):
            # beta_x = x - alpha_x
            # alpha_x = theta_E * c + a*(r - theta_E)*c + b*theta_E*(C*c + S*s)
            # beta_x = x - theta_E*c - a*r*c + a*theta_E*c - b*theta_E*(C*c + S*s)
            # beta_x = x - theta_E*c*(1 - a) - a*r*c - b*theta_E*(C*c + S*s)
            #
            # Rearrange: beta_x + theta_E*c*(1-a) + a*r*c + b*theta_E*(C*c + S*s) = x
            # But this has a*theta_E term (nonlinear).
            #
            # Use approximation: near ring, 1-a ≈ 1, so:
            # beta_x + theta_E*c + a*(r - theta_E)*c + b*theta_E*(C*c + S*s) = x
            #
            # Let's define auxiliary: rho_i = r_i (measured)
            # Then we solve for unknowns at first order.
            
            # From the lens equation:
            # x = beta_x + theta_E * [c + a*(r/theta_E - 1)*c + b*(C*c + S*s)]
            # y = beta_y + theta_E * [s + a*(r/theta_E - 1)*s + b*(C*s - S*c)]
            
            # At leading order (theta_E ~ r_mean, a,b small):
            # x ≈ beta_x + theta_E*c + a*(r - theta_E)*c + b*theta_E*(C*c + S*s)
            # y ≈ beta_y + theta_E*s + a*(r - theta_E)*s + b*theta_E*(C*s - S*c)
            
            # Rewrite with unknowns [beta_x, beta_y, theta_E, a, b]:
            # beta_x + theta_E*(c*(1-a) + b*(C*c + S*s)) + a*r*c = x
            #
            # This is nonlinear in (theta_E, a). Use iterative or substitute.
            #
            # Practical approach: Use mean radius as theta_E estimate, then refine.
            # For exact solve, use the formulation:
            #
            # x = beta_x + (theta_E - a*theta_E)*c + a*r*c + b*theta_E*(C*c + S*s)
            # x = beta_x + theta_E*c - a*theta_E*c + a*r*c + b*theta_E*(C*c + S*s)
            # x = beta_x + theta_E*c + a*(r - theta_E)*c + b*theta_E*(C*c + S*s)
            #
            # Define: q = theta_E (unknown), p = a (unknown), m = b (unknown)
            # x = beta_x + q*c + p*(r - q)*c + m*q*(C*c + S*s)
            # x = beta_x + q*c + p*r*c - p*q*c + m*q*(C*c + S*s)
            # x = beta_x + q*c*(1 - p) + p*r*c + m*q*(C*c + S*s)
            #
            # If p << 1 (weak quadrupole), then (1-p) ≈ 1:
            # x ≈ beta_x + q*c + p*r*c + m*q*(C*c + S*s)
            # x ≈ beta_x + q*(c + m*(C*c + S*s)) + p*r*c
            #
            # This IS linear in [beta_x, q, p, m] = [beta_x, theta_E, a, b]!
            # (Assuming weak quadrupole approximation)
            
            # Coefficients:
            # x = beta_x * 1 + theta_E * (c + b*(C*c + S*s)) + a * r*c + b * theta_E*(C*c + S*s)
            # Hmm, still has b*theta_E term.
            
            # Let's use the formulation with combined parameters:
            # Define: B = b * theta_E (effective quadrupole)
            # Then: x = beta_x + theta_E*c + a*r*c + B*(C*c + S*s) - a*theta_E*c
            #       x = beta_x + theta_E*(c - a*c) + a*r*c + B*(C*c + S*s)
            #       x = beta_x + theta_E*c*(1-a) + a*r*c + B*(C*c + S*s)
            #
            # If we solve for [beta_x, beta_y, theta_E*(1-a), a, B]:
            # Let T = theta_E*(1-a), then:
            # x = beta_x + T*c + a*r*c + B*(C*c + S*s)
            # This IS linear in [beta_x, beta_y, T, a, B]!
            #
            # After solving, recover theta_E = T/(1-a), b = B/theta_E
            
            T_coeff_x = c
            a_coeff_x = r * c
            B_coeff_x = C*c + S*s
            
            T_coeff_y = s
            a_coeff_y = r * s
            B_coeff_y = C*s - S*c
            
            # Row for x-equation
            A[2*i, 0] = 1           # beta_x
            A[2*i, 1] = 0           # beta_y
            A[2*i, 2] = T_coeff_x   # T = theta_E*(1-a)
            A[2*i, 3] = a_coeff_x   # a
            A[2*i, 4] = B_coeff_x   # B = b*theta_E
            b_vec[2*i] = x
            
            # Row for y-equation
            A[2*i+1, 0] = 0           # beta_x
            A[2*i+1, 1] = 1           # beta_y
            A[2*i+1, 2] = T_coeff_y   # T
            A[2*i+1, 3] = a_coeff_y   # a
            A[2*i+1, 4] = B_coeff_y   # B
            b_vec[2*i+1] = y
        
        # Note: param_names are now [beta_x, beta_y, T, a, B]
        # where T = theta_E*(1-a), B = b*theta_E
        # Need to convert back after solving
        param_names = ['beta_x', 'beta_y', 'T', 'a', 'B']
        
        return A, b_vec, param_names
    
    def convert_params(self, solved: Dict[str, float]) -> Dict[str, float]:
        """
        Convert from solved parameters [T, a, B] to physical [theta_E, a, b].
        
        T = theta_E * (1 - a)
        B = b * theta_E
        
        Therefore:
        theta_E = T / (1 - a)
        b = B / theta_E
        """
        T = solved['T']
        a = solved['a']
        B = solved['B']
        
        theta_E = T / (1 - a)
        b = B / theta_E
        
        return {
            'beta_x': solved['beta_x'],
            'beta_y': solved['beta_y'],
            'theta_E': theta_E,
            'a': a,
            'b': b
        }
    
    def consistency_function(
        self,
        images: np.ndarray,
        nonlinear_value: float,
        nonlinear_name: str = 'phi_gamma'
    ) -> float:
        """
        Consistency function h(phi_gamma) whose zeros give valid solutions.
        
        Strategy: solve 5 equations for 5 unknowns, check 6th equation.
        The sign change indicates a root.
        """
        if nonlinear_name != 'phi_gamma':
            raise ValueError(f"Unknown nonlinear parameter: {nonlinear_name}")
        
        phi_gamma = nonlinear_value
        
        # Build linear system
        A, b_vec, param_names = self._build_linear_system_original(images, phi_gamma)
        
        n_eq = A.shape[0]
        n_params = A.shape[1]
        
        if n_eq < n_params:
            return float('nan')
        
        # Try to solve using first 5 rows
        from ..inversion.exact_solvers import solve_linear_subset
        
        rows = [0, 1, 2, 3, 4]  # First 5 equations
        try:
            p, success = solve_linear_subset(A, b_vec, rows)
            if not success:
                return float('nan')
        except Exception:
            return float('nan')
        
        # Evaluate residual on 6th equation (or any unused equation)
        if n_eq > 5:
            # Use equation 5 (0-indexed) which is 6th equation
            residual_row = 5
            h = np.dot(A[residual_row], p) - b_vec[residual_row]
            return h
        
        return 0.0
    
    def initial_guess(self, images: np.ndarray) -> Dict[str, float]:
        """
        Heuristic initial guess from image positions.
        
        Uses moment method:
        - theta_E from mean radius
        - beta from centroid offset
        - phi_gamma from second moment orientation
        """
        n = len(images)
        
        # Mean position (should be near lens center, i.e., origin)
        x_mean = np.mean(images[:, 0])
        y_mean = np.mean(images[:, 1])
        
        # Radii
        radii = np.sqrt(images[:, 0]**2 + images[:, 1]**2)
        theta_E_est = np.mean(radii)
        
        # Source offset estimate from centroid
        # beta ≈ centroid offset (for symmetric configurations)
        beta_x_est = x_mean
        beta_y_est = y_mean
        
        # Quadrupole orientation from second moment
        # M_2 = sum(z_i^2) where z = x + iy
        z = images[:, 0] + 1j * images[:, 1]
        m2 = np.mean(z**2)
        phi_gamma_est = 0.5 * np.angle(m2)  # Divide by 2 for m=2
        
        # Quadrupole amplitude estimate
        # |b| ≈ spread in radii / theta_E
        r_spread = np.std(radii)
        b_est = r_spread / theta_E_est if theta_E_est > 0 else 0.1
        
        return {
            'beta_x': beta_x_est,
            'beta_y': beta_y_est,
            'theta_E': theta_E_est,
            'a': 0.0,  # Start with no radial slope
            'b': b_est,
            'phi_gamma': phi_gamma_est
        }
    
    def invert(
        self,
        images: np.ndarray,
        phi_gamma_range: Tuple[float, float] = (0, np.pi),
        tol: float = 1e-12
    ) -> List[Dict]:
        """
        Full inversion: find all valid parameter sets.
        
        Returns list of solutions with residual info.
        """
        from ..inversion.root_solvers import find_all_roots
        from ..inversion.exact_solvers import solve_linear_subset, choose_invertible_subset
        from ..inversion.diagnostics import residual_report
        
        # Find roots of consistency function
        def h(phi):
            return self.consistency_function(images, phi, 'phi_gamma')
        
        roots = find_all_roots(h, phi_gamma_range[0], phi_gamma_range[1], 
                               n_samples=500, tol=tol)
        
        solutions = []
        for phi_gamma in roots:
            # Build and solve linear system
            A, b_vec, param_names = self._build_linear_system_original(images, phi_gamma)
            
            # Find best invertible subset
            rows, success = choose_invertible_subset(A, b_vec, 5)
            if not success:
                continue
            
            p, ok = solve_linear_subset(A, b_vec, rows)
            if not ok:
                continue
            
            # Convert to physical parameters
            solved = dict(zip(param_names, p))
            physical = self.convert_params(solved)
            physical['phi_gamma'] = phi_gamma
            
            # Validate
            valid, msg = self.validate_params(physical)
            if not valid:
                continue
            
            # Compute residuals
            source_params = {'beta_x': physical['beta_x'], 'beta_y': physical['beta_y']}
            lens_params = {k: physical[k] for k in ['theta_E', 'a', 'b', 'phi_gamma']}
            residuals = self.equations(images, source_params, lens_params)
            
            report = residual_report(residuals)
            
            solutions.append({
                'params': physical,
                'residuals': residuals,
                'report': report
            })
        
        return solutions
