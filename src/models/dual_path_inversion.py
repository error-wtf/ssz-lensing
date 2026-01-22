"""
Dual-Path Inversion Framework: Algebraic (A) + Phase Scan (B)

Two strictly separated inversion paths sharing the SAME forward model:

PATH A - Algebraic Components (No-Fit, CANONICAL):
    - Parameters: (a_m, b_m), (gamma_1, gamma_2), theta_E, beta^(k)
    - Solve: exact linear (or exact-subset + consistency check)
    - Phase is OUTPUT: phi_m = atan2(b_m, a_m)
    - This is the REFERENCE mode - deterministic, DOF-compliant

PATH B - Phase Scan Mode (Hypothesis Test):
    - Parameters: (A_m, phi_m), (gamma, phi_gamma) kept explicitly
    - Controlled scan over phi values (NOT free-flight optimization)
    - For each phi: solve all OTHER params linearly exact
    - Report: residual landscape, best candidates, degeneracy hints
    - Explicitly labeled as "Scan/Hypothesis Test mode"

HARD RULE: Both paths use IDENTICAL forward_model for alpha_red.
Scan results MUST be cross-checked against Path A derived phases.

Authors: Carmen N. Wrede, Lino P. Casu
License: Anti-Capitalist Software License v1.4
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# SHARED FORWARD MODEL (used by BOTH paths)
# =============================================================================

def reduced_deflection(
    theta: np.ndarray,
    theta_E: float,
    multipoles: Dict[int, Tuple[float, float]],
    gamma_1: float = 0.0,
    gamma_2: float = 0.0
) -> np.ndarray:
    """
    Compute reduced deflection alpha_red(theta) for given parameters.
    
    THIS IS THE SHARED PHYSICS - both paths MUST use this function.
    
    Parameters
    ----------
    theta : ndarray, shape (N, 2)
        Image positions (x, y)
    theta_E : float
        Einstein radius
    multipoles : dict
        {m: (a_m, b_m)} in component form
    gamma_1, gamma_2 : float
        External shear components
    
    Returns
    -------
    alpha : ndarray, shape (N, 2)
        Reduced deflection at each position
    """
    x, y = theta[:, 0], theta[:, 1]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    # Monopole (Einstein ring)
    alpha_r = theta_E * np.ones_like(r)
    alpha_phi = np.zeros_like(r)
    
    # Multipoles: alpha_r += a_m*cos(m*phi) + b_m*sin(m*phi)
    for m, (a_m, b_m) in multipoles.items():
        alpha_r += a_m * np.cos(m * phi) + b_m * np.sin(m * phi)
    
    # Convert radial deflection to Cartesian
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    alpha_x = alpha_r * cos_phi - alpha_phi * sin_phi
    alpha_y = alpha_r * sin_phi + alpha_phi * cos_phi
    
    # External shear contribution
    alpha_x += gamma_1 * x + gamma_2 * y
    alpha_y += gamma_2 * x - gamma_1 * y
    
    return np.column_stack([alpha_x, alpha_y])


def lens_equation(
    theta: np.ndarray,
    beta: Tuple[float, float],
    theta_E: float,
    multipoles: Dict[int, Tuple[float, float]],
    gamma_1: float = 0.0,
    gamma_2: float = 0.0
) -> np.ndarray:
    """
    Evaluate lens equation: beta = theta - alpha_red(theta)
    
    Returns residual: should be zero for correct solution.
    """
    alpha = reduced_deflection(theta, theta_E, multipoles, gamma_1, gamma_2)
    beta_arr = np.array([[beta[0], beta[1]]])
    return theta - alpha - beta_arr


# =============================================================================
# PATH A: ALGEBRAIC COMPONENTS (NO-FIT, CANONICAL)
# =============================================================================

@dataclass
class AlgebraicResult:
    """Result from Path A (algebraic solve)."""
    params: Dict[str, float]
    derived_phases: Dict[str, float]  # phi_m = atan2(b_m, a_m)
    residuals: np.ndarray
    max_residual: float
    consistency: str
    dof_status: str
    n_params: int
    n_constraints: int


class AlgebraicSolver:
    """
    Path A: Solve in component form (a_m, b_m) - fully linear, no search.
    
    This is the REFERENCE/CANONICAL mode.
    """
    
    def __init__(self, m_max: int = 2, include_shear: bool = False):
        self.m_max = m_max
        self.include_shear = include_shear
    
    def unknowns(self, n_sources: int) -> List[str]:
        """All unknowns - ALL are linear."""
        params = ['theta_E']
        for m in range(2, self.m_max + 1):
            params.extend([f'a_{m}', f'b_{m}'])
        if self.include_shear:
            params.extend(['gamma_1', 'gamma_2'])
        for k in range(n_sources):
            params.extend([f'beta_x_{k}', f'beta_y_{k}'])
        return params
    
    def solve(self, sources: List[np.ndarray]) -> AlgebraicResult:
        """
        Solve using algebraic components - exact linear solve.
        
        Phase is computed as OUTPUT from components.
        """
        n_sources = len(sources)
        unknowns = self.unknowns(n_sources)
        n_params = len(unknowns)
        n_total_images = sum(len(s) for s in sources)
        n_constraints = 2 * n_total_images
        
        # DOF check
        if n_params > n_constraints:
            raise ValueError(
                f"FORBIDDEN: {n_params} params > {n_constraints} constraints"
            )
        
        dof_status = (
            f"OVERDETERMINED: +{n_constraints - n_params} redundancy"
            if n_constraints > n_params else "EXACT"
        )
        
        # Build linear system
        A = np.zeros((n_constraints, n_params))
        b = np.zeros(n_constraints)
        
        row = 0
        for k, images in enumerate(sources):
            for x, y in images:
                phi = np.arctan2(y, x)
                cos_phi, sin_phi = np.cos(phi), np.sin(phi)
                
                col = 0
                # theta_E
                A[row, col] = cos_phi
                A[row + 1, col] = sin_phi
                col += 1
                
                # Multipoles
                for m in range(2, self.m_max + 1):
                    cos_m, sin_m = np.cos(m * phi), np.sin(m * phi)
                    A[row, col] = cos_m * cos_phi
                    A[row, col + 1] = sin_m * cos_phi
                    A[row + 1, col] = cos_m * sin_phi
                    A[row + 1, col + 1] = sin_m * sin_phi
                    col += 2
                
                # Shear
                if self.include_shear:
                    A[row, col] = x
                    A[row, col + 1] = y
                    A[row + 1, col] = -y
                    A[row + 1, col + 1] = x
                    col += 2
                
                # Beta for this source
                beta_col = col + 2 * k
                A[row, beta_col] = 1.0
                A[row + 1, beta_col + 1] = 1.0
                
                b[row] = x
                b[row + 1] = y
                row += 2
        
        # Solve
        if n_constraints > n_params:
            p = np.linalg.lstsq(A[:n_params], b[:n_params], rcond=None)[0]
        else:
            p = np.linalg.solve(A, b)
        
        residuals = A @ p - b
        max_res = np.max(np.abs(residuals))
        
        # Build result
        params = {name: val for name, val in zip(unknowns, p)}
        
        # Derive phases as OUTPUT
        derived_phases = {}
        for m in range(2, self.m_max + 1):
            a_m = params.get(f'a_{m}', 0.0)
            b_m = params.get(f'b_{m}', 0.0)
            derived_phases[f'A_{m}'] = np.sqrt(a_m**2 + b_m**2)
            derived_phases[f'phi_{m}'] = np.arctan2(b_m, a_m) / m
        
        if self.include_shear:
            g1 = params.get('gamma_1', 0.0)
            g2 = params.get('gamma_2', 0.0)
            derived_phases['gamma'] = np.sqrt(g1**2 + g2**2)
            derived_phases['phi_gamma'] = np.arctan2(g2, g1) / 2
        
        consistency = "PASS" if max_res < 1e-8 else f"FAIL ({max_res:.2e})"
        
        return AlgebraicResult(
            params=params,
            derived_phases=derived_phases,
            residuals=residuals,
            max_residual=max_res,
            consistency=consistency,
            dof_status=dof_status,
            n_params=n_params,
            n_constraints=n_constraints
        )


# =============================================================================
# PATH B: PHASE SCAN MODE (HYPOTHESIS TEST)
# =============================================================================

@dataclass
class ScanPoint:
    """Single point in phase scan."""
    phases: Dict[str, float]  # {phi_2: ..., phi_gamma: ...}
    linear_params: Dict[str, float]  # {A_2, theta_E, beta_x, ...}
    residual: float
    rank: int
    condition: float


@dataclass
class PhaseScanResult:
    """Result from Path B (phase scan)."""
    best_candidate: ScanPoint
    all_candidates: List[ScanPoint]
    residual_landscape: np.ndarray
    phase_grid: np.ndarray
    degeneracy_hints: List[str]
    cross_check: Dict[str, float]  # Comparison with Path A


class PhaseScanSolver:
    """
    Path B: Phase Scan Mode - controlled scan over phi values.
    
    EXPLICITLY LABELED: "Scan/Hypothesis Test mode (nonlinear search)"
    
    For each scan point:
    1. Fix phi values
    2. Solve ALL OTHER params linearly exact
    3. Compute residuals
    
    This is NOT a free-flight optimizer - it's a hypothesis test.
    """
    
    MODE_LABEL = "Scan/Hypothesis Test mode (nonlinear search)"
    
    def __init__(self, m_max: int = 2, include_shear: bool = False):
        self.m_max = m_max
        self.include_shear = include_shear
    
    def scan_phases_then_solve_linear(
        self,
        sources: List[np.ndarray],
        phi_2_range: Tuple[float, float, int] = (0, 2*np.pi, 36),
        phi_gamma_range: Optional[Tuple[float, float, int]] = None,
        algebraic_result: Optional[AlgebraicResult] = None
    ) -> PhaseScanResult:
        """
        Controlled phase scan with linear solve at each point.
        
        Parameters
        ----------
        sources : list of ndarray
            Image positions for each source
        phi_2_range : tuple
            (min, max, n_steps) for phi_2 scan
        phi_gamma_range : optional tuple
            (min, max, n_steps) for phi_gamma scan (if shear included)
        algebraic_result : optional AlgebraicResult
            Result from Path A for cross-checking
        
        Returns
        -------
        PhaseScanResult with best candidates and residual landscape
        """
        print(f"[{self.MODE_LABEL}]")
        
        # Build phase grid
        phi_2_vals = np.linspace(*phi_2_range)
        
        if self.include_shear and phi_gamma_range:
            phi_gamma_vals = np.linspace(*phi_gamma_range)
            phase_grid = np.array(np.meshgrid(
                phi_2_vals, phi_gamma_vals
            )).T.reshape(-1, 2)
        else:
            phase_grid = phi_2_vals.reshape(-1, 1)
        
        # Scan
        candidates = []
        residuals = []
        
        for phases in phase_grid:
            phi_2 = phases[0]
            phi_gamma = phases[1] if len(phases) > 1 else 0.0
            
            try:
                result = self._solve_at_fixed_phase(
                    sources, phi_2, phi_gamma
                )
                candidates.append(result)
                residuals.append(result.residual)
            except np.linalg.LinAlgError:
                residuals.append(np.inf)
        
        residuals = np.array(residuals)
        
        # Find best
        best_idx = np.argmin(residuals)
        best = candidates[best_idx] if candidates else None
        
        # Sort by residual
        sorted_candidates = sorted(candidates, key=lambda x: x.residual)[:10]
        
        # Degeneracy hints
        hints = self._analyze_degeneracy(residuals, phase_grid)
        
        # Cross-check against Path A
        cross_check = {}
        if algebraic_result:
            cross_check = self._cross_check(best, algebraic_result)
        
        return PhaseScanResult(
            best_candidate=best,
            all_candidates=sorted_candidates,
            residual_landscape=residuals.reshape(
                len(phi_2_vals), -1
            ) if self.include_shear else residuals,
            phase_grid=phase_grid,
            degeneracy_hints=hints,
            cross_check=cross_check
        )
    
    def _solve_at_fixed_phase(
        self,
        sources: List[np.ndarray],
        phi_2: float,
        phi_gamma: float = 0.0
    ) -> ScanPoint:
        """
        At fixed phase, solve for amplitudes and other params linearly.
        
        Unknown: A_2 (amplitude), theta_E, beta^(k)
        Fixed: phi_2, phi_gamma
        """
        n_sources = len(sources)
        n_total_images = sum(len(s) for s in sources)
        n_constraints = 2 * n_total_images
        
        # Unknowns: theta_E, A_2, [gamma], beta_x_k, beta_y_k
        n_params = 1 + 1  # theta_E, A_2
        if self.include_shear:
            n_params += 1  # gamma (amplitude)
        n_params += 2 * n_sources  # betas
        
        A = np.zeros((n_constraints, n_params))
        b = np.zeros(n_constraints)
        
        row = 0
        for k, images in enumerate(sources):
            for x, y in images:
                phi = np.arctan2(y, x)
                cos_phi, sin_phi = np.cos(phi), np.sin(phi)
                
                col = 0
                
                # theta_E
                A[row, col] = cos_phi
                A[row + 1, col] = sin_phi
                col += 1
                
                # A_2 * cos(2*phi - phi_2) = A_2 * (cos terms)
                deflection_2 = np.cos(2 * phi - 2 * phi_2)
                A[row, col] = deflection_2 * cos_phi
                A[row + 1, col] = deflection_2 * sin_phi
                col += 1
                
                # Shear: gamma * cos(2*phi - 2*phi_gamma)
                if self.include_shear:
                    deflection_g = np.cos(2 * phi - 2 * phi_gamma)
                    A[row, col] = (x * np.cos(2*phi_gamma) +
                                   y * np.sin(2*phi_gamma))
                    A[row + 1, col] = (x * np.sin(2*phi_gamma) -
                                       y * np.cos(2*phi_gamma))
                    col += 1
                
                # Beta
                beta_col = col + 2 * k
                A[row, beta_col] = 1.0
                A[row + 1, beta_col + 1] = 1.0
                
                b[row] = x
                b[row + 1] = y
                row += 2
        
        # Solve
        p, residuals_arr, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        residuals = A @ p - b
        max_res = np.max(np.abs(residuals))
        condition = s[0] / s[-1] if len(s) > 0 and s[-1] > 0 else np.inf
        
        # Build result
        linear_params = {
            'theta_E': p[0],
            'A_2': p[1]
        }
        col = 2
        if self.include_shear:
            linear_params['gamma'] = p[col]
            col += 1
        for k in range(n_sources):
            linear_params[f'beta_x_{k}'] = p[col + 2*k]
            linear_params[f'beta_y_{k}'] = p[col + 2*k + 1]
        
        return ScanPoint(
            phases={'phi_2': phi_2, 'phi_gamma': phi_gamma},
            linear_params=linear_params,
            residual=max_res,
            rank=rank,
            condition=condition
        )
    
    def _analyze_degeneracy(
        self,
        residuals: np.ndarray,
        phase_grid: np.ndarray
    ) -> List[str]:
        """Analyze residual landscape for degeneracy."""
        hints = []
        
        finite = residuals[np.isfinite(residuals)]
        if len(finite) == 0:
            hints.append("WARNING: All scan points failed")
            return hints
        
        min_res = np.min(finite)
        threshold = min_res * 1.1  # 10% above minimum
        
        n_good = np.sum(finite < threshold)
        if n_good > 1:
            hints.append(
                f"DEGENERACY: {n_good} scan points within 10% of minimum"
            )
        
        # Check for periodicity
        if len(finite) > 10:
            fft = np.abs(np.fft.fft(finite - np.mean(finite)))
            if fft[2] > 0.5 * fft[1]:
                hints.append("HINT: 2-fold symmetry detected in residuals")
        
        return hints
    
    def _cross_check(
        self,
        scan_best: ScanPoint,
        algebraic: AlgebraicResult
    ) -> Dict[str, float]:
        """
        SANITY CHECK: Compare scan result against algebraic solution.
        
        The phases from algebraic (derived) should match scan (found).
        """
        check = {}
        
        # Get algebraic-derived phase
        alg_phi_2 = algebraic.derived_phases.get('phi_2', 0.0)
        scan_phi_2 = scan_best.phases.get('phi_2', 0.0)
        
        # Phase difference (mod periodicity)
        diff = np.abs(alg_phi_2 - scan_phi_2)
        diff = min(diff, np.pi - diff)  # Account for pi periodicity of m=2
        
        check['phi_2_algebraic'] = alg_phi_2
        check['phi_2_scan'] = scan_phi_2
        check['phi_2_diff'] = diff
        check['consistent'] = diff < 0.1  # ~6 degrees tolerance
        
        if not check['consistent']:
            print(f"WARNING: Scan phase ({scan_phi_2:.3f}) differs from "
                  f"algebraic ({alg_phi_2:.3f}) by {diff:.3f} rad")
        
        return check


# =============================================================================
# UNIFIED INTERFACE: Run both paths and compare
# =============================================================================

def dual_path_inversion(
    sources: List[np.ndarray],
    m_max: int = 2,
    include_shear: bool = False,
    run_scan: bool = True,
    phi_2_steps: int = 36
) -> Dict:
    """
    Run BOTH inversion paths and compare results.
    
    Path A (Algebraic) is ALWAYS run as the reference.
    Path B (Scan) is optional and cross-checked against A.
    
    Returns
    -------
    dict with:
        'algebraic': AlgebraicResult (Path A)
        'scan': PhaseScanResult (Path B, if run)
        'comparison': cross-check results
    """
    results = {}
    
    # PATH A: Always run (reference)
    print("=" * 60)
    print("PATH A: Algebraic Components (No-Fit, CANONICAL)")
    print("=" * 60)
    
    solver_a = AlgebraicSolver(m_max=m_max, include_shear=include_shear)
    result_a = solver_a.solve(sources)
    results['algebraic'] = result_a
    
    print(f"  theta_E = {result_a.params.get('theta_E', 0):.6f}")
    print(f"  Derived phi_2 = {result_a.derived_phases.get('phi_2', 0):.4f} rad")
    print(f"  Max residual = {result_a.max_residual:.2e}")
    print(f"  Status: {result_a.consistency}")
    
    # PATH B: Optional scan
    if run_scan:
        print()
        print("=" * 60)
        print("PATH B: Phase Scan Mode (Hypothesis Test)")
        print("=" * 60)
        
        solver_b = PhaseScanSolver(m_max=m_max, include_shear=include_shear)
        result_b = solver_b.scan_phases_then_solve_linear(
            sources,
            phi_2_range=(0, np.pi, phi_2_steps),  # pi due to m=2 symmetry
            algebraic_result=result_a
        )
        results['scan'] = result_b
        
        if result_b.best_candidate:
            print(f"  Best phi_2 = {result_b.best_candidate.phases['phi_2']:.4f} rad")
            print(f"  Best residual = {result_b.best_candidate.residual:.2e}")
        
        if result_b.degeneracy_hints:
            for hint in result_b.degeneracy_hints:
                print(f"  {hint}")
        
        # Cross-check
        print()
        print("-" * 40)
        print("CROSS-CHECK: Scan vs Algebraic")
        print("-" * 40)
        cc = result_b.cross_check
        print(f"  Algebraic phi_2: {cc.get('phi_2_algebraic', 0):.4f} rad")
        print(f"  Scan phi_2:      {cc.get('phi_2_scan', 0):.4f} rad")
        print(f"  Difference:      {cc.get('phi_2_diff', 0):.4f} rad")
        print(f"  Consistent:      {'YES' if cc.get('consistent') else 'NO'}")
        
        results['comparison'] = cc
    
    return results
