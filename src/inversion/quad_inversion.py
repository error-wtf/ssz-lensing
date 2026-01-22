"""
Quad Inversion: Exact algebraic solution for Einstein Cross (4 images)

The lens equation β = θ - α(θ; p) gives 8 constraints (2 per image).
For model with n_params parameters, we need:
- n_params ≤ 8: solvable (exactly or overdetermined)
- n_params > 8: underdetermined (need regularization or more data)

Key insight: Source position β is SHARED across all images.
This is the consistency condition for model validity.

NO scipy.optimize, NO least squares fitting.
Only: exact linear algebra + bisection rootfinding.

Authors: Carmen N. Wrede, Lino P. Casu
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from .lens_equation import (
    DEFLECTION_MODELS, 
    compute_source_positions, 
    check_source_consistency,
    SourceConsistency
)


@dataclass
class InversionResult:
    """Result of lens inversion."""
    model_name: str
    params: Dict[str, float]
    source_position: np.ndarray  # β = (β_x, β_y)
    source_consistency: SourceConsistency
    residuals: np.ndarray  # Per-constraint residuals
    max_residual: float
    rms_residual: float
    is_exact: bool  # True if max_residual < tolerance
    rank: int
    condition_number: float
    regime: str  # 'determined', 'overdetermined', 'underdetermined'
    message: str


@dataclass  
class ModelComparison:
    """Comparison of multiple models on same data."""
    results: List[InversionResult]
    best_model: str
    ranking: List[str]  # Models sorted by residual
    recommendation: str


def build_linear_system_quad(
    theta: np.ndarray,
    model_name: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build linear system for quad inversion.
    
    The lens equation β = θ - α(θ; p) can be written as:
    
    For each image i: β = θ_i - α(θ_i; p)
    
    Rearranging with β as unknown:
    β_x = θ_ix - α_x(θ_i; p)
    β_y = θ_iy - α_y(θ_i; p)
    
    For linear models (all multipoles in c_m, s_m form), this becomes:
    A @ [β_x, β_y, theta_E, c2, s2, ...] = b
    
    Parameters
    ----------
    theta : ndarray, shape (4, 2)
        Four image positions
    model_name : str
        Model identifier from DEFLECTION_MODELS
        
    Returns
    -------
    A : ndarray
        Coefficient matrix
    b : ndarray
        Right-hand side
    param_names : list
        Names of parameters in solution vector
    """
    if model_name not in DEFLECTION_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    _, param_names = DEFLECTION_MODELS[model_name]
    n_images = len(theta)
    n_constraints = 2 * n_images  # x and y for each image
    n_params = 2 + len(param_names)  # β_x, β_y + lens params
    
    full_param_names = ['beta_x', 'beta_y'] + param_names
    
    A = np.zeros((n_constraints, n_params))
    b = np.zeros(n_constraints)
    
    for i in range(n_images):
        x, y = theta[i]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        
        row_x = 2 * i
        row_y = 2 * i + 1
        
        # β appears with coefficient 1
        A[row_x, 0] = 1.0  # β_x
        A[row_y, 1] = 1.0  # β_y
        
        # Right-hand side: θ_i
        b[row_x] = x
        b[row_y] = y
        
        # Now subtract deflection terms (move to LHS with sign flip)
        # α_r = θ_E²/r + c2*cos(2φ) + s2*sin(2φ) + ...
        # α_x = α_r * cos(φ), α_y = α_r * sin(φ)
        
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # θ_E contribution: α_r = θ_E²/r → need θ_E, so linearize around θ_E=1
        # Actually for exact linear solve, we parameterize as θ_E² directly
        # Or we use the fact that at leading order, we can absorb scaling
        
        # For now, use the simplified form where we solve for θ_E directly
        # assuming small perturbations. The full nonlinear case uses bisection.
        
        # θ_E term (monopole): coefficient is 1/r * cos(φ), 1/r * sin(φ)
        if 'theta_E' in param_names:
            idx = full_param_names.index('theta_E')
            # θ_E²/r in radial → linearize: θ_E ≈ r for Einstein ring
            # More precisely: α = θ_E²/r, so ∂α/∂θ_E = 2θ_E/r
            # For linear system, use θ_E directly
            A[row_x, idx] = -cos_phi  # -α_x contribution
            A[row_y, idx] = -sin_phi  # -α_y contribution
        
        # c2 term: c2 * cos(2φ)
        if 'c2' in param_names:
            idx = full_param_names.index('c2')
            A[row_x, idx] = -np.cos(2*phi) * cos_phi
            A[row_y, idx] = -np.cos(2*phi) * sin_phi
        
        # s2 term: s2 * sin(2φ)
        if 's2' in param_names:
            idx = full_param_names.index('s2')
            A[row_x, idx] = -np.sin(2*phi) * cos_phi
            A[row_y, idx] = -np.sin(2*phi) * sin_phi
        
        # gamma1, gamma2 (external shear)
        if 'gamma1' in param_names:
            idx = full_param_names.index('gamma1')
            A[row_x, idx] = -x  # α_x += γ1*x
            A[row_y, idx] = y   # α_y += -γ1*y
        
        if 'gamma2' in param_names:
            idx = full_param_names.index('gamma2')
            A[row_x, idx] = -y  # α_x += γ2*y
            A[row_y, idx] = -x  # α_y += γ2*x
        
        # c3, s3 (octupole)
        if 'c3' in param_names:
            idx = full_param_names.index('c3')
            A[row_x, idx] = -np.cos(3*phi) * cos_phi
            A[row_y, idx] = -np.cos(3*phi) * sin_phi
        
        if 's3' in param_names:
            idx = full_param_names.index('s3')
            A[row_x, idx] = -np.sin(3*phi) * cos_phi
            A[row_y, idx] = -np.sin(3*phi) * sin_phi
        
        # c4, s4 (hexadecapole)
        if 'c4' in param_names:
            idx = full_param_names.index('c4')
            A[row_x, idx] = -np.cos(4*phi) * cos_phi
            A[row_y, idx] = -np.cos(4*phi) * sin_phi
        
        if 's4' in param_names:
            idx = full_param_names.index('s4')
            A[row_x, idx] = -np.sin(4*phi) * cos_phi
            A[row_y, idx] = -np.sin(4*phi) * sin_phi
    
    return A, b, full_param_names


def analyze_system(A: np.ndarray) -> Tuple[int, float, str]:
    """
    Analyze linear system to determine regime.
    
    Returns
    -------
    rank : int
    condition_number : float
    regime : str
    """
    m, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    tol = max(m, n) * np.finfo(float).eps * s[0] if len(s) > 0 else 1e-15
    rank = np.sum(s > tol)
    
    if s[-1] > tol:
        condition = s[0] / s[-1]
    else:
        condition = float('inf')
    
    if m == n and rank == n:
        regime = 'determined'
    elif m > n and rank == n:
        regime = 'overdetermined'
    elif rank < n:
        regime = 'underdetermined'
    else:
        regime = 'unknown'
    
    if condition > 1e10:
        regime = 'ill_conditioned'
    
    return rank, condition, regime


def solve_exact(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Solve Ax = b exactly via Gaussian elimination.
    
    No scipy, no least squares.
    """
    n = A.shape[1]
    m = A.shape[0]
    
    if m < n:
        return np.zeros(n), False
    
    # For overdetermined, try to find exact solution
    if m > n:
        # Use first n equations, check consistency with rest
        A_sub = A[:n, :]
        b_sub = b[:n]
        x, success = solve_exact(A_sub, b_sub)
        if success:
            # Check residual on withheld equations
            residual = np.max(np.abs(A @ x - b))
            if residual < 1e-10:
                return x, True
        return x, False
    
    # Square system: Gaussian elimination with pivoting
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    
    for col in range(n):
        max_row = col + np.argmax(np.abs(Ab[col:, col]))
        if abs(Ab[max_row, col]) < 1e-15:
            return np.zeros(n), False
        Ab[[col, max_row]] = Ab[[max_row, col]]
        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]
    
    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        if abs(Ab[row, row]) < 1e-15:
            return np.zeros(n), False
        x[row] = (Ab[row, n] - np.dot(Ab[row, row+1:n], x[row+1:n])) / Ab[row, row]
    
    return x, True


def invert_quad(
    theta: np.ndarray,
    model_name: str = 'm2',
    tolerance: float = 1e-10
) -> InversionResult:
    """
    Perform quad inversion for given model.
    
    Parameters
    ----------
    theta : ndarray, shape (4, 2)
        Four image positions
    model_name : str
        Model from DEFLECTION_MODELS
    tolerance : float
        Threshold for 'exact' classification
        
    Returns
    -------
    InversionResult with full diagnostics
    """
    A, b, param_names = build_linear_system_quad(theta, model_name)
    rank, condition, regime = analyze_system(A)
    
    x, success = solve_exact(A, b)
    
    residuals = A @ x - b
    max_residual = np.max(np.abs(residuals))
    rms_residual = np.sqrt(np.mean(residuals**2))
    
    # Extract parameters
    params = {name: x[i] for i, name in enumerate(param_names)}
    source_position = np.array([params['beta_x'], params['beta_y']])
    
    # Compute source consistency using actual deflection model
    deflection_func, _ = DEFLECTION_MODELS[model_name]
    beta_positions = compute_source_positions(theta, deflection_func, params)
    source_consistency = check_source_consistency(beta_positions, tolerance)
    
    is_exact = max_residual < tolerance
    
    if is_exact:
        message = f"EXACT solution found (max residual = {max_residual:.2e})"
    elif success:
        message = f"Solution found but not exact (max residual = {max_residual:.2e})"
    else:
        message = f"Solution failed (regime: {regime})"
    
    return InversionResult(
        model_name=model_name,
        params=params,
        source_position=source_position,
        source_consistency=source_consistency,
        residuals=residuals,
        max_residual=max_residual,
        rms_residual=rms_residual,
        is_exact=is_exact,
        rank=rank,
        condition_number=condition,
        regime=regime,
        message=message
    )


def compare_models(
    theta: np.ndarray,
    models: List[str] = None,
    tolerance: float = 1e-10
) -> ModelComparison:
    """
    Compare multiple models on same quad data.
    
    Parameters
    ----------
    theta : ndarray, shape (4, 2)
        Four image positions
    models : list
        Model names to compare (default: all)
    tolerance : float
        Threshold for exact classification
        
    Returns
    -------
    ModelComparison with ranking and recommendation
    """
    if models is None:
        models = list(DEFLECTION_MODELS.keys())
    
    results = []
    for model in models:
        try:
            result = invert_quad(theta, model, tolerance)
            results.append(result)
        except Exception as e:
            print(f"Model {model} failed: {e}")
    
    # Sort by residual (lower is better)
    results.sort(key=lambda r: r.max_residual)
    ranking = [r.model_name for r in results]
    
    # Find best (lowest residual that is exact, or just lowest)
    exact_results = [r for r in results if r.is_exact]
    if exact_results:
        # Among exact, prefer simpler (fewer params)
        exact_results.sort(key=lambda r: len(r.params))
        best = exact_results[0]
        recommendation = f"Use '{best.model_name}' (exact, simplest)"
    else:
        best = results[0]
        recommendation = f"Use '{best.model_name}' (lowest residual, but not exact)"
    
    return ModelComparison(
        results=results,
        best_model=best.model_name,
        ranking=ranking,
        recommendation=recommendation
    )


# ============================================================================
# SYNTHETIC DATA FOR TESTING
# ============================================================================

def generate_synthetic_quad(
    theta_E: float = 1.0,
    beta: Tuple[float, float] = (0.1, 0.05),
    c2: float = 0.05,
    s2: float = 0.03,
    gamma1: float = 0.0,
    gamma2: float = 0.0,
    c3: float = 0.0,
    s3: float = 0.0,
    c4: float = 0.0,
    s4: float = 0.0
) -> Tuple[np.ndarray, Dict]:
    """
    Generate synthetic quad images from known truth.
    
    Uses bisection rootfinding on the angular condition,
    then computes radii from the radial condition.
    
    Returns
    -------
    theta : ndarray, shape (n, 2)
        Image positions
    truth : dict
        True parameters
    """
    beta_x, beta_y = beta
    beta_mag = np.sqrt(beta_x**2 + beta_y**2)
    phi_beta = np.arctan2(beta_y, beta_x)
    
    # Angular condition: sum of sin terms = 0
    def angular_condition(phi):
        result = beta_mag * np.sin(phi - phi_beta)
        result += s2 * np.sin(2*phi)  # simplified, actual depends on model
        return result
    
    # Find roots via sign change scanning
    phi_test = np.linspace(0, 2*np.pi, 1000)
    f_vals = np.array([angular_condition(p) for p in phi_test])
    
    roots = []
    for i in range(len(phi_test) - 1):
        if f_vals[i] * f_vals[i+1] < 0:
            # Bisection
            a, b = phi_test[i], phi_test[i+1]
            for _ in range(50):
                mid = 0.5 * (a + b)
                if angular_condition(a) * angular_condition(mid) < 0:
                    b = mid
                else:
                    a = mid
            roots.append(0.5 * (a + b))
    
    if len(roots) == 0:
        raise ValueError("No image solutions found")
    
    phi_solutions = np.array(roots)
    
    # Compute radii
    radii = theta_E + c2 * np.cos(2*phi_solutions) + s2 * np.sin(2*phi_solutions)
    radii += c3 * np.cos(3*phi_solutions) + s3 * np.sin(3*phi_solutions)
    radii += c4 * np.cos(4*phi_solutions) + s4 * np.sin(4*phi_solutions)
    radii += beta_mag * np.cos(phi_solutions - phi_beta)
    
    theta = np.column_stack([
        radii * np.cos(phi_solutions),
        radii * np.sin(phi_solutions)
    ])
    
    truth = {
        'beta_x': beta_x, 'beta_y': beta_y,
        'theta_E': theta_E,
        'c2': c2, 's2': s2,
        'gamma1': gamma1, 'gamma2': gamma2,
        'c3': c3, 's3': s3,
        'c4': c4, 's4': s4
    }
    
    return theta, truth
