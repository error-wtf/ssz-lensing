"""
Constraint and Degrees of Freedom Management

Tracks:
- Number of equations vs unknowns
- Linear vs nonlinear unknowns
- Observable constraints
- Determinacy status (under/over/exact)

NO FITTING ALLOWED - only exact constraint counting.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConstraintStatus:
    """Status of constraint system."""
    n_equations: int
    n_unknowns: int
    n_linear: int
    n_nonlinear: int
    status: str  # 'underdetermined', 'determined', 'overdetermined'
    dof_deficit: int  # positive = need more equations
    message: str


def count_degrees_of_freedom(
    n_images: int,
    model_unknowns: List[str],
    additional_constraints: int = 0
) -> ConstraintStatus:
    """
    Count degrees of freedom for inversion.
    
    Each image provides 2 equations (x, y components of lens equation).
    
    Parameters
    ----------
    n_images : int
        Number of observed images
    model_unknowns : list of str
        Names of unknown parameters
    additional_constraints : int
        Extra constraints (e.g., time delays, flux ratios)
        
    Returns
    -------
    status : ConstraintStatus
        Full status report
    """
    n_equations = 2 * n_images + additional_constraints
    n_unknowns = len(model_unknowns)
    
    # Classify unknowns (heuristic based on naming)
    linear_keywords = ['beta', 'theta_E', 'a_', 'b_', 'A_', 'B_', 'T']
    nonlinear_keywords = ['phi_', 'phase']
    
    n_linear = sum(1 for u in model_unknowns 
                   if any(k in u for k in linear_keywords))
    n_nonlinear = sum(1 for u in model_unknowns 
                      if any(k in u for k in nonlinear_keywords))
    
    # Adjust for unclassified
    n_other = n_unknowns - n_linear - n_nonlinear
    n_linear += n_other  # Assume unclassified are linear
    
    dof_deficit = n_unknowns - n_equations
    
    if dof_deficit > 0:
        status = 'underdetermined'
        msg = f"Need {dof_deficit} more constraints"
    elif dof_deficit < 0:
        status = 'overdetermined'
        msg = f"Have {-dof_deficit} extra constraints (good for validation)"
    else:
        status = 'determined'
        msg = "Exactly determined system"
    
    return ConstraintStatus(
        n_equations=n_equations,
        n_unknowns=n_unknowns,
        n_linear=n_linear,
        n_nonlinear=n_nonlinear,
        status=status,
        dof_deficit=dof_deficit,
        message=msg
    )


def check_image_multiplicity(n_images: int, model_name: str) -> Dict:
    """
    Check if image count is consistent with model requirements.
    
    Different models require different minimum images:
    - Ring only (m=0): 1 image (Einstein ring)
    - Quadrupole (m=2): 4 images typical
    - Higher multipoles: more images needed
    
    Returns
    -------
    dict with keys: valid, required, message
    """
    requirements = {
        'ring': 1,
        'quadrupole': 4,
        'multipole_2': 4,
        'multipole_3': 6,
        'multipole_4': 8,
        'general': 4
    }
    
    # Normalize model name
    model_lower = model_name.lower()
    
    if 'ring' in model_lower and 'quad' not in model_lower:
        required = requirements['ring']
    elif 'quad' in model_lower or 'm=2' in model_lower or 'm_max=2' in model_lower:
        required = requirements['quadrupole']
    elif 'm=3' in model_lower or 'm_max=3' in model_lower:
        required = requirements['multipole_3']
    elif 'm=4' in model_lower or 'm_max=4' in model_lower:
        required = requirements['multipole_4']
    else:
        required = requirements['general']
    
    valid = n_images >= required
    
    if valid:
        msg = f"Image count {n_images} >= {required} required by {model_name}"
    else:
        msg = f"Image count {n_images} < {required} required by {model_name}"
    
    return {
        'valid': valid,
        'required': required,
        'actual': n_images,
        'message': msg
    }


def observable_equations(observables: Dict) -> int:
    """
    Count equations provided by observables.
    
    Observable types and their equation counts:
    - image_positions: 2 per image (x, y)
    - time_delays: 1 per pair (relative delay)
    - flux_ratios: 1 per pair (relative magnification)
    - absolute_magnifications: 1 per image
    - image_parities: 0 (discrete constraint, not equation)
    """
    count = 0
    
    if 'image_positions' in observables:
        positions = observables['image_positions']
        if hasattr(positions, 'shape'):
            n_images = positions.shape[0]
        else:
            n_images = len(positions)
        count += 2 * n_images
    
    if 'time_delays' in observables:
        delays = observables['time_delays']
        count += len(delays)
    
    if 'flux_ratios' in observables:
        ratios = observables['flux_ratios']
        count += len(ratios)
    
    if 'absolute_magnifications' in observables:
        mags = observables['absolute_magnifications']
        count += len(mags)
    
    return count


def suggest_additional_observables(
    status: ConstraintStatus,
    available_observables: List[str]
) -> List[str]:
    """
    Suggest what observables to add if underdetermined.
    """
    if status.status != 'underdetermined':
        return []
    
    needed = status.dof_deficit
    suggestions = []
    
    # Prioritize time delays (usually most informative)
    if 'time_delays' in available_observables and needed >= 1:
        suggestions.append(f"Add {needed} time delay measurements")
    
    if 'flux_ratios' in available_observables and needed >= 1:
        suggestions.append(f"Add {needed} flux ratio measurements")
    
    if needed > 0:
        suggestions.append(f"Or reduce model complexity by {needed} parameters")
    
    return suggestions


def inversion_summary(
    images: np.ndarray,
    model_name: str,
    unknowns: List[str],
    nonlinear: List[str]
) -> str:
    """
    Generate human-readable summary of inversion setup.
    """
    n_images = len(images)
    status = count_degrees_of_freedom(n_images, unknowns)
    mult_check = check_image_multiplicity(n_images, model_name)
    
    lines = [
        "=" * 50,
        "INVERSION CONSTRAINT SUMMARY",
        "=" * 50,
        f"Model: {model_name}",
        f"Images: {n_images}",
        "",
        "Unknowns:",
    ]
    
    for u in unknowns:
        marker = " [NL]" if u in nonlinear else ""
        lines.append(f"  - {u}{marker}")
    
    lines.extend([
        "",
        f"Equations: {status.n_equations}",
        f"Unknowns: {status.n_unknowns}",
        f"  Linear: {status.n_linear}",
        f"  Nonlinear: {status.n_nonlinear}",
        "",
        f"Status: {status.status.upper()}",
        f"  {status.message}",
        "",
        f"Multiplicity: {mult_check['message']}",
        "=" * 50
    ])
    
    return "\n".join(lines)


def validate_constraint_system(
    A: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-12
) -> Dict:
    """
    Validate the linear constraint system Ax = b.
    
    Checks:
    - Rank of A
    - Consistency (is b in column space of A?)
    - Condition number estimate
    """
    m, n = A.shape
    
    # Compute rank
    from .exact_solvers import matrix_rank, condition_estimate
    
    rank_A = matrix_rank(A, tol)
    
    # Augmented rank
    Ab = np.hstack([A, b.reshape(-1, 1)])
    rank_Ab = matrix_rank(Ab, tol)
    
    # Consistency check
    consistent = (rank_A == rank_Ab)
    
    # Condition estimate (for square or near-square)
    if m >= n:
        cond = condition_estimate(A[:n, :])
    else:
        cond = float('inf')
    
    return {
        'rows': m,
        'cols': n,
        'rank_A': rank_A,
        'rank_Ab': rank_Ab,
        'consistent': consistent,
        'condition': cond,
        'full_rank': (rank_A == min(m, n)),
        'message': (
            "System consistent" if consistent 
            else "INCONSISTENT - no exact solution exists"
        )
    }
