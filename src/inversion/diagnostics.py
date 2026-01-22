"""
Diagnostic Tools for No-Fit Inversion

Provides residual analysis and validation without optimization.
Residuals are used to CHECK solutions, never to FIND them.

Key principle: Large residuals indicate:
- Incorrect model
- Bad phase (nonlinear parameter) choice
- Data errors
They do NOT trigger minimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ResidualReport:
    """Comprehensive residual analysis."""
    residuals: np.ndarray
    max_abs: float
    rms: float
    per_image: np.ndarray
    worst_image: int
    acceptable: bool
    message: str


def residual_report(
    residuals: np.ndarray,
    tol: float = 1e-10
) -> Dict:
    """
    Generate residual analysis report.
    
    Parameters
    ----------
    residuals : ndarray
        Residual values (pred - obs)
    tol : float
        Tolerance for "acceptable" classification
        
    Returns
    -------
    dict : Report with statistics and assessment
    """
    residuals = np.asarray(residuals)
    
    max_abs = np.max(np.abs(residuals))
    rms = np.sqrt(np.mean(residuals ** 2))
    
    # Per-image residuals (pairs of x, y)
    n_pairs = len(residuals) // 2
    per_image = np.zeros(n_pairs)
    for i in range(n_pairs):
        per_image[i] = np.sqrt(residuals[2*i]**2 + residuals[2*i+1]**2)
    
    worst_image = int(np.argmax(per_image)) if n_pairs > 0 else -1
    
    acceptable = max_abs < tol
    
    if acceptable:
        message = f"EXACT: max residual {max_abs:.2e} < {tol:.0e}"
    else:
        message = f"NOT EXACT: max residual {max_abs:.2e} > {tol:.0e}"
    
    return {
        'residuals': residuals,
        'max_abs': max_abs,
        'rms': rms,
        'per_image': per_image,
        'worst_image': worst_image,
        'acceptable': acceptable,
        'message': message
    }


def compare_solutions(
    solutions: List[Dict],
    true_params: Optional[Dict] = None
) -> str:
    """
    Compare multiple solutions from inversion.
    
    Useful when rootfinding gives multiple candidates.
    """
    lines = [
        "=" * 60,
        "SOLUTION COMPARISON",
        "=" * 60
    ]
    
    for i, sol in enumerate(solutions):
        params = sol.get('params', sol)
        report = sol.get('report', {})
        
        lines.append(f"\nSolution {i+1}:")
        lines.append("-" * 40)
        
        for key, val in params.items():
            if isinstance(val, float):
                lines.append(f"  {key}: {val:.6f}")
            else:
                lines.append(f"  {key}: {val}")
        
        if 'max_abs' in report:
            lines.append(f"  Max residual: {report['max_abs']:.2e}")
        if 'rms' in report:
            lines.append(f"  RMS residual: {report['rms']:.2e}")
        
        if true_params:
            lines.append("  Errors vs true:")
            for key in params:
                if key in true_params:
                    error = params[key] - true_params[key]
                    lines.append(f"    {key}: {error:+.6f}")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def consistency_check(
    params: Dict,
    images: np.ndarray,
    model
) -> Dict:
    """
    Full consistency check of recovered parameters.
    
    1. Forward model: generate images from params
    2. Compare with observed images
    3. Report discrepancies
    """
    # Extract source and lens params
    source_params = {
        'beta_x': params.get('beta_x', 0),
        'beta_y': params.get('beta_y', 0)
    }
    lens_params = {k: v for k, v in params.items() 
                   if k not in ['beta_x', 'beta_y']}
    
    # Forward model
    try:
        predicted = model.predict_images(source_params, lens_params)
    except Exception as e:
        return {
            'success': False,
            'message': f"Forward model failed: {e}",
            'n_predicted': 0,
            'n_observed': len(images)
        }
    
    n_pred = len(predicted)
    n_obs = len(images)
    
    # Match images (nearest neighbor)
    if n_pred == 0:
        return {
            'success': False,
            'message': "Forward model produced no images",
            'n_predicted': 0,
            'n_observed': n_obs
        }
    
    # Compute distances
    matches = []
    for i, obs in enumerate(images):
        dists = [np.linalg.norm(obs - pred) for pred in predicted]
        best_j = int(np.argmin(dists))
        matches.append({
            'observed_idx': i,
            'predicted_idx': best_j,
            'distance': dists[best_j]
        })
    
    max_dist = max(m['distance'] for m in matches)
    mean_dist = np.mean([m['distance'] for m in matches])
    
    success = (n_pred >= n_obs and max_dist < 1e-6)
    
    return {
        'success': success,
        'message': "Consistent" if success else "Inconsistent",
        'n_predicted': n_pred,
        'n_observed': n_obs,
        'max_distance': max_dist,
        'mean_distance': mean_dist,
        'matches': matches
    }


def parameter_bounds_check(
    params: Dict,
    bounds: Optional[Dict] = None
) -> Dict:
    """
    Check if parameters are within physical bounds.
    
    Default bounds:
    - theta_E > 0
    - |a|, |b| < 1 (quadrupole coefficients)
    - 0 <= phi < 2*pi
    """
    if bounds is None:
        bounds = {
            'theta_E': (0, np.inf),
            'a': (-1, 1),
            'b': (-1, 1),
            'a_1': (-1, 1),
            'b_1': (-1, 1),
            'a_2': (-1, 1),
            'b_2': (-1, 1),
        }
    
    violations = []
    
    for key, val in params.items():
        if key in bounds:
            lo, hi = bounds[key]
            if val < lo or val > hi:
                violations.append(f"{key}={val:.4f} outside [{lo}, {hi}]")
        
        # Phase bounds (any phi_* should be in [0, 2*pi])
        if key.startswith('phi_'):
            # Normalize to [0, 2*pi]
            normalized = val % (2 * np.pi)
            if normalized < 0:
                normalized += 2 * np.pi
            # This is just informational, phases can be any real number
    
    return {
        'valid': len(violations) == 0,
        'violations': violations,
        'message': "All parameters in bounds" if not violations else "; ".join(violations)
    }


def solution_quality_score(report: Dict) -> float:
    """
    Compute overall solution quality score.
    
    Score in [0, 1] where 1 = perfect exact solution.
    
    Based on:
    - Max residual (logarithmic scale)
    - RMS residual
    - Whether solution is exact
    """
    max_res = report.get('max_abs', 1.0)
    rms = report.get('rms', 1.0)
    
    # Log scale score: residual of 1e-12 -> score 1, 1e-3 -> score ~0.75
    if max_res <= 0:
        log_score = 1.0
    else:
        log_max = -np.log10(max_res + 1e-15)
        log_score = min(1.0, max(0.0, log_max / 12))  # Normalize to [0,1]
    
    if rms <= 0:
        rms_score = 1.0
    else:
        log_rms = -np.log10(rms + 1e-15)
        rms_score = min(1.0, max(0.0, log_rms / 12))
    
    # Combine
    score = 0.7 * log_score + 0.3 * rms_score
    
    return score


def print_diagnostic_summary(
    params: Dict,
    report: Dict,
    model_name: str = "Unknown"
) -> str:
    """
    Generate printable diagnostic summary.
    """
    score = solution_quality_score(report)
    
    lines = [
        "",
        "=" * 50,
        f"DIAGNOSTIC SUMMARY - {model_name}",
        "=" * 50,
        "",
        "Recovered Parameters:",
    ]
    
    for key, val in sorted(params.items()):
        if isinstance(val, float):
            if 'phi' in key:
                lines.append(f"  {key}: {np.degrees(val):.2f} deg")
            else:
                lines.append(f"  {key}: {val:.6f}")
        else:
            lines.append(f"  {key}: {val}")
    
    lines.extend([
        "",
        "Residual Analysis:",
        f"  Max |residual|: {report.get('max_abs', 0):.2e}",
        f"  RMS residual:   {report.get('rms', 0):.2e}",
        f"  Worst image:    {report.get('worst_image', -1) + 1}",
        "",
        f"Quality Score: {score:.3f} / 1.000",
        "",
        f"Assessment: {report.get('message', 'Unknown')}",
        "=" * 50
    ])
    
    return "\n".join(lines)
