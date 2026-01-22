#!/usr/bin/env python3
"""
Gauge Lens Inversion: No-Fit Exact Solution for Einstein Cross

This module implements the "quadrature first -> break inversion" approach:
- Einstein ring scale theta_E as reference (quadrature calibration)
- Quadrupole (a, b, phi_gamma) + offset (beta) as the "break" from perfect ring

The inversion is EXACT (no least-squares fitting, no scipy.optimize):
1. phi_gamma determined via bisection rootfinding on a consistency condition
2. (beta_x, beta_y, theta_E, a, b) solved linearly-exactly
3. Residuals computed to check model validity

IMPORTANT: This code uses NO curve fitting, NO least squares, NO optimization.
Only: exact linear algebra + bisection rootfinding + residual validation.

Authors: Carmen N. Wrede, Lino P. Casu
License: ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import numpy as np
import json
import argparse


# Default parameters for synthetic test (in cross regime)
# For 4-image cross: need |b| > beta/2 approximately
DEFAULT_PARAMS = {
    'theta_E': 1.0,
    'a': 0.05,
    'b': 0.15,
    'beta': 0.08,
    'phi_beta': np.radians(30),
    'phi_gamma': np.radians(20),
}


def bisection(f, a, b, tol=1e-12, max_iter=100):
    """
    Simple bisection root-finding. No scipy required.
    
    Finds x such that f(x) = 0 in [a, b], assuming f(a) and f(b) have opposite signs.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None  # No sign change
    
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) < tol or (b - a) < tol:
            return mid
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
    return 0.5 * (a + b)


def find_all_roots(f, x_min, x_max, n_samples=500, tol=1e-12):
    """
    Find all roots of f in [x_min, x_max] by scanning for sign changes
    and applying bisection to each bracket.
    """
    x_test = np.linspace(x_min, x_max, n_samples)
    f_vals = np.array([f(x) for x in x_test])
    
    roots = []
    for i in range(len(x_test) - 1):
        if f_vals[i] * f_vals[i+1] < 0:
            root = bisection(f, x_test[i], x_test[i+1], tol=tol)
            if root is not None:
                roots.append(root)
    
    return np.array(roots)


def generate_synthetic_cross(theta_E, a, b, beta, phi_beta, phi_gamma):
    """
    Generate synthetic Einstein Cross image positions from the local model.
    
    Model:
        alpha(theta, phi) = theta_E * e_r + a * cos(2*Delta) * e_r + b * sin(2*Delta) * e_phi
        where Delta = phi - phi_gamma
    
    The lens equation beta = theta - alpha leads to:
        Angular condition: beta * sin(phi - phi_beta) + b * sin(2*(phi - phi_gamma)) = 0
        Radial condition: r = theta_E + a * cos(2*(phi - phi_gamma)) + beta * cos(phi - phi_beta)
    
    Parameters
    ----------
    theta_E : float
        Einstein ring radius (quadrature scale)
    a : float
        Radial quadrupole amplitude
    b : float
        Tangential quadrupole amplitude
    beta : float
        Source offset magnitude
    phi_beta : float
        Source offset angle (radians)
    phi_gamma : float
        Quadrupole axis angle (radians)
    
    Returns
    -------
    points : ndarray, shape (n, 2)
        Image positions (x, y)
    phi_solutions : ndarray
        Azimuthal angles of images
    diagnostics : dict
        Diagnostic information
    """
    # Angular condition: f(phi) = beta*sin(phi - phi_beta) + b*sin(2*(phi - phi_gamma)) = 0
    def angular_condition(phi):
        return beta * np.sin(phi - phi_beta) + b * np.sin(2 * (phi - phi_gamma))
    
    # Find all roots in [0, 2*pi)
    phi_solutions = find_all_roots(angular_condition, 0, 2*np.pi, n_samples=1000)
    
    n_images = len(phi_solutions)
    diagnostics = {
        'n_images': n_images,
        'in_cross_regime': n_images == 4,
        'phi_solutions_deg': np.degrees(phi_solutions) if n_images > 0 else []
    }
    
    if n_images == 0:
        return np.array([]).reshape(0, 2), np.array([]), diagnostics
    
    # Compute radii: r_i = theta_E + a*cos(2*(phi_i - phi_gamma)) + beta*cos(phi_i - phi_beta)
    radii = (theta_E 
             + a * np.cos(2 * (phi_solutions - phi_gamma)) 
             + beta * np.cos(phi_solutions - phi_beta))
    
    # Convert to Cartesian
    points = np.column_stack([
        radii * np.cos(phi_solutions),
        radii * np.sin(phi_solutions)
    ])
    
    return points, phi_solutions, diagnostics


def build_linear_system(points, phi_gamma):
    """
    Build the linear system for p = [beta_x, beta_y, theta_E, a, b] given phi_gamma.
    
    For each image point i at (x_i, y_i), we have the lens equation components.
    The deflection model gives:
        theta_i - alpha_i = beta
    where:
        alpha = theta_E * e_r + a * cos(2*Delta) * e_r + b * sin(2*Delta) * e_phi
        Delta = phi - phi_gamma
    
    In Cartesian coordinates, this yields 2 equations per point (8 total for 4 points).
    
    Returns A (8x5 matrix) and b_vec (8-vector) such that A @ p = b_vec
    """
    n = len(points)
    A = np.zeros((2 * n, 5))
    b_vec = np.zeros(2 * n)
    
    for i, (x, y) in enumerate(points):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        Delta = phi - phi_gamma
        
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_2D = np.cos(2 * Delta)
        sin_2D = np.sin(2 * Delta)
        
        # Unit vectors at this point
        # e_r = (cos_phi, sin_phi)
        # e_phi = (-sin_phi, cos_phi)
        
        # Deflection in x: alpha_x = theta_E*cos_phi + a*cos_2D*cos_phi + b*sin_2D*(-sin_phi)
        # Deflection in y: alpha_y = theta_E*sin_phi + a*cos_2D*sin_phi + b*sin_2D*(cos_phi)
        
        # Lens equation: theta - alpha = beta
        # x - alpha_x = beta_x  =>  beta_x + theta_E*cos_phi + a*cos_2D*cos_phi - b*sin_2D*sin_phi = x
        # y - alpha_y = beta_y  =>  beta_y + theta_E*sin_phi + a*cos_2D*sin_phi + b*sin_2D*cos_phi = y
        
        # Row for x-component
        row_x = 2 * i
        A[row_x, 0] = 1.0                        # beta_x
        A[row_x, 1] = 0.0                        # beta_y
        A[row_x, 2] = cos_phi                    # theta_E
        A[row_x, 3] = cos_2D * cos_phi           # a
        A[row_x, 4] = -sin_2D * sin_phi          # b
        b_vec[row_x] = x
        
        # Row for y-component
        row_y = 2 * i + 1
        A[row_y, 0] = 0.0                        # beta_x
        A[row_y, 1] = 1.0                        # beta_y
        A[row_y, 2] = sin_phi                    # theta_E
        A[row_y, 3] = cos_2D * sin_phi           # a
        A[row_y, 4] = sin_2D * cos_phi           # b
        b_vec[row_y] = y
    
    return A, b_vec


def solve_5x5_subset(A, b_vec, rows):
    """
    Solve a 5x5 subsystem exactly using the specified row indices.
    Returns solution p and whether the matrix was invertible.
    """
    A_sub = A[rows, :]
    b_sub = b_vec[rows]
    
    try:
        det = np.linalg.det(A_sub)
        if abs(det) < 1e-14:
            return None, False
        p = np.linalg.solve(A_sub, b_sub)
        return p, True
    except np.linalg.LinAlgError:
        return None, False


def compute_residuals(A, b_vec, p):
    """Compute residual vector and statistics."""
    residual = A @ p - b_vec
    return {
        'residual_vector': residual,
        'max_abs': np.max(np.abs(residual)),
        'rms': np.sqrt(np.mean(residual**2))
    }


def consistency_residual(phi_gamma, points, row_subset, check_row):
    """
    For a given phi_gamma:
    1. Build the linear system
    2. Solve using 5 rows (row_subset)
    3. Return the residual of a 6th equation (check_row)
    
    This residual should be zero at the correct phi_gamma.
    """
    A, b_vec = build_linear_system(points, phi_gamma)
    p, ok = solve_5x5_subset(A, b_vec, row_subset)
    
    if not ok:
        return np.inf  # Singular matrix
    
    # Residual of the check equation
    return A[check_row, :] @ p - b_vec[check_row]


def invert_no_fit(points, center=(0, 0)):
    """
    Perform no-fit inversion on 4 image points.
    
    Strategy:
    1. For each candidate phi_gamma in [0, pi/2]:
       - Build the 8x5 linear system
       - Solve a 5x5 subset exactly
       - Use a 6th equation as consistency check: h(phi_gamma) = 0
    2. Find phi_gamma via bisection rootfinding on h
    3. Solve the full system with the found phi_gamma
    4. Compute residuals over all 8 equations
    
    Parameters
    ----------
    points : array-like, shape (4, 2)
        Image positions (x, y) relative to lens center
    center : tuple
        Lens center position (subtracted from points)
    
    Returns
    -------
    params : dict
        Recovered parameters
    residuals : dict
        Residual statistics
    diagnostics : dict
        Diagnostic information
    """
    points = np.asarray(points) - np.asarray(center)
    
    if len(points) != 4:
        return None, None, {'error': f'Expected 4 points, got {len(points)}'}
    
    # Try different row combinations for robustness
    # We need 5 rows for the linear solve, and 1 row for the root condition
    row_combinations = [
        ([0, 1, 2, 3, 4], 5),  # First 5 rows, check 6th
        ([0, 1, 2, 3, 5], 4),
        ([0, 1, 2, 4, 5], 3),
        ([0, 1, 3, 4, 5], 2),
        ([0, 2, 3, 4, 5], 1),
        ([1, 2, 3, 4, 5], 0),
        ([0, 1, 2, 5, 6], 7),
        ([0, 1, 4, 5, 6], 7),
    ]
    
    best_result = None
    best_max_residual = np.inf
    
    for row_subset, check_row in row_combinations:
        # Define the consistency function for this row combination
        def h(phi_gamma):
            return consistency_residual(phi_gamma, points, row_subset, check_row)
        
        # Find all roots in [0, pi/2] (phi_gamma is only defined mod pi/2 for m=2)
        roots = find_all_roots(h, 0, np.pi/2, n_samples=200, tol=1e-10)
        
        if len(roots) == 0:
            continue
        
        # Evaluate each root and pick the one with smallest max residual
        for phi_gamma_candidate in roots:
            A, b_vec = build_linear_system(points, phi_gamma_candidate)
            p, ok = solve_5x5_subset(A, b_vec, row_subset)
            
            if not ok:
                continue
            
            res = compute_residuals(A, b_vec, p)
            
            if res['max_abs'] < best_max_residual:
                best_max_residual = res['max_abs']
                best_result = {
                    'phi_gamma': phi_gamma_candidate,
                    'p': p,
                    'residuals': res,
                    'row_subset': row_subset,
                    'check_row': check_row
                }
    
    if best_result is None:
        return None, None, {'error': 'No valid solution found. Check data or centering.'}
    
    # Extract parameters
    p = best_result['p']
    beta_x, beta_y, theta_E, a, b = p
    phi_gamma = best_result['phi_gamma']
    
    beta = np.sqrt(beta_x**2 + beta_y**2)
    phi_beta = np.arctan2(beta_y, beta_x)
    
    # Normalize phi_gamma to [0, 90) degrees
    phi_gamma_deg = np.degrees(phi_gamma) % 90
    
    params = {
        'beta_x': beta_x,
        'beta_y': beta_y,
        'beta': beta,
        'phi_beta': phi_beta,
        'phi_beta_deg': np.degrees(phi_beta) % 360,
        'theta_E': theta_E,
        'a': a,
        'b': b,
        'phi_gamma': phi_gamma,
        'phi_gamma_deg': phi_gamma_deg,
    }
    
    residuals = best_result['residuals']
    
    diagnostics = {
        'row_subset_used': best_result['row_subset'],
        'check_row_used': best_result['check_row'],
        'n_roots_found': len(roots) if 'roots' in dir() else 'N/A'
    }
    
    return params, residuals, diagnostics


def moment_estimate(points):
    """
    Compute m2 moment as initial estimate for phi_gamma.
    This is for diagnostic purposes only; the final phi_gamma comes from rootfinding.
    
    m2 = sum(exp(i * 2 * phi_j)) gives approximate quadrupole axis.
    """
    phis = np.arctan2(points[:, 1], points[:, 0])
    m2 = np.sum(np.exp(2j * phis))
    phi_gamma_est = 0.5 * np.angle(m2)
    if phi_gamma_est < 0:
        phi_gamma_est += np.pi/2
    return phi_gamma_est, np.abs(m2)


def print_results(params, residuals, diagnostics, title="Inversion Results"):
    """Pretty-print the inversion results."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    
    if params is None:
        print(f"\n  ERROR: {diagnostics.get('error', 'Unknown error')}")
        return
    
    print(f"\n{'-'*40}")
    print(" Recovered Parameters")
    print(f"{'-'*40}")
    print(f"  theta_E (Einstein radius): {params['theta_E']:.6f}")
    print(f"  beta (source offset):      {params['beta']:.6f}")
    print(f"  phi_beta (offset angle):   {params['phi_beta_deg']:.2f} deg")
    print(f"  a (radial quadrupole):     {params['a']:.6f}")
    print(f"  b (tangential quadrupole): {params['b']:.6f}")
    print(f"  phi_gamma (quad axis):     {params['phi_gamma_deg']:.2f} deg (mod 90)")
    
    print(f"\n{'-'*40}")
    print(" Residuals (Model Consistency)")
    print(f"{'-'*40}")
    print(f"  Max |residual|:  {residuals['max_abs']:.2e}")
    print(f"  RMS residual:    {residuals['rms']:.2e}")
    
    if residuals['max_abs'] < 1e-10:
        print("\n  [OK] Residuals at numerical precision -> exact model fit")
    elif residuals['max_abs'] < 0.01:
        print("\n  [OK] Small residuals -> local model adequate")
    else:
        print("\n  [!] Significant residuals -> higher modes, substructure, or centering error")
    
    print(f"{'='*60}\n")


def run_synthetic_test():
    """Run a synthetic test: generate cross, then invert, compare."""
    print("\n" + "="*60)
    print(" SYNTHETIC TEST: Generate -> Invert -> Compare")
    print("="*60)
    
    # Use default parameters (known to be in cross regime)
    true_params = DEFAULT_PARAMS.copy()
    
    print("\nTrue parameters:")
    print(f"  theta_E   = {true_params['theta_E']}")
    print(f"  a         = {true_params['a']}")
    print(f"  b         = {true_params['b']}")
    print(f"  beta      = {true_params['beta']}")
    print(f"  phi_beta  = {np.degrees(true_params['phi_beta']):.2f} deg")
    print(f"  phi_gamma = {np.degrees(true_params['phi_gamma']):.2f} deg")
    
    # Generate synthetic cross
    points, phi_sol, gen_diag = generate_synthetic_cross(**true_params)
    
    print(f"\nGeneration diagnostics:")
    print(f"  Number of images: {gen_diag['n_images']}")
    print(f"  In cross regime:  {gen_diag['in_cross_regime']}")
    
    if gen_diag['n_images'] != 4:
        print("\n  [!] Not in 4-image regime. Adjust parameters.")
        print(f"      Try increasing |b| or adjusting beta.")
        return None, None
    
    print(f"\nGenerated {len(points)} image points:")
    for i, (x, y) in enumerate(points):
        r = np.sqrt(x**2 + y**2)
        phi = np.degrees(np.arctan2(y, x))
        print(f"  Image {i+1}: ({x:+.6f}, {y:+.6f})  r={r:.4f}, phi={phi:+.1f} deg")
    
    # Moment estimate (diagnostic only)
    phi_gamma_est, m2_mag = moment_estimate(points)
    print(f"\nMoment estimate (diagnostic):")
    print(f"  m2 magnitude: {m2_mag:.4f}")
    print(f"  phi_gamma_est: {np.degrees(phi_gamma_est):.2f} deg")
    
    # Invert
    params, residuals, diagnostics = invert_no_fit(points)
    print_results(params, residuals, diagnostics, "Recovered Parameters (Synthetic)")
    
    if params is None:
        return None, None
    
    # Compare
    print("Parameter Recovery Check:")
    print(f"  theta_E: true={true_params['theta_E']:.4f}, recovered={params['theta_E']:.4f}, "
          f"diff={abs(true_params['theta_E'] - params['theta_E']):.2e}")
    print(f"  beta:    true={true_params['beta']:.4f}, recovered={params['beta']:.4f}, "
          f"diff={abs(true_params['beta'] - params['beta']):.2e}")
    print(f"  a:       true={true_params['a']:.4f}, recovered={params['a']:.4f}, "
          f"diff={abs(true_params['a'] - params['a']):.2e}")
    print(f"  b:       true={true_params['b']:.4f}, recovered={params['b']:.4f}, "
          f"diff={abs(true_params['b'] - params['b']):.2e}")
    
    # Save example points
    return points, params


def load_points_from_json(filepath):
    """Load image points from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'points' in data:
        points = np.array(data['points'])
    elif 'images' in data:
        points = np.array([img['position'] for img in data['images']])
    else:
        raise ValueError("JSON must contain 'points' or 'images' key")
    
    center = data.get('center', [0, 0])
    return points, center, data


def save_example_json(points, filepath):
    """Save points to example JSON file."""
    data = {
        "center": [0.0, 0.0],
        "points": points.tolist(),
        "units": "arbitrary",
        "note": "Synthetic Einstein-cross points from local ring+quadrupole+offset model."
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved example points to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Gauge Lens Inversion: No-Fit Exact Solution for Einstein Cross"
    )
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to JSON file with image points')
    parser.add_argument('--save-example', '-s', type=str, default=None,
                        help='Save synthetic example to JSON file')
    
    args = parser.parse_args()
    
    if args.input:
        # Load from file
        points, center, data = load_points_from_json(args.input)
        print(f"\nLoaded {len(points)} points from {args.input}")
        if 'note' in data:
            print(f"Note: {data['note']}")
        
        print(f"\nImage positions (relative to center {center}):")
        for i, (x, y) in enumerate(points):
            r = np.sqrt(x**2 + y**2)
            phi = np.degrees(np.arctan2(y, x))
            print(f"  Point {i+1}: ({x:+.4f}, {y:+.4f})  r={r:.4f}, phi={phi:+.1f} deg")
        
        params, residuals, diagnostics = invert_no_fit(points, center)
        print_results(params, residuals, diagnostics, f"Inversion: {args.input}")
        
    else:
        # Run synthetic test
        points, params = run_synthetic_test()
        
        # Optionally save example
        if args.save_example and points is not None:
            save_example_json(points, args.save_example)


if __name__ == "__main__":
    main()
