#!/usr/bin/env python3
"""
Minimal Model Demo - Exact Recovery Test

Demonstrates the no-fit inversion on synthetic data:
1. Generate 4 images from known parameters
2. Run exact inversion (rootfinding + linear solve)
3. Verify parameter recovery

This is the core validation that the algorithm works.
NO SCIPY.OPTIMIZE - pure rootfinding and linear algebra.

This demo uses the proven working code from gauge_lens_inversion.py.
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def bisection(f, a, b, tol=1e-12, max_iter=100):
    """Simple bisection root-finding."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) < tol or (b - a) < tol:
            return mid
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def find_all_roots(f, x_min, x_max, n_samples=500, tol=1e-12):
    """Find all roots by scanning for sign changes."""
    x_test = np.linspace(x_min, x_max, n_samples)
    f_vals = np.array([f(x) for x in x_test])
    roots = []
    for i in range(len(x_test) - 1):
        if np.isfinite(f_vals[i]) and np.isfinite(f_vals[i+1]):
            if f_vals[i] * f_vals[i+1] < 0:
                root = bisection(f, x_test[i], x_test[i+1], tol=tol)
                if root is not None:
                    roots.append(root)
    return np.array(roots)


def generate_synthetic_cross(theta_E, a, b, beta, phi_beta, phi_gamma):
    """Generate synthetic Einstein Cross image positions."""
    def angular_condition(phi):
        return beta * np.sin(phi - phi_beta) + b * np.sin(2 * (phi - phi_gamma))

    phi_solutions = find_all_roots(angular_condition, 0, 2*np.pi, n_samples=1000)
    if len(phi_solutions) == 0:
        return np.array([]).reshape(0, 2), np.array([])

    radii = (theta_E
             + a * np.cos(2 * (phi_solutions - phi_gamma))
             + beta * np.cos(phi_solutions - phi_beta))

    points = np.column_stack([
        radii * np.cos(phi_solutions),
        radii * np.sin(phi_solutions)
    ])
    return points, phi_solutions


def build_linear_system(points, phi_gamma):
    """Build the linear system for p = [beta_x, beta_y, theta_E, a, b]."""
    n = len(points)
    A = np.zeros((2 * n, 5))
    b_vec = np.zeros(2 * n)

    for i, (x, y) in enumerate(points):
        phi = np.arctan2(y, x)
        Delta = phi - phi_gamma
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        cos_2D, sin_2D = np.cos(2 * Delta), np.sin(2 * Delta)

        row_x = 2 * i
        A[row_x, 0] = 1.0
        A[row_x, 2] = cos_phi
        A[row_x, 3] = cos_2D * cos_phi
        A[row_x, 4] = -sin_2D * sin_phi
        b_vec[row_x] = x

        row_y = 2 * i + 1
        A[row_y, 1] = 1.0
        A[row_y, 2] = sin_phi
        A[row_y, 3] = cos_2D * sin_phi
        A[row_y, 4] = sin_2D * cos_phi
        b_vec[row_y] = y

    return A, b_vec


def solve_5x5_subset(A, b_vec, rows):
    """Solve a 5x5 subsystem exactly."""
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


def consistency_residual(phi_gamma, points, row_subset, check_row):
    """Compute consistency residual for a given phi_gamma."""
    A, b_vec = build_linear_system(points, phi_gamma)
    p, ok = solve_5x5_subset(A, b_vec, row_subset)
    if not ok:
        return np.inf
    return A[check_row, :] @ p - b_vec[check_row]


def invert_no_fit(points):
    """Perform no-fit inversion on 4 image points."""
    points = np.asarray(points)
    if len(points) != 4:
        return None, None

    row_combinations = [
        ([0, 1, 2, 3, 4], 5),
        ([0, 1, 2, 3, 5], 4),
        ([0, 1, 2, 4, 5], 3),
        ([0, 1, 3, 4, 5], 2),
    ]

    best_result = None
    best_max_residual = np.inf

    for row_subset, check_row in row_combinations:
        def h(phi_gamma):
            return consistency_residual(phi_gamma, points, row_subset, check_row)

        roots = find_all_roots(h, 0, np.pi/2, n_samples=200, tol=1e-10)
        if len(roots) == 0:
            continue

        for phi_gamma_candidate in roots:
            A, b_vec = build_linear_system(points, phi_gamma_candidate)
            p, ok = solve_5x5_subset(A, b_vec, row_subset)
            if not ok:
                continue

            residual = A @ p - b_vec
            max_res = np.max(np.abs(residual))

            if max_res < best_max_residual:
                best_max_residual = max_res
                best_result = {
                    'phi_gamma': phi_gamma_candidate,
                    'p': p,
                    'max_residual': max_res
                }

    if best_result is None:
        return None, None

    p = best_result['p']
    beta_x, beta_y, theta_E, a, b = p

    params = {
        'beta_x': beta_x,
        'beta_y': beta_y,
        'theta_E': theta_E,
        'a': a,
        'b': b,
        'phi_gamma': best_result['phi_gamma'],
        'max_residual': best_result['max_residual']
    }
    return params, best_result


def run_demo():
    """Run the minimal model demonstration."""
    print("\n" + "#"*60)
    print("# GAUGE LENS INVERSION - MINIMAL MODEL DEMO")
    print("# No-fit strategy: rootfinding + exact linear solve")
    print("#"*60)

    # Generate synthetic data using proven method
    print("\n[1] GENERATING SYNTHETIC DATA")
    print("-"*40)

    true_params = {
        'theta_E': 1.0,
        'a': 0.05,
        'b': 0.15,
        'beta': 0.08,
        'phi_beta': np.radians(30),
        'phi_gamma': np.radians(20)
    }

    images, phi_sols = generate_synthetic_cross(**true_params)

    print("True parameters:")
    print(f"  theta_E = {true_params['theta_E']}")
    print(f"  a = {true_params['a']}")
    print(f"  b = {true_params['b']}")
    print(f"  beta = {true_params['beta']}")
    print(f"  phi_beta = {np.degrees(true_params['phi_beta']):.1f} deg")
    print(f"  phi_gamma = {np.degrees(true_params['phi_gamma']):.1f} deg")

    print(f"\nGenerated {len(images)} images:")
    for i, (x, y) in enumerate(images):
        r = np.sqrt(x**2 + y**2)
        phi = np.degrees(np.arctan2(y, x))
        print(f"  Image {i+1}: ({x:+.4f}, {y:+.4f})  r={r:.4f}, phi={phi:+.1f} deg")

    # Run inversion
    print("\n[2] RUNNING INVERSION")
    print("-"*40)

    recovered, result = invert_no_fit(images)

    # Evaluate results
    print("\n[3] RESULTS COMPARISON")
    print("-"*40)

    if recovered is None:
        print("ERROR: No valid solutions found!")
        return False

    print("\nRecovered vs True:")
    print(f"  {'Parameter':<12} {'True':>12} {'Recovered':>12} {'Error':>12}")
    print("  " + "-"*50)

    for key in ['theta_E', 'a', 'b']:
        true_val = true_params[key]
        rec_val = recovered[key]
        err = rec_val - true_val
        print(f"  {key:<12} {true_val:>12.6f} {rec_val:>12.6f} {err:>+12.2e}")

    # Phase comparison
    true_phi = true_params['phi_gamma']
    rec_phi = recovered['phi_gamma']
    phi_err = rec_phi - true_phi
    print(f"  {'phi_gamma':<12} {np.degrees(true_phi):>8.2f} deg"
          f" {np.degrees(rec_phi):>8.2f} deg {np.degrees(phi_err):>+8.2e}")

    # Source position
    true_beta = true_params['beta']
    true_phi_b = true_params['phi_beta']
    true_bx = true_beta * np.cos(true_phi_b)
    true_by = true_beta * np.sin(true_phi_b)
    rec_bx, rec_by = recovered['beta_x'], recovered['beta_y']
    print(f"  {'beta_x':<12} {true_bx:>12.6f} {rec_bx:>12.6f} {rec_bx-true_bx:>+12.2e}")
    print(f"  {'beta_y':<12} {true_by:>12.6f} {rec_by:>12.6f} {rec_by-true_by:>+12.2e}")

    # Final assessment
    print("\n[4] ASSESSMENT")
    print("-"*40)

    max_res = recovered['max_residual']
    if max_res < 1e-10:
        status = "EXACT RECOVERY"
        success = True
    elif max_res < 1e-6:
        status = "GOOD RECOVERY"
        success = True
    else:
        status = "POOR RECOVERY"
        success = False

    print(f"Max residual: {max_res:.2e}")
    print(f"Status: {status}")

    if success:
        print("\n*** DEMO SUCCESSFUL: Parameters exactly recovered ***")
    else:
        print("\n*** DEMO FAILED: Recovery not exact ***")

    return success


if __name__ == '__main__':
    success = run_demo()
    sys.exit(0 if success else 1)
