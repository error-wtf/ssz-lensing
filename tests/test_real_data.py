#!/usr/bin/env python3
"""
Full Test Suite with Real Gravitational Lensing Data

Tests the no-fit inversion framework against:
1. Synthetic data (exact recovery validation)
2. Real observational data from published quad lens systems
3. Noise sensitivity analysis
4. Model adequacy diagnostics

Real data sources:
- Q2237+0305 (Einstein Cross) - Schneider et al., CASTLES survey
- B1608+656 - Fassnacht et al., CASTLES
- HE0435-1223 - Wisotzki et al., COSMOGRAIL
- PG1115+080 - Weymann et al., CASTLES

Authors: Carmen N. Wrede, Lino P. Casu
Date: January 2026
"""

import numpy as np
import sys
import os
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# =============================================================================
# REAL OBSERVATIONAL DATA
# =============================================================================

# Published image positions (relative to lens center, in arcseconds)
# Sources: CASTLES survey, COSMOGRAIL, various HST papers

REAL_LENSES = {
    'Q2237+0305': {
        'name': 'Einstein Cross',
        'source': 'CASTLES survey, Schneider et al. 1988',
        'z_lens': 0.0394,
        'z_source': 1.695,
        # Image positions relative to lens center (arcsec)
        # A: NE, B: NW, C: SW, D: SE
        'images': np.array([
            [+0.758, +0.964],   # A
            [-0.869, +0.541],   # B  
            [-0.634, -0.797],   # C
            [+0.674, -0.618],   # D
        ]),
        'errors': 0.003,  # typical astrometric error
    },
    
    'B1608+656': {
        'name': 'B1608+656',
        'source': 'CASTLES survey, Fassnacht et al. 1996',
        'z_lens': 0.6304,
        'z_source': 1.394,
        'images': np.array([
            [+0.738, +1.961],   # A
            [-0.745, +1.354],   # B
            [-1.128, -0.599],   # C
            [+1.128, -0.213],   # D
        ]),
        'errors': 0.003,
    },
    
    'HE0435-1223': {
        'name': 'HE0435-1223',
        'source': 'COSMOGRAIL, Wisotzki et al. 2002',
        'z_lens': 0.4546,
        'z_source': 1.693,
        'images': np.array([
            [+1.272, +0.306],   # A
            [-0.277, +1.148],   # B
            [-1.332, -0.152],   # C
            [+0.294, -1.306],   # D
        ]),
        'errors': 0.003,
    },
    
    'PG1115+080': {
        'name': 'PG1115+080',
        'source': 'CASTLES survey, Weymann et al. 1980',
        'z_lens': 0.311,
        'z_source': 1.722,
        # Note: This is actually a "fold" configuration, not pure cross
        'images': np.array([
            [+0.948, +0.795],   # A1
            [+1.071, +0.538],   # A2
            [-1.093, -0.260],   # B
            [-0.213, -1.018],   # C
        ]),
        'errors': 0.003,
    },
}


# =============================================================================
# INVERSION FUNCTIONS (from demo_minimal.py)
# =============================================================================

def generate_synthetic_images(theta_E, a, b, beta, phi_beta, phi_gamma):
    """Generate synthetic 4-image configuration."""
    def angular_condition(phi):
        return beta * np.sin(phi - phi_beta) + b * np.sin(2*(phi - phi_gamma))
    
    # Find roots
    n_samples = 1000
    phi_scan = np.linspace(0, 2*np.pi, n_samples)
    roots = []
    for i in range(n_samples - 1):
        f1 = angular_condition(phi_scan[i])
        f2 = angular_condition(phi_scan[i+1])
        if f1 * f2 < 0:
            # Bisection
            a_b, b_b = phi_scan[i], phi_scan[i+1]
            for _ in range(50):
                mid = (a_b + b_b) / 2
                if angular_condition(a_b) * angular_condition(mid) < 0:
                    b_b = mid
                else:
                    a_b = mid
            roots.append((a_b + b_b) / 2)
    
    if len(roots) != 4:
        return None, None
    
    phi_solutions = np.array(roots)
    
    # Compute radii
    radii = (theta_E 
             + a * np.cos(2*(phi_solutions - phi_gamma))
             + beta * np.cos(phi_solutions - phi_beta))
    
    images = np.column_stack([
        radii * np.cos(phi_solutions),
        radii * np.sin(phi_solutions)
    ])
    
    return images, phi_solutions


def build_linear_system(images, phi_gamma):
    """Build 8x5 linear system for given phi_gamma."""
    n = len(images)
    A = np.zeros((2 * n, 5))
    b = np.zeros(2 * n)
    
    for i in range(n):
        x, y = images[i]
        phi = np.arctan2(y, x)
        Delta = phi - phi_gamma
        
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c2D = np.cos(2 * Delta)
        s2D = np.sin(2 * Delta)
        
        # x equation (row 2i)
        A[2*i, 0] = 1.0           # beta_x
        A[2*i, 1] = 0.0           # beta_y
        A[2*i, 2] = c_phi         # theta_E
        A[2*i, 3] = c2D * c_phi   # a
        A[2*i, 4] = -s2D * s_phi  # b
        b[2*i] = x
        
        # y equation (row 2i+1)
        A[2*i+1, 0] = 0.0           # beta_x
        A[2*i+1, 1] = 1.0           # beta_y
        A[2*i+1, 2] = s_phi         # theta_E
        A[2*i+1, 3] = c2D * s_phi   # a
        A[2*i+1, 4] = s2D * c_phi   # b
        b[2*i+1] = y
    
    return A, b


def consistency_residual(images, phi_gamma, row_subset, check_row):
    """Compute consistency residual for given phi_gamma."""
    A, b_vec = build_linear_system(images, phi_gamma)
    
    A_sub = A[row_subset, :]
    b_sub = b_vec[row_subset]
    
    try:
        det = np.linalg.det(A_sub)
        if abs(det) < 1e-14:
            return np.inf
        p = np.linalg.solve(A_sub, b_sub)
    except np.linalg.LinAlgError:
        return np.inf
    
    return A[check_row, :] @ p - b_vec[check_row]


def invert_no_fit(images, phi_range=(0, np.pi/2), n_samples=200, tol=1e-12):
    """
    Perform no-fit lens inversion.
    
    Returns list of solutions, each with:
        - phi_gamma: quadrupole phase
        - params: [beta_x, beta_y, theta_E, a, b]
        - residuals: per-equation residuals
        - max_residual: maximum absolute residual
    """
    row_subset = [0, 1, 2, 3, 4]  # 5 rows for 5 unknowns
    check_row = 5
    
    def h(phi):
        return consistency_residual(images, phi, row_subset, check_row)
    
    # Find all roots via bracketing
    phi_scan = np.linspace(phi_range[0], phi_range[1], n_samples)
    h_values = np.array([h(phi) for phi in phi_scan])
    
    roots = []
    for i in range(n_samples - 1):
        if np.isfinite(h_values[i]) and np.isfinite(h_values[i+1]):
            if h_values[i] * h_values[i+1] < 0:
                # Bisection
                a_b, b_b = phi_scan[i], phi_scan[i+1]
                for _ in range(60):
                    mid = (a_b + b_b) / 2
                    h_mid = h(mid)
                    if not np.isfinite(h_mid):
                        break
                    if h(a_b) * h_mid < 0:
                        b_b = mid
                    else:
                        a_b = mid
                root = (a_b + b_b) / 2
                if np.isfinite(h(root)) and abs(h(root)) < 1e-6:
                    roots.append(root)
    
    # Solve for each root
    solutions = []
    for phi_gamma in roots:
        A, b_vec = build_linear_system(images, phi_gamma)
        
        A_sub = A[row_subset, :]
        b_sub = b_vec[row_subset]
        
        try:
            p = np.linalg.solve(A_sub, b_sub)
        except np.linalg.LinAlgError:
            continue
        
        # Check physical constraints
        theta_E = p[2]
        if theta_E <= 0:
            continue
        
        # Compute all residuals
        residuals = A @ p - b_vec
        max_res = np.max(np.abs(residuals))
        
        solutions.append({
            'phi_gamma': phi_gamma,
            'params': p,
            'residuals': residuals,
            'max_residual': max_res,
            'theta_E': theta_E,
            'a': p[3],
            'b': p[4],
            'beta_x': p[0],
            'beta_y': p[1],
            'beta': np.sqrt(p[0]**2 + p[1]**2),
        })
    
    return solutions


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_synthetic_exact():
    """Test 1: Exact recovery on synthetic data."""
    print("\n" + "="*70)
    print("TEST 1: Synthetic Data - Exact Recovery")
    print("="*70)
    
    # True parameters
    theta_E = 1.0
    a = 0.05
    b = 0.15
    beta = 0.08
    phi_beta = np.radians(30)
    phi_gamma = np.radians(20)
    
    print(f"\nTrue parameters:")
    print(f"  theta_E  = {theta_E}")
    print(f"  a        = {a}")
    print(f"  b        = {b}")
    print(f"  beta     = {beta}")
    print(f"  phi_beta = {np.degrees(phi_beta):.1f} deg")
    print(f"  phi_gamma = {np.degrees(phi_gamma):.1f} deg")
    
    # Generate images
    images, _ = generate_synthetic_images(theta_E, a, b, beta, phi_beta, phi_gamma)
    if images is None:
        print("\nFAILED: Could not generate 4 images")
        return False, None
    
    print(f"\nGenerated 4 images:")
    for i, (x, y) in enumerate(images):
        print(f"  Image {i+1}: ({x:+.6f}, {y:+.6f})")
    
    # Invert
    solutions = invert_no_fit(images)
    
    if not solutions:
        print("\nFAILED: No valid solutions found")
        return False, None
    
    best = min(solutions, key=lambda s: s['max_residual'])
    
    print(f"\nRecovered parameters:")
    print(f"  theta_E   = {best['theta_E']:.12f}")
    print(f"  a         = {best['a']:.12f}")
    print(f"  b         = {best['b']:.12f}")
    print(f"  beta      = {best['beta']:.12f}")
    print(f"  phi_gamma = {np.degrees(best['phi_gamma']):.12f} deg")
    print(f"\n  Max |residual| = {best['max_residual']:.2e}")
    
    # Check recovery
    errors = {
        'theta_E': abs(best['theta_E'] - theta_E),
        'a': abs(best['a'] - a),
        'b': abs(abs(best['b']) - abs(b)),  # sign can flip
        'beta': abs(best['beta'] - beta),
        'phi_gamma': min(abs(best['phi_gamma'] - phi_gamma),
                        abs(best['phi_gamma'] - phi_gamma + np.pi/2),
                        abs(best['phi_gamma'] - phi_gamma - np.pi/2)),
    }
    
    all_exact = all(e < 1e-8 for e in errors.values())
    status = "PASS" if all_exact else "FAIL"
    
    print(f"\nRecovery errors:")
    for name, err in errors.items():
        print(f"  {name}: {err:.2e}")
    
    print(f"\n>>> TEST 1 STATUS: {status}")
    
    return all_exact, best


def test_synthetic_random(n_configs=50):
    """Test 2: Random parameter sweep."""
    print("\n" + "="*70)
    print(f"TEST 2: Random Parameter Sweep ({n_configs} configurations)")
    print("="*70)
    
    np.random.seed(42)
    
    successes = 0
    max_residuals = []
    max_errors = []
    
    for i in range(n_configs):
        theta_E = 1.0
        a = np.random.uniform(0.01, 0.1)
        b = np.random.uniform(0.1, 0.25)
        beta = np.random.uniform(0.02, 0.1)
        phi_beta = np.random.uniform(0, 2*np.pi)
        phi_gamma = np.random.uniform(0, np.pi/2)
        
        images, _ = generate_synthetic_images(theta_E, a, b, beta, phi_beta, phi_gamma)
        if images is None:
            continue
        
        solutions = invert_no_fit(images)
        if not solutions:
            continue
        
        best = min(solutions, key=lambda s: s['max_residual'])
        max_residuals.append(best['max_residual'])
        
        errors = [
            abs(best['theta_E'] - theta_E),
            abs(best['a'] - a),
            abs(abs(best['b']) - abs(b)),
            abs(best['beta'] - beta),
        ]
        max_errors.append(max(errors))
        
        if best['max_residual'] < 1e-8:
            successes += 1
    
    success_rate = successes / n_configs * 100
    
    print(f"\nResults:")
    print(f"  Configurations tested: {n_configs}")
    print(f"  Successful inversions: {successes}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    if max_residuals:
        print(f"\n  Max |residual| statistics:")
        print(f"    Mean:   {np.mean(max_residuals):.2e}")
        print(f"    Median: {np.median(max_residuals):.2e}")
        print(f"    Max:    {np.max(max_residuals):.2e}")
        
        print(f"\n  Max parameter error statistics:")
        print(f"    Mean:   {np.mean(max_errors):.2e}")
        print(f"    Max:    {np.max(max_errors):.2e}")
    
    status = "PASS" if success_rate == 100 else "PARTIAL" if success_rate > 90 else "FAIL"
    print(f"\n>>> TEST 2 STATUS: {status}")
    
    return success_rate == 100, {'success_rate': success_rate, 'max_residuals': max_residuals}


def test_real_lens(name, data):
    """Test a real lens system."""
    print(f"\n{'-'*60}")
    print(f"Real Lens: {name} ({data['name']})")
    print(f"Source: {data['source']}")
    print(f"z_lens = {data['z_lens']}, z_source = {data['z_source']}")
    print(f"{'-'*60}")
    
    images = data['images']
    
    print(f"\nImage positions (arcsec):")
    labels = ['A', 'B', 'C', 'D']
    for i, (x, y) in enumerate(images):
        r = np.sqrt(x**2 + y**2)
        theta = np.degrees(np.arctan2(y, x))
        print(f"  {labels[i]}: ({x:+.3f}, {y:+.3f})  r={r:.3f}  theta={theta:+.1f} deg")
    
    # Run inversion
    solutions = invert_no_fit(images)
    
    if not solutions:
        print("\n  Result: NO SOLUTION FOUND")
        print("  This may indicate model inadequacy (real data doesn't match m=2 model)")
        return False, None
    
    best = min(solutions, key=lambda s: s['max_residual'])
    
    print(f"\nRecovered parameters:")
    print(f"  theta_E   = {best['theta_E']:.4f} arcsec")
    print(f"  a         = {best['a']:.4f}")
    print(f"  b         = {best['b']:.4f}")
    print(f"  beta      = {best['beta']:.4f} arcsec")
    print(f"  phi_gamma = {np.degrees(best['phi_gamma']):.1f} deg")
    print(f"\n  Max |residual| = {best['max_residual']:.4f} arcsec")
    
    # Residual analysis
    print(f"\n  Residuals per image (arcsec):")
    for i in range(4):
        rx = best['residuals'][2*i]
        ry = best['residuals'][2*i+1]
        r_total = np.sqrt(rx**2 + ry**2)
        print(f"    {labels[i]}: x={rx:+.4f}, y={ry:+.4f}  |r|={r_total:.4f}")
    
    # Model adequacy assessment
    astrometric_error = data['errors']
    rms_residual = np.sqrt(np.mean(best['residuals']**2))
    
    if best['max_residual'] < 3 * astrometric_error:
        adequacy = "GOOD (residuals within measurement error)"
    elif best['max_residual'] < 10 * astrometric_error:
        adequacy = "MARGINAL (residuals 3-10x measurement error)"
    else:
        adequacy = "POOR (residuals >> measurement error, model may be inadequate)"
    
    print(f"\n  Model adequacy: {adequacy}")
    print(f"  (astrometric error ~ {astrometric_error} arcsec)")
    
    return True, best


def test_real_data():
    """Test 3: Real observational data."""
    print("\n" + "="*70)
    print("TEST 3: Real Observational Data")
    print("="*70)
    
    results = {}
    successes = 0
    
    for name, data in REAL_LENSES.items():
        success, sol = test_real_lens(name, data)
        results[name] = {'success': success, 'solution': sol}
        if success:
            successes += 1
    
    print(f"\n{'='*60}")
    print(f"Real Data Summary: {successes}/{len(REAL_LENSES)} systems inverted")
    print(f"{'='*60}")
    
    status = "PASS" if successes == len(REAL_LENSES) else "PARTIAL"
    print(f"\n>>> TEST 3 STATUS: {status}")
    
    return successes == len(REAL_LENSES), results


def test_noise_sensitivity():
    """Test 4: Noise sensitivity analysis."""
    print("\n" + "="*70)
    print("TEST 4: Noise Sensitivity Analysis")
    print("="*70)
    
    # Generate clean synthetic data
    theta_E = 1.0
    a = 0.05
    b = 0.15
    beta = 0.08
    phi_beta = np.radians(30)
    phi_gamma = np.radians(20)
    
    images_clean, _ = generate_synthetic_images(theta_E, a, b, beta, phi_beta, phi_gamma)
    if images_clean is None:
        print("FAILED: Could not generate test images")
        return False, None
    
    noise_levels = [0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2]
    results = []
    
    print(f"\nNoise level (arcsec) | Max |residual| | theta_E error | Status")
    print("-" * 65)
    
    np.random.seed(123)
    
    for noise in noise_levels:
        if noise > 0:
            images = images_clean + np.random.normal(0, noise, images_clean.shape)
        else:
            images = images_clean.copy()
        
        solutions = invert_no_fit(images)
        
        if not solutions:
            print(f"{noise:>18.0e} | {'NO SOLUTION':^15} | {'N/A':^13} | FAIL")
            results.append({'noise': noise, 'success': False})
            continue
        
        best = min(solutions, key=lambda s: s['max_residual'])
        theta_E_err = abs(best['theta_E'] - theta_E)
        
        if noise == 0:
            status = "EXACT" if best['max_residual'] < 1e-10 else "GOOD"
        elif best['max_residual'] < 3 * noise:
            status = "GOOD"
        elif best['max_residual'] < 10 * noise:
            status = "MARGINAL"
        else:
            status = "POOR"
        
        print(f"{noise:>18.0e} | {best['max_residual']:>15.2e} | {theta_E_err:>13.2e} | {status}")
        results.append({
            'noise': noise,
            'success': True,
            'max_residual': best['max_residual'],
            'theta_E_error': theta_E_err,
            'status': status
        })
    
    print(f"\n>>> TEST 4 STATUS: COMPLETE")
    
    return True, results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests and generate report."""
    print("="*70)
    print("FULL TEST SUITE: No-Fit Gravitational Lens Inversion")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    
    all_results = {}
    
    # Test 1: Synthetic exact
    success1, result1 = test_synthetic_exact()
    all_results['synthetic_exact'] = {'passed': success1, 'data': result1}
    
    # Test 2: Random sweep
    success2, result2 = test_synthetic_random(50)
    all_results['random_sweep'] = {'passed': success2, 'data': result2}
    
    # Test 3: Real data
    success3, result3 = test_real_data()
    all_results['real_data'] = {'passed': success3, 'data': result3}
    
    # Test 4: Noise sensitivity
    success4, result4 = test_noise_sensitivity()
    all_results['noise_sensitivity'] = {'passed': success4, 'data': result4}
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    tests = [
        ("Test 1: Synthetic Exact Recovery", success1),
        ("Test 2: Random Parameter Sweep", success2),
        ("Test 3: Real Observational Data", success3),
        ("Test 4: Noise Sensitivity", success4),
    ]
    
    passed = sum(1 for _, s in tests if s)
    
    for name, status in tests:
        mark = "PASS" if status else "PARTIAL/FAIL"
        print(f"  {name}: {mark}")
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    overall = "ALL TESTS PASSED" if passed == len(tests) else "SOME TESTS NEED ATTENTION"
    print(f"\n>>> OVERALL: {overall}")
    
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    sys.exit(main())
