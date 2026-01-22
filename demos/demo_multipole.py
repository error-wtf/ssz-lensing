#!/usr/bin/env python3
"""
Multipole Model Demo

Demonstrates the general multipole framework:
1. Generate images from m=2 multipole model
2. Run inversion using conditional linearity
3. Show how higher multipoles would work

This showcases the extensibility of the no-fit approach.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from dataio.datasets import generate_cross_images, generate_multipole_images
from inversion.root_solvers import find_all_roots
from inversion.exact_solvers import solve_linear_exact
from inversion.diagnostics import residual_report
from inversion.constraints import count_degrees_of_freedom, inversion_summary


def multipole_linear_system(images, phases):
    """
    Build linear system for multipole model.

    When phases are fixed, the system is linear in:
    [beta_x, beta_y, theta_E, A_1, B_1, A_2, B_2, ...]

    where A_m = theta_E * a_m, B_m = theta_E * b_m
    """
    n = len(images)
    m_max = max(phases.keys()) if phases else 2

    # Parameters: beta_x, beta_y, theta_E, then A_m, B_m for each m
    n_params = 3 + 2 * m_max
    A = np.zeros((2*n, n_params))
    b = np.zeros(2*n)

    for i, (x, y) in enumerate(images):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        c, s = np.cos(phi), np.sin(phi)

        # x-equation
        A[2*i, 0] = 1  # beta_x
        A[2*i, 1] = 0  # beta_y
        A[2*i, 2] = c  # theta_E (monopole)

        # y-equation
        A[2*i+1, 0] = 0
        A[2*i+1, 1] = 1
        A[2*i+1, 2] = s

        col = 3
        for m in range(1, m_max + 1):
            phi_m = phases.get(m, 0.0)
            delta = phi - phi_m
            cos_m = np.cos(m * delta)
            sin_m = np.sin(m * delta)

            # Coefficients for A_m and B_m in alpha_x, alpha_y
            coeff_A_x = cos_m * c + m * sin_m * s
            coeff_B_x = sin_m * c - m * cos_m * s
            coeff_A_y = cos_m * s - m * sin_m * c
            coeff_B_y = sin_m * s + m * cos_m * c

            A[2*i, col] = coeff_A_x
            A[2*i, col+1] = coeff_B_x
            A[2*i+1, col] = coeff_A_y
            A[2*i+1, col+1] = coeff_B_y

            col += 2

        b[2*i] = x
        b[2*i+1] = y

    return A, b


def multipole_consistency(images, phi_2, m_max=2):
    """Consistency function for m=2 model."""
    phases = {1: 0.0, 2: phi_2}

    A, b_vec = multipole_linear_system(images, phases)
    n_params = A.shape[1]

    if A.shape[0] < n_params:
        return float('nan')

    # Solve first n_params equations
    A_sub = A[:n_params, :]
    b_sub = b_vec[:n_params]

    p, ok = solve_linear_exact(A_sub, b_sub)
    if not ok:
        return float('nan')

    # Residual on next equation
    if A.shape[0] > n_params:
        h = np.dot(A[n_params, :], p) - b_vec[n_params]
        return h

    return 0.0


def invert_multipole_m2(images, tol=1e-12):
    """Invert using m=2 multipole model."""
    print("\nSearching for phi_2 roots...")

    def h(phi):
        return multipole_consistency(images, phi, m_max=2)

    roots = find_all_roots(h, 0, np.pi, n_samples=500, tol=tol)
    print(f"Found {len(roots)} root(s)")

    solutions = []

    for phi_2 in roots:
        phases = {1: 0.0, 2: phi_2}
        A, b_vec = multipole_linear_system(images, phases)

        n_params = A.shape[1]
        A_sub = A[:n_params, :]
        b_sub = b_vec[:n_params]

        p, ok = solve_linear_exact(A_sub, b_sub)
        if not ok:
            continue

        # Convert to physical parameters
        beta_x, beta_y, theta_E = p[0], p[1], p[2]
        if theta_E <= 0:
            continue

        params = {
            'beta_x': beta_x,
            'beta_y': beta_y,
            'theta_E': theta_E
        }

        col = 3
        for m in range(1, 3):
            A_m, B_m = p[col], p[col+1]
            a_m = A_m / theta_E
            b_m = B_m / theta_E
            params[f'a_{m}'] = a_m
            params[f'b_{m}'] = b_m
            params[f'phi_{m}'] = phases[m]
            col += 2

        residuals = A @ p - b_vec
        report = residual_report(residuals, tol)

        solutions.append({
            'params': params,
            'residuals': residuals,
            'report': report
        })

    return solutions


def run_demo():
    """Run the multipole model demonstration."""
    print("\n" + "#"*60)
    print("# MULTIPOLE MODEL DEMO")
    print("# General framework with conditional linearity")
    print("#"*60)

    # Part 1: DoF analysis
    print("\n[1] DEGREES OF FREEDOM ANALYSIS")
    print("-"*40)

    unknowns_m2 = ['beta_x', 'beta_y', 'theta_E', 'a_1', 'b_1', 'a_2', 'b_2', 'phi_2']
    status = count_degrees_of_freedom(4, unknowns_m2)

    print(f"m=2 model with 4 images:")
    print(f"  Equations: {status.n_equations}")
    print(f"  Unknowns: {status.n_unknowns}")
    print(f"    Linear: {status.n_linear}")
    print(f"    Nonlinear: {status.n_nonlinear}")
    print(f"  Status: {status.status}")
    print(f"  {status.message}")

    # Part 2: Generate and invert
    print("\n[2] SYNTHETIC DATA GENERATION")
    print("-"*40)

    true_params = {
        'theta_E': 1.0,
        'beta': 0.08,
        'phi_beta': 0.4,
        'a': 0.0,
        'b': 0.12,
        'phi_gamma': 0.6
    }

    images, params = generate_cross_images(**true_params)

    print("True parameters (quadrupole form):")
    print(f"  theta_E = {params['theta_E']}")
    print(f"  b (quadrupole) = {params['b']}")
    print(f"  phi_gamma = {np.degrees(params['phi_gamma']):.1f} deg")

    print("\nImages:")
    for i, (x, y) in enumerate(images):
        print(f"  {i+1}: ({x:+.4f}, {y:+.4f})")

    # Part 3: Multipole inversion
    print("\n[3] MULTIPOLE INVERSION (m_max=2)")
    print("-"*40)

    solutions = invert_multipole_m2(images)

    if not solutions:
        print("No solutions found!")
        return False

    best = min(solutions, key=lambda s: s['report']['max_abs'])
    recovered = best['params']

    print("\nRecovered multipole parameters:")
    print(f"  theta_E = {recovered['theta_E']:.6f}")
    print(f"  beta = ({recovered['beta_x']:.6f}, {recovered['beta_y']:.6f})")

    for m in [1, 2]:
        a_m = recovered.get(f'a_{m}', 0)
        b_m = recovered.get(f'b_{m}', 0)
        phi_m = recovered.get(f'phi_{m}', 0)
        amp = np.sqrt(a_m**2 + b_m**2)
        print(f"  m={m}: amp={amp:.6f}, phi={np.degrees(phi_m):.1f} deg")

    print(f"\nMax residual: {best['report']['max_abs']:.2e}")

    # Part 4: Framework extensibility
    print("\n[4] FRAMEWORK EXTENSIBILITY")
    print("-"*40)

    print("The multipole framework supports:")
    print("  - Any m_max (limited by observables)")
    print("  - Conditional linearity: phases nonlinear, amplitudes linear")
    print("  - Nested rootfinding for multiple phases")
    print("  - Natural extension to higher multipoles")

    print("\nDoF requirements for higher m_max:")
    for m_max in [2, 3, 4]:
        n_phases = m_max - 1  # phi_1 = 0 by convention
        n_linear = 3 + 2*m_max  # beta_x, beta_y, theta_E + amplitudes
        n_unknowns = n_linear + n_phases
        n_images_needed = (n_unknowns + 1) // 2

        print(f"  m_max={m_max}: {n_unknowns} unknowns -> need >= {n_images_needed} images")

    # Assessment
    print("\n[5] ASSESSMENT")
    print("-"*40)

    max_res = best['report']['max_abs']
    success = max_res < 1e-8

    if success:
        print("*** MULTIPOLE DEMO SUCCESSFUL ***")
    else:
        print(f"*** Max residual {max_res:.2e} - check implementation ***")

    return success


if __name__ == '__main__':
    success = run_demo()
    sys.exit(0 if success else 1)
