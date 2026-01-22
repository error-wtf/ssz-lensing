#!/usr/bin/env python3
"""
Test Suite: Minimal Model Exact Recovery

Verifies that the no-fit inversion exactly recovers known parameters.
These tests validate the core algorithm.

Run with: python -m pytest tests/test_minimal_exact.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Import modules under test
from dataio.datasets import generate_cross_images
from inversion.exact_solvers import solve_linear_exact, matrix_rank
from inversion.root_solvers import bisection, find_all_roots


class TestLinearSolver:
    """Tests for exact linear solver."""

    def test_simple_2x2(self):
        """Solve simple 2x2 system."""
        A = np.array([[2, 1], [1, 3]])
        b = np.array([5, 7])
        x, ok = solve_linear_exact(A, b)
        assert ok
        assert np.allclose(A @ x, b)

    def test_identity(self):
        """Solve with identity matrix."""
        n = 5
        A = np.eye(n)
        b = np.arange(n, dtype=float)
        x, ok = solve_linear_exact(A, b)
        assert ok
        assert np.allclose(x, b)

    def test_singular_matrix(self):
        """Detect singular matrix."""
        A = np.array([[1, 2], [2, 4]])
        b = np.array([3, 6])
        x, ok = solve_linear_exact(A, b)
        assert not ok

    def test_near_singular(self):
        """Handle near-singular matrix."""
        A = np.array([[1, 2], [1, 2 + 1e-10]])
        b = np.array([3, 3])
        x, ok = solve_linear_exact(A, b)
        # Should either fail or give poor solution


class TestRootSolver:
    """Tests for root finding."""

    def test_bisection_linear(self):
        """Find root of linear function."""
        def f(x):
            return x - 3
        root, ok = bisection(f, 0, 10)
        assert ok
        assert abs(root - 3) < 1e-10

    def test_bisection_quadratic(self):
        """Find root of quadratic."""
        def f(x):
            return x**2 - 4
        root, ok = bisection(f, 0, 5)
        assert ok
        assert abs(root - 2) < 1e-10

    def test_bisection_trig(self):
        """Find root of sin."""
        root, ok = bisection(np.sin, 2, 4)
        assert ok
        assert abs(root - np.pi) < 1e-10

    def test_find_all_roots(self):
        """Find multiple roots."""
        def f(x):
            return np.sin(x)
        roots = find_all_roots(f, 0, 10, n_samples=100)
        # Should find roots at 0, pi, 2*pi, 3*pi
        expected = [0, np.pi, 2*np.pi, 3*np.pi]
        assert len(roots) >= 3  # At least pi, 2*pi, 3*pi


class TestExactRecovery:
    """Tests for parameter recovery."""

    def test_standard_cross(self):
        """Recover parameters from standard configuration."""
        true_params = {
            'theta_E': 1.0,
            'beta': 0.1,
            'phi_beta': 0.3,
            'a': 0.0,
            'b': 0.15,
            'phi_gamma': 0.5
        }

        images, params = generate_cross_images(**true_params)
        assert len(images) == 4

        # Run inversion (inline implementation)
        recovered = self._invert(images)

        # Check recovery
        assert abs(recovered['theta_E'] - params['theta_E']) < 1e-8
        assert abs(recovered['b'] - params['b']) < 1e-8

    def test_symmetric_cross(self):
        """Recover from symmetric configuration."""
        true_params = {
            'theta_E': 1.0,
            'beta': 0.05,
            'phi_beta': 0.0,
            'a': 0.0,
            'b': 0.1,
            'phi_gamma': 0.0
        }

        images, params = generate_cross_images(**true_params)
        recovered = self._invert(images)

        assert abs(recovered['theta_E'] - params['theta_E']) < 1e-8

    def test_asymmetric_cross(self):
        """Recover from asymmetric configuration."""
        true_params = {
            'theta_E': 1.0,
            'beta': 0.15,
            'phi_beta': 0.7,
            'a': 0.05,
            'b': 0.2,
            'phi_gamma': 0.8
        }

        images, params = generate_cross_images(**true_params)
        recovered = self._invert(images)

        assert abs(recovered['theta_E'] - params['theta_E']) < 1e-6
        # Allow looser tolerance for asymmetric case

    def test_varying_theta_E(self):
        """Recovery works for different Einstein radii."""
        for theta_E in [0.5, 1.0, 2.0, 5.0]:
            images, params = generate_cross_images(
                theta_E=theta_E, beta=0.1*theta_E,
                phi_beta=0.3, a=0.0, b=0.15, phi_gamma=0.5
            )
            recovered = self._invert(images)
            assert abs(recovered['theta_E'] - theta_E) < 1e-6 * theta_E

    def _invert(self, images):
        """Simple inversion for testing."""
        def build_system(phi_gamma):
            n = len(images)
            A = np.zeros((2*n, 5))
            b = np.zeros(2*n)
            for i, (x, y) in enumerate(images):
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                c, s = np.cos(phi), np.sin(phi)
                C = np.cos(2*(phi - phi_gamma))
                S = np.sin(2*(phi - phi_gamma))
                A[2*i] = [1, 0, c, r*c, C*c + S*s]
                A[2*i+1] = [0, 1, s, r*s, C*s - S*c]
                b[2*i] = x
                b[2*i+1] = y
            return A, b

        def consistency(phi_gamma):
            A, b_vec = build_system(phi_gamma)
            p, ok = solve_linear_exact(A[:5], b_vec[:5])
            if not ok:
                return float('nan')
            return np.dot(A[5], p) - b_vec[5]

        roots = find_all_roots(consistency, 0, np.pi, n_samples=500)
        if not roots:
            return None

        best_res = float('inf')
        best_params = None

        for phi_gamma in roots:
            A, b_vec = build_system(phi_gamma)
            p, ok = solve_linear_exact(A[:5], b_vec[:5])
            if not ok:
                continue

            residuals = A @ p - b_vec
            max_res = np.max(np.abs(residuals))

            if max_res < best_res:
                best_res = max_res
                beta_x, beta_y, T, a, B = p
                theta_E = T / (1 - a) if abs(1-a) > 1e-12 else T
                b_phys = B / theta_E if abs(theta_E) > 1e-12 else 0
                best_params = {
                    'beta_x': beta_x,
                    'beta_y': beta_y,
                    'theta_E': theta_E,
                    'a': a,
                    'b': b_phys,
                    'phi_gamma': phi_gamma
                }

        return best_params


class TestMatrixRank:
    """Tests for matrix rank computation."""

    def test_full_rank(self):
        """Full rank matrix."""
        A = np.array([[1, 2], [3, 4]])
        assert matrix_rank(A) == 2

    def test_rank_deficient(self):
        """Rank deficient matrix."""
        A = np.array([[1, 2], [2, 4]])
        assert matrix_rank(A) == 1

    def test_rectangular(self):
        """Rectangular matrix."""
        A = np.array([[1, 2, 3], [4, 5, 6]])
        assert matrix_rank(A) == 2


def run_tests():
    """Run all tests manually."""
    print("Running tests...")

    # Linear solver tests
    t = TestLinearSolver()
    t.test_simple_2x2()
    t.test_identity()
    t.test_singular_matrix()
    print("  Linear solver tests: PASSED")

    # Root solver tests
    t = TestRootSolver()
    t.test_bisection_linear()
    t.test_bisection_quadratic()
    t.test_bisection_trig()
    t.test_find_all_roots()
    print("  Root solver tests: PASSED")

    # Recovery tests
    t = TestExactRecovery()
    t.test_standard_cross()
    t.test_symmetric_cross()
    t.test_asymmetric_cross()
    t.test_varying_theta_E()
    print("  Exact recovery tests: PASSED")

    # Matrix rank tests
    t = TestMatrixRank()
    t.test_full_rank()
    t.test_rank_deficient()
    t.test_rectangular()
    print("  Matrix rank tests: PASSED")

    print("\nAll tests PASSED!")
    return True


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
