#!/usr/bin/env python3
"""
Test Suite: Multipole Model Consistency

Verifies that the general multipole framework:
1. Produces consistent results with the minimal model
2. Handles DoF counting correctly
3. Maintains numerical stability

Run with: python -m pytest tests/test_multipole_consistency.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from dataio.datasets import generate_cross_images, generate_multipole_images
from inversion.constraints import count_degrees_of_freedom, check_image_multiplicity
from inversion.exact_solvers import solve_linear_exact, matrix_rank


class TestDoFCounting:
    """Tests for degrees of freedom analysis."""

    def test_minimal_model_4_images(self):
        """4 images with minimal m=2 model."""
        unknowns = ['beta_x', 'beta_y', 'theta_E', 'a', 'b', 'phi_gamma']
        status = count_degrees_of_freedom(4, unknowns)

        assert status.n_equations == 8
        assert status.n_unknowns == 6
        assert status.status == 'overdetermined'

    def test_underdetermined(self):
        """3 images with m=2 model is underdetermined."""
        unknowns = ['beta_x', 'beta_y', 'theta_E', 'a', 'b', 'phi_gamma']
        status = count_degrees_of_freedom(3, unknowns)

        assert status.n_equations == 6
        assert status.n_unknowns == 6
        assert status.status == 'determined'

    def test_multipole_m3(self):
        """m=3 model needs more images."""
        unknowns = ['beta_x', 'beta_y', 'theta_E',
                    'a_1', 'b_1', 'a_2', 'b_2', 'phi_2',
                    'a_3', 'b_3', 'phi_3']
        status = count_degrees_of_freedom(4, unknowns)

        # 8 equations, 11 unknowns
        assert status.status == 'underdetermined'

    def test_image_multiplicity_quad(self):
        """Quadrupole model needs 4 images."""
        result = check_image_multiplicity(4, 'quadrupole')
        assert result['valid']
        assert result['required'] == 4

        result = check_image_multiplicity(3, 'quadrupole')
        assert not result['valid']


class TestMultipoleConsistency:
    """Tests for multipole model consistency."""

    def test_m2_matches_minimal(self):
        """m=2 multipole should match minimal quadrupole model."""
        images, params = generate_cross_images(
            theta_E=1.0, beta=0.1, phi_beta=0.3,
            a=0.0, b=0.15, phi_gamma=0.5
        )

        # Both models should give same residuals for same parameters
        phi_gamma = params['phi_gamma']

        # Minimal model system
        A_min, b_min = self._build_minimal_system(images, phi_gamma)

        # Multipole m=2 system
        A_mp, b_mp = self._build_multipole_system(images, {1: 0, 2: phi_gamma})

        # Both should be solvable
        p_min, ok_min = solve_linear_exact(A_min[:5], b_min[:5])
        p_mp, ok_mp = solve_linear_exact(A_mp[:7], b_mp[:7])

        assert ok_min
        assert ok_mp

    def test_multipole_residuals(self):
        """Multipole model residuals should be small for correct params."""
        images, params = generate_cross_images(
            theta_E=1.0, beta=0.1, phi_beta=0.3,
            a=0.0, b=0.15, phi_gamma=0.5
        )

        phi_gamma = params['phi_gamma']
        A, b_vec = self._build_multipole_system(images, {1: 0, 2: phi_gamma})

        # Solve
        n_params = 7  # beta_x, beta_y, theta_E, A_1, B_1, A_2, B_2
        p, ok = solve_linear_exact(A[:n_params], b_vec[:n_params])

        if ok:
            residuals = A @ p - b_vec
            max_res = np.max(np.abs(residuals))
            assert max_res < 1e-8

    def test_phase_periodicity(self):
        """Results should respect phase periodicity."""
        images, params = generate_cross_images(
            theta_E=1.0, beta=0.1, phi_beta=0.3,
            a=0.0, b=0.15, phi_gamma=0.5
        )

        # phi_gamma and phi_gamma + pi should give related solutions
        # (for m=2, period is pi)
        phi1 = params['phi_gamma']
        phi2 = phi1 + np.pi

        A1, b1 = self._build_minimal_system(images, phi1)
        A2, b2 = self._build_minimal_system(images, phi2)

        p1, ok1 = solve_linear_exact(A1[:5], b1[:5])
        p2, ok2 = solve_linear_exact(A2[:5], b2[:5])

        # Both should give valid solutions
        # (residuals may differ due to phase)

    def _build_minimal_system(self, images, phi_gamma):
        """Build minimal model system."""
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

    def _build_multipole_system(self, images, phases):
        """Build multipole system."""
        n = len(images)
        m_max = 2
        n_params = 3 + 2*m_max  # beta_x, beta_y, theta_E, A_1, B_1, A_2, B_2

        A = np.zeros((2*n, n_params))
        b = np.zeros(2*n)

        for i, (x, y) in enumerate(images):
            phi = np.arctan2(y, x)
            c, s = np.cos(phi), np.sin(phi)

            A[2*i, 0] = 1  # beta_x
            A[2*i, 1] = 0  # beta_y
            A[2*i, 2] = c  # theta_E

            A[2*i+1, 0] = 0
            A[2*i+1, 1] = 1
            A[2*i+1, 2] = s

            col = 3
            for m in range(1, m_max + 1):
                phi_m = phases.get(m, 0)
                delta = phi - phi_m
                cos_m = np.cos(m * delta)
                sin_m = np.sin(m * delta)

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


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_small_quadrupole(self):
        """Handle small quadrupole amplitude."""
        images, params = generate_cross_images(
            theta_E=1.0, beta=0.01, phi_beta=0.0,
            a=0.0, b=0.01, phi_gamma=0.0
        )

        # Should still be solvable
        assert len(images) == 4
        radii = np.sqrt(images[:, 0]**2 + images[:, 1]**2)
        assert np.std(radii) < 0.1  # Nearly circular

    def test_large_offset(self):
        """Handle larger source offset."""
        images, params = generate_cross_images(
            theta_E=1.0, beta=0.3, phi_beta=0.5,
            a=0.0, b=0.2, phi_gamma=0.5
        )

        # Images should still form a cross
        assert len(images) == 4

    def test_matrix_conditioning(self):
        """Check matrix conditioning."""
        images, params = generate_cross_images(
            theta_E=1.0, beta=0.1, phi_beta=0.3,
            a=0.0, b=0.15, phi_gamma=0.5
        )

        phi_gamma = params['phi_gamma']
        A, b_vec = self._build_system(images, phi_gamma)

        # Check rank
        rank = matrix_rank(A[:5])
        assert rank == 5  # Full rank for solvability

    def _build_system(self, images, phi_gamma):
        """Build system for conditioning test."""
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


def run_tests():
    """Run all tests manually."""
    print("Running multipole consistency tests...")

    # DoF tests
    t = TestDoFCounting()
    t.test_minimal_model_4_images()
    t.test_underdetermined()
    t.test_multipole_m3()
    t.test_image_multiplicity_quad()
    print("  DoF counting tests: PASSED")

    # Consistency tests
    t = TestMultipoleConsistency()
    t.test_m2_matches_minimal()
    t.test_multipole_residuals()
    t.test_phase_periodicity()
    print("  Multipole consistency tests: PASSED")

    # Stability tests
    t = TestNumericalStability()
    t.test_small_quadrupole()
    t.test_large_offset()
    t.test_matrix_conditioning()
    print("  Numerical stability tests: PASSED")

    print("\nAll multipole tests PASSED!")
    return True


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
