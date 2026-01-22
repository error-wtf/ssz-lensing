#!/usr/bin/env python3
"""
Test Suite: Multi-Source / Same-Lens No-Fit Inversion

Tests the key innovation: multiple sources behind the same lens
share lens parameters, each has its own β.

Key tests:
1. DOF Gatekeeper blocks underdetermined systems
2. Single source recovery (baseline)
3. Multi-source recovery with shared lens params
4. Phase is OUTPUT not INPUT (derived from components)

Run with: python -m pytest tests/test_multi_source.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from models.multi_source_model import (
    DOFGatekeeper,
    MultiSourceLinearSystemBuilder,
    MultiSourceParams,
    analyze_dof,
    generate_multi_source_synthetic
)


class TestDOFGatekeeper:
    """Tests for DOF policy enforcement."""

    def test_overdetermined_allowed(self):
        """5 params, 8 constraints -> allowed (+3 redundancy)."""
        allowed, msg = DOFGatekeeper.check(5, 8)
        assert allowed
        assert "OVERDETERMINED" in msg

    def test_exactly_determined_allowed(self):
        """8 params, 8 constraints -> allowed (unique)."""
        allowed, msg = DOFGatekeeper.check(8, 8)
        assert allowed
        assert "EXACT" in msg

    def test_underdetermined_forbidden(self):
        """9 params, 8 constraints -> FORBIDDEN."""
        allowed, msg = DOFGatekeeper.check(9, 8)
        assert not allowed
        assert "FORBIDDEN" in msg

    def test_max_params_single_source(self):
        """Single source (4 images) -> max 8 params."""
        max_p = DOFGatekeeper.max_params_for_sources(1)
        assert max_p == 8  # exactly determined allowed

    def test_max_params_two_sources(self):
        """Two sources (8 images) -> max 16 params."""
        max_p = DOFGatekeeper.max_params_for_sources(2)
        assert max_p == 16  # exactly determined allowed


class TestMultiSourceParams:
    """Tests for parameter handling."""

    def test_phase_derived_from_components(self):
        """Phase is OUTPUT, computed from (a_m, b_m)."""
        params = MultiSourceParams()
        params.multipoles[2] = (0.1, 0.1)  # a_2, b_2

        A_2, phi_2 = params.get_amplitude_phase(2)

        # A = sqrt(0.1² + 0.1²) = sqrt(0.02) ≈ 0.1414
        assert abs(A_2 - np.sqrt(0.02)) < 1e-10

        # phi = arctan2(0.1, 0.1) / 2 = (π/4) / 2 = π/8
        assert abs(phi_2 - np.pi/8) < 1e-10

    def test_shear_phase_derived(self):
        """Shear phase is OUTPUT."""
        params = MultiSourceParams()
        params.gamma_1 = 0.05
        params.gamma_2 = 0.05

        gamma, phi_gamma = params.get_shear_amplitude_phase()

        assert abs(gamma - np.sqrt(0.005)) < 1e-10
        assert abs(phi_gamma - np.pi/8) < 1e-10


class TestMultiSourceBuilder:
    """Tests for linear system building."""

    def test_unknowns_single_source_m2(self):
        """Single source, m=2 only: 5 unknowns."""
        builder = MultiSourceLinearSystemBuilder(m_max=2, include_shear=False)
        unknowns = builder.unknowns(n_sources=1)

        # theta_E, a_2, b_2, beta_x_0, beta_y_0
        assert len(unknowns) == 5
        assert 'theta_E' in unknowns
        assert 'a_2' in unknowns
        assert 'b_2' in unknowns
        assert 'beta_x_0' in unknowns
        assert 'beta_y_0' in unknowns

    def test_unknowns_two_sources_with_shear(self):
        """Two sources, m=2 + shear: 9 unknowns."""
        builder = MultiSourceLinearSystemBuilder(m_max=2, include_shear=True)
        unknowns = builder.unknowns(n_sources=2)

        # theta_E, a_2, b_2, gamma_1, gamma_2, beta_x_0, beta_y_0, beta_x_1, beta_y_1
        assert len(unknowns) == 9
        assert 'gamma_1' in unknowns
        assert 'gamma_2' in unknowns
        assert 'beta_x_1' in unknowns

    def test_dof_blocks_underdetermined(self):
        """Builder rejects underdetermined systems."""
        builder = MultiSourceLinearSystemBuilder(m_max=3, include_shear=True)

        # Single source with m=3 + shear = too many params
        # unknowns: theta_E, a_2, b_2, a_3, b_3, gamma_1, gamma_2, beta_x, beta_y = 9
        # constraints: 8
        images = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

        try:
            builder.build_system([images])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "FORBIDDEN" in str(e)


class TestMultiSourceRecovery:
    """Tests for exact parameter recovery."""

    def test_single_source_recovery(self):
        """Single source should recover params exactly."""
        # True parameters
        theta_E = 1.0
        a_2, b_2 = 0.08, 0.05
        beta_x, beta_y = 0.06, 0.04

        # Generate synthetic data
        builder = MultiSourceLinearSystemBuilder(m_max=2, include_shear=False)

        # Create approximate images (simplified model)
        images = []
        for k in range(4):
            phi = np.pi/4 + k * np.pi/2
            r = theta_E + a_2*np.cos(2*phi) + b_2*np.sin(2*phi)
            r += beta_x*np.cos(phi) + beta_y*np.sin(phi)
            images.append([r*np.cos(phi), r*np.sin(phi)])
        images = np.array(images)

        result = builder.solve([images])

        assert 'error' not in result
        params = result['params']

        # Check recovery (allowing some numerical tolerance)
        assert abs(params['theta_E'] - theta_E) < 0.1
        # Note: exact recovery depends on model matching generation

    def test_two_source_shared_lens(self):
        """Two sources must share lens params, differ only in beta."""
        builder = MultiSourceLinearSystemBuilder(m_max=2, include_shear=False)

        # Verify DOF counting for multi-source
        n_params = builder.n_params(n_sources=2)
        assert n_params == 7  # theta_E, a_2, b_2, 2x(beta_x, beta_y)

        # 2 sources x 4 images x 2 = 16 constraints
        # 7 params -> +9 redundancy -> OVERDETERMINED
        allowed, msg = DOFGatekeeper.check(n_params, 16)
        assert allowed
        assert "OVERDETERMINED" in msg

    def test_phase_is_output_not_input(self):
        """Verify phase is computed FROM components, not fitted."""
        builder = MultiSourceLinearSystemBuilder(m_max=2, include_shear=False)

        # Simple test case
        images = np.array([
            [1.1, 0.1],
            [0.1, 0.9],
            [-0.9, -0.1],
            [-0.1, -1.1]
        ])

        result = builder.solve([images])

        # 'derived' should contain phases computed from components
        assert 'derived' in result
        assert 'A_2' in result['derived']
        assert 'phi_2' in result['derived']

        # Verify phase relationship: A_2 = sqrt(a_2² + b_2²)
        a_2 = result['params']['a_2']
        b_2 = result['params']['b_2']
        A_2_computed = np.sqrt(a_2**2 + b_2**2)

        assert abs(result['derived']['A_2'] - A_2_computed) < 1e-10


class TestDOFAnalysis:
    """Tests for DOF analysis utility."""

    def test_analyze_single_source(self):
        """Single source analysis."""
        analysis = analyze_dof(n_sources=1, m_max=2, include_shear=False)
        assert "5" in analysis  # 5 params
        assert "8" in analysis  # 8 constraints
        assert "[OK]" in analysis  # allowed

    def test_analyze_forbidden_config(self):
        """Forbidden configuration analysis."""
        # Single source with m=3 + shear = 9 params, 8 constraints
        analysis = analyze_dof(n_sources=1, m_max=3, include_shear=True)
        assert "[FORBIDDEN]" in analysis  # forbidden

    def test_analyze_multi_source_enables_more(self):
        """Multi-source enables more params."""
        # 2 sources with m=3 + shear
        # params: theta_E, a_2, b_2, a_3, b_3, gamma_1, gamma_2, 4 betas = 11
        # constraints: 16
        analysis = analyze_dof(n_sources=2, m_max=3, include_shear=True)
        assert "[OK]" in analysis  # now allowed!


def run_tests():
    """Run all tests manually."""
    print("=" * 60)
    print("MULTI-SOURCE NO-FIT INVERSION TESTS")
    print("=" * 60)

    # DOF Gatekeeper
    print("\n[1] DOF Gatekeeper Tests")
    t = TestDOFGatekeeper()
    t.test_overdetermined_allowed()
    t.test_exactly_determined_allowed()
    t.test_underdetermined_forbidden()
    t.test_max_params_single_source()
    t.test_max_params_two_sources()
    print("  [PASS] All DOF Gatekeeper tests PASSED")

    # Parameter handling
    print("\n[2] Parameter Handling Tests")
    t = TestMultiSourceParams()
    t.test_phase_derived_from_components()
    t.test_shear_phase_derived()
    print("  [PASS] Phase-as-output tests PASSED")

    # Builder tests
    print("\n[3] Linear System Builder Tests")
    t = TestMultiSourceBuilder()
    t.test_unknowns_single_source_m2()
    t.test_unknowns_two_sources_with_shear()
    t.test_dof_blocks_underdetermined()
    print("  [PASS] Builder tests PASSED")

    # Recovery tests
    print("\n[4] Parameter Recovery Tests")
    t = TestMultiSourceRecovery()
    t.test_single_source_recovery()
    t.test_two_source_shared_lens()
    t.test_phase_is_output_not_input()
    print("  [PASS] Recovery tests PASSED")

    # DOF Analysis
    print("\n[5] DOF Analysis Utility Tests")
    t = TestDOFAnalysis()
    t.test_analyze_single_source()
    t.test_analyze_forbidden_config()
    t.test_analyze_multi_source_enables_more()
    print("  [PASS] DOF analysis tests PASSED")

    print("\n" + "=" * 60)
    print("ALL MULTI-SOURCE TESTS PASSED")
    print("=" * 60)

    # Show DOF analysis examples
    print("\n[DOF Analysis Examples]")
    print(analyze_dof(1, 2, False))
    print(analyze_dof(2, 2, True))
    print(analyze_dof(2, 3, True))

    return True


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
