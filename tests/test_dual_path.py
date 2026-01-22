#!/usr/bin/env python3
"""
Test Suite: Dual-Path Inversion (Algebraic + Phase Scan)

Tests that:
1. Path A (Algebraic) works as canonical reference
2. Path B (Phase Scan) is explicitly labeled as hypothesis test
3. Both paths use SAME forward model
4. Cross-check catches inconsistencies

Run: python tests/test_dual_path.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from models.dual_path_inversion import (
    AlgebraicSolver,
    PhaseScanSolver,
    dual_path_inversion,
    reduced_deflection,
    lens_equation
)


class TestSharedForwardModel:
    """Verify both paths use the same physics."""

    def test_reduced_deflection_basic(self):
        """Test forward model computes deflection correctly."""
        theta = np.array([[1.0, 0.0], [0.0, 1.0]])
        theta_E = 1.0
        multipoles = {2: (0.1, 0.0)}  # Pure cos(2*phi)
        
        alpha = reduced_deflection(theta, theta_E, multipoles)
        
        # At phi=0: deflection should be theta_E + a_2
        assert abs(alpha[0, 0] - 1.1) < 1e-10
        
        # At phi=pi/2: cos(2*pi/2) = cos(pi) = -1, so theta_E - a_2
        assert abs(alpha[1, 1] - 0.9) < 1e-10

    def test_lens_equation_zero_residual(self):
        """Lens equation gives zero for correct solution."""
        theta = np.array([[1.1, 0.0]])
        beta = (0.0, 0.0)  # Source at center
        theta_E = 1.1
        multipoles = {}
        
        residual = lens_equation(theta, beta, theta_E, multipoles)
        
        # For circular ring with beta=0, residual should be zero
        assert np.max(np.abs(residual)) < 1e-10


class TestPathA_Algebraic:
    """Test Path A: Algebraic Components (canonical)."""

    def test_algebraic_solver_basic(self):
        """Basic algebraic solve with known solution."""
        solver = AlgebraicSolver(m_max=2, include_shear=False)
        
        # Create synthetic images
        theta_E = 1.0
        a_2, b_2 = 0.1, 0.05
        beta_x, beta_y = 0.05, 0.03
        
        images = []
        for k in range(4):
            phi = np.pi/4 + k * np.pi/2
            r = theta_E + a_2*np.cos(2*phi) + b_2*np.sin(2*phi)
            r += beta_x*np.cos(phi) + beta_y*np.sin(phi)
            images.append([r*np.cos(phi), r*np.sin(phi)])
        
        result = solver.solve([np.array(images)])
        
        # Should have derived phases
        assert 'phi_2' in result.derived_phases
        assert 'A_2' in result.derived_phases

    def test_phase_is_output_not_input(self):
        """Verify phase is computed FROM components."""
        solver = AlgebraicSolver(m_max=2)
        
        images = np.array([
            [1.1, 0.1], [0.1, 0.9], [-0.9, -0.1], [-0.1, -1.1]
        ])
        
        result = solver.solve([images])
        
        # Phase derived from components
        a_2 = result.params.get('a_2', 0)
        b_2 = result.params.get('b_2', 0)
        expected_A = np.sqrt(a_2**2 + b_2**2)
        
        assert abs(result.derived_phases['A_2'] - expected_A) < 1e-10


class TestPathB_PhaseScan:
    """Test Path B: Phase Scan Mode (hypothesis test)."""

    def test_scan_is_labeled(self):
        """Verify scan mode is explicitly labeled."""
        solver = PhaseScanSolver(m_max=2)
        
        assert "Scan" in solver.MODE_LABEL
        assert "Hypothesis" in solver.MODE_LABEL
        assert "nonlinear" in solver.MODE_LABEL

    def test_scan_finds_candidates(self):
        """Phase scan should find multiple candidates."""
        solver = PhaseScanSolver(m_max=2)
        
        images = np.array([
            [1.1, 0.1], [0.1, 0.9], [-0.9, -0.1], [-0.1, -1.1]
        ])
        
        result = solver.scan_phases_then_solve_linear(
            [images],
            phi_2_range=(0, np.pi, 18)
        )
        
        assert result.best_candidate is not None
        assert len(result.all_candidates) > 0
        assert result.residual_landscape is not None


class TestCrossCheck:
    """Test cross-validation between paths."""

    def test_dual_path_runs_both(self):
        """Dual path should run both A and B."""
        images = np.array([
            [1.1, 0.1], [0.1, 0.9], [-0.9, -0.1], [-0.1, -1.1]
        ])
        
        results = dual_path_inversion(
            [images],
            m_max=2,
            run_scan=True,
            phi_2_steps=18
        )
        
        assert 'algebraic' in results
        assert 'scan' in results
        assert 'comparison' in results

    def test_cross_check_reports_consistency(self):
        """Cross-check should report if paths agree."""
        images = np.array([
            [1.1, 0.1], [0.1, 0.9], [-0.9, -0.1], [-0.1, -1.1]
        ])
        
        results = dual_path_inversion(
            [images],
            m_max=2,
            run_scan=True,
            phi_2_steps=36
        )
        
        cc = results['comparison']
        assert 'phi_2_algebraic' in cc
        assert 'phi_2_scan' in cc
        assert 'consistent' in cc


def run_tests():
    """Run all dual-path tests."""
    print("=" * 60)
    print("DUAL-PATH INVERSION TESTS")
    print("=" * 60)
    
    print("\n[1] Shared Forward Model Tests")
    t = TestSharedForwardModel()
    t.test_reduced_deflection_basic()
    t.test_lens_equation_zero_residual()
    print("  [PASS] Forward model tests passed")
    
    print("\n[2] Path A (Algebraic) Tests")
    t = TestPathA_Algebraic()
    t.test_algebraic_solver_basic()
    t.test_phase_is_output_not_input()
    print("  [PASS] Algebraic solver tests passed")
    
    print("\n[3] Path B (Phase Scan) Tests")
    t = TestPathB_PhaseScan()
    t.test_scan_is_labeled()
    t.test_scan_finds_candidates()
    print("  [PASS] Phase scan tests passed")
    
    print("\n[4] Cross-Check Tests")
    t = TestCrossCheck()
    t.test_dual_path_runs_both()
    t.test_cross_check_reports_consistency()
    print("  [PASS] Cross-check tests passed")
    
    print("\n" + "=" * 60)
    print("ALL DUAL-PATH TESTS PASSED")
    print("=" * 60)
    
    # Demo run
    print("\n[DEMO] Running dual_path_inversion...")
    print("-" * 60)
    
    images = np.array([
        [1.08, 0.12], [0.12, 0.92], [-0.88, -0.08], [-0.08, -1.12]
    ])
    
    results = dual_path_inversion(
        [images],
        m_max=2,
        run_scan=True,
        phi_2_steps=36
    )
    
    return True


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
