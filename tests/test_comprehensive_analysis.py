#!/usr/bin/env python3
"""
Comprehensive Analysis and Comparison Suite
Tests all three paths (A, B, C) under various conditions and generates analysis.

Authors: Carmen N. Wrede, Lino P. Casu
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Dict, List, Optional

from models.dual_path_inversion import AlgebraicSolver, PhaseScanSolver
from models.regime_classifier import RegimeClassifier, Regime, UnderdeterminedExplorer
from dataio.datasets import generate_cross_images

np.random.seed(42)


@dataclass
class PathResult:
    """Result from one inversion path."""
    path: str
    regime: str
    residual: float
    params: Dict[str, float]
    success: bool
    notes: str


@dataclass
class ComparisonResult:
    """Comparison across all paths for one scenario."""
    scenario: str
    n_constraints: int
    n_params: int
    path_a: Optional[PathResult]
    path_b: Optional[PathResult]
    path_c: Optional[PathResult]
    ground_truth: Dict[str, float]
    winner: str
    analysis: str


def run_path_a(images_list, m_max=2):
    """Run Path A (Algebraic)."""
    try:
        solver = AlgebraicSolver(m_max=m_max)
        result = solver.solve(images_list)
        return PathResult(
            path='A',
            regime='algebraic',
            residual=result.max_residual,
            params=result.params,
            success=result.max_residual < 1.0,
            notes=f"DOF: {result.dof_status}, {result.n_constraints}C vs {result.n_params}P"
        )
    except Exception as e:
        return PathResult('A', 'error', float('inf'), {}, False, str(e))


def run_path_b(images_list, m_max=2, n_phases=36):
    """Run Path B (Phase Scan)."""
    try:
        solver = PhaseScanSolver(m_max=m_max)
        result = solver.scan_phases_then_solve_linear(
            images_list, phi_2_range=(0, np.pi, n_phases)
        )
        if result.best_candidate:
            hints = result.degeneracy_hints if result.degeneracy_hints else []
            return PathResult(
                path='B',
                regime='phase_scan',
                residual=result.best_candidate.residual,
                params=result.best_candidate.linear_params,
                success=True,
                notes=f"Candidates: {len(result.all_candidates)}, hints: {len(hints)}"
            )
        return PathResult('B', 'no_candidate', float('inf'), {}, False, 'No valid')
    except Exception as e:
        return PathResult('B', 'error', float('inf'), {}, False, str(e))


def run_path_c(A, b, param_names):
    """Run Path C (Underdetermined Explorer)."""
    try:
        analysis = RegimeClassifier.classify(A, param_names)
        if analysis.regime == Regime.UNDERDETERMINED:
            explorer = UnderdeterminedExplorer(param_names)
            result = explorer.explore(A, b, analysis)
            return PathResult(
                path='C',
                regime=analysis.regime.value,
                residual=result.solutions[0].residual if result.solutions else float('inf'),
                params=result.particular_solution,
                success=True,
                notes=f"Nullspace: {analysis.nullspace_dim}D, "
                      f"Solutions: {len(result.solutions)}"
            )
        return PathResult(
            path='C',
            regime=analysis.regime.value,
            residual=0,
            params={},
            success=True,
            notes=f"Not underdetermined: {analysis.regime.value}"
        )
    except Exception as e:
        return PathResult('C', 'error', float('inf'), {}, False, str(e))


class TestScenarioSuite:
    """Test various scenarios comparing all paths."""

    def test_scenario_determined_standard(self):
        """Scenario 1: Standard determined system (4 images, m_max=2)."""
        imgs, true_params = generate_cross_images(theta_E=1.0, beta=0.1, b=0.15)
        
        result_a = run_path_a([imgs], m_max=2)
        result_b = run_path_b([imgs], m_max=2)
        
        # Both should succeed with similar residuals
        assert result_a.success, f"Path A failed: {result_a.notes}"
        assert result_b.success, f"Path B failed: {result_b.notes}"
        assert result_a.residual < 0.1, f"Path A residual too high: {result_a.residual}"
        
        print(f"\n[Scenario 1: Determined Standard]")
        print(f"  Path A residual: {result_a.residual:.6f}")
        print(f"  Path B residual: {result_b.residual:.6f}")
        print(f"  theta_E recovered: A={result_a.params.get('theta_E', 'N/A'):.4f}")

    def test_scenario_overdetermined(self):
        """Scenario 2: Overdetermined (8 images from 2 sources, m_max=2)."""
        imgs1, _ = generate_cross_images(theta_E=1.0, beta=0.1, b=0.15)
        imgs2, _ = generate_cross_images(theta_E=1.0, beta=0.15, b=0.15)
        
        result_a = run_path_a([imgs1, imgs2], m_max=2)
        result_b = run_path_b([imgs1, imgs2], m_max=2)
        
        assert result_a.success
        # Overdetermined: residual tells us about model adequacy
        print(f"\n[Scenario 2: Overdetermined]")
        print(f"  Path A residual: {result_a.residual:.6f} (model check)")
        print(f"  Path B residual: {result_b.residual:.6f}")
        print(f"  DOF info: {result_a.notes}")

    def test_scenario_underdetermined_high_mmax(self):
        """Scenario 3: Underdetermined (4 images, m_max=4)."""
        imgs, _ = generate_cross_images(theta_E=1.0, beta=0.1, b=0.15)
        
        # Build the system matrix to analyze
        n_images = 4
        n_sources = 1
        m_max = 4
        n_multipole = 2 * m_max - 1  # a_2..a_4, b_2..b_4 = 7 params
        n_params = 1 + n_multipole + 2 + 2 * n_sources  # theta_E + multipoles + shear + betas
        n_constraints = 2 * n_images  # x,y per image
        
        # This is underdetermined: 8 constraints, 12 params
        A = np.random.randn(n_constraints, n_params)
        b = np.random.randn(n_constraints)
        params = ['theta_E'] + [f'a_{m}' for m in range(2, m_max+1)] + \
                 [f'b_{m}' for m in range(2, m_max+1)] + ['gamma_1', 'gamma_2', 'beta_x', 'beta_y']
        
        analysis = RegimeClassifier.classify(A, params)
        assert analysis.regime == Regime.UNDERDETERMINED
        
        result_c = run_path_c(A, b, params)
        
        print(f"\n[Scenario 3: Underdetermined (high m_max)]")
        print(f"  Constraints: {n_constraints}, Params: {n_params}")
        print(f"  Regime: {analysis.regime.value}")
        print(f"  Nullspace: {analysis.nullspace_dim} dimensions")
        print(f"  Path C notes: {result_c.notes}")
        print(f"  Non-identifiable: {analysis.non_identifiable_params[:5]}...")

    def test_scenario_rescue_with_source(self):
        """Scenario 4: DOF rescue - adding source reduces nullspace."""
        m_max = 3
        n_multipole = 2 * m_max - 1  # 5 params
        
        # 1 source: 8 constraints, 10 params -> underdetermined
        n_params_1 = 1 + n_multipole + 2 + 2  # 10
        A1 = np.random.randn(8, n_params_1)
        params1 = ['theta_E', 'a_2', 'b_2', 'a_3', 'b_3', 'g1', 'g2', 'bx1', 'by1', 'extra']
        analysis1 = RegimeClassifier.classify(A1, params1)
        
        # 2 sources: 16 constraints, 12 params -> overdetermined
        n_params_2 = 1 + n_multipole + 2 + 4  # 12
        A2 = np.random.randn(16, n_params_2)
        params2 = params1[:8] + ['bx2', 'by2', 'extra2', 'extra3']
        analysis2 = RegimeClassifier.classify(A2, params2)
        
        print(f"\n[Scenario 4: DOF Rescue]")
        print(f"  1 source: {analysis1.n_constraints}C vs {analysis1.n_params}P -> {analysis1.regime.value}")
        print(f"  2 sources: {analysis2.n_constraints}C vs {analysis2.n_params}P -> {analysis2.regime.value}")
        print(f"  Nullspace: {analysis1.nullspace_dim} -> {analysis2.nullspace_dim}")
        
        assert analysis1.nullspace_dim > analysis2.nullspace_dim

    def test_scenario_ill_conditioned(self):
        """Scenario 5: Ill-conditioned system."""
        # Create near-singular matrix
        A = np.eye(8)
        A[7, 7] = 1e-12
        b = np.ones(8)
        params = [f'p{i}' for i in range(8)]
        
        analysis = RegimeClassifier.classify(A, params)
        
        print(f"\n[Scenario 5: Ill-Conditioned]")
        print(f"  Condition number: {analysis.condition_number:.2e}")
        print(f"  Regime: {analysis.regime.value}")
        print(f"  Recommendation: {analysis.recommendations[0] if analysis.recommendations else 'None'}")
        
        assert analysis.regime == Regime.ILL_CONDITIONED

    def test_scenario_phase_degeneracy(self):
        """Scenario 6: Phase degeneracy detection."""
        imgs, _ = generate_cross_images(theta_E=1.0, beta=0.1, b=0.15)
        
        solver = PhaseScanSolver(m_max=2)
        result = solver.scan_phases_then_solve_linear(
            [imgs], phi_2_range=(0, np.pi, 72)
        )
        
        # Check for degeneracy in residual landscape
        residuals = [c.residual for c in result.all_candidates]
        min_res = min(residuals)
        near_optimal = sum(1 for r in residuals if r < min_res * 1.2)
        
        print(f"\n[Scenario 6: Phase Degeneracy]")
        print(f"  Phase points scanned: {len(residuals)}")
        print(f"  Min residual: {min_res:.6f}")
        print(f"  Points within 20%: {near_optimal}")
        print(f"  Degeneracy hints: {len(result.degeneracy_hints)}")


class TestPathConsistency:
    """Test that paths give consistent results when applicable."""

    def test_path_a_b_consistency(self):
        """Path A and B should agree on theta_E for determined systems."""
        imgs, true_params = generate_cross_images(theta_E=1.0, beta=0.1, b=0.15)
        
        result_a = run_path_a([imgs])
        result_b = run_path_b([imgs])
        
        theta_a = result_a.params.get('theta_E', 0)
        theta_b = result_b.params.get('theta_E', 0)
        
        print(f"\n[Consistency: Path A vs B]")
        print(f"  theta_E (A): {theta_a:.6f}")
        print(f"  theta_E (B): {theta_b:.6f}")
        print(f"  Difference: {abs(theta_a - theta_b):.6f}")
        
        # Should be very close
        assert abs(theta_a - theta_b) < 0.01 or abs(theta_a - theta_b) < 0.1 * max(abs(theta_a), abs(theta_b))

    def test_regime_matches_dof(self):
        """Regime classification should match DOF count."""
        scenarios = [
            (8, 8, Regime.DETERMINED),
            (12, 8, Regime.OVERDETERMINED),
            (6, 10, Regime.UNDERDETERMINED),
        ]
        
        print(f"\n[Consistency: Regime vs DOF]")
        for n_c, n_p, expected in scenarios:
            A = np.random.randn(n_c, n_p) + np.eye(n_c, n_p) * 0.5
            params = [f'p{i}' for i in range(n_p)]
            analysis = RegimeClassifier.classify(A, params)
            
            print(f"  {n_c}C vs {n_p}P: {analysis.regime.value} (expected: {expected.value})")
            assert analysis.regime == expected


def run_full_analysis():
    """Run full analysis and generate report."""
    print("=" * 60)
    print("RSG LENSING INVERSION: COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    test_suite = TestScenarioSuite()
    test_suite.test_scenario_determined_standard()
    test_suite.test_scenario_overdetermined()
    test_suite.test_scenario_underdetermined_high_mmax()
    test_suite.test_scenario_rescue_with_source()
    test_suite.test_scenario_ill_conditioned()
    test_suite.test_scenario_phase_degeneracy()
    
    print("\n" + "=" * 60)
    print("CONSISTENCY CHECKS")
    print("=" * 60)
    
    consistency = TestPathConsistency()
    consistency.test_path_a_b_consistency()
    consistency.test_regime_matches_dof()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    PATH A (Algebraic): Best for determined systems, fast, deterministic
    PATH B (Phase Scan): Detects degeneracies, residual landscape visible
    PATH C (Explorer): Handles underdetermined, shows solution space
    
    KEY INSIGHT: The regime determines which path is most informative.
    - DETERMINED: A or B (same result)
    - OVERDETERMINED: A (residual = model adequacy)
    - UNDERDETERMINED: C (nullspace exploration)
    - ILL-CONDITIONED: All paths with uncertainty bounds
    """)


if __name__ == '__main__':
    run_full_analysis()
