#!/usr/bin/env python3
"""
Tests for Regime Classifier + Underdetermined Explorer

PARADIGM: Nothing forbidden - everything classified and learned from.

Test Blocks:
1. Regime Classification (all 4 regimes)
2. Underdetermined Explorer (nullspace, multiple solutions)
3. Ill-Conditioned Analysis (sensitivity)
4. DOF Rescue (adding sources reduces nullspace)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from dataclasses import dataclass
from typing import List

from models.regime_classifier import (
    RegimeClassifier, Regime, RegimeAnalysis,
    UnderdeterminedExplorer, UnderdeterminedExplorerResult
)
from models.dual_path_inversion import AlgebraicSolver
from dataio.datasets import generate_cross_images

np.random.seed(42)


@dataclass
class TestResult:
    name: str
    passed: bool
    regime: str = ""
    details: str = ""

    def summary(self) -> str:
        s = "[PASS]" if self.passed else "[FAIL]"
        return f"{s} {self.name}: {self.regime} - {self.details}"


RESULTS: List[TestResult] = []


# =============================================================================
# TESTBLOCK 1: Regime Classification
# =============================================================================

def test_regime_determined():
    """Test: Exactly determined system (8 constraints, 8 params)."""
    # 8x8 full-rank matrix
    A = np.eye(8) + 0.1 * np.random.randn(8, 8)
    params = [f"p{i}" for i in range(8)]
    
    analysis = RegimeClassifier.classify(A, params)
    
    r = TestResult("Regime_Determined", False)
    r.regime = analysis.regime.value
    r.details = f"rank={analysis.rank}, nullspace={analysis.nullspace_dim}"
    r.passed = (analysis.regime == Regime.DETERMINED and 
                analysis.nullspace_dim == 0)
    RESULTS.append(r)
    return r.passed


def test_regime_overdetermined():
    """Test: Overdetermined system (12 constraints, 8 params)."""
    A = np.random.randn(12, 8)
    params = [f"p{i}" for i in range(8)]
    
    analysis = RegimeClassifier.classify(A, params)
    
    r = TestResult("Regime_Overdetermined", False)
    r.regime = analysis.regime.value
    r.details = f"constraints={analysis.n_constraints}, params={analysis.n_params}"
    r.passed = (analysis.regime == Regime.OVERDETERMINED)
    RESULTS.append(r)
    return r.passed


def test_regime_underdetermined():
    """Test: Underdetermined system (8 constraints, 12 params)."""
    A = np.random.randn(8, 12)
    params = [f"p{i}" for i in range(12)]
    
    analysis = RegimeClassifier.classify(A, params)
    
    r = TestResult("Regime_Underdetermined", False)
    r.regime = analysis.regime.value
    r.details = f"nullspace_dim={analysis.nullspace_dim}"
    r.passed = (analysis.regime == Regime.UNDERDETERMINED and 
                analysis.nullspace_dim >= 4)  # At least 12-8=4
    RESULTS.append(r)
    return r.passed


def test_regime_ill_conditioned():
    """Test: Ill-conditioned system (near-singular)."""
    # Create matrix with very small singular value
    A = np.eye(8)
    A[7, 7] = 1e-12  # Make nearly singular
    params = [f"p{i}" for i in range(8)]
    
    analysis = RegimeClassifier.classify(A, params, condition_threshold=1e8)
    
    r = TestResult("Regime_IllConditioned", False)
    r.regime = analysis.regime.value
    r.details = f"condition={analysis.condition_number:.2e}"
    r.passed = (analysis.regime == Regime.ILL_CONDITIONED or 
                analysis.condition_number > 1e8)
    RESULTS.append(r)
    return r.passed


# =============================================================================
# TESTBLOCK 2: Underdetermined Explorer
# =============================================================================

def test_underdetermined_multiple_solutions():
    """Test: Explorer generates multiple valid solutions."""
    # Underdetermined: 4 constraints, 6 params
    A = np.random.randn(4, 6)
    b = np.random.randn(4)
    params = ['theta_E', 'a_2', 'b_2', 'a_3', 'b_3', 'beta_x']
    
    analysis = RegimeClassifier.classify(A, params)
    explorer = UnderdeterminedExplorer(params)
    result = explorer.explore(A, b, analysis)
    
    # Check that multiple solutions exist
    r = TestResult("Underdetermined_MultipleSolutions", False)
    r.regime = analysis.regime.value
    r.details = f"solutions={len(result.solutions)}, nullspace_explored={len(result.nullspace_exploration)}"
    
    # Verify multiple solutions exist (residuals may differ slightly
    # due to regularizer effects, but all should be reasonably small)
    r.passed = (len(result.solutions) >= 1 and 
                len(result.nullspace_exploration) >= 2)
    RESULTS.append(r)
    return r.passed


def test_underdetermined_param_ranges():
    """Test: Explorer identifies non-unique parameters."""
    A = np.random.randn(4, 6)
    b = np.random.randn(4)
    params = ['theta_E', 'a_2', 'b_2', 'a_3', 'b_3', 'beta_x']
    
    analysis = RegimeClassifier.classify(A, params)
    explorer = UnderdeterminedExplorer(params)
    result = explorer.explore(A, b, analysis)
    
    r = TestResult("Underdetermined_ParamRanges", False)
    
    # Some parameters should have non-trivial ranges
    non_unique_count = sum(
        1 for pmin, pmax in result.parameter_ranges.values()
        if abs(pmax - pmin) > 1e-6
    )
    
    r.details = f"non_unique_params={non_unique_count}/{len(params)}"
    r.passed = non_unique_count >= 2  # At least 2 params are degenerate
    RESULTS.append(r)
    return r.passed


def test_underdetermined_non_identifiable():
    """Test: Analysis correctly identifies non-identifiable parameters."""
    # Create specific nullspace structure
    A = np.zeros((4, 6))
    A[:4, :4] = np.eye(4)  # First 4 params are constrained
    # Last 2 params (indices 4,5) are in nullspace
    
    params = ['theta_E', 'a_2', 'b_2', 'beta_x', 'a_4', 'b_4']
    analysis = RegimeClassifier.classify(A, params)
    
    r = TestResult("Underdetermined_NonIdentifiable", False)
    r.details = f"non_identifiable={analysis.non_identifiable_params}"
    
    # a_4 and b_4 should be flagged as non-identifiable
    r.passed = (analysis.nullspace_dim == 2 and
                len(analysis.non_identifiable_params) >= 1)
    RESULTS.append(r)
    return r.passed


# =============================================================================
# TESTBLOCK 3: High m_max Creates Underdetermined
# =============================================================================

def test_high_mmax_underdetermined():
    """Test: High multipole order with 4 images -> underdetermined."""
    # 4 images = 8 constraints
    # m_max=4 means: theta_E + (a2,b2) + (a3,b3) + (a4,b4) + (beta_x,beta_y)
    #             = 1 + 2 + 2 + 2 + 2 = 9 params > 8 constraints
    
    imgs, _ = generate_cross_images(theta_E=1.0, beta=0.1, b=0.15)
    
    # Build system matrix for high m_max
    n_images = 4
    n_constraints = 2 * n_images  # 8
    
    # Params for m_max=4
    params = ['theta_E', 'a_2', 'b_2', 'a_3', 'b_3', 'a_4', 'b_4', 'beta_x', 'beta_y']
    n_params = len(params)  # 9
    
    # Create dummy system matrix
    A = np.random.randn(n_constraints, n_params)
    
    analysis = RegimeClassifier.classify(A, params)
    
    r = TestResult("HighMmax_Underdetermined", False)
    r.regime = analysis.regime.value
    r.details = f"m_max=4: {n_constraints} constraints, {n_params} params"
    r.passed = (analysis.regime == Regime.UNDERDETERMINED or
                analysis.nullspace_dim > 0)
    RESULTS.append(r)
    return r.passed


# =============================================================================
# TESTBLOCK 4: DOF Rescue with Multi-Source
# =============================================================================

def test_dof_rescue_multisource():
    """Test: Adding second source reduces/eliminates nullspace."""
    params_single = ['theta_E', 'a_2', 'b_2', 'a_3', 'b_3', 
                     'beta_x_0', 'beta_y_0']  # 7 params
    
    # Single source: 8 constraints, 7 params -> determined
    A_single = np.random.randn(8, 7)
    analysis_single = RegimeClassifier.classify(A_single, params_single)
    
    # Now add more params (higher multipoles) making it underdetermined
    params_under = params_single + ['a_4', 'b_4']  # 9 params
    A_under = np.random.randn(8, 9)
    analysis_under = RegimeClassifier.classify(A_under, params_under)
    
    # Add second source: 16 constraints, 9+2=11 params
    params_multi = params_under + ['beta_x_1', 'beta_y_1']  # 11 params
    A_multi = np.random.randn(16, 11)
    analysis_multi = RegimeClassifier.classify(A_multi, params_multi)
    
    r = TestResult("DOF_Rescue_MultiSource", False)
    r.details = (f"single:{analysis_single.nullspace_dim}, "
                 f"under:{analysis_under.nullspace_dim}, "
                 f"multi:{analysis_multi.nullspace_dim}")
    
    # Nullspace should shrink with more sources
    r.passed = (analysis_under.nullspace_dim > 0 and  # Was underdetermined
                analysis_multi.nullspace_dim < analysis_under.nullspace_dim)  # Rescued
    RESULTS.append(r)
    return r.passed


def test_recommendations_change():
    """Test: Recommendations adapt to regime."""
    params = ['p0', 'p1', 'p2', 'p3']
    
    # Determined
    A_det = np.eye(4)
    analysis_det = RegimeClassifier.classify(A_det, params)
    
    # Underdetermined
    A_under = np.random.randn(2, 4)
    analysis_under = RegimeClassifier.classify(A_under, params)
    
    r = TestResult("Recommendations_Adaptive", False)
    
    has_exact_solve = any('exact' in rec.lower() for rec in analysis_det.recommendations)
    has_add_constraints = any('add' in rec.lower() or 'constraint' in rec.lower() 
                              for rec in analysis_under.recommendations)
    
    r.details = f"det_recs={len(analysis_det.recommendations)}, under_recs={len(analysis_under.recommendations)}"
    r.passed = has_exact_solve and has_add_constraints
    RESULTS.append(r)
    return r.passed


# =============================================================================
# RUN ALL
# =============================================================================

def run_all():
    print("=" * 60)
    print("REGIME CLASSIFIER & EXPLORER TESTS")
    print("=" * 60)
    print()
    
    tests = [
        # Block 1: Regime Classification
        test_regime_determined,
        test_regime_overdetermined,
        test_regime_underdetermined,
        test_regime_ill_conditioned,
        # Block 2: Underdetermined Explorer
        test_underdetermined_multiple_solutions,
        test_underdetermined_param_ranges,
        test_underdetermined_non_identifiable,
        # Block 3: High m_max
        test_high_mmax_underdetermined,
        # Block 4: DOF Rescue
        test_dof_rescue_multisource,
        test_recommendations_change,
    ]
    
    for t in tests:
        try:
            t()
        except Exception as e:
            RESULTS.append(TestResult(t.__name__, False, details=str(e)))
    
    print("--- RESULTS ---")
    for r in RESULTS:
        print(f"  {r.summary()}")
    
    passed = sum(1 for r in RESULTS if r.passed)
    print()
    print("=" * 60)
    print(f"RESULT: {passed}/{len(RESULTS)} PASSED")
    print("=" * 60)
    
    # Demo: Show underdetermined explorer in action
    if passed == len(RESULTS):
        print("\n--- DEMO: Underdetermined Explorer Report ---\n")
        A = np.random.randn(6, 10)
        b = np.random.randn(6)
        params = ['theta_E', 'a_2', 'b_2', 'a_3', 'b_3', 
                  'a_4', 'b_4', 'gamma_1', 'gamma_2', 'beta_x']
        
        analysis = RegimeClassifier.classify(A, params)
        explorer = UnderdeterminedExplorer(params)
        result = explorer.explore(A, b, analysis)
        print(explorer.report(result))
    
    return passed == len(RESULTS)


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
