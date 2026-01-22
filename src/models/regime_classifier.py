"""
Regime Classifier + Underdetermined Explorer

PARADIGM SHIFT: Nothing is forbidden - everything is classified and learned from.

Regimes:
    1. DETERMINED: constraints == params, unique solution
    2. OVERDETERMINED: constraints > params, consistency check via residuals
    3. UNDERDETERMINED: constraints < params, nullspace analysis + multiple solutions
    4. ILL_CONDITIONED: high condition number, sensitivity analysis

For underdetermined systems:
    - Show nullspace dimension and basis vectors
    - Generate multiple equivalent solutions
    - Apply regularizers (minimal-norm, smoothness, etc.) as EXPLICIT hypotheses
    - Explain which parameters are not identifiable

Authors: Carmen N. Wrede, Lino P. Casu
License: Anti-Capitalist Software License v1.4
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class Regime(Enum):
    """System regime classification."""
    DETERMINED = "determined"
    OVERDETERMINED = "overdetermined"
    UNDERDETERMINED = "underdetermined"
    ILL_CONDITIONED = "ill_conditioned"


@dataclass
class RegimeAnalysis:
    """Complete analysis of system regime and structure."""
    regime: Regime
    n_constraints: int
    n_params: int
    rank: int
    nullspace_dim: int
    condition_number: float
    
    # Nullspace analysis (for underdetermined)
    nullspace_basis: Optional[np.ndarray] = None
    non_identifiable_params: List[str] = field(default_factory=list)
    
    # Sensitivity (for ill-conditioned)
    sensitivity_vectors: Optional[np.ndarray] = None
    
    # Human-readable explanation
    explanation: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        lines = [
            f"REGIME: {self.regime.value.upper()}",
            f"  Constraints: {self.n_constraints}",
            f"  Parameters:  {self.n_params}",
            f"  Rank:        {self.rank}",
            f"  Nullspace:   {self.nullspace_dim} dimensions",
            f"  Condition:   {self.condition_number:.2e}",
        ]
        if self.explanation:
            lines.append(f"\n{self.explanation}")
        if self.recommendations:
            lines.append("\nRecommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")
        return "\n".join(lines)


class RegimeClassifier:
    """
    Classify system regime and provide strategy recommendations.
    
    REPLACES DOFGatekeeper - no more FORBIDDEN, only LEARNING.
    """
    
    CONDITION_THRESHOLD = 1e10  # Above this = ill-conditioned
    
    @classmethod
    def classify(
        cls,
        A: np.ndarray,
        param_names: List[str],
        condition_threshold: float = None
    ) -> RegimeAnalysis:
        """
        Analyze system matrix and classify regime.
        
        Parameters
        ----------
        A : ndarray, shape (n_constraints, n_params)
            System matrix
        param_names : list
            Names of parameters for interpretability
        condition_threshold : float
            Threshold for ill-conditioned warning
            
        Returns
        -------
        RegimeAnalysis with full diagnostics
        """
        if condition_threshold is None:
            condition_threshold = cls.CONDITION_THRESHOLD
            
        n_constraints, n_params = A.shape
        
        # SVD for rank and condition
        U, s, Vt = np.linalg.svd(A, full_matrices=True)
        
        # Numerical rank (count significant singular values)
        tol = max(n_constraints, n_params) * np.finfo(float).eps * s[0]
        rank = np.sum(s > tol)
        
        # Condition number
        if s[-1] > tol:
            condition = s[0] / s[-1]
        else:
            condition = float('inf')
        
        # Nullspace dimension
        nullspace_dim = n_params - rank
        
        # Determine regime
        if condition > condition_threshold:
            regime = Regime.ILL_CONDITIONED
        elif n_constraints < n_params or nullspace_dim > 0:
            regime = Regime.UNDERDETERMINED
        elif n_constraints == n_params and nullspace_dim == 0:
            regime = Regime.DETERMINED
        else:
            regime = Regime.OVERDETERMINED
        
        # Build analysis
        analysis = RegimeAnalysis(
            regime=regime,
            n_constraints=n_constraints,
            n_params=n_params,
            rank=rank,
            nullspace_dim=nullspace_dim,
            condition_number=condition
        )
        
        # Regime-specific analysis
        if regime == Regime.UNDERDETERMINED or nullspace_dim > 0:
            cls._analyze_nullspace(analysis, Vt, s, tol, param_names)
        
        if regime == Regime.ILL_CONDITIONED:
            cls._analyze_sensitivity(analysis, U, s, Vt, param_names)
        
        cls._generate_explanation(analysis, param_names)
        cls._generate_recommendations(analysis)
        
        return analysis
    
    @classmethod
    def _analyze_nullspace(
        cls,
        analysis: RegimeAnalysis,
        Vt: np.ndarray,
        s: np.ndarray,
        tol: float,
        param_names: List[str]
    ):
        """Analyze nullspace structure for underdetermined systems."""
        # Nullspace = rows of Vt corresponding to zero singular values
        null_indices = np.where(s < tol)[0]
        if len(null_indices) == 0 and analysis.nullspace_dim > 0:
            # Use smallest singular values
            null_indices = np.argsort(s)[:analysis.nullspace_dim]
        
        if len(null_indices) > 0:
            analysis.nullspace_basis = Vt[null_indices, :]
            
            # Find which parameters are involved in nullspace
            for idx in null_indices:
                null_vec = Vt[idx, :]
                significant = np.where(np.abs(null_vec) > 0.1)[0]
                for param_idx in significant:
                    if param_idx < len(param_names):
                        param = param_names[param_idx]
                        if param not in analysis.non_identifiable_params:
                            analysis.non_identifiable_params.append(param)
    
    @classmethod
    def _analyze_sensitivity(
        cls,
        analysis: RegimeAnalysis,
        U: np.ndarray,
        s: np.ndarray,
        Vt: np.ndarray,
        param_names: List[str]
    ):
        """Analyze parameter sensitivity for ill-conditioned systems."""
        # Directions of high sensitivity = large singular values in V
        # Small changes in data along U directions cause large param changes
        analysis.sensitivity_vectors = Vt[:3, :]  # Top 3 sensitive directions
    
    @classmethod
    def _generate_explanation(cls, analysis: RegimeAnalysis, param_names: List[str]):
        """Generate human-readable explanation."""
        if analysis.regime == Regime.DETERMINED:
            analysis.explanation = (
                "System is exactly determined. Unique solution exists.\n"
                "Linear algebra will yield the exact parameters."
            )
        
        elif analysis.regime == Regime.OVERDETERMINED:
            redundancy = analysis.n_constraints - analysis.n_params
            analysis.explanation = (
                f"System is overdetermined with {redundancy} extra constraints.\n"
                "Residuals will indicate model adequacy (not fitting error).\n"
                "Non-zero residuals suggest model limitations or data noise."
            )
        
        elif analysis.regime == Regime.UNDERDETERMINED:
            analysis.explanation = (
                f"System is underdetermined: {analysis.nullspace_dim} free parameters.\n"
                "INFINITELY MANY solutions exist that fit the data equally well.\n"
            )
            if analysis.non_identifiable_params:
                params_str = ", ".join(analysis.non_identifiable_params)
                analysis.explanation += f"Non-identifiable parameters: {params_str}\n"
            analysis.explanation += (
                "To select ONE solution, an explicit regularizer/hypothesis is needed.\n"
                "This is NOT fitting - it's choosing among equivalent solutions."
            )
        
        elif analysis.regime == Regime.ILL_CONDITIONED:
            analysis.explanation = (
                f"System is ill-conditioned (condition={analysis.condition_number:.2e}).\n"
                "Small data perturbations cause large parameter changes.\n"
                "Results should be interpreted with caution."
            )
    
    @classmethod
    def _generate_recommendations(cls, analysis: RegimeAnalysis):
        """Generate actionable recommendations."""
        if analysis.regime == Regime.DETERMINED:
            analysis.recommendations = [
                "Proceed with exact linear solve",
                "Check residuals for sanity (should be ~machine precision)"
            ]
        
        elif analysis.regime == Regime.OVERDETERMINED:
            analysis.recommendations = [
                "Solve using exact subset or least-norm (not least-squares fit!)",
                "Use residuals as model diagnostic, not fitting metric",
                "Large residuals indicate model needs refinement"
            ]
        
        elif analysis.regime == Regime.UNDERDETERMINED:
            analysis.recommendations = [
                f"Add {analysis.nullspace_dim} more constraints (e.g., second source)",
                "Or: reduce model complexity (lower m_max)",
                "Or: apply explicit regularizer and document the choice",
                "Generate multiple solutions to understand degeneracy"
            ]
        
        elif analysis.regime == Regime.ILL_CONDITIONED:
            analysis.recommendations = [
                "Run sensitivity analysis (perturb data, observe param changes)",
                "Consider if configuration is near-caustic",
                "Results are valid but uncertain - report with error bars"
            ]


@dataclass
class UnderdeterminedSolution:
    """One solution from the underdetermined family."""
    params: Dict[str, float]
    residual: float
    regularizer_used: str
    regularizer_value: float  # e.g., norm of multipoles


@dataclass
class UnderdeterminedExplorerResult:
    """Results from exploring underdetermined solution space."""
    regime_analysis: RegimeAnalysis
    particular_solution: Dict[str, float]
    solutions: List[UnderdeterminedSolution]
    nullspace_exploration: List[Dict[str, float]]  # params at nullspace extremes
    parameter_ranges: Dict[str, Tuple[float, float]]  # min/max for each param


class UnderdeterminedExplorer:
    """
    Explore solution space when constraints < params.
    
    Instead of FORBIDDEN, we:
    1. Find a particular solution (minimal norm)
    2. Explore the nullspace to show degeneracy
    3. Apply different regularizers as explicit hypotheses
    4. Report which parameters are/aren't identifiable
    """
    
    REGULARIZERS = {
        'minimal_norm': 'Minimize ||params||_2 (Occam)',
        'minimal_multipole': 'Minimize sum(a_m^2 + b_m^2) (smooth lens)',
        'minimal_shear': 'Minimize gamma^2 (intrinsic over external)',
        'zero_higher_multipoles': 'Set highest multipoles to zero'
    }
    
    def __init__(self, param_names: List[str]):
        self.param_names = param_names
    
    def explore(
        self,
        A: np.ndarray,
        b: np.ndarray,
        regime_analysis: RegimeAnalysis
    ) -> UnderdeterminedExplorerResult:
        """
        Explore the underdetermined solution space.
        
        Parameters
        ----------
        A : ndarray (n_constraints, n_params)
        b : ndarray (n_constraints,)
        regime_analysis : RegimeAnalysis from classifier
        
        Returns
        -------
        UnderdeterminedExplorerResult with multiple solutions
        """
        n_params = A.shape[1]
        
        # 1. Particular solution: minimal norm (pseudoinverse)
        x_min_norm = np.linalg.lstsq(A, b, rcond=None)[0]
        particular = dict(zip(self.param_names, x_min_norm))
        
        result = UnderdeterminedExplorerResult(
            regime_analysis=regime_analysis,
            particular_solution=particular,
            solutions=[],
            nullspace_exploration=[],
            parameter_ranges={}
        )
        
        # 2. Add minimal-norm as first solution
        res_norm = np.linalg.norm(A @ x_min_norm - b)
        result.solutions.append(UnderdeterminedSolution(
            params=particular.copy(),
            residual=res_norm,
            regularizer_used='minimal_norm',
            regularizer_value=np.linalg.norm(x_min_norm)
        ))
        
        # 3. Explore nullspace if it exists
        if regime_analysis.nullspace_basis is not None:
            null_basis = regime_analysis.nullspace_basis
            
            for i, null_vec in enumerate(null_basis):
                # Move along nullspace direction
                for alpha in [-1.0, 1.0]:
                    x_variant = x_min_norm + alpha * null_vec
                    params_variant = dict(zip(self.param_names, x_variant))
                    result.nullspace_exploration.append(params_variant)
        
        # 4. Compute parameter ranges across all explored solutions
        all_solutions = [particular] + result.nullspace_exploration
        for param in self.param_names:
            values = [sol.get(param, 0) for sol in all_solutions]
            result.parameter_ranges[param] = (min(values), max(values))
        
        # 5. Try other regularizers
        result.solutions.extend(
            self._apply_regularizers(A, b, x_min_norm, regime_analysis)
        )
        
        return result
    
    def _apply_regularizers(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x_min_norm: np.ndarray,
        regime_analysis: RegimeAnalysis
    ) -> List[UnderdeterminedSolution]:
        """Apply different regularizers to generate alternative solutions."""
        solutions = []
        
        # Minimal multipole power
        multipole_indices = [
            i for i, name in enumerate(self.param_names)
            if name.startswith('a_') or name.startswith('b_')
        ]
        
        if multipole_indices and regime_analysis.nullspace_basis is not None:
            # Project minimal-norm solution to minimize multipole power
            # This is a simplified version - full implementation would solve
            # constrained optimization
            x_smooth = x_min_norm.copy()
            for idx in multipole_indices:
                # Reduce multipole if in nullspace direction
                for null_vec in regime_analysis.nullspace_basis:
                    if abs(null_vec[idx]) > 0.1:
                        # Can reduce this parameter along nullspace
                        reduction = 0.5 * x_min_norm[idx]
                        x_smooth = x_smooth - reduction * null_vec / null_vec[idx]
            
            res = np.linalg.norm(A @ x_smooth - b)
            multipole_power = sum(x_smooth[i]**2 for i in multipole_indices)
            
            solutions.append(UnderdeterminedSolution(
                params=dict(zip(self.param_names, x_smooth)),
                residual=res,
                regularizer_used='minimal_multipole',
                regularizer_value=multipole_power
            ))
        
        return solutions
    
    def report(self, result: UnderdeterminedExplorerResult) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "UNDERDETERMINED EXPLORER REPORT",
            "=" * 60,
            "",
            result.regime_analysis.summary(),
            "",
            "--- PARAMETER RANGES (showing degeneracy) ---"
        ]
        
        for param, (pmin, pmax) in result.parameter_ranges.items():
            if abs(pmax - pmin) > 1e-10:
                lines.append(f"  {param}: [{pmin:.4f}, {pmax:.4f}] (NOT UNIQUE)")
            else:
                lines.append(f"  {param}: {pmin:.4f} (identifiable)")
        
        lines.extend([
            "",
            "--- SOLUTIONS WITH DIFFERENT REGULARIZERS ---"
        ])
        
        for sol in result.solutions:
            lines.append(f"\n[{sol.regularizer_used}]")
            lines.append(f"  Residual: {sol.residual:.2e}")
            lines.append(f"  Regularizer value: {sol.regularizer_value:.4f}")
            lines.append(f"  Rationale: {self.REGULARIZERS.get(sol.regularizer_used, 'N/A')}")
        
        lines.extend([
            "",
            "--- KEY INSIGHT ---",
            "All solutions above fit the data EQUALLY WELL.",
            "The choice between them requires PHYSICS input, not fitting.",
            "=" * 60
        ])
        
        return "\n".join(lines)
