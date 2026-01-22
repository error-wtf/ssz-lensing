"""
RegimeExplorer: Analyze underdetermined systems without aborting.

Add-only design: Does NOT abort or delete models.
Instead, explains WHY a model needs more observables.

Authors: Carmen N. Wrede, Lino P. Casu
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class NullspaceAnalysis:
    """Analysis of the nullspace (non-identifiable directions)."""
    dimension: int
    basis_vectors: np.ndarray
    non_identifiable_params: List[str]
    identifiable_params: List[str]


@dataclass
class RegimeAnalysis:
    """Complete regime analysis for a model."""
    regime: str
    rank: int
    n_constraints: int
    n_params: int
    dof: int
    condition_number: float
    nullspace: Optional[NullspaceAnalysis] = None
    suggestions: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [f"Regime: {self.regime}"]
        lines.append(f"Rank: {self.rank}")
        lines.append(f"Constraints: {self.n_constraints}")
        lines.append(f"Parameters: {self.n_params}")
        lines.append(f"DOF: {self.dof}")
        lines.append(f"Condition: {self.condition_number:.2e}")
        
        if self.nullspace and self.nullspace.dimension > 0:
            lines.append(f"\nNullspace dimension: {self.nullspace.dimension}")
            lines.append("Non-identifiable parameters:")
            for p in self.nullspace.non_identifiable_params:
                lines.append(f"  - {p}")
        
        if self.suggestions:
            lines.append("\nSuggestions to resolve:")
            for s in self.suggestions:
                lines.append(f"  -> {s}")
        
        return "\n".join(lines)


class RegimeExplorer:
    """
    Explore and explain regime properties without aborting.
    
    Key principle: FORBIDDEN is not an error, it's information.
    This class explains what would make the system solvable.
    """
    
    CONDITION_THRESHOLD = 1e10
    
    def __init__(self, param_names: List[str]):
        """
        Initialize with parameter names.
        
        Args:
            param_names: Names of all parameters in order
        """
        self.param_names = param_names
    
    def analyze(self, A: np.ndarray) -> RegimeAnalysis:
        """
        Analyze the design matrix.
        
        Args:
            A: Design matrix (n_constraints x n_params)
            
        Returns:
            RegimeAnalysis with full diagnostic information
        """
        n_constraints, n_params = A.shape
        
        U, s, Vt = np.linalg.svd(A, full_matrices=True)
        
        tol = max(n_constraints, n_params) * np.finfo(float).eps * s[0]
        rank = np.sum(s > tol)
        
        if s[-1] > tol:
            condition = s[0] / s[-1]
        else:
            condition = float('inf')
        
        dof = n_constraints - n_params
        
        if dof > 0 and condition < self.CONDITION_THRESHOLD:
            regime = "OVERDETERMINED"
        elif dof == 0 and condition < self.CONDITION_THRESHOLD:
            regime = "DETERMINED"
        elif dof < 0:
            regime = "UNDERDETERMINED"
        else:
            regime = "ILL_CONDITIONED"
        
        nullspace = None
        if rank < n_params:
            nullspace = self._analyze_nullspace(Vt, rank, n_params)
        
        suggestions = self._generate_suggestions(regime, dof, nullspace)
        
        return RegimeAnalysis(
            regime=regime,
            rank=rank,
            n_constraints=n_constraints,
            n_params=n_params,
            dof=dof,
            condition_number=condition,
            nullspace=nullspace,
            suggestions=suggestions
        )
    
    def _analyze_nullspace(
        self, Vt: np.ndarray, rank: int, n_params: int
    ) -> NullspaceAnalysis:
        """Analyze the nullspace to find non-identifiable parameters."""
        null_dim = n_params - rank
        null_basis = Vt[rank:].T
        
        param_in_null = np.any(np.abs(null_basis) > 0.1, axis=1)
        
        non_ident = [self.param_names[i] for i in range(n_params) 
                     if param_in_null[i]]
        ident = [self.param_names[i] for i in range(n_params) 
                 if not param_in_null[i]]
        
        return NullspaceAnalysis(
            dimension=null_dim,
            basis_vectors=null_basis,
            non_identifiable_params=non_ident,
            identifiable_params=ident
        )
    
    def _generate_suggestions(
        self, regime: str, dof: int,
        nullspace: Optional[NullspaceAnalysis]
    ) -> List[str]:
        """Generate suggestions for resolving issues."""
        suggestions = []
        
        if regime == "UNDERDETERMINED":
            missing = abs(dof)
            suggestions.append(f"Need {missing} more constraint(s)")
            suggestions.append("Options:")
            suggestions.append("  1. Add flux ratio measurements")
            suggestions.append("  2. Add time delay measurements")
            suggestions.append("  3. Add arc point positions")
            suggestions.append("  4. Add second background source")
            
            if nullspace:
                suggestions.append("")
                suggestions.append("Non-identifiable combinations:")
                for p in nullspace.non_identifiable_params[:3]:
                    suggestions.append(f"  - {p}")
        
        elif regime == "ILL_CONDITIONED":
            suggestions.append("System is numerically unstable")
            suggestions.append("Possible causes:")
            suggestions.append("  - Near-degenerate image configuration")
            suggestions.append("  - Images close to caustic")
            suggestions.append("Consider regularization or more data")
        
        elif regime == "DETERMINED":
            suggestions.append("Exactly determined (no redundancy)")
            suggestions.append("Consider adding observables for consistency check")
        
        return suggestions
    
    def explain_forbidden(
        self, A: np.ndarray, model_label: str
    ) -> str:
        """
        Generate detailed explanation of why model is FORBIDDEN.
        
        This is the key didactic function: explains the derivation
        step that requires more data.
        """
        analysis = self.analyze(A)
        
        lines = [f"Model: {model_label}"]
        lines.append("=" * 50)
        lines.append("")
        lines.append(analysis.summary())
        lines.append("")
        lines.append("EXPLANATION:")
        lines.append("-" * 50)
        
        if analysis.regime == "UNDERDETERMINED":
            lines.append(
                f"This model has {analysis.n_params} parameters but only "
                f"{analysis.n_constraints} constraints."
            )
            lines.append("")
            lines.append(
                "The derivation chain shows WHY this extension is logical:"
            )
            lines.append("  m=2 -> explains basic quadrupole")
            lines.append("  +shear -> adds external tidal field")
            lines.append("  +m=3 -> adds bar/asymmetry structure")
            lines.append("")
            lines.append(
                "But the combined model (m=2+shear+m=3) needs more data "
                "to be uniquely determined."
            )
            lines.append("")
            lines.append(
                "This is NOT a failure - it's information about what "
                "additional observations would enable this model."
            )
        
        return "\n".join(lines)
