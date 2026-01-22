"""
ModelZooRunner: Wrapper that calls existing solvers in sequence.

Add-only design: Does NOT modify existing solvers.
Calls them and compares results side-by-side.

Authors: Carmen N. Wrede, Lino P. Casu
"""

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .models import ModelFamily, MODEL_CONFIGS, get_derivation_chain


@dataclass
class RunResult:
    """Result from running one model."""
    family: ModelFamily
    success: bool
    params: Optional[Dict[str, float]] = None
    residuals: Optional[np.ndarray] = None
    max_residual: float = float('inf')
    regime: str = "UNKNOWN"
    n_constraints: int = 0
    n_params: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class DerivationReport:
    """Report showing stepwise improvement chain."""
    results: Dict[ModelFamily, RunResult]
    derivation_chain: List[ModelFamily]
    bundle_summary: str
    
    def generate(self) -> str:
        """Generate human-readable derivation report."""
        lines = ["=" * 60]
        lines.append("DERIVATION CHAIN REPORT")
        lines.append("=" * 60)
        lines.append(self.bundle_summary)
        lines.append("")
        
        prev_residual = None
        for i, family in enumerate(self.derivation_chain):
            result = self.results.get(family)
            config = MODEL_CONFIGS[family]
            
            lines.append(f"\n[Step {i+1}] {config.label}")
            lines.append("-" * 40)
            
            if result is None:
                lines.append("  Not run")
                continue
            
            lines.append(f"  Regime: {result.regime}")
            lines.append(f"  Constraints: {result.n_constraints}")
            lines.append(f"  Parameters: {result.n_params}")
            
            if result.success:
                lines.append(f"  Max Residual: {result.max_residual:.4e}")
                if prev_residual is not None and prev_residual > 0:
                    improvement = (prev_residual - result.max_residual)
                    pct = improvement / prev_residual * 100
                    if pct > 0:
                        lines.append(f"  Improvement: {pct:.1f}% vs previous")
                    elif pct < -1:
                        lines.append(f"  Worse: {-pct:.1f}% vs previous")
                prev_residual = result.max_residual
            else:
                lines.append(f"  Status: {result.regime}")
                for note in result.notes:
                    lines.append(f"    -> {note}")
        
        lines.append("\n" + "=" * 60)
        lines.append("SUMMARY")
        lines.append("=" * 60)
        
        successful = [f for f, r in self.results.items() if r.success]
        if successful:
            best = min(successful, key=lambda f: self.results[f].max_residual)
            lines.append(f"Best model: {MODEL_CONFIGS[best].label}")
            lines.append(f"Best residual: {self.results[best].max_residual:.4e}")
        
        forbidden = [f for f, r in self.results.items() 
                     if r.regime == "FORBIDDEN"]
        if forbidden:
            lines.append(f"\nFORBIDDEN models (need more observables):")
            for f in forbidden:
                lines.append(f"  - {MODEL_CONFIGS[f].label}")
                for note in self.results[f].notes:
                    lines.append(f"      {note}")
        
        return "\n".join(lines)


class ModelZooRunner:
    """
    Wrapper that runs existing solvers and compares results.
    
    Does NOT modify existing solvers - just calls them.
    Shows the derivation chain: why each extension helps (or doesn't).
    """
    
    def __init__(self, bundle):
        """
        Initialize with ObservablesBundle.
        
        Args:
            bundle: ObservablesBundle with all observables
        """
        self.bundle = bundle
        self.results: Dict[ModelFamily, RunResult] = {}
    
    def run_derivation_chain(self, include_m4: bool = False) -> DerivationReport:
        """
        Run all models in derivation order.
        
        Args:
            include_m4: If True, include m4 multipole models
        
        Returns report showing stepwise improvement.
        """
        chain = get_derivation_chain(include_m4=include_m4)
        
        for family in chain:
            result = self._run_model(family)
            self.results[family] = result
        
        return DerivationReport(
            results=self.results,
            derivation_chain=chain,
            bundle_summary=self.bundle.summary()
        )
    
    def _run_model(self, family: ModelFamily) -> RunResult:
        """Run one model family."""
        config = MODEL_CONFIGS[family]
        images = self.bundle.get_primary_images()
        
        if len(images) == 0:
            return RunResult(
                family=family,
                success=False,
                regime="NO_DATA",
                notes=["No image positions provided"]
            )
        
        n_constraints = self.bundle.count_constraints()
        n_source_params = 2 * self.bundle.n_sources
        n_params = config.n_lens_params + n_source_params
        dof = n_constraints - n_params
        
        if dof < 0:
            return self._make_forbidden_result(
                family, n_constraints, n_params, dof
            )
        
        try:
            params, residuals = self._solve_linear(family, images)
            max_res = np.max(np.abs(residuals))
            
            regime = "OVERDETERMINED" if dof > 0 else "DETERMINED"
            
            return RunResult(
                family=family,
                success=True,
                params=params,
                residuals=residuals,
                max_residual=max_res,
                regime=regime,
                n_constraints=n_constraints,
                n_params=n_params
            )
        except Exception as e:
            return RunResult(
                family=family,
                success=False,
                regime="ERROR",
                n_constraints=n_constraints,
                n_params=n_params,
                notes=[str(e)]
            )
    
    def _make_forbidden_result(
        self, family: ModelFamily,
        n_constraints: int, n_params: int, dof: int
    ) -> RunResult:
        """Create result for FORBIDDEN (underdetermined) model."""
        missing = abs(dof)
        notes = [
            f"Need {missing} more constraint(s)",
            "Options: flux ratios, time delays, arc points, or multi-source"
        ]
        
        return RunResult(
            family=family,
            success=False,
            regime="FORBIDDEN",
            n_constraints=n_constraints,
            n_params=n_params,
            notes=notes
        )
    
    def _solve_linear(
        self, family: ModelFamily, images: np.ndarray
    ) -> tuple:
        """Solve linear system for this model."""
        config = MODEL_CONFIGS[family]
        n_images = len(images)
        n_constraints = 2 * n_images
        n_params = config.n_lens_params + 2
        
        A = np.zeros((n_constraints, n_params))
        b = np.zeros(n_constraints)
        
        for i, (x, y) in enumerate(images):
            row = 2 * i
            phi = np.arctan2(y, x)
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)
            
            col = 0
            A[row, col] = cos_phi
            A[row + 1, col] = sin_phi
            col += 1
            
            cos_2, sin_2 = np.cos(2*phi), np.sin(2*phi)
            A[row, col] = cos_2 * cos_phi
            A[row, col + 1] = sin_2 * cos_phi
            A[row + 1, col] = cos_2 * sin_phi
            A[row + 1, col + 1] = sin_2 * sin_phi
            col += 2
            
            if config.include_shear:
                A[row, col] = x
                A[row, col + 1] = y
                A[row + 1, col] = -y
                A[row + 1, col + 1] = x
                col += 2
            
            if config.m_max >= 3:
                cos_3, sin_3 = np.cos(3*phi), np.sin(3*phi)
                A[row, col] = cos_3 * cos_phi
                A[row, col + 1] = sin_3 * cos_phi
                A[row + 1, col] = cos_3 * sin_phi
                A[row + 1, col + 1] = sin_3 * sin_phi
                col += 2
            
            A[row, col] = 1.0
            A[row + 1, col + 1] = 1.0
            
            b[row] = x
            b[row + 1] = y
        
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        residuals = A @ x - b
        
        param_names = config.param_names + ['beta_x', 'beta_y']
        params = dict(zip(param_names, x))
        
        return params, residuals
