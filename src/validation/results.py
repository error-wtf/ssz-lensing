"""
Inversion Results & Model Comparison

Ported from: segmented-calculation-suite (result dictionaries, tie handling)

(C) 2025 Carmen Wrede & Lino Casu
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class InversionResult:
    """
    Complete result of lens inversion.

    Separates g1 (observables) from g2 (formal parameters).
    """
    # g2: Formal parameters (recovered)
    params: Dict[str, float]

    # g1: Observable predictions
    predicted_images: np.ndarray
    residuals: np.ndarray

    # Diagnostics
    max_residual: float
    rms_residual: float
    consistency: float  # How well redundant equations agree

    # DOF info
    n_constraints: int
    n_parameters: int
    dof: int
    dof_status: str

    # Metadata
    model_name: str
    solver_converged: bool
    iterations: int = 0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Model: {self.model_name}",
            f"DOF: {self.dof_status} ({self.dof})",
            f"max|res|: {self.max_residual:.4f} arcsec",
            f"RMS: {self.rms_residual:.4f} arcsec",
            f"Converged: {self.solver_converged}",
        ]
        return "\n".join(lines)

    def is_acceptable(self, threshold: float = 0.01) -> bool:
        """
        Check if result is acceptable.

        Note: "Acceptable" means solver converged, NOT that model is adequate.
        Model adequacy is diagnosed via residuals.
        """
        return self.solver_converged and self.max_residual < threshold


@dataclass
class ModelComparison:
    """
    Compare multiple models on same data.

    Following SSZ principle: simpler model wins if residuals comparable.
    """
    results: List[InversionResult]
    winner: str
    winner_reason: str
    is_tie: bool = False

    def summary(self) -> str:
        """Human-readable comparison."""
        lines = ["Model Comparison:"]
        lines.append("-" * 40)

        for r in self.results:
            marker = " <-- WINNER" if r.model_name == self.winner else ""
            lines.append(
                f"  {r.model_name}: max|res|={r.max_residual:.4f}\"{marker}"
            )

        lines.append("-" * 40)
        lines.append(f"Winner: {self.winner}")
        lines.append(f"Reason: {self.winner_reason}")

        return "\n".join(lines)


def compare_models(results: List[InversionResult],
                   prefer_simple: bool = True,
                   tie_epsilon: float = 1e-3) -> ModelComparison:
    """
    Compare inversion results from multiple models.

    Args:
        results: List of InversionResult from different models
        prefer_simple: If True, simpler model wins ties
        tie_epsilon: Relative threshold for tie detection

    Returns:
        ModelComparison with winner determination
    """
    if not results:
        raise ValueError("No results to compare")

    if len(results) == 1:
        return ModelComparison(
            results=results,
            winner=results[0].model_name,
            winner_reason="Only model tested",
            is_tie=False
        )

    # Sort by complexity (fewer parameters = simpler)
    sorted_results = sorted(results, key=lambda r: r.n_parameters)

    # Find best residual
    valid_results = [r for r in results if r.solver_converged]

    if not valid_results:
        # No valid results - return first as "winner"
        return ModelComparison(
            results=results,
            winner=results[0].model_name,
            winner_reason="No model converged",
            is_tie=False
        )

    best_residual = min(r.max_residual for r in valid_results)

    # Check for ties (within epsilon)
    tied = []
    for r in valid_results:
        if r.max_residual <= best_residual * (1 + tie_epsilon):
            tied.append(r)

    if len(tied) == 1:
        winner = tied[0]
        return ModelComparison(
            results=results,
            winner=winner.model_name,
            winner_reason=f"Best residual: {winner.max_residual:.4f}\"",
            is_tie=False
        )

    # Multiple models tied - prefer simpler
    if prefer_simple:
        # Sort tied models by complexity
        tied_sorted = sorted(tied, key=lambda r: r.n_parameters)
        winner = tied_sorted[0]
        reason = (f"Tie (within {tie_epsilon*100:.1f}%), "
                  f"prefer simpler ({winner.n_parameters} params)")
    else:
        # Take best residual even if complex
        winner = min(tied, key=lambda r: r.max_residual)
        reason = f"Best residual among tied: {winner.max_residual:.4f}\""

    return ModelComparison(
        results=results,
        winner=winner.model_name,
        winner_reason=reason,
        is_tie=True
    )


def interpret_residuals(result: InversionResult,
                        astrometry_precision: float = 0.003) -> Dict[str, Any]:
    """
    Interpret residuals following SSZ methodology.

    Key insight: Large residuals indicate MODEL INADEQUACY, not solver failure.

    Args:
        result: Inversion result to interpret
        astrometry_precision: Typical measurement precision (arcsec)

    Returns:
        Dict with interpretation
    """
    ratio = result.max_residual / astrometry_precision

    if ratio < 1:
        quality = "EXCELLENT"
        interpretation = "Residuals below measurement noise"
        recommendation = "Model adequate for this data"
    elif ratio < 3:
        quality = "GOOD"
        interpretation = "Residuals comparable to noise"
        recommendation = "Model likely adequate"
    elif ratio < 10:
        quality = "MARGINAL"
        interpretation = "Residuals significantly above noise"
        recommendation = "Consider more complex model or check data"
    else:
        quality = "POOR"
        interpretation = f"Residuals {ratio:.0f}x measurement precision"
        recommendation = "Model inadequate - need extension"

    return {
        "quality": quality,
        "residual_to_noise": ratio,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "solver_ok": result.solver_converged,
        "model_adequate": ratio < 10,
    }
