"""
Model Regime Selection & DOF Analysis

Ported from: ssz-qubits (regime auto-selection)
             g79-cygnus-test (calibration philosophy)

Key principle: Never fit more parameters than (constraints - 1).

(C) 2025 Carmen Wrede & Lino Casu
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum


class ModelRegime(Enum):
    """Model complexity regime based on DOF analysis."""
    MINIMAL = "minimal"           # m=2 only (5 params)
    STANDARD = "standard"         # m=2 + shear OR m=3 (7 params)
    EXTENDED = "extended"         # m=2 + shear + m=3 (9 params)
    FORBIDDEN = "forbidden"       # n_params >= n_constraints


@dataclass
class DOFAnalysis:
    """Degrees of freedom analysis result."""
    n_images: int
    n_constraints: int
    n_parameters: int
    dof: int                      # constraints - parameters
    regime: ModelRegime
    status: str                   # "OVERDETERMINED", "EXACT", "UNDERDETERMINED"
    recommendation: str
    extra_data_needed: List[str]  # What data would help


# Model parameter counts (from DOF_ANALYSIS.md)
MODEL_PARAMS: Dict[str, int] = {
    "m2_only": 5,                 # beta_x, beta_y, theta_E, c_2, s_2
    "m2_shear": 7,                # + gamma_1, gamma_2
    "m2_m3": 7,                   # + c_3, s_3
    "m2_shear_m3": 9,             # All of above
    "m2_m3_m4": 9,                # + c_4, s_4
}


def dof_analysis(n_images: int,
                 model_type: str = "m2_only",
                 has_fluxes: bool = False,
                 has_time_delays: bool = False) -> DOFAnalysis:
    """
    Analyze degrees of freedom for given configuration.

    Args:
        n_images: Number of lensed images
        model_type: One of MODEL_PARAMS keys
        has_fluxes: Whether flux ratios are available
        has_time_delays: Whether time delays are available

    Returns:
        DOFAnalysis with regime classification
    """
    # Count constraints
    n_constraints = 2 * n_images  # Position constraints

    if has_fluxes:
        n_constraints += n_images - 1  # Flux ratios

    if has_time_delays:
        n_constraints += n_images - 1  # Time delay constraints

    # Get parameter count
    if model_type not in MODEL_PARAMS:
        raise ValueError(f"Unknown model: {model_type}")

    n_params = MODEL_PARAMS[model_type]
    dof = n_constraints - n_params

    # Determine status
    if dof > 0:
        status = "OVERDETERMINED"
        regime = ModelRegime.MINIMAL if dof >= 3 else ModelRegime.STANDARD
    elif dof == 0:
        status = "EXACT"
        regime = ModelRegime.STANDARD
    else:
        status = "UNDERDETERMINED"
        regime = ModelRegime.FORBIDDEN

    # Recommendation
    if status == "OVERDETERMINED":
        rec = f"Good: {dof} redundant equations for consistency check"
    elif status == "EXACT":
        rec = "Minimal: No redundancy, solution unique if exists"
    else:
        rec = f"FORBIDDEN: Need {-dof} more constraints!"

    # What extra data would help
    extra_data = []
    if status == "UNDERDETERMINED":
        if not has_fluxes:
            extra_data.append(f"flux_ratios (+{n_images-1})")
        if not has_time_delays:
            extra_data.append(f"time_delays (+{n_images-1})")
        extra_data.append("extended_arcs (+many)")
        extra_data.append("stellar_kinematics (+1-3)")

    return DOFAnalysis(
        n_images=n_images,
        n_constraints=n_constraints,
        n_parameters=n_params,
        dof=dof,
        regime=regime,
        status=status,
        recommendation=rec,
        extra_data_needed=extra_data
    )


def select_model_regime(n_images: int,
                        has_fluxes: bool = False,
                        has_time_delays: bool = False,
                        prefer_simple: bool = True) -> Tuple[str, DOFAnalysis]:
    """
    Auto-select best model based on available constraints.

    Follows SSZ principle: UNDER-fitting is better than OVER-fitting.

    Args:
        n_images: Number of images
        has_fluxes: Flux ratios available
        has_time_delays: Time delays available
        prefer_simple: If True, prefer simpler models

    Returns:
        (model_type, DOFAnalysis) tuple
    """
    # Try models in order of complexity
    model_order = ["m2_only", "m2_shear", "m2_m3", "m2_shear_m3"]

    if prefer_simple:
        # Find simplest model that's not underdetermined
        for model in model_order:
            analysis = dof_analysis(
                n_images, model, has_fluxes, has_time_delays
            )
            if analysis.regime != ModelRegime.FORBIDDEN:
                return model, analysis

    else:
        # Find most complex model that's still valid
        best = None
        for model in model_order:
            analysis = dof_analysis(
                n_images, model, has_fluxes, has_time_delays
            )
            if analysis.regime != ModelRegime.FORBIDDEN:
                best = (model, analysis)

        if best:
            return best

    # Fallback: return minimal model with warning
    return "m2_only", dof_analysis(
        n_images, "m2_only", has_fluxes, has_time_delays
    )


def print_dof_table(n_images: int = 4):
    """Print DOF table for quad lens (debugging utility)."""
    print(f"\nDOF Analysis for {n_images}-image system:")
    print("-" * 60)
    print(f"{'Model':<15} {'Params':>8} {'Constr':>8} {'DOF':>6} {'Status':<20}")
    print("-" * 60)

    for model, n_params in MODEL_PARAMS.items():
        analysis = dof_analysis(n_images, model)
        print(f"{model:<15} {n_params:>8} {analysis.n_constraints:>8} "
              f"{analysis.dof:>6} {analysis.status:<20}")

    print("-" * 60)
