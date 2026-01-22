"""
Inversion Tools for Gauge Gravitational Lensing

NO curve fitting, NO least squares.
Only exact solvers, rootfinding, and consistency checks.

Modules:
- exact_solvers: Linear system solvers (no lstsq)
- root_solvers: Bracketing + bisection for nonlinear variables
- constraints: DoF bookkeeping and observable requirements
- diagnostics: Residual analysis and model adequacy checks
"""

from .exact_solvers import (
    solve_linear_exact,
    solve_linear_subset,
    choose_invertible_subset,
    matrix_rank,
    determinant
)
from .root_solvers import (
    bisection,
    brent,
    find_all_roots,
    newton_raphson,
    secant_method
)
from .constraints import (
    count_degrees_of_freedom,
    check_image_multiplicity,
    observable_equations,
    inversion_summary
)
from .diagnostics import (
    residual_report,
    compare_solutions,
    consistency_check,
    solution_quality_score,
    print_diagnostic_summary
)
from .lens_equation import (
    InputMode,
    MorphologyResult,
    SourceConsistency,
    classify_morphology,
    compute_source_positions,
    check_source_consistency,
    DEFLECTION_MODELS
)
from .quad_inversion import (
    InversionResult,
    ModelComparison,
    invert_quad,
    compare_models,
    generate_synthetic_quad
)

__all__ = [
    'solve_linear_exact', 'solve_linear_subset', 'choose_invertible_subset',
    'matrix_rank', 'determinant',
    'bisection', 'brent', 'find_all_roots', 'newton_raphson', 'secant_method',
    'count_degrees_of_freedom', 'check_image_multiplicity',
    'observable_equations', 'inversion_summary',
    'residual_report', 'compare_solutions', 'consistency_check',
    'solution_quality_score', 'print_diagnostic_summary',
    # Real inversion framework
    'InputMode', 'MorphologyResult', 'SourceConsistency',
    'classify_morphology', 'compute_source_positions', 'check_source_consistency',
    'DEFLECTION_MODELS',
    'InversionResult', 'ModelComparison',
    'invert_quad', 'compare_models', 'generate_synthetic_quad'
]
