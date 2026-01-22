"""
SSZ-Inspired Validation Framework for Gravitational Lensing

Ported patterns from:
- segmented-calculation-suite: Schema validation, ValidationResult
- ssz-qubits: Regime auto-selection, dataclasses
- g79-cygnus-test: Calibration vs Fitting philosophy

(C) 2025 Carmen Wrede & Lino Casu
Licensed under the ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

from .data_model import (
    ColumnSpec,
    ColumnStatus,
    ValidationError,
    ValidationResult,
    ImageSchema,
    validate_images
)

from .regime import (
    ModelRegime,
    select_model_regime,
    dof_analysis
)

from .results import (
    InversionResult,
    ModelComparison,
    compare_models
)

__all__ = [
    'ColumnSpec',
    'ColumnStatus',
    'ValidationError',
    'ValidationResult',
    'ImageSchema',
    'validate_images',
    'ModelRegime',
    'select_model_regime',
    'dof_analysis',
    'InversionResult',
    'ModelComparison',
    'compare_models',
]
