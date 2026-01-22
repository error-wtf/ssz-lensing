"""
Data Model & Schema Validation for Lens Inversion

Ported from: segmented-calculation-suite/segcalc/core/data_model.py

Key principle: No silent fallbacks. Every deviation must be logged.

(C) 2025 Carmen Wrede & Lino Casu
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import numpy as np


class ColumnStatus(Enum):
    """Column presence status."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    COMPUTED = "computed"


@dataclass
class ColumnSpec:
    """Specification for a single data column."""
    name: str
    dtype: str
    unit: str
    status: ColumnStatus
    description: str
    valid_range: Tuple[float, float] = (float('-inf'), float('inf'))
    default: Any = None

    def validate_value(self, value: Any) -> Tuple[bool, str]:
        """Validate a single value. Returns (valid, error_message)."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if self.status == ColumnStatus.REQUIRED:
                return False, f"NaN not allowed in required column '{self.name}'"
            return True, ""

        try:
            if self.dtype == "float64":
                v = float(value)
                if not (self.valid_range[0] <= v <= self.valid_range[1]):
                    return False, f"Value {v} outside range {self.valid_range}"
        except (ValueError, TypeError) as e:
            return False, f"Type error: {e}"

        return True, ""


@dataclass
class ValidationError:
    """A single validation error."""
    index: int
    field: str
    value: Any
    message: str

    def __str__(self):
        return f"[{self.index}] {self.field}: {self.message} (got: {self.value})"


@dataclass
class ValidationResult:
    """Result of data validation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_images: int = 0
    n_constraints: int = 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.valid:
            return f"VALID: {self.n_images} images, {self.n_constraints} constraints"
        else:
            err_str = "\n".join(str(e) for e in self.errors[:5])
            return f"INVALID: {len(self.errors)} errors\n{err_str}"


# =============================================================================
# IMAGE SCHEMA (g1 observables)
# =============================================================================

ImageSchema: Dict[str, ColumnSpec] = {
    "x": ColumnSpec(
        name="x",
        dtype="float64",
        unit="arcsec",
        status=ColumnStatus.REQUIRED,
        description="Image x-position (RA offset)",
        valid_range=(-100, 100)
    ),
    "y": ColumnSpec(
        name="y",
        dtype="float64",
        unit="arcsec",
        status=ColumnStatus.REQUIRED,
        description="Image y-position (Dec offset)",
        valid_range=(-100, 100)
    ),
    "flux": ColumnSpec(
        name="flux",
        dtype="float64",
        unit="mJy",
        status=ColumnStatus.OPTIONAL,
        description="Image flux (optional constraint)",
        valid_range=(0, 1e6),
        default=None
    ),
    "time_delay": ColumnSpec(
        name="time_delay",
        dtype="float64",
        unit="days",
        status=ColumnStatus.OPTIONAL,
        description="Time delay relative to first image",
        valid_range=(-1e4, 1e4),
        default=None
    ),
    "sigma_pos": ColumnSpec(
        name="sigma_pos",
        dtype="float64",
        unit="arcsec",
        status=ColumnStatus.OPTIONAL,
        description="Position uncertainty",
        valid_range=(0, 1),
        default=0.003  # Typical HST precision
    ),
}


def validate_images(images: np.ndarray,
                    fluxes: Optional[np.ndarray] = None,
                    time_delays: Optional[np.ndarray] = None) -> ValidationResult:
    """
    Validate image data against schema.

    Args:
        images: (N, 2) array of (x, y) positions
        fluxes: Optional (N,) array of fluxes
        time_delays: Optional (N,) array of time delays

    Returns:
        ValidationResult with errors/warnings
    """
    errors = []
    warnings = []

    # Check shape
    if images.ndim != 2 or images.shape[1] != 2:
        errors.append(ValidationError(
            index=-1,
            field="images",
            value=images.shape,
            message="Expected shape (N, 2)"
        ))
        return ValidationResult(valid=False, errors=errors)

    n_images = len(images)

    # Validate each position
    for i, (x, y) in enumerate(images):
        # X coordinate
        valid, msg = ImageSchema["x"].validate_value(x)
        if not valid:
            errors.append(ValidationError(i, "x", x, msg))

        # Y coordinate
        valid, msg = ImageSchema["y"].validate_value(y)
        if not valid:
            errors.append(ValidationError(i, "y", y, msg))

    # Count constraints
    n_constraints = 2 * n_images  # Position constraints

    # Validate optional data
    if fluxes is not None:
        if len(fluxes) != n_images:
            errors.append(ValidationError(
                -1, "fluxes", len(fluxes),
                f"Length mismatch: {len(fluxes)} vs {n_images} images"
            ))
        else:
            n_constraints += n_images - 1  # Flux ratios
            for i, f in enumerate(fluxes):
                valid, msg = ImageSchema["flux"].validate_value(f)
                if not valid:
                    errors.append(ValidationError(i, "flux", f, msg))

    if time_delays is not None:
        if len(time_delays) != n_images:
            errors.append(ValidationError(
                -1, "time_delays", len(time_delays),
                f"Length mismatch: {len(time_delays)} vs {n_images} images"
            ))
        else:
            n_constraints += n_images - 1  # Time delay constraints
            for i, t in enumerate(time_delays):
                valid, msg = ImageSchema["time_delay"].validate_value(t)
                if not valid:
                    errors.append(ValidationError(i, "time_delay", t, msg))

    # Warnings for edge cases
    if n_images < 2:
        warnings.append("Less than 2 images: single-image systems not solvable")
    elif n_images == 2:
        warnings.append("Double lens: limited model complexity possible")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        n_images=n_images,
        n_constraints=n_constraints
    )
