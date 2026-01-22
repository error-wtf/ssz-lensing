"""Quicklook Schema v1.0 - geometry estimates and diagnostics."""
import json
from typing import Dict, List, Any

SCHEMA_VERSION = "1.0"

QUICKLOOK_SCHEMA = {
    "schema_version": str,
    "case_id": str,
    "center_estimate": dict,
    "theta_E_estimate_rad": float,
    "radial_rms_rad": float,
    "harmonic_signature": dict,
    "morphology": dict
}


def create_quicklook_json(
    case_id: str,
    center_x_rad: float,
    center_y_rad: float,
    center_method: str,
    theta_E_rad: float,
    radial_rms_rad: float,
    m2_amp_rad: float,
    m4_amp_rad: float,
    morphology_guess: str,
    criteria: List[str]
) -> Dict[str, Any]:
    """Create quicklook.json content per schema v1.0."""
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "center_estimate": {
            "x_rad": float(center_x_rad),
            "y_rad": float(center_y_rad),
            "method": center_method
        },
        "theta_E_estimate_rad": float(theta_E_rad),
        "radial_rms_rad": float(radial_rms_rad),
        "harmonic_signature": {
            "m2_amp_rad": float(m2_amp_rad),
            "m4_amp_rad": float(m4_amp_rad),
            "label": "diagnostic_only"
        },
        "morphology": {
            "guess": morphology_guess,
            "criteria": criteria
        }
    }


def validate_quicklook(data: Dict) -> List[str]:
    """Validate quicklook.json."""
    errors = []
    required = ["schema_version", "case_id", "center_estimate",
                "theta_E_estimate_rad", "radial_rms_rad", "morphology"]
    for key in required:
        if key not in data:
            errors.append(f"Missing: {key}")
    return errors


def save_quicklook(data: Dict, filepath: str) -> None:
    """Save quicklook.json with validation."""
    errors = validate_quicklook(data)
    if errors:
        raise ValueError(f"Validation failed: {errors}")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
