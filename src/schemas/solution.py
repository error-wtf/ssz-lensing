"""Solution Schema v1.0 - model inversion results."""
import json
from typing import Dict, List, Any

SCHEMA_VERSION = "1.0"

SOLUTION_SCHEMA = {
    "schema_version": str,
    "case_id": str,
    "model": str,
    "regime": dict,
    "params": dict,
    "beta_solution_rad": list,
    "residuals": dict,
    "notes": str
}


def create_solution_json(
    case_id: str,
    model_name: str,
    constraints: int,
    parameters: int,
    rank: int,
    nullspace_dim: int,
    condition_number: float,
    status: str,
    params: Dict[str, float],
    beta_x_rad: float,
    beta_y_rad: float,
    image_plane_rms_rad: float,
    source_plane_scatter_rad: float,
    notes: str = "No fitting performed; solved algebraically."
) -> Dict[str, Any]:
    """Create solutions/<model>.json content per schema v1.0."""
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "model": model_name,
        "regime": {
            "constraints": constraints,
            "parameters": parameters,
            "rank": rank,
            "nullspace_dim": nullspace_dim,
            "condition_number": float(condition_number),
            "status": status
        },
        "params": {k: float(v) for k, v in params.items()},
        "beta_solution_rad": [float(beta_x_rad), float(beta_y_rad)],
        "residuals": {
            "image_plane_rms_rad": float(image_plane_rms_rad),
            "source_plane_scatter_rad": float(source_plane_scatter_rad)
        },
        "notes": notes
    }


def validate_solution(data: Dict) -> List[str]:
    """Validate solution JSON."""
    errors = []
    required = ["schema_version", "case_id", "model", "regime", "params",
                "beta_solution_rad", "residuals"]
    for key in required:
        if key not in data:
            errors.append(f"Missing: {key}")
    
    if "regime" in data:
        regime = data["regime"]
        for k in ["constraints", "parameters", "rank", "status"]:
            if k not in regime:
                errors.append(f"Missing regime.{k}")
    
    return errors


def save_solution(data: Dict, filepath: str) -> None:
    """Save solution JSON with validation."""
    errors = validate_solution(data)
    if errors:
        raise ValueError(f"Validation failed: {errors}")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
