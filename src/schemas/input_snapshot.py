"""Input Snapshot Schema v1.0 - exact reproducibility of user inputs."""
import json
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import numpy as np

SCHEMA_VERSION = "1.0"

# Allowed enum values
MODES = ["QUAD", "RING", "ARC", "DOUBLE", "UNKNOWN"]
ANGLE_UNITS = ["rad", "arcsec", "mas", "uas"]
DISTANCE_UNITS = ["m", "km", "AU", "ly", "pc", "kpc", "Mpc", "Gpc"]
DISTANCE_MODES = ["normalized", "direct", "redshift"]
COSMOLOGY_PRESETS = ["Planck18", "Planck15", "WMAP9", "custom"]

INPUT_SNAPSHOT_SCHEMA = {
    "schema_version": str,
    "created_utc": str,
    "repo": {"name": str, "git_commit": str},
    "case": {"case_id": str, "case_name": str, "mode": str, "notes": str},
    "inputs": {
        "positions": {
            "format": str,
            "raw_text": str,
            "unit": str,
            "parsed": list
        }
    },
    "distance_config": {"distance_mode": str},
    "normalization": {"internal_units": dict, "converted": dict}
}


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "UNKNOWN_OR_HASH"


def create_input_snapshot(
    case_id: str,
    case_name: str,
    mode: str,
    raw_text: str,
    position_unit: str,
    positions_parsed: List[Dict],
    positions_rad: np.ndarray,
    distance_mode: str = "normalized",
    d_l_value: Optional[float] = None,
    d_l_unit: Optional[str] = None,
    d_s_value: Optional[float] = None,
    d_s_unit: Optional[str] = None,
    z_l: Optional[float] = None,
    z_s: Optional[float] = None,
    cosmology_preset: str = "Planck18",
    lens_center_known: bool = False,
    lens_center_x: Optional[float] = None,
    lens_center_y: Optional[float] = None,
    lens_center_unit: str = "arcsec",
    notes: str = ""
) -> Dict[str, Any]:
    """
    Create input_snapshot.json content per schema v1.0.
    
    Parameters
    ----------
    case_id : str
        Unique case identifier (e.g., "Q2237_20260122_112345")
    case_name : str
        Human-readable name (e.g., "Q2237+0305")
    mode : str
        QUAD, RING, ARC, DOUBLE, or UNKNOWN
    raw_text : str
        Original user input text (for exact reproducibility)
    position_unit : str
        Unit of input positions (arcsec, mas, uas, rad)
    positions_parsed : list
        List of dicts with id, x, y, sx, sy
    positions_rad : np.ndarray
        Converted positions in radians (internal unit)
    distance_mode : str
        "normalized", "direct", or "redshift"
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Build parsed positions list
    parsed = []
    for i, p in enumerate(positions_parsed):
        parsed.append({
            "id": p.get("id", chr(65 + i)),
            "x": float(p["x"]),
            "y": float(p["y"]),
            "sx": p.get("sx"),
            "sy": p.get("sy")
        })
    
    # Build converted positions
    converted_positions = []
    converted_sigmas = []
    for i, (p, rad) in enumerate(zip(positions_parsed, positions_rad)):
        pid = p.get("id", chr(65 + i))
        converted_positions.append({
            "id": pid,
            "theta_x": float(rad[0]),
            "theta_y": float(rad[1])
        })
        converted_sigmas.append({
            "id": pid,
            "sx": None,
            "sy": None
        })
    
    # Validation checks
    checks = [
        {
            "name": "min_points",
            "ok": len(parsed) >= (4 if mode == "QUAD" else 2),
            "detail": f"{mode} needs >={4 if mode == 'QUAD' else 2} points"
        },
        {
            "name": "finite_values",
            "ok": all(np.isfinite(positions_rad).flatten()),
            "detail": ""
        }
    ]
    
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": now,
        "repo": {
            "name": "ssz-lensing",
            "git_commit": get_git_commit()
        },
        
        "case": {
            "case_id": case_id,
            "case_name": case_name,
            "mode": mode,
            "notes": notes
        },
        
        "inputs": {
            "positions": {
                "format": "xy_lines",
                "raw_text": raw_text,
                "unit": position_unit,
                "parsed": parsed
            },
            
            "arc_points": {
                "present": False,
                "unit": None,
                "points": []
            },
            
            "flux_ratios": {
                "present": False,
                "unit": "dimensionless",
                "ratios": []
            },
            
            "time_delays": {
                "present": False,
                "unit": "day",
                "delays": []
            },
            
            "lens_center": {
                "known": lens_center_known,
                "unit": lens_center_unit,
                "x0": lens_center_x,
                "y0": lens_center_y
            }
        },
        
        "distance_config": {
            "distance_mode": distance_mode,
            
            "normalized": {
                "D_L": 1.0,
                "D_S": 2.0
            },
            
            "direct": {
                "present": distance_mode == "direct",
                "D_L_value": d_l_value,
                "D_L_unit": d_l_unit,
                "D_S_value": d_s_value,
                "D_S_unit": d_s_unit
            },
            
            "redshift": {
                "present": distance_mode == "redshift",
                "z_L": z_l,
                "z_S": z_s,
                "cosmology": {
                    "preset": cosmology_preset,
                    "H0_km_s_Mpc": None,
                    "Omega_m": None,
                    "Omega_Lambda": None
                }
            }
        },
        
        "normalization": {
            "internal_units": {
                "angle": "rad",
                "length": "m",
                "time": "s",
                "mass": "kg"
            },
            
            "converted": {
                "positions_rad": converted_positions,
                "position_sigmas_rad": converted_sigmas
            },
            
            "validation": {
                "checks": checks,
                "warnings": []
            }
        }
    }
    
    return snapshot


def validate_input_snapshot(data: Dict) -> List[str]:
    """
    Validate input_snapshot.json against schema v1.0.
    
    Returns list of errors (empty = valid).
    """
    errors = []
    
    # Required top-level keys
    required = ["schema_version", "created_utc", "repo", "case", "inputs",
                "distance_config", "normalization"]
    for key in required:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        return errors
    
    # Schema version
    if data["schema_version"] != SCHEMA_VERSION:
        errors.append(f"Schema version mismatch: {data['schema_version']} != {SCHEMA_VERSION}")
    
    # Case validation
    case = data.get("case", {})
    if not case.get("case_id"):
        errors.append("Missing case.case_id")
    if case.get("mode") not in MODES:
        errors.append(f"Invalid mode: {case.get('mode')}. Must be one of {MODES}")
    
    # Inputs validation
    inputs = data.get("inputs", {})
    pos = inputs.get("positions", {})
    if pos.get("unit") not in ANGLE_UNITS:
        errors.append(f"Invalid position unit: {pos.get('unit')}")
    if not pos.get("raw_text"):
        errors.append("Missing inputs.positions.raw_text")
    
    # Distance config validation
    dc = data.get("distance_config", {})
    dm = dc.get("distance_mode")
    if dm not in DISTANCE_MODES:
        errors.append(f"Invalid distance_mode: {dm}")
    
    if dm == "redshift":
        rs = dc.get("redshift", {})
        if rs.get("z_L") is None or rs.get("z_S") is None:
            errors.append("Redshift mode requires z_L and z_S")
    
    if dm == "direct":
        dr = dc.get("direct", {})
        if dr.get("D_L_value") is None or dr.get("D_S_value") is None:
            errors.append("Direct mode requires D_L_value and D_S_value")
    
    # Normalization validation
    norm = data.get("normalization", {})
    iu = norm.get("internal_units", {})
    if iu.get("angle") != "rad":
        errors.append("Internal angle unit must be 'rad'")
    if iu.get("length") != "m":
        errors.append("Internal length unit must be 'm'")
    
    # QUAD specific
    if case.get("mode") == "QUAD":
        parsed = pos.get("parsed", [])
        if len(parsed) < 4:
            errors.append(f"QUAD mode requires >= 4 points, got {len(parsed)}")
    
    return errors


def save_input_snapshot(snapshot: Dict, filepath: str) -> None:
    """Save input_snapshot.json with validation."""
    errors = validate_input_snapshot(snapshot)
    if errors:
        raise ValueError(f"Schema validation failed: {errors}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
