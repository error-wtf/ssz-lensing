"""Scene3D Schema v1.0 - 3D scene + auto-units decisions."""
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import numpy as np

SCHEMA_VERSION = "1.0"

SCENE3D_SCHEMA = {
    "schema_version": str,
    "created_utc": str,
    "case_id": str,
    "scene": dict,
    "lensing_scales": dict,
    "rays": dict,
    "display_units": dict
}


def create_scene3d_json(
    case_id: str,
    D_L_m: float,
    D_S_m: float,
    distance_mode: str,
    theta_E_rad: float,
    beta_rad: tuple,
    positions_rad: Optional[np.ndarray] = None,
    theta_E_source: str = "quicklook_or_solution",
    beta_source: str = "inversion_best_model_or_quicklook",
    display_angle: str = "arcsec",
    display_distance: str = "Mpc",
    display_length: str = "kpc",
    formatted_D_L: Optional[Dict] = None,
    formatted_D_S: Optional[Dict] = None,
    formatted_theta_E: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create scene3d.json content per schema v1.0.
    
    All internal values in meters (distances) and radians (angles).
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    D_LS_m = D_S_m - D_L_m
    R_E_m = D_L_m * theta_E_rad
    beta_mag = np.sqrt(beta_rad[0]**2 + beta_rad[1]**2)
    R_beta_m = D_S_m * beta_mag
    
    # Build ray intersections if positions provided
    rays_enabled = positions_rad is not None and len(positions_rad) > 0
    theta_images = []
    lens_plane_points = []
    
    if rays_enabled:
        for i, pos in enumerate(positions_rad):
            pid = chr(65 + i)
            theta_images.append({
                "id": pid,
                "theta": [float(pos[0]), float(pos[1])]
            })
            lens_plane_points.append({
                "id": pid,
                "pos": [float(pos[0]), float(pos[1]), float(D_L_m)]
            })
    
    # Determine if normalized
    is_normalized = distance_mode == "normalized"
    
    scene = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": now,
        "case_id": case_id,
        
        "scene": {
            "coordinate_system": {
                "name": "observer_frame",
                "convention": "Observer at origin; Lens on +z axis; Source behind lens",
                "axes": {"x": "right", "y": "up", "z": "towards source"}
            },
            
            "objects": {
                "observer": {"id": "O", "pos_m": [0.0, 0.0, 0.0]},
                "lens": {
                    "id": "L",
                    "pos_m": [0.0, 0.0, float(D_L_m)],
                    "center_offset_rad": [0.0, 0.0],
                    "notes": "Lens plane at z = D_L"
                },
                "source": {
                    "id": "S",
                    "pos_m": [0.0, 0.0, float(D_S_m)],
                    "notes": "Source plane at z = D_S"
                }
            },
            
            "distances": {
                "D_L_m": float(D_L_m),
                "D_S_m": float(D_S_m),
                "D_LS_m": float(D_LS_m),
                "distance_mode": distance_mode,
                "distance_source": "normalized_defaults" if is_normalized else "user_input"
            }
        },
        
        "lensing_scales": {
            "theta_E_rad": float(theta_E_rad),
            "theta_E_source": theta_E_source,
            "beta_rad": [float(beta_rad[0]), float(beta_rad[1])],
            "beta_source": beta_source,
            "R_E_m": float(R_E_m),
            "R_beta_m": float(R_beta_m)
        },
        
        "rays": {
            "enabled": rays_enabled,
            "theta_images_rad": theta_images,
            "intersections": {
                "lens_plane_points_m": lens_plane_points
            }
        },
        
        "display_units": {
            "primary": {
                "angle": display_angle,
                "distance": display_distance,
                "length": display_length,
                "time": "day"
            },
            "secondary": {
                "angle": "rad",
                "distance": None,
                "length": "AU"
            },
            "auto_rules_used": {
                "angle_rule": "arcsec/mas/uas thresholds",
                "distance_rule": "pc/kpc/Mpc/Gpc thresholds",
                "length_rule": "AU/pc/kpc thresholds"
            },
            "formatted_values": {
                "D_L": formatted_D_L or {
                    "value": float(D_L_m),
                    "unit": "normalized" if is_normalized else "m"
                },
                "D_S": formatted_D_S or {
                    "value": float(D_S_m),
                    "unit": "normalized" if is_normalized else "m"
                },
                "D_LS": {
                    "value": float(D_LS_m),
                    "unit": "normalized" if is_normalized else "m"
                },
                "theta_E": formatted_theta_E or {
                    "value": float(theta_E_rad),
                    "unit": "rad"
                },
                "R_E": {
                    "value": float(R_E_m),
                    "unit": "normalized_length" if is_normalized else "m"
                },
                "beta": {
                    "value": float(beta_mag),
                    "unit": "rad"
                }
            }
        }
    }
    
    return scene


def validate_scene3d(data: Dict) -> List[str]:
    """Validate scene3d.json against schema v1.0."""
    errors = []
    
    required = ["schema_version", "created_utc", "case_id", "scene",
                "lensing_scales", "rays", "display_units"]
    for key in required:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        return errors
    
    if data["schema_version"] != SCHEMA_VERSION:
        errors.append(f"Schema version mismatch: {data['schema_version']}")
    
    if not data.get("case_id"):
        errors.append("Missing case_id")
    
    scene = data.get("scene", {})
    distances = scene.get("distances", {})
    
    if distances.get("D_L_m") is None:
        errors.append("Missing scene.distances.D_L_m")
    if distances.get("D_S_m") is None:
        errors.append("Missing scene.distances.D_S_m")
    
    dm = distances.get("distance_mode")
    if dm not in ["normalized", "direct", "redshift"]:
        errors.append(f"Invalid distance_mode: {dm}")
    
    scales = data.get("lensing_scales", {})
    if scales.get("theta_E_rad") is None:
        errors.append("Missing lensing_scales.theta_E_rad")
    
    return errors


def save_scene3d(scene: Dict, filepath: str) -> None:
    """Save scene3d.json with validation."""
    errors = validate_scene3d(scene)
    if errors:
        raise ValueError(f"Schema validation failed: {errors}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(scene, f, indent=2, ensure_ascii=False)
