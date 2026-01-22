"""Fallback dataset loader with No-NaN guarantee."""
import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .no_nan import assert_finite, NaNDetectedError

FALLBACK_DIR = Path(__file__).parent.parent.parent / "data" / "fallback"


def load_manifest() -> Dict:
    """Load fallback manifest.json."""
    manifest_path = FALLBACK_DIR / "manifest.json"
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_available_datasets() -> List[Dict]:
    """Get list of available fallback datasets."""
    manifest = load_manifest()
    return manifest.get("datasets", [])


def get_dataset_by_mode(mode: str) -> Optional[Dict]:
    """Get first dataset matching mode (QUAD/RING)."""
    datasets = get_available_datasets()
    for ds in datasets:
        if ds.get("mode") == mode:
            return ds
    return None


def load_quad_images(dataset_id: str = "Q2237+0305_quad") -> Tuple[np.ndarray, Dict]:
    """
    Load QUAD image positions from fallback dataset.
    
    Returns
    -------
    positions : np.ndarray, shape (N, 2)
        Image positions in arcseconds
    meta : dict
        Dataset metadata
    """
    manifest = load_manifest()
    dataset = None
    for ds in manifest["datasets"]:
        if ds["id"] == dataset_id:
            dataset = ds
            break
    
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    # Load CSV
    csv_path = FALLBACK_DIR / dataset["files"]["images_csv"]
    positions = []
    image_ids = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            positions.append([x, y])
            image_ids.append(row["image_id"])
    
    positions = np.array(positions)
    assert_finite(positions, f"positions from {dataset_id}")
    
    # Load meta
    meta_path = FALLBACK_DIR / dataset["files"]["meta_json"]
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    meta["image_ids"] = image_ids
    meta["unit"] = "arcsec"
    
    return positions, meta


def load_ring_arc_points(dataset_id: str = "B1938+666_ring") -> Tuple[np.ndarray, Dict]:
    """
    Load RING/ARC points from fallback dataset.
    
    Returns
    -------
    positions : np.ndarray, shape (N, 2)
        Arc point positions in arcseconds
    meta : dict
        Dataset metadata
    """
    manifest = load_manifest()
    dataset = None
    for ds in manifest["datasets"]:
        if ds["id"] == dataset_id:
            dataset = ds
            break
    
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    # Load CSV
    csv_path = FALLBACK_DIR / dataset["files"]["arc_points_csv"]
    positions = []
    point_ids = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            positions.append([x, y])
            point_ids.append(row["point_id"])
    
    positions = np.array(positions)
    assert_finite(positions, f"arc_points from {dataset_id}")
    
    # Load meta
    meta_path = FALLBACK_DIR / dataset["files"]["meta_json"]
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    meta["point_ids"] = point_ids
    meta["unit"] = "arcsec"
    
    return positions, meta


def load_fallback_by_mode(mode: str) -> Tuple[np.ndarray, Dict]:
    """
    Load appropriate fallback dataset based on mode.
    
    Parameters
    ----------
    mode : str
        "QUAD" or "RING"
        
    Returns
    -------
    positions : np.ndarray
    meta : dict
    """
    if mode == "QUAD":
        return load_quad_images()
    elif mode in ("RING", "ARC"):
        return load_ring_arc_points()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use QUAD or RING.")


def get_fallback_text(mode: str) -> str:
    """Get fallback positions as text (for UI)."""
    positions, meta = load_fallback_by_mode(mode)
    lines = []
    for pos in positions:
        lines.append(f"{pos[0]:.3f}, {pos[1]:.3f}")
    return "\n".join(lines)


def validate_all_fallback_datasets() -> List[str]:
    """
    Validate all fallback datasets have no NaN.
    
    Returns list of issues (empty = all valid).
    """
    issues = []
    manifest = load_manifest()
    
    for ds in manifest["datasets"]:
        try:
            if ds["mode"] == "QUAD":
                pos, _ = load_quad_images(ds["id"])
            else:
                pos, _ = load_ring_arc_points(ds["id"])
            assert_finite(pos, ds["id"])
        except NaNDetectedError as e:
            issues.append(f"{ds['id']}: {e}")
        except Exception as e:
            issues.append(f"{ds['id']}: Error loading - {e}")
    
    return issues
