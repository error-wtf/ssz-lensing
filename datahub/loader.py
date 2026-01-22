"""DataHub loader: load validated snapshots for pipeline use."""
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

from .validate import validate_snapshot, DatasetValidationError, load_manifest

DATAHUB_ROOT = Path(__file__).parent


def load_quad_snapshot(dataset_id: str = "Q2237+0305") -> Tuple[np.ndarray, Dict]:
    """
    Load QUAD snapshot with validation.
    
    Returns (positions, meta) where positions shape is (4, 2).
    Raises DatasetValidationError if snapshot invalid.
    """
    valid, issues = validate_snapshot(dataset_id)
    if not valid:
        raise DatasetValidationError(f"{dataset_id}: {issues}")
    
    snapshot_dir = DATAHUB_ROOT / "snapshots" / dataset_id
    
    # Load CSV
    csv_path = snapshot_dir / "images.csv"
    positions = []
    image_ids = []
    sx_list = []
    sy_list = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append([float(row['x']), float(row['y'])])
            image_ids.append(row['image_id'])
            if 'sx' in row and row['sx']:
                sx_list.append(float(row['sx']))
            if 'sy' in row and row['sy']:
                sy_list.append(float(row['sy']))
    
    positions = np.array(positions)
    
    # Load meta
    meta_path = snapshot_dir / "meta.json"
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    meta['image_ids'] = image_ids
    meta['sx'] = sx_list if sx_list else None
    meta['sy'] = sy_list if sy_list else None
    meta['unit'] = 'arcsec'
    meta['data_source'] = 'datahub_snapshot'
    meta['snapshot_id'] = dataset_id
    
    return positions, meta


def load_ring_snapshot(dataset_id: str = "B1938+666") -> Tuple[np.ndarray, Dict]:
    """
    Load RING/ARC snapshot with validation.
    
    Returns (positions, meta) where positions shape is (N, 2).
    Raises DatasetValidationError if snapshot invalid.
    """
    valid, issues = validate_snapshot(dataset_id)
    if not valid:
        raise DatasetValidationError(f"{dataset_id}: {issues}")
    
    snapshot_dir = DATAHUB_ROOT / "snapshots" / dataset_id
    
    # Load CSV
    csv_path = snapshot_dir / "arc_points.csv"
    positions = []
    point_ids = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append([float(row['x']), float(row['y'])])
            point_ids.append(row['point_id'])
    
    positions = np.array(positions)
    
    # Load meta
    meta_path = snapshot_dir / "meta.json"
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    meta['point_ids'] = point_ids
    meta['unit'] = 'arcsec'
    meta['data_source'] = 'datahub_snapshot'
    meta['snapshot_id'] = dataset_id
    
    return positions, meta


def load_fallback_by_mode(mode: str) -> Tuple[np.ndarray, Dict]:
    """
    Load appropriate fallback based on mode.
    
    Parameters
    ----------
    mode : str
        "QUAD" or "RING"/"ARC"
        
    Returns
    -------
    positions : np.ndarray
    meta : dict
    
    Raises
    ------
    DatasetValidationError
        If no valid fallback available
    ValueError
        If mode unknown
    """
    manifest = load_manifest()
    fallback_rules = manifest.get("fallback_rules", {})
    
    if mode == "QUAD":
        candidates = fallback_rules.get("QUAD", ["Q2237+0305"])
        for ds_id in candidates:
            try:
                return load_quad_snapshot(ds_id)
            except DatasetValidationError:
                continue
        raise DatasetValidationError(f"No valid QUAD fallback available")
    
    elif mode in ("RING", "ARC"):
        candidates = fallback_rules.get("RING", ["B1938+666"])
        for ds_id in candidates:
            try:
                return load_ring_snapshot(ds_id)
            except DatasetValidationError:
                continue
        raise DatasetValidationError(f"No valid RING fallback available")
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use QUAD or RING.")


def get_fallback_text(mode: str) -> str:
    """Get fallback positions as text for UI."""
    positions, _ = load_fallback_by_mode(mode)
    lines = [f"{p[0]:.3f}, {p[1]:.3f}" for p in positions]
    return "\n".join(lines)


def list_available_datasets() -> List[Dict]:
    """List all available validated datasets."""
    manifest = load_manifest()
    available = []
    
    for ds in manifest["datasets"]:
        valid, _ = validate_snapshot(ds["id"])
        if valid:
            available.append({
                "id": ds["id"],
                "name": ds["name"],
                "type": ds["type"],
                "mode": ds["mode"]
            })
    
    return available
