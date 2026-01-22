"""UI State management: dataset_state and run_state as single source of truth."""
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from datahub import load_fallback_by_mode, validate_snapshot, list_available_datasets
from datahub.validate import validate_csv_no_nan_no_null


@dataclass
class DatasetState:
    """Single source of truth for dataset in UI."""
    dataset_id: str = ""
    mode: str = ""  # QUAD or RING
    source: str = ""  # "user" or "fallback"
    points: List[List[float]] = field(default_factory=list)
    point_ids: List[str] = field(default_factory=list)
    unit: str = "arcsec"
    z_lens: Optional[float] = None
    z_source: Optional[float] = None
    theta_E_arcsec: Optional[float] = None
    center_x: float = 0.0
    center_y: float = 0.0
    center_known: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)
    validated: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'DatasetState':
        return cls(**d)

    def get_positions_array(self) -> np.ndarray:
        return np.array(self.points)


@dataclass
class RunState:
    """Run configuration state."""
    selected_models: List[str] = field(default_factory=lambda: ["m2", "shear"])
    distance_mode: str = "normalized"  # normalized, direct, redshift
    D_L: Optional[float] = None
    D_S: Optional[float] = None
    D_LS: Optional[float] = None
    D_unit: str = "Gpc"
    cosmology: str = "Planck18"
    lens_mass_Msun: Optional[float] = None
    last_quicklook: Optional[Dict] = None
    last_inversion: Optional[Dict] = None
    last_scene3d: Optional[Dict] = None
    run_dir: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'RunState':
        return cls(**d)


def empty_dataset_state() -> Dict:
    """Create empty dataset state dict."""
    return DatasetState().to_dict()


def default_run_state() -> Dict:
    """Create default run state dict."""
    return RunState().to_dict()


def parse_user_points(text: str, mode: str) -> tuple:
    """
    Parse user text input to points list.
    
    Returns (points, point_ids, errors).
    """
    points = []
    point_ids = []
    errors = []
    
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    
    for i, line in enumerate(lines):
        # Try comma-separated
        parts = [p.strip() for p in line.replace(',', ' ').split()]
        if len(parts) < 2:
            errors.append(f"Line {i+1}: need at least x,y values")
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            if not np.isfinite(x) or not np.isfinite(y):
                errors.append(f"Line {i+1}: non-finite value")
                continue
            points.append([x, y])
            point_ids.append(chr(65 + i) if mode == "QUAD" else f"P{i+1:02d}")
        except ValueError as e:
            errors.append(f"Line {i+1}: {e}")
    
    # Mode validation
    if mode == "QUAD" and len(points) != 4:
        errors.append(f"QUAD mode requires exactly 4 points, got {len(points)}")
    elif mode == "RING" and len(points) < 8:
        errors.append(f"RING mode requires at least 8 points, got {len(points)}")
    
    return points, point_ids, errors


def build_user_dataset(
    text: str,
    mode: str,
    unit: str,
    center_known: bool = False,
    center_x: float = 0.0,
    center_y: float = 0.0,
    z_lens: Optional[float] = None,
    z_source: Optional[float] = None
) -> DatasetState:
    """Build dataset state from user input."""
    ds = DatasetState()
    ds.source = "user"
    ds.mode = mode
    ds.unit = unit
    ds.center_known = center_known
    ds.center_x = center_x
    ds.center_y = center_y
    ds.z_lens = z_lens
    ds.z_source = z_source
    
    points, point_ids, errors = parse_user_points(text, mode)
    ds.points = points
    ds.point_ids = point_ids
    ds.errors = errors
    ds.dataset_id = f"user_{mode.lower()}_{len(points)}pts"
    
    ds.provenance = {
        "source": "user_input",
        "input_lines": len(text.strip().split('\n')),
        "parsed_points": len(points)
    }
    
    return ds


def load_fallback_dataset(dataset_id: str) -> DatasetState:
    """Load fallback dataset from datahub."""
    ds = DatasetState()
    ds.source = "fallback"
    ds.dataset_id = dataset_id
    
    try:
        # Determine mode from manifest
        datasets = list_available_datasets()
        ds_info = next((d for d in datasets if d['id'] == dataset_id), None)
        if not ds_info:
            ds.errors.append(f"Dataset {dataset_id} not found in manifest")
            return ds
        
        ds.mode = ds_info['mode']
        
        # Load via datahub
        positions, meta = load_fallback_by_mode(ds.mode)
        
        ds.points = positions.tolist()
        ds.point_ids = meta.get('image_ids') or meta.get('point_ids', [])
        ds.unit = meta.get('unit', 'arcsec')
        ds.z_lens = meta.get('z_lens')
        ds.z_source = meta.get('z_source')
        ds.theta_E_arcsec = meta.get('theta_E_arcsec')
        ds.provenance = {
            "source": "datahub_snapshot",
            "dataset_id": dataset_id,
            "references": meta.get('primary_references', [])
        }
        
    except Exception as e:
        ds.errors.append(f"Failed to load {dataset_id}: {e}")
    
    return ds


def validate_dataset(ds: DatasetState) -> DatasetState:
    """
    Validate dataset state. Sets validated=True if all checks pass.
    
    Enforces: no NaN, no Inf, required fields present.
    """
    ds.errors = []
    ds.warnings = []
    
    # Check points exist
    if not ds.points:
        ds.errors.append("No points loaded")
        ds.validated = False
        return ds
    
    # Check all points finite
    arr = np.array(ds.points)
    if np.any(np.isnan(arr)):
        ds.errors.append("NaN values in points")
    if np.any(np.isinf(arr)):
        ds.errors.append("Inf values in points")
    
    # Check mode consistency
    if ds.mode == "QUAD" and len(ds.points) != 4:
        ds.errors.append(f"QUAD mode requires 4 points, got {len(ds.points)}")
    elif ds.mode == "RING" and len(ds.points) < 8:
        ds.errors.append(f"RING mode requires ≥8 points, got {len(ds.points)}")
    
    # Check unit is set
    if not ds.unit:
        ds.errors.append("Unit not specified")
    
    # Warnings for missing optional fields
    if ds.z_lens is None or ds.z_source is None:
        ds.warnings.append("Redshifts not set (3D scene will use normalized distances)")
    
    if not ds.center_known:
        ds.warnings.append("Lens center not specified (will estimate from points)")
    
    ds.validated = len(ds.errors) == 0
    return ds


def get_validation_report(ds: DatasetState) -> str:
    """Generate validation report markdown."""
    lines = ["## Validation Report\n"]
    
    if ds.validated:
        lines.append("✅ **Dataset VALID**\n")
    else:
        lines.append("❌ **Dataset INVALID**\n")
    
    lines.append(f"- **ID:** {ds.dataset_id}")
    lines.append(f"- **Mode:** {ds.mode}")
    lines.append(f"- **Source:** {ds.source}")
    lines.append(f"- **Points:** {len(ds.points)}")
    lines.append(f"- **Unit:** {ds.unit}")
    
    if ds.z_lens is not None:
        lines.append(f"- **z_lens:** {ds.z_lens}")
    if ds.z_source is not None:
        lines.append(f"- **z_source:** {ds.z_source}")
    
    if ds.errors:
        lines.append("\n### Errors")
        for e in ds.errors:
            lines.append(f"- ❌ {e}")
    
    if ds.warnings:
        lines.append("\n### Warnings")
        for w in ds.warnings:
            lines.append(f"- ⚠️ {w}")
    
    return "\n".join(lines)


def get_dataset_summary(ds: DatasetState) -> str:
    """Get short dataset summary for display."""
    if not ds.validated:
        return "⚠️ No active dataset"
    
    return f"✅ **{ds.dataset_id}** | {ds.mode} | {len(ds.points)} points | {ds.source}"
