"""Data Tab functions for Gradio UI."""
import numpy as np
from .state import (
    DatasetState, build_user_dataset, load_fallback_dataset,
    validate_dataset, get_validation_report, get_dataset_summary
)

QUAD_EXAMPLE = """0.758, 0.560
-0.619, 0.480
-0.472, -0.761
0.857, -0.196"""

RING_EXAMPLE = """0.475, 0.155
0.410, 0.280
0.295, 0.390
0.150, 0.460
0.000, 0.500
-0.150, 0.460
-0.295, 0.390
-0.410, 0.280"""


def get_fallback_choices():
    """Get list of available fallback datasets."""
    try:
        from datahub import list_available_datasets
        datasets = list_available_datasets()
        return [f"{d['id']} ({d['mode']})" for d in datasets]
    except Exception:
        return ["Q2237+0305 (QUAD)", "B1938+666 (RING)"]


def load_fallback_btn(fallback_id):
    """Load fallback dataset."""
    dataset_id = fallback_id.split(" ")[0]
    ds = load_fallback_dataset(dataset_id)
    ds = validate_dataset(ds)
    points_text = "\n".join(f"{p[0]:.3f}, {p[1]:.3f}" for p in ds.points)
    report = get_validation_report(ds)
    return points_text, report, ds.to_dict()


def build_user_btn(text, mode, unit, center_known, cx, cy, z_l, z_s):
    """Build dataset from user input."""
    mode_clean = "QUAD" if "QUAD" in mode else "RING"
    z_lens = z_l if z_l and z_l > 0 else None
    z_source = z_s if z_s and z_s > 0 else None
    ds = build_user_dataset(text, mode_clean, unit, center_known, cx, cy, z_lens, z_source)
    ds = validate_dataset(ds)
    return get_validation_report(ds), ds.to_dict()


def activate_btn(ds_dict):
    """Activate validated dataset."""
    if not ds_dict or not ds_dict.get("validated"):
        return ds_dict, "❌ Dataset not validated", False
    summary = get_dataset_summary(DatasetState.from_dict(ds_dict))
    return ds_dict, f"✅ Active: {summary}", True
