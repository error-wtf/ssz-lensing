"""Utilities for ssz-lensing: No-NaN enforcement and fallback data."""

from .no_nan import (
    NaNDetectedError,
    assert_finite,
    sanitize_no_nan,
    sanitize_dict,
    validate_no_nan_in_dict,
    safe_divide,
    safe_sqrt,
    parse_float_safe,
    dump_json_no_nan,
)

from .fallback_loader import (
    load_manifest,
    get_available_datasets,
    get_dataset_by_mode,
    load_quad_images,
    load_ring_arc_points,
    load_fallback_by_mode,
    get_fallback_text,
    validate_all_fallback_datasets,
)

__all__ = [
    'NaNDetectedError',
    'assert_finite',
    'sanitize_no_nan',
    'sanitize_dict',
    'validate_no_nan_in_dict',
    'safe_divide',
    'safe_sqrt',
    'parse_float_safe',
    'dump_json_no_nan',
    'load_manifest',
    'get_available_datasets',
    'get_dataset_by_mode',
    'load_quad_images',
    'load_ring_arc_points',
    'load_fallback_by_mode',
    'get_fallback_text',
    'validate_all_fallback_datasets',
]
