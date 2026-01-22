"""Utilities for ssz-lensing: No-Null/No-NaN enforcement and fallback data."""

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

from .defaults import (
    ValueWithProvenance,
    FullNumericPoint,
    compute_default_sigma,
    fill_position_uncertainties,
    get_normalized_distances,
    estimate_center,
    estimate_theta_E,
    create_full_numeric_points,
    count_assumptions,
)

from .validate_no_null import (
    NullOrNaNFoundError,
    is_null_or_nan,
    validate_dict_no_null,
    validate_json_file_no_null,
    validate_csv_file_no_null,
    validate_run_bundle_no_null,
    assert_no_null_no_nan,
    summarize_provenance,
)

__all__ = [
    # No-NaN basics
    'NaNDetectedError',
    'assert_finite',
    'sanitize_no_nan',
    'sanitize_dict',
    'validate_no_nan_in_dict',
    'safe_divide',
    'safe_sqrt',
    'parse_float_safe',
    'dump_json_no_nan',
    # Fallback data
    'load_manifest',
    'get_available_datasets',
    'get_dataset_by_mode',
    'load_quad_images',
    'load_ring_arc_points',
    'load_fallback_by_mode',
    'get_fallback_text',
    'validate_all_fallback_datasets',
    # Defaults with provenance
    'ValueWithProvenance',
    'FullNumericPoint',
    'compute_default_sigma',
    'fill_position_uncertainties',
    'get_normalized_distances',
    'estimate_center',
    'estimate_theta_E',
    'create_full_numeric_points',
    'count_assumptions',
    # No-Null validation
    'NullOrNaNFoundError',
    'is_null_or_nan',
    'validate_dict_no_null',
    'validate_json_file_no_null',
    'validate_csv_file_no_null',
    'validate_run_bundle_no_null',
    'assert_no_null_no_nan',
    'summarize_provenance',
]
