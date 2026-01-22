"""DataHub: real-only fallback datasets with validation."""

from .validate import (
    DatasetValidationError,
    validate_snapshot,
    validate_all_snapshots,
    assert_snapshot_valid,
)

from .loader import (
    load_quad_snapshot,
    load_ring_snapshot,
    load_fallback_by_mode,
    get_fallback_text,
    list_available_datasets,
)

__all__ = [
    'DatasetValidationError',
    'validate_snapshot',
    'validate_all_snapshots',
    'assert_snapshot_valid',
    'load_quad_snapshot',
    'load_ring_snapshot',
    'load_fallback_by_mode',
    'get_fallback_text',
    'list_available_datasets',
]
