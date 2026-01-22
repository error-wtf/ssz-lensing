"""Tests for DataHub: real-only fallback datasets."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datahub import (
    validate_snapshot,
    validate_all_snapshots,
    load_quad_snapshot,
    load_ring_snapshot,
    load_fallback_by_mode,
    list_available_datasets,
    DatasetValidationError,
)


class TestSnapshotValidation:
    """Test snapshot validation."""

    def test_quad_snapshot_valid(self):
        """Q2237+0305 snapshot passes validation."""
        valid, issues = validate_snapshot("Q2237+0305")
        assert valid, f"Issues: {issues}"

    def test_ring_snapshot_valid(self):
        """B1938+666 snapshot passes validation."""
        valid, issues = validate_snapshot("B1938+666")
        assert valid, f"Issues: {issues}"

    def test_all_snapshots_valid(self):
        """All manifest snapshots pass validation."""
        results = validate_all_snapshots()
        for ds_id, result in results.items():
            assert result["valid"], f"{ds_id} invalid: {result['issues']}"


class TestQuadSnapshot:
    """Test QUAD snapshot loading."""

    def test_load_quad_positions(self):
        """Load Q2237+0305 returns 4 positions."""
        positions, meta = load_quad_snapshot("Q2237+0305")
        assert positions.shape == (4, 2)
        assert np.all(np.isfinite(positions))

    def test_quad_has_redshifts(self):
        """QUAD meta has z_lens and z_source."""
        _, meta = load_quad_snapshot("Q2237+0305")
        assert meta['z_lens'] is not None
        assert meta['z_source'] is not None
        assert isinstance(meta['z_lens'], (int, float))
        assert isinstance(meta['z_source'], (int, float))

    def test_quad_no_nan(self):
        """QUAD positions have no NaN."""
        positions, _ = load_quad_snapshot("Q2237+0305")
        assert not np.any(np.isnan(positions))

    def test_quad_no_inf(self):
        """QUAD positions have no Inf."""
        positions, _ = load_quad_snapshot("Q2237+0305")
        assert not np.any(np.isinf(positions))

    def test_quad_has_theta_E(self):
        """QUAD meta has theta_E."""
        _, meta = load_quad_snapshot("Q2237+0305")
        assert 'theta_E_arcsec' in meta
        assert meta['theta_E_arcsec'] > 0


class TestRingSnapshot:
    """Test RING snapshot loading."""

    def test_load_ring_positions(self):
        """Load B1938+666 returns >= 20 positions."""
        positions, meta = load_ring_snapshot("B1938+666")
        assert positions.shape[0] >= 20
        assert positions.shape[1] == 2
        assert np.all(np.isfinite(positions))

    def test_ring_has_redshifts(self):
        """RING meta has z_lens and z_source."""
        _, meta = load_ring_snapshot("B1938+666")
        assert meta['z_lens'] is not None
        assert meta['z_source'] is not None

    def test_ring_no_nan(self):
        """RING positions have no NaN."""
        positions, _ = load_ring_snapshot("B1938+666")
        assert not np.any(np.isnan(positions))

    def test_ring_no_inf(self):
        """RING positions have no Inf."""
        positions, _ = load_ring_snapshot("B1938+666")
        assert not np.any(np.isinf(positions))


class TestFallbackByMode:
    """Test fallback selection by mode."""

    def test_quad_mode(self):
        """Mode QUAD returns QUAD data."""
        positions, meta = load_fallback_by_mode("QUAD")
        assert positions.shape == (4, 2)
        assert meta['mode'] == 'QUAD'

    def test_ring_mode(self):
        """Mode RING returns RING data."""
        positions, meta = load_fallback_by_mode("RING")
        assert positions.shape[0] > 4
        assert meta['mode'] == 'RING'

    def test_arc_mode(self):
        """Mode ARC returns RING data."""
        positions, meta = load_fallback_by_mode("ARC")
        assert positions.shape[0] > 4

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            load_fallback_by_mode("INVALID")


class TestDataQuality:
    """Test real data quality requirements."""

    def test_quad_all_fields_from_source(self):
        """QUAD: all values from published sources."""
        _, meta = load_quad_snapshot("Q2237+0305")
        # Check provenance
        assert meta.get('data_source') == 'datahub_snapshot'
        assert len(meta.get('primary_references', [])) > 0

    def test_ring_all_fields_from_source(self):
        """RING: all values from published sources."""
        _, meta = load_ring_snapshot("B1938+666")
        assert meta.get('data_source') == 'datahub_snapshot'
        assert len(meta.get('primary_references', [])) > 0

    def test_available_datasets(self):
        """At least 2 datasets available."""
        datasets = list_available_datasets()
        assert len(datasets) >= 2


class TestNoDefaultsNoNull:
    """Verify no defaults, no null, no NaN policy."""

    def test_quad_complete_numeric(self):
        """QUAD has complete numeric data, no placeholders."""
        positions, meta = load_quad_snapshot("Q2237+0305")
        # Positions
        assert positions.dtype in (np.float64, np.float32)
        assert np.all(np.isfinite(positions))
        # Redshifts
        assert isinstance(meta['z_lens'], (int, float))
        assert isinstance(meta['z_source'], (int, float))
        assert np.isfinite(meta['z_lens'])
        assert np.isfinite(meta['z_source'])

    def test_ring_complete_numeric(self):
        """RING has complete numeric data, no placeholders."""
        positions, meta = load_ring_snapshot("B1938+666")
        # Positions
        assert positions.dtype in (np.float64, np.float32)
        assert np.all(np.isfinite(positions))
        # Redshifts
        assert isinstance(meta['z_lens'], (int, float))
        assert isinstance(meta['z_source'], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
