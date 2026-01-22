"""Tests for No-Null/No-NaN Contract with provenance flags."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.defaults import (
    compute_default_sigma,
    fill_position_uncertainties,
    get_normalized_distances,
    estimate_center,
    estimate_theta_E,
    create_full_numeric_points,
    count_assumptions,
    ValueWithProvenance,
)
from utils.validate_no_null import (
    is_null_or_nan,
    validate_dict_no_null,
    assert_no_null_no_nan,
    NullOrNaNFoundError,
    summarize_provenance,
)
from utils.fallback_loader import load_quad_images, load_ring_arc_points


class TestIsNullOrNaN:
    """Test null/NaN detection."""

    def test_none_is_null(self):
        assert is_null_or_nan(None) is True

    def test_nan_is_null(self):
        assert is_null_or_nan(float('nan')) is True

    def test_inf_is_null(self):
        assert is_null_or_nan(float('inf')) is True

    def test_empty_string_is_null(self):
        assert is_null_or_nan('') is True
        assert is_null_or_nan('nan') is True
        assert is_null_or_nan('NULL') is True

    def test_valid_number_not_null(self):
        assert is_null_or_nan(0.0) is False
        assert is_null_or_nan(1.5) is False
        assert is_null_or_nan(-100) is False

    def test_valid_string_not_null(self):
        assert is_null_or_nan('hello') is False
        assert is_null_or_nan('0.5') is False


class TestDictValidation:
    """Test recursive dict validation."""

    def test_valid_dict_passes(self):
        data = {'a': 1.0, 'b': 2.0, 'nested': {'c': 3.0}}
        issues = validate_dict_no_null(data)
        assert len(issues) == 0

    def test_null_detected(self):
        data = {'a': 1.0, 'b': None}
        issues = validate_dict_no_null(data)
        assert len(issues) == 1
        assert 'b' in issues[0]

    def test_nested_null_detected(self):
        data = {'a': 1.0, 'nested': {'b': None}}
        issues = validate_dict_no_null(data)
        assert len(issues) == 1
        assert 'nested.b' in issues[0]

    def test_list_null_detected(self):
        data = {'items': [1.0, None, 3.0]}
        issues = validate_dict_no_null(data)
        assert len(issues) == 1
        assert 'items[1]' in issues[0]


class TestDefaultSigma:
    """Test default sigma computation."""

    def test_quad_sigma_positive(self):
        positions = np.array([
            [0.740, 0.565],
            [-0.635, 0.470],
            [-0.480, -0.755],
            [0.870, -0.195]
        ])
        sigma, source = compute_default_sigma(positions, "QUAD")
        assert sigma > 0
        assert np.isfinite(sigma)
        assert 'auto' in source

    def test_ring_sigma_positive(self):
        # Create ring-like points
        theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
        positions = np.column_stack([0.5*np.cos(theta), 0.5*np.sin(theta)])
        sigma, source = compute_default_sigma(positions, "RING")
        assert sigma > 0
        assert np.isfinite(sigma)

    def test_single_point_fallback(self):
        positions = np.array([[0.5, 0.5]])
        sigma, source = compute_default_sigma(positions, "QUAD")
        assert sigma > 0
        assert 'fallback' in source


class TestFillUncertainties:
    """Test uncertainty filling with provenance."""

    def test_all_defaults(self):
        positions = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]])
        sx, sy = fill_position_uncertainties(positions, None, None, "QUAD")
        assert len(sx) == 4
        assert len(sy) == 4
        for s in sx + sy:
            assert s.value > 0
            assert s.is_measured is False
            assert 'auto' in s.source

    def test_partial_input(self):
        positions = np.array([[0.5, 0.5], [0.5, -0.5]])
        sx_in = [0.01, None]
        sy_in = [0.01, None]
        sx, sy = fill_position_uncertainties(positions, sx_in, sy_in, "QUAD")
        assert sx[0].is_measured is True
        assert sx[1].is_measured is False
        assert sy[0].is_measured is True
        assert sy[1].is_measured is False


class TestFullNumericPoints:
    """Test complete numeric point creation."""

    def test_creates_all_fields(self):
        positions = np.array([
            [0.740, 0.565],
            [-0.635, 0.470],
            [-0.480, -0.755],
            [0.870, -0.195]
        ])
        points = create_full_numeric_points(positions, mode="QUAD")
        assert len(points) == 4
        for p in points:
            assert p.x is not None and np.isfinite(p.x)
            assert p.y is not None and np.isfinite(p.y)
            assert p.sx is not None and np.isfinite(p.sx) and p.sx > 0
            assert p.sy is not None and np.isfinite(p.sy) and p.sy > 0
            assert isinstance(p.x_is_measured, bool)
            assert isinstance(p.sx_is_measured, bool)
            assert isinstance(p.sx_source, str)

    def test_to_dict_no_null(self):
        positions = np.array([[0.5, 0.5], [-0.5, -0.5]])
        points = create_full_numeric_points(positions, mode="QUAD")
        for p in points:
            d = p.to_dict()
            issues = validate_dict_no_null(d)
            assert len(issues) == 0, f"Null found: {issues}"


class TestNormalizedDistances:
    """Test normalized distance defaults."""

    def test_all_values_present(self):
        dists = get_normalized_distances()
        assert 'D_L' in dists
        assert 'D_S' in dists
        assert 'D_LS' in dists
        for key, vp in dists.items():
            assert vp.value > 0
            assert vp.is_measured is False
            assert 'normalized' in vp.source


class TestEstimates:
    """Test estimation functions."""

    def test_center_estimate(self):
        positions = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
        cx, cy = estimate_center(positions)
        assert np.isfinite(cx.value)
        assert np.isfinite(cy.value)
        assert cx.is_measured is False
        assert cy.is_measured is False

    def test_theta_E_estimate(self):
        positions = np.array([[0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5]])
        theta_E = estimate_theta_E(positions)
        assert theta_E.value > 0
        assert np.isfinite(theta_E.value)
        assert theta_E.is_measured is False


class TestAssertNoNullNoNaN:
    """Test assertion function."""

    def test_valid_dict_passes(self):
        data = {'a': 1.0, 'b': [2.0, 3.0], 'c': {'d': 4.0}}
        assert_no_null_no_nan(data, "test")  # Should not raise

    def test_null_raises(self):
        data = {'a': 1.0, 'b': None}
        with pytest.raises(NullOrNaNFoundError):
            assert_no_null_no_nan(data, "test")

    def test_nan_raises(self):
        data = {'a': float('nan')}
        with pytest.raises(NullOrNaNFoundError):
            assert_no_null_no_nan(data, "test")


class TestProvenanceSummary:
    """Test provenance summary."""

    def test_counts_flags(self):
        data = {
            'x': 1.0, 'x_is_measured': True,
            'y': 2.0, 'y_is_measured': True,
            'sx': 0.01, 'sx_is_measured': False,
            'sy': 0.01, 'sy_is_measured': False,
        }
        summary = summarize_provenance(data)
        assert summary['measured'] == 2
        assert summary['assumed'] == 2
        assert summary['total'] == 4


class TestFallbackDatasetsComplete:
    """Test fallback datasets are 100% complete (no nulls)."""

    def test_quad_fallback_complete(self):
        positions, meta = load_quad_images()
        # All positions must be finite
        assert np.all(np.isfinite(positions))
        # Meta should have redshifts
        assert meta['z_lens'] is not None and np.isfinite(meta['z_lens'])
        assert meta['z_source'] is not None and np.isfinite(meta['z_source'])

    def test_ring_fallback_complete(self):
        positions, meta = load_ring_arc_points()
        # All positions must be finite
        assert np.all(np.isfinite(positions))
        # Meta should have redshifts
        assert meta['z_lens'] is not None and np.isfinite(meta['z_lens'])
        assert meta['z_source'] is not None and np.isfinite(meta['z_source'])

    def test_quad_full_numeric_output(self):
        """User minimal input (just positions) still produces full numeric."""
        positions, _ = load_quad_images()
        # Create full numeric points with no sx/sy input
        points = create_full_numeric_points(positions, mode="QUAD")
        for p in points:
            d = p.to_dict()
            issues = validate_dict_no_null(d)
            assert len(issues) == 0

    def test_ring_full_numeric_output(self):
        """Ring fallback produces full numeric."""
        positions, _ = load_ring_arc_points()
        points = create_full_numeric_points(positions, mode="RING")
        for p in points:
            d = p.to_dict()
            issues = validate_dict_no_null(d)
            assert len(issues) == 0


class TestUserMinimalInput:
    """Test pipeline with minimal user input produces full numeric."""

    def test_4_points_no_uncertainties(self):
        """Just 4 positions, nothing else - must produce complete output."""
        positions = np.array([
            [0.740, 0.565],
            [-0.635, 0.470],
            [-0.480, -0.755],
            [0.870, -0.195]
        ])
        # Get full numeric points
        points = create_full_numeric_points(positions, mode="QUAD")
        # Get distances
        dists = get_normalized_distances()
        # Get estimates
        cx, cy = estimate_center(positions)
        theta_E = estimate_theta_E(positions)

        # Build complete output dict
        output = {
            'points': [p.to_dict() for p in points],
            'distances': {k: v.to_dict() for k, v in dists.items()},
            'center': {'x': cx.to_dict(), 'y': cy.to_dict()},
            'theta_E': theta_E.to_dict(),
        }

        # Validate no nulls
        issues = validate_dict_no_null(output)
        assert len(issues) == 0, f"Nulls found: {issues}"

        # Count assumptions
        assumptions = count_assumptions(points)
        assert assumptions['total_points'] == 4
        # sx/sy should be assumed for all 4
        assert assumptions['sx_assumed'] == 4
        assert assumptions['sy_assumed'] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
