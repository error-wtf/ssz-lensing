"""Tests for fallback datasets and No-NaN Contract."""
import pytest
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.no_nan import (
    assert_finite, NaNDetectedError, sanitize_no_nan,
    validate_no_nan_in_dict, safe_divide, safe_sqrt
)
from utils.fallback_loader import (
    load_quad_images, load_ring_arc_points, load_fallback_by_mode,
    validate_all_fallback_datasets, get_fallback_text
)


class TestNoNaNUtilities:
    """Test No-NaN utility functions."""
    
    def test_assert_finite_valid(self):
        """Valid array passes."""
        arr = np.array([1.0, 2.0, 3.0])
        result = assert_finite(arr, "test")
        assert np.allclose(result, arr)
    
    def test_assert_finite_nan_raises(self):
        """NaN in array raises error."""
        arr = np.array([1.0, np.nan, 3.0])
        with pytest.raises(NaNDetectedError):
            assert_finite(arr, "test")
    
    def test_assert_finite_inf_raises(self):
        """Inf in array raises error."""
        arr = np.array([1.0, np.inf, 3.0])
        with pytest.raises(NaNDetectedError):
            assert_finite(arr, "test")
    
    def test_sanitize_no_nan_converts_nan_to_none(self):
        """NaN values become None."""
        data = {"a": 1.0, "b": float('nan'), "c": [1.0, float('nan')]}
        result = sanitize_no_nan(data)
        assert result["a"] == 1.0
        assert result["b"] is None
        assert result["c"][0] == 1.0
        assert result["c"][1] is None
    
    def test_validate_no_nan_finds_issues(self):
        """Validation finds NaN in dict."""
        data = {"a": 1.0, "b": float('nan')}
        issues = validate_no_nan_in_dict(data)
        assert len(issues) == 1
        assert "b" in issues[0]
    
    def test_safe_divide_zero(self):
        """Division by zero returns None."""
        result = safe_divide(1.0, 0.0)
        assert result is None
    
    def test_safe_divide_valid(self):
        """Valid division works."""
        result = safe_divide(4.0, 2.0)
        assert result == 2.0
    
    def test_safe_sqrt_negative(self):
        """Sqrt of negative returns None."""
        result = safe_sqrt(-1.0)
        assert result is None
    
    def test_safe_sqrt_valid(self):
        """Valid sqrt works."""
        result = safe_sqrt(4.0)
        assert result == 2.0


class TestFallbackQuad:
    """Test QUAD fallback dataset (Q2237+0305)."""
    
    def test_load_quad_images_no_nan(self):
        """QUAD images have no NaN."""
        positions, meta = load_quad_images()
        assert_finite(positions, "Q2237+0305 positions")
    
    def test_load_quad_has_4_images(self):
        """QUAD has exactly 4 images."""
        positions, meta = load_quad_images()
        assert len(positions) == 4
        assert meta["mode"] == "QUAD"
    
    def test_quad_has_redshift_info(self):
        """QUAD meta has redshift."""
        _, meta = load_quad_images()
        assert meta["z_lens"] is not None
        assert meta["z_source"] is not None
        assert meta["z_source"] > meta["z_lens"]
    
    def test_quad_positions_finite(self):
        """All QUAD positions are finite."""
        positions, _ = load_quad_images()
        assert np.all(np.isfinite(positions))


class TestFallbackRing:
    """Test RING fallback dataset (B1938+666)."""
    
    def test_load_ring_no_nan(self):
        """RING points have no NaN."""
        positions, meta = load_ring_arc_points()
        assert_finite(positions, "B1938+666 arc points")
    
    def test_ring_has_multiple_points(self):
        """RING has >4 points."""
        positions, meta = load_ring_arc_points()
        assert len(positions) >= 8
        assert meta["mode"] == "RING"
    
    def test_ring_has_redshift_info(self):
        """RING meta has redshift."""
        _, meta = load_ring_arc_points()
        assert meta["z_lens"] is not None
        assert meta["z_source"] is not None
    
    def test_ring_positions_finite(self):
        """All RING positions are finite."""
        positions, _ = load_ring_arc_points()
        assert np.all(np.isfinite(positions))


class TestFallbackByMode:
    """Test fallback selection by mode."""
    
    def test_load_quad_by_mode(self):
        """Load QUAD by mode."""
        positions, meta = load_fallback_by_mode("QUAD")
        assert len(positions) == 4
    
    def test_load_ring_by_mode(self):
        """Load RING by mode."""
        positions, meta = load_fallback_by_mode("RING")
        assert len(positions) > 4
    
    def test_invalid_mode_raises(self):
        """Invalid mode raises error."""
        with pytest.raises(ValueError):
            load_fallback_by_mode("INVALID")


class TestAllFallbackDatasets:
    """Validate all fallback datasets."""
    
    def test_all_datasets_no_nan(self):
        """All datasets pass No-NaN validation."""
        issues = validate_all_fallback_datasets()
        assert len(issues) == 0, f"NaN issues found: {issues}"
    
    def test_fallback_text_parseable(self):
        """Fallback text is parseable."""
        for mode in ["QUAD", "RING"]:
            text = get_fallback_text(mode)
            lines = text.strip().split("\n")
            for line in lines:
                parts = line.split(",")
                assert len(parts) == 2
                x, y = float(parts[0]), float(parts[1])
                assert np.isfinite(x)
                assert np.isfinite(y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
