"""Tests for UI state management."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.state import (
    DatasetState, RunState,
    empty_dataset_state, default_run_state,
    parse_user_points, build_user_dataset, load_fallback_dataset,
    validate_dataset, get_validation_report, get_dataset_summary
)


class TestDatasetState:
    """Test DatasetState dataclass."""

    def test_empty_state(self):
        ds = DatasetState()
        assert ds.validated is False
        assert ds.points == []
        assert ds.mode == ""

    def test_to_dict(self):
        ds = DatasetState(mode="QUAD", source="user")
        d = ds.to_dict()
        assert d["mode"] == "QUAD"
        assert d["source"] == "user"

    def test_from_dict(self):
        d = {"mode": "RING", "source": "fallback", "points": [[1, 2]],
             "point_ids": [], "unit": "arcsec", "z_lens": None,
             "z_source": None, "theta_E_arcsec": None, "center_x": 0,
             "center_y": 0, "center_known": False, "provenance": {},
             "validated": False, "errors": [], "warnings": [],
             "dataset_id": "test"}
        ds = DatasetState.from_dict(d)
        assert ds.mode == "RING"
        assert ds.points == [[1, 2]]


class TestParseUserPoints:
    """Test user input parsing."""

    def test_parse_quad(self):
        text = "0.5, 0.5\n-0.5, 0.5\n-0.5, -0.5\n0.5, -0.5"
        pts, ids, errs = parse_user_points(text, "QUAD")
        assert len(pts) == 4
        assert len(errs) == 0
        assert ids == ['A', 'B', 'C', 'D']

    def test_parse_ring(self):
        text = "\n".join(f"{i}, {i}" for i in range(10))
        pts, ids, errs = parse_user_points(text, "RING")
        assert len(pts) == 10
        assert len(errs) == 0

    def test_wrong_count_quad(self):
        text = "0.5, 0.5\n-0.5, 0.5"
        pts, ids, errs = parse_user_points(text, "QUAD")
        assert len(pts) == 2
        assert any("4 points" in e for e in errs)

    def test_invalid_line(self):
        text = "0.5, 0.5\nbad line\n-0.5, 0.5\n0.5, -0.5"
        pts, ids, errs = parse_user_points(text, "QUAD")
        assert len(errs) > 0


class TestBuildUserDataset:
    """Test building dataset from user input."""

    def test_build_quad(self):
        text = "0.5, 0.5\n-0.5, 0.5\n-0.5, -0.5\n0.5, -0.5"
        ds = build_user_dataset(text, "QUAD", "arcsec")
        assert ds.mode == "QUAD"
        assert ds.source == "user"
        assert len(ds.points) == 4

    def test_build_with_redshifts(self):
        text = "0.5, 0.5\n-0.5, 0.5\n-0.5, -0.5\n0.5, -0.5"
        ds = build_user_dataset(text, "QUAD", "arcsec", z_lens=0.5, z_source=2.0)
        assert ds.z_lens == 0.5
        assert ds.z_source == 2.0


class TestLoadFallbackDataset:
    """Test fallback dataset loading."""

    def test_load_quad(self):
        ds = load_fallback_dataset("Q2237+0305")
        assert ds.mode == "QUAD"
        assert ds.source == "fallback"
        assert len(ds.points) == 4
        assert ds.z_lens is not None

    def test_load_ring(self):
        ds = load_fallback_dataset("B1938+666")
        assert ds.mode == "RING"
        assert len(ds.points) >= 20


class TestValidateDataset:
    """Test dataset validation."""

    def test_valid_quad(self):
        ds = load_fallback_dataset("Q2237+0305")
        ds = validate_dataset(ds)
        assert ds.validated is True
        assert len(ds.errors) == 0

    def test_empty_fails(self):
        ds = DatasetState()
        ds = validate_dataset(ds)
        assert ds.validated is False
        assert "No points" in ds.errors[0]

    def test_wrong_mode_count(self):
        ds = DatasetState(mode="QUAD", points=[[1, 2], [3, 4]])
        ds = validate_dataset(ds)
        assert ds.validated is False

    def test_nan_fails(self):
        ds = DatasetState(mode="QUAD", points=[[1, 2], [3, np.nan], [5, 6], [7, 8]])
        ds = validate_dataset(ds)
        assert ds.validated is False
        assert any("NaN" in e for e in ds.errors)


class TestValidationReport:
    """Test validation report generation."""

    def test_valid_report(self):
        ds = load_fallback_dataset("Q2237+0305")
        ds = validate_dataset(ds)
        report = get_validation_report(ds)
        assert "VALID" in report
        assert "Q2237+0305" in report

    def test_invalid_report(self):
        ds = DatasetState()
        ds = validate_dataset(ds)
        report = get_validation_report(ds)
        assert "INVALID" in report


class TestDatasetSummary:
    """Test dataset summary generation."""

    def test_summary_valid(self):
        ds = load_fallback_dataset("Q2237+0305")
        ds = validate_dataset(ds)
        summary = get_dataset_summary(ds)
        assert "Q2237" in summary
        assert "QUAD" in summary

    def test_summary_invalid(self):
        ds = DatasetState()
        summary = get_dataset_summary(ds)
        assert "No active" in summary


class TestRunState:
    """Test RunState dataclass."""

    def test_default(self):
        rs = RunState()
        assert rs.distance_mode == "normalized"
        assert "m2" in rs.selected_models

    def test_to_from_dict(self):
        rs = RunState(distance_mode="direct", D_L=1.5)
        d = rs.to_dict()
        rs2 = RunState.from_dict(d)
        assert rs2.distance_mode == "direct"
        assert rs2.D_L == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
