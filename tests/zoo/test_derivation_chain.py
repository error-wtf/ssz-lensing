"""
Didactic Tests: Show the derivation chain and when extra observables are needed.

These tests are designed to DOCUMENT the stepwise improvement,
not just pass/fail. They show WHY each model extension helps.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from observables.bundle import ObservablesBundle, ImageSet, ArcPoints
from model_zoo.runner import ModelZooRunner
from model_zoo.models import ModelFamily


def gen_images(theta_E=1.0, c2=0.1, s2=0.05, g1=0.0, g2=0.0,
               c3=0.0, s3=0.0, beta=(0.1, -0.05)):
    """Generate synthetic quad images."""
    angles = np.array([0.3, 1.8, 3.5, 5.2])
    images = []
    for phi in angles:
        r = theta_E + c2*np.cos(2*phi) + s2*np.sin(2*phi)
        r += c3*np.cos(3*phi) + s3*np.sin(3*phi)
        x = r*np.cos(phi) + g1*r*np.cos(phi) + g2*r*np.sin(phi) + beta[0]
        y = r*np.sin(phi) - g1*r*np.sin(phi) + g2*r*np.cos(phi) + beta[1]
        images.append([x, y])
    return np.array(images)


class TestDerivationChain:
    """Test A: Herleitungskette - shows stepwise improvement."""

    def test_shear_data_shear_wins(self):
        """When data has shear, m=2+shear should beat m=2 alone."""
        images = gen_images(g1=0.04, g2=-0.03, c3=0.0, s3=0.0)
        bundle = ObservablesBundle(
            name="shear_test",
            image_sets=[ImageSet(positions=images)]
        )
        runner = ModelZooRunner(bundle)
        report = runner.run_derivation_chain()

        m2_res = runner.results[ModelFamily.M2].max_residual
        shear_res = runner.results[ModelFamily.M2_SHEAR].max_residual

        print(f"\nShear data: m=2 residual={m2_res:.4f}, "
              f"m=2+shear residual={shear_res:.4f}")

        assert shear_res <= m2_res * 1.1

    def test_m3_data_m3_wins(self):
        """When data has m=3, m=2+m=3 should beat m=2 alone."""
        images = gen_images(g1=0.0, g2=0.0, c3=0.03, s3=-0.02)
        bundle = ObservablesBundle(
            name="m3_test",
            image_sets=[ImageSet(positions=images)]
        )
        runner = ModelZooRunner(bundle)
        runner.run_derivation_chain()

        m2_res = runner.results[ModelFamily.M2].max_residual
        m3_res = runner.results[ModelFamily.M2_M3].max_residual

        print(f"\nm=3 data: m=2 residual={m2_res:.4f}, "
              f"m=2+m=3 residual={m3_res:.4f}")

        assert m3_res <= m2_res * 1.1

    def test_full_model_forbidden_without_extras(self):
        """m=2+shear+m=3 should be FORBIDDEN with only 4 images."""
        images = gen_images(g1=0.03, g2=-0.02, c3=0.02, s3=-0.01)
        bundle = ObservablesBundle(
            name="full_test",
            image_sets=[ImageSet(positions=images)]
        )
        runner = ModelZooRunner(bundle)
        runner.run_derivation_chain()

        full_result = runner.results[ModelFamily.M2_SHEAR_M3]

        print(f"\nFull model: regime={full_result.regime}")

        assert full_result.regime == "FORBIDDEN"
        assert not full_result.success

    def test_report_shows_derivation(self):
        """Report should show the derivation chain clearly."""
        images = gen_images()
        bundle = ObservablesBundle(
            name="report_test",
            image_sets=[ImageSet(positions=images)]
        )
        runner = ModelZooRunner(bundle)
        report = runner.run_derivation_chain()

        report_text = report.generate()
        print("\n" + report_text)

        assert "Step 1" in report_text
        assert "m=2" in report_text
        assert "FORBIDDEN" in report_text


class TestForbiddenToAllowed:
    """Test B: FORBIDDEN becomes ALLOWED with extra observables."""

    def test_arc_points_rescue_full_model(self):
        """Arc points should rescue the full model."""
        images = gen_images(g1=0.03, g2=-0.02, c3=0.02, s3=-0.01)

        bundle_basic = ObservablesBundle(
            name="basic",
            image_sets=[ImageSet(positions=images)]
        )
        runner1 = ModelZooRunner(bundle_basic)
        runner1.run_derivation_chain()
        assert runner1.results[ModelFamily.M2_SHEAR_M3].regime == "FORBIDDEN"

        arc_pts = np.array([[1.1, 0.15], [0.95, 0.25], [1.05, 0.1]])
        bundle_extended = ObservablesBundle(
            name="extended",
            image_sets=[ImageSet(positions=images)],
            arc_points=ArcPoints(positions=arc_pts)
        )
        runner2 = ModelZooRunner(bundle_extended)
        runner2.run_derivation_chain()

        result = runner2.results[ModelFamily.M2_SHEAR_M3]
        print(f"\nWith arc points: regime={result.regime}, "
              f"constraints={result.n_constraints}")

        assert result.n_constraints > 8

    def test_multi_source_rescue_full_model(self):
        """Two sources should rescue the full model."""
        images1 = gen_images(beta=(0.1, -0.05))
        images2 = gen_images(beta=(-0.08, 0.12))

        bundle = ObservablesBundle(
            name="multi_source",
            image_sets=[
                ImageSet(positions=images1, source_id=0),
                ImageSet(positions=images2, source_id=1)
            ]
        )
        runner = ModelZooRunner(bundle)
        runner.run_derivation_chain()

        result = runner.results[ModelFamily.M2_SHEAR_M3]
        print(f"\nWith 2 sources: constraints={result.n_constraints}, "
              f"params={result.n_params}")

        assert result.n_constraints == 16


class TestRegression:
    """Test C: Existing tests must still pass."""

    def test_basic_m2_still_works(self):
        """Basic m=2 model should still work as before."""
        images = gen_images()
        bundle = ObservablesBundle.from_positions("basic", images)
        runner = ModelZooRunner(bundle)
        runner.run_derivation_chain()

        result = runner.results[ModelFamily.M2]
        assert result.success
        assert result.max_residual < 1.0

    def test_bundle_backward_compatible(self):
        """ObservablesBundle.from_positions should work like old API."""
        images = np.array([[1.0, 0.1], [-0.9, 0.2], [0.1, 1.0], [-0.2, -0.95]])
        bundle = ObservablesBundle.from_positions("compat", images)

        assert bundle.n_sources == 1
        assert bundle.total_images == 4
        assert np.array_equal(bundle.get_primary_images(), images)
