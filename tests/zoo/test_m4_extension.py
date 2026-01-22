"""
Tests for m4 multipole extension.

Verifies:
1. m4 models work correctly
2. Derivation chain includes m4 when requested
3. Real-data pipeline with Q2237+0305
4. Artifact storage works
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from observables.bundle import ObservablesBundle, ImageSet
from observables.realdata import load_observation_pack, list_available_systems
from model_zoo.runner import ModelZooRunner
from model_zoo.models import ModelFamily, get_derivation_chain
from model_zoo.artifacts import save_run_artifacts


def gen_images_m4(theta_E=1.0, c2=0.1, s2=0.05, g1=0.0, g2=0.0,
                  c3=0.0, s3=0.0, c4=0.02, s4=-0.01, beta=(0.1, -0.05)):
    """Generate synthetic quad images including m4."""
    angles = np.array([0.3, 1.8, 3.5, 5.2])
    images = []
    for phi in angles:
        r = theta_E
        r += c2*np.cos(2*phi) + s2*np.sin(2*phi)
        r += c3*np.cos(3*phi) + s3*np.sin(3*phi)
        r += c4*np.cos(4*phi) + s4*np.sin(4*phi)
        x = r*np.cos(phi) + g1*r*np.cos(phi) + g2*r*np.sin(phi) + beta[0]
        y = r*np.sin(phi) - g1*r*np.sin(phi) + g2*r*np.cos(phi) + beta[1]
        images.append([x, y])
    return np.array(images)


class TestM4Models:
    """Test m4 multipole models."""

    def test_derivation_chain_includes_m4(self):
        """Derivation chain should include m4 when requested."""
        chain_basic = get_derivation_chain(include_m4=False)
        chain_full = get_derivation_chain(include_m4=True)
        
        assert len(chain_basic) == 4
        assert len(chain_full) == 8
        assert ModelFamily.M2_M4 in chain_full
        assert ModelFamily.M2_SHEAR_M3_M4 in chain_full

    def test_m4_data_m4_model_works(self):
        """When data has m4, m4 models should work."""
        images = gen_images_m4(c4=0.03, s4=-0.02)
        bundle = ObservablesBundle(
            name="m4_test",
            image_sets=[ImageSet(positions=images)]
        )
        runner = ModelZooRunner(bundle)
        runner.run_derivation_chain(include_m4=True)
        
        m4_result = runner.results.get(ModelFamily.M2_M4)
        assert m4_result is not None
        print(f"\nm4 model: regime={m4_result.regime}")

    def test_full_chain_report(self):
        """Full chain with m4 should generate report."""
        images = gen_images_m4()
        bundle = ObservablesBundle(
            name="full_chain",
            image_sets=[ImageSet(positions=images)]
        )
        runner = ModelZooRunner(bundle)
        report = runner.run_derivation_chain(include_m4=True)
        
        report_text = report.generate()
        print("\n" + report_text[:500] + "...")
        
        assert "m=2" in report_text
        assert "m=4" in report_text


class TestRealDataPipeline:
    """Test real-data loading and processing."""

    def test_list_available_systems(self):
        """Should list available observation packs."""
        base = os.path.join(os.path.dirname(__file__), 
                           '..', '..', 'data', 'observations')
        systems = list_available_systems(base)
        print(f"\nAvailable systems: {systems}")
        assert "Q2237+0305" in systems

    def test_load_q2237(self):
        """Should load Q2237+0305 observation pack."""
        base = os.path.join(os.path.dirname(__file__),
                           '..', '..', 'data', 'observations', 'Q2237+0305')
        bundle = load_observation_pack(base)
        
        assert bundle.name == "Q2237+0305"
        assert bundle.total_images == 4
        assert "Einstein Cross" in bundle.metadata.get("common_name", "")

    def test_q2237_derivation_chain(self):
        """Run full derivation chain on Q2237+0305."""
        base = os.path.join(os.path.dirname(__file__),
                           '..', '..', 'data', 'observations', 'Q2237+0305')
        bundle = load_observation_pack(base)
        
        runner = ModelZooRunner(bundle)
        report = runner.run_derivation_chain(include_m4=True)
        
        print("\n" + report.generate())
        
        m2_res = runner.results[ModelFamily.M2].max_residual
        assert m2_res < 1.0


class TestArtifacts:
    """Test artifact storage."""

    def test_save_artifacts(self):
        """Should save run artifacts."""
        images = gen_images_m4()
        bundle = ObservablesBundle(
            name="artifact_test",
            image_sets=[ImageSet(positions=images)]
        )
        runner = ModelZooRunner(bundle)
        report = runner.run_derivation_chain(include_m4=False)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_run_artifacts(
                report, 
                "test_system",
                base_dir=tmpdir,
                config={'test': True}
            )
            assert os.path.exists(path)
            assert os.path.exists(os.path.join(path, "report.md"))
            assert os.path.exists(os.path.join(path, "config.json"))
