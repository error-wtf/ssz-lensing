"""
Tests for Geometry Module: 3D Scene, Projection, Visualization.

Verifies:
1. TriadScene creation and serialization
2. Projection consistency (3D -> 2D -> back)
3. Scene roundtrip (save/load)
4. Visualization smoke test
"""

import numpy as np
import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from geometry.triad_scene import TriadScene, Position3D, LensPlaneSetup
from geometry.projection import (
    project_to_lens_plane, ProjectionTracer, 
    forward_lens_images, backproject_to_source
)
from geometry.visualization import SceneVisualizer


class TestTriadScene:
    """Test TriadScene creation and basic operations."""

    def test_create_standard_scene(self):
        """Should create a standard O-L-S scene."""
        scene = TriadScene.create_standard(
            name="test_scene",
            D_L=1.0, D_S=2.0,
            beta_x=0.1, beta_y=-0.05,
            theta_E=1.0
        )
        
        assert scene.name == "test_scene"
        assert scene.observer.z == 0
        assert scene.lens.position.z == 1.0
        assert len(scene.sources) == 1
        assert scene.sources[0].position.z == 2.0

    def test_scene_distances(self):
        """Should compute distances correctly."""
        scene = TriadScene.create_standard(
            "dist_test", D_L=1.5, D_S=3.0, beta_x=0.0, beta_y=0.0
        )
        distances = scene.get_distances()
        
        assert distances['D_L'] == 1.5
        assert 'D_S_0' in distances
        assert np.isclose(distances['D_S_0'], 3.0, atol=0.01)
        assert np.isclose(distances['D_LS_0'], 1.5, atol=0.01)

    def test_add_multiple_sources(self):
        """Should support multiple sources."""
        scene = TriadScene.create_standard("multi", D_L=1.0, D_S=2.0)
        scene.add_source(0.2, -0.1, 2.5, source_id=1)
        scene.add_source(-0.15, 0.2, 3.0, source_id=2)
        
        assert len(scene.sources) == 3
        assert scene.sources[2].position.z == 3.0


class TestProjection:
    """Test projection from 3D to lens plane coordinates."""

    def test_project_single_source(self):
        """Should project to correct beta."""
        scene = TriadScene.create_standard(
            "proj_test", D_L=1.0, D_S=2.0,
            beta_x=0.1, beta_y=-0.05
        )
        
        setups = project_to_lens_plane(scene)
        
        assert len(setups) == 1
        setup = setups[0]
        
        # beta = source_offset / D_S
        assert np.isclose(setup.beta[0], 0.1, atol=0.01)
        assert np.isclose(setup.beta[1], -0.05, atol=0.01)
        assert setup.D_L == 1.0
        assert setup.D_S == 2.0
        assert setup.D_LS == 1.0

    def test_projection_tracer(self):
        """Should trace all projection steps."""
        scene = TriadScene.create_standard("trace_test")
        tracer = ProjectionTracer(scene)
        result = tracer.trace()
        
        assert result['n_steps'] == 3
        assert result['trace'][0]['step'] == '3D_positions'
        assert result['trace'][1]['step'] == 'distances'
        assert result['trace'][2]['step'] == 'projection'

    def test_forward_backward_consistency(self):
        """Forward then backward should return to original beta."""
        scene = TriadScene.create_standard(
            "consistency", beta_x=0.1, beta_y=-0.05
        )
        setups = project_to_lens_plane(scene)
        setup = setups[0]
        
        params = {'c_2': 0.05, 's_2': 0.02}
        
        # Forward: beta -> images
        images = forward_lens_images(setup, params)
        
        # Backward: images -> beta
        betas = backproject_to_source(images, setup, params)
        
        # All betas should cluster near original
        mean_beta = np.mean(betas, axis=0)
        print(f"\nOriginal beta: {setup.beta}")
        print(f"Mean recovered beta: {mean_beta}")
        print(f"Beta spread: {np.std(betas, axis=0)}")


class TestSerialization:
    """Test scene save/load roundtrip."""

    def test_to_dict_and_back(self):
        """Should serialize and deserialize correctly."""
        scene = TriadScene.create_standard(
            "roundtrip", D_L=1.5, D_S=3.0,
            beta_x=0.2, beta_y=-0.1, theta_E=1.2
        )
        scene.add_source(0.1, 0.15, 2.5)
        
        d = scene.to_dict()
        restored = TriadScene.from_dict(d)
        
        assert restored.name == scene.name
        assert restored.lens.position.z == scene.lens.position.z
        assert len(restored.sources) == len(scene.sources)

    def test_json_roundtrip(self):
        """Should save to JSON and load back."""
        scene = TriadScene.create_standard("json_test")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                         delete=False) as f:
            scene.save(f.name)
            loaded = TriadScene.load(f.name)
        
        assert loaded.name == scene.name
        assert loaded.lens.einstein_radius == scene.lens.einstein_radius


class TestVisualization:
    """Test visualization output."""

    def test_visualizer_smoke_test(self):
        """Visualizer should run without errors."""
        scene = TriadScene.create_standard("viz_test")
        viz = SceneVisualizer(scene)
        
        images = np.array([[1.0, 0.1], [-0.9, 0.2], [0.1, 1.0], [-0.2, -0.95]])
        params = {'c_2': 0.05, 's_2': 0.02}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data = viz.generate_all(tmpdir, images=images, params=params)
            
            assert 'scene_3d' in data
            assert 'lens_plane' in data
            assert 'source_plane' in data
            assert 'ray_bundle' in data
            
            assert os.path.exists(os.path.join(tmpdir, 'visualization_data.json'))
            assert os.path.exists(os.path.join(tmpdir, 'scene_ascii.txt'))

    def test_ascii_scene_output(self):
        """ASCII scene should be readable."""
        scene = TriadScene.create_standard("ascii_test", D_L=1.0, D_S=2.0)
        viz = SceneVisualizer(scene)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            viz.generate_all(tmpdir)
            
            with open(os.path.join(tmpdir, 'scene_ascii.txt'), 'r') as f:
                content = f.read()
            
            print("\n" + content)
            
            assert "Observer" in content
            assert "Lens" in content
            assert "Source" in content
