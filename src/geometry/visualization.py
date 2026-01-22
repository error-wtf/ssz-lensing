"""
Geometry Visualization: 3D Scene, Lens/Source Planes, Ray Bundles.

Add-only: Creates visual representations of the derivation chain.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path

from .triad_scene import TriadScene, LensPlaneSetup
from .projection import project_to_lens_plane, backproject_to_source


class SceneVisualizer:
    """
    Visualize the 3D lensing scene and projections.
    
    Outputs:
    1. 3D scene view (O, L, S, optical axis)
    2. Lens plane + source plane plots
    3. Ray bundle diagram
    """
    
    def __init__(self, scene: TriadScene):
        self.scene = scene
        self.setups = project_to_lens_plane(scene)
    
    def generate_all(self, output_dir: str, 
                     images: Optional[np.ndarray] = None,
                     params: Optional[Dict[str, float]] = None):
        """Generate all visualizations."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            'scene_3d': self._scene_3d_data(),
            'lens_plane': self._lens_plane_data(images),
            'source_plane': self._source_plane_data(images, params),
            'ray_bundle': self._ray_bundle_data(images)
        }
        
        # Save as JSON for external plotting
        import json
        with open(out_path / "visualization_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=self._json_default)
        
        # Generate ASCII representations
        self._save_ascii_scene(out_path / "scene_ascii.txt")
        
        return data
    
    def _json_default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        return str(obj)
    
    def _scene_3d_data(self) -> Dict[str, Any]:
        """Data for 3D scene plot."""
        return {
            'observer': {'pos': [0, 0, 0], 'label': 'Observer'},
            'lens': {
                'pos': self.scene.lens.position.to_array().tolist(),
                'label': 'Lens',
                'theta_E': self.scene.lens.einstein_radius
            },
            'sources': [
                {
                    'pos': s.position.to_array().tolist(),
                    'label': f'Source_{s.source_id}'
                }
                for s in self.scene.sources
            ],
            'optical_axis': {
                'start': [0, 0, 0],
                'end': self.scene.sources[0].position.to_array().tolist()
                       if self.scene.sources else [0, 0, 2]
            }
        }
    
    def _lens_plane_data(self, images: Optional[np.ndarray]) -> Dict[str, Any]:
        """Data for lens plane plot."""
        data = {
            'theta_E': self.scene.lens.einstein_radius,
            'center': [0, 0]
        }
        if images is not None:
            data['observed_images'] = images.tolist()
        return data
    
    def _source_plane_data(self, images: Optional[np.ndarray],
                           params: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Data for source plane plot with back-projected betas."""
        data = {'true_beta': []}
        
        for setup in self.setups:
            data['true_beta'].append(setup.beta.tolist())
        
        if images is not None and params is not None and self.setups:
            betas = backproject_to_source(images, self.setups[0], params)
            data['backprojected_betas'] = betas.tolist()
            
            # Consistency check: how tight is the beta cluster?
            spread = np.std(betas, axis=0)
            data['beta_spread'] = spread.tolist()
        
        return data
    
    def _ray_bundle_data(self, images: Optional[np.ndarray]) -> Dict[str, Any]:
        """Data for ray bundle visualization."""
        D_L = self.scene.lens.position.z
        D_S = self.scene.sources[0].position.z if self.scene.sources else 2.0
        
        rays = []
        if images is not None:
            for i, img in enumerate(images):
                ray = {
                    'image_id': i,
                    'segments': [
                        {'from': [0, 0, 0], 'to': [img[0]*D_L, img[1]*D_L, D_L]},
                        {'from': [img[0]*D_L, img[1]*D_L, D_L], 
                         'to': [img[0]*D_S, img[1]*D_S, D_S]}
                    ]
                }
                rays.append(ray)
        
        return {
            'D_L': D_L,
            'D_S': D_S,
            'rays': rays
        }
    
    def _save_ascii_scene(self, filepath: Path):
        """Save ASCII art representation of scene."""
        lines = [
            "=" * 50,
            f"SCENE: {self.scene.name}",
            "=" * 50,
            "",
            "Observer [O] -----> Lens [L] -----> Source(s) [S]",
            f"         z=0        z={self.scene.lens.position.z:.2f}",
            "",
            "3D Positions:",
            f"  Observer: (0, 0, 0)",
            f"  Lens:     ({self.scene.lens.position.x:.3f}, "
            f"{self.scene.lens.position.y:.3f}, "
            f"{self.scene.lens.position.z:.3f})",
        ]
        
        for src in self.scene.sources:
            lines.append(
                f"  Source_{src.source_id}: ({src.position.x:.3f}, "
                f"{src.position.y:.3f}, {src.position.z:.3f})"
            )
        
        lines.extend([
            "",
            "Projected to Lens Plane:",
        ])
        
        for setup in self.setups:
            lines.append(
                f"  Source {setup.source_id}: beta = "
                f"({setup.beta[0]:.4f}, {setup.beta[1]:.4f})"
            )
        
        lines.extend([
            "",
            "Distances:",
            f"  D_L  = {self.scene.lens.position.z:.3f}",
        ])
        
        for setup in self.setups:
            lines.append(f"  D_S  = {setup.D_S:.3f}")
            lines.append(f"  D_LS = {setup.D_LS:.3f}")
        
        with open(filepath, 'w') as f:
            f.write("\n".join(lines))
