"""
CasePipeline: Orchestrates the full analysis workflow.

Stages:
1. load_data - Load observation pack
2. validate - Check data schema
3. select_models - Choose which models to run
4. invert - Run Model Zoo
5. predict - Generate predictions (if withheld data)
6. visualize - Create all plots
7. export - Save artifacts

Authors: Carmen N. Wrede, Lino P. Casu
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from observables.realdata import load_observation_pack
from observables.bundle import ObservablesBundle
from model_zoo.runner import ModelZooRunner
from model_zoo.models import ModelFamily, MODEL_CONFIGS
from model_zoo.artifacts import save_run_artifacts
from geometry.triad_scene import TriadScene
from geometry.visualization import SceneVisualizer
from geometry.plots import GeometryPlotter, HAS_MATPLOTLIB


class CasePipeline:
    """
    Full analysis pipeline for a lensing system.
    
    Usage:
        pipeline = CasePipeline("Q2237+0305", "data/observations/Q2237+0305")
        result = pipeline.run()
    """
    
    def __init__(self, system_name: str, data_path: str,
                 include_m4: bool = False, withhold: Optional[str] = None):
        self.system_name = system_name
        self.data_path = Path(data_path)
        self.include_m4 = include_m4
        self.withhold = withhold
        
        self.bundle: Optional[ObservablesBundle] = None
        self.runner: Optional[ModelZooRunner] = None
        self.scene: Optional[TriadScene] = None
        self.run_dir: Optional[Path] = None
        
        self.log: List[str] = []
    
    def run(self) -> Dict[str, Any]:
        """Run full pipeline."""
        self._log("Starting pipeline")
        
        # Stage 1: Load data
        self._log("Stage 1: Loading data")
        self.bundle = load_observation_pack(str(self.data_path))
        self._log(f"  Loaded {self.bundle.total_images} images")
        
        # Stage 2: Validate
        self._log("Stage 2: Validating")
        self._validate()
        
        # Stage 3: Create scene
        self._log("Stage 3: Creating 3D scene")
        self._create_scene()
        
        # Stage 4: Run Model Zoo
        self._log("Stage 4: Running Model Zoo")
        self.runner = ModelZooRunner(self.bundle)
        report = self.runner.run_derivation_chain(include_m4=self.include_m4)
        n_models = len([r for r in self.runner.results.values() if r.success])
        self._log(f"  Ran {n_models} models successfully")
        
        # Stage 5: Find best model
        self._log("Stage 5: Analyzing results")
        best_family, best_result = self._find_best()
        self._log(f"  Best: {MODEL_CONFIGS[best_family].label}")
        self._log(f"  Residual: {best_result.max_residual:.4e}")
        
        # Stage 6: Create run directory
        self._log("Stage 6: Saving artifacts")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path("runs") / f"{timestamp}_{self.system_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save artifacts
        save_run_artifacts(
            report, self.system_name, 
            base_dir=str(self.run_dir.parent),
            config={
                'include_m4': self.include_m4,
                'withhold': self.withhold,
                'data_path': str(self.data_path)
            }
        )
        
        # Stage 7: Visualize
        self._log("Stage 7: Generating visualizations")
        self._visualize()
        
        # Stage 8: Save scene
        self._log("Stage 8: Saving scene")
        if self.scene:
            self.scene.save(str(self.run_dir / "scene.json"))
        
        # Save log
        with open(self.run_dir / "pipeline.log", 'w') as f:
            f.write("\n".join(self.log))
        
        self._log("Pipeline complete!")
        
        return {
            'run_dir': str(self.run_dir),
            'n_models': n_models,
            'best_model': MODEL_CONFIGS[best_family].label,
            'best_residual': best_result.max_residual
        }
    
    def _log(self, msg: str):
        """Add to log."""
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.log.append(entry)
        print(entry)
    
    def _validate(self):
        """Validate data."""
        if self.bundle.total_images < 2:
            raise ValueError("Need at least 2 images")
        self._log(f"  Constraints: {self.bundle.count_constraints()}")
    
    def _create_scene(self):
        """Create 3D scene from bundle."""
        images = self.bundle.get_primary_images()
        if len(images) == 0:
            return
        
        # Estimate beta from image centroid
        centroid = images.mean(axis=0)
        beta_x = centroid[0] * 0.1
        beta_y = centroid[1] * 0.1
        
        self.scene = TriadScene.create_standard(
            name=self.system_name,
            D_L=1.0, D_S=2.0,
            beta_x=beta_x, beta_y=beta_y,
            theta_E=1.0
        )
        self._log(f"  Scene: O(0,0,0) -> L(0,0,1) -> S({beta_x:.3f},{beta_y:.3f},2)")
    
    def _find_best(self):
        """Find best model by residual."""
        best = None
        best_res = float('inf')
        
        for family, result in self.runner.results.items():
            if result.success and result.max_residual < best_res:
                best = family
                best_res = result.max_residual
        
        return best, self.runner.results[best]
    
    def _visualize(self):
        """Generate visualizations."""
        if self.scene is None:
            return
        
        images = self.bundle.get_primary_images()
        best_family, best_result = self._find_best()
        params = best_result.params or {}
        
        fig_dir = self.run_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # ASCII/JSON visualizations (always)
        viz = SceneVisualizer(self.scene)
        viz.generate_all(str(fig_dir), images=images, params=params)
        
        # Matplotlib plots (if available)
        if HAS_MATPLOTLIB:
            plotter = GeometryPlotter(self.scene)
            plotter.generate_all(str(fig_dir), images=images, params=params)
            self._log(f"  Generated PNG plots in {fig_dir}")
        else:
            self._log(f"  Matplotlib not available, ASCII only")
        
        self._log(f"  Saved figures to {fig_dir}")
