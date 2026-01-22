"""
Artifact Storage: Save reproducible run bundles.

Each run saves:
    runs/<timestamp>_<system>/<model>/
        solution.json
        predictions.json
        residuals.csv
        config.json
        report.md
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from .models import ModelFamily, MODEL_CONFIGS
from .runner import RunResult, DerivationReport


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def save_run_artifacts(
    report: DerivationReport,
    system_name: str,
    base_dir: str = "runs",
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save complete run as artifact bundle.
    
    Returns path to saved artifacts.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{timestamp}_{system_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    if config is None:
        config = {}
    config['timestamp'] = timestamp
    config['system_name'] = system_name
    
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder)
    
    # Save each model result
    for family, result in report.results.items():
        model_dir = run_dir / family.value
        model_dir.mkdir(exist_ok=True)
        
        _save_model_result(result, model_dir)
    
    # Save summary report
    with open(run_dir / "report.md", 'w') as f:
        f.write(report.generate())
    
    return str(run_dir)


def _save_model_result(result: RunResult, model_dir: Path):
    """Save single model result."""
    # Solution
    solution = {
        'family': result.family.value,
        'regime': result.regime,
        'success': result.success,
        'n_constraints': result.n_constraints,
        'n_params': result.n_params,
        'max_residual': result.max_residual,
    }
    if result.params:
        solution['params'] = result.params
    if result.notes:
        solution['notes'] = result.notes
    
    with open(model_dir / "solution.json", 'w') as f:
        json.dump(solution, f, indent=2, cls=NumpyEncoder)
    
    # Residuals
    if result.residuals is not None:
        with open(model_dir / "residuals.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'residual'])
            for i, r in enumerate(result.residuals):
                writer.writerow([i, r])


def load_run_artifacts(run_dir: str) -> Dict[str, Any]:
    """Load previously saved run artifacts."""
    run_path = Path(run_dir)
    
    with open(run_path / "config.json", 'r') as f:
        config = json.load(f)
    
    with open(run_path / "report.md", 'r') as f:
        report_text = f.read()
    
    models = {}
    for model_dir in run_path.iterdir():
        if model_dir.is_dir():
            sol_file = model_dir / "solution.json"
            if sol_file.exists():
                with open(sol_file, 'r') as f:
                    models[model_dir.name] = json.load(f)
    
    return {
        'config': config,
        'report': report_text,
        'models': models
    }
