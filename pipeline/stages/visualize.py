"""
Visualization Stage: Generate all standard plots.

Outputs:
1. Image plane plot (observed vs predicted)
2. Source plane plot (beta cluster)
3. Model Zoo comparison
4. 3D scene diagram
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


def generate_all_plots(run_dir: Path, output_dir: Path):
    """Generate all visualizations from a completed run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load run data
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    
    # Generate model comparison
    _generate_model_comparison(run_dir, output_dir)
    
    # Generate summary
    _generate_summary_txt(run_dir, output_dir)


def _generate_model_comparison(run_dir: Path, output_dir: Path):
    """Generate model comparison table."""
    lines = ["MODEL COMPARISON", "=" * 50, ""]
    
    for model_dir in run_dir.iterdir():
        if not model_dir.is_dir():
            continue
        sol_file = model_dir / "solution.json"
        if not sol_file.exists():
            continue
        
        with open(sol_file) as f:
            sol = json.load(f)
        
        name = sol.get('family', model_dir.name)
        regime = sol.get('regime', 'UNKNOWN')
        success = sol.get('success', False)
        residual = sol.get('max_residual', float('inf'))
        
        status = "OK" if success else regime
        res_str = f"{residual:.4e}" if success else "N/A"
        
        lines.append(f"{name:20} | {status:15} | {res_str}")
    
    with open(output_dir / "model_comparison.txt", 'w') as f:
        f.write("\n".join(lines))


def _generate_summary_txt(run_dir: Path, output_dir: Path):
    """Generate text summary."""
    report_path = run_dir / "report.md"
    if report_path.exists():
        with open(report_path) as f:
            content = f.read()
        with open(output_dir / "summary.txt", 'w') as f:
            f.write(content)
