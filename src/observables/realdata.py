"""
RealData Loader: Load observation packs with provenance.

Add-only design: Creates ObservablesBundle from CSV/JSON files.
"""

import json
import csv
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from .bundle import ObservablesBundle, ImageSet, FluxRatios, TimeDelays, ArcPoints


def load_observation_pack(data_dir: str) -> ObservablesBundle:
    """
    Load observation pack from directory.
    
    Expected structure:
        data_dir/
            images.csv      (required)
            meta.json       (optional)
            flux_ratios.csv (optional)
            time_delays.csv (optional)
            arc_points.csv  (optional)
    """
    data_path = Path(data_dir)
    
    # Load metadata
    meta = {}
    meta_file = data_path / "meta.json"
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            meta = json.load(f)
    
    name = meta.get("system_name", data_path.name)
    
    # Load images (required)
    images_file = data_path / "images.csv"
    if not images_file.exists():
        raise FileNotFoundError(f"images.csv not found in {data_dir}")
    
    positions = []
    with open(images_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row['x'])
            y = float(row['y'])
            positions.append([x, y])
    
    image_set = ImageSet(
        positions=np.array(positions),
        source_id=0,
        label=name
    )
    
    # Load optional flux ratios
    flux_ratios = None
    flux_file = data_path / "flux_ratios.csv"
    if flux_file.exists():
        ratios = []
        with open(flux_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ratios.append(float(row['ratio']))
        flux_ratios = FluxRatios(ratios=np.array(ratios))
    
    # Load optional time delays
    time_delays = None
    delay_file = data_path / "time_delays.csv"
    if delay_file.exists():
        delays = []
        with open(delay_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                delays.append(float(row['delay']))
        time_delays = TimeDelays(delays=np.array(delays))
    
    # Load optional arc points
    arc_points = None
    arc_file = data_path / "arc_points.csv"
    if arc_file.exists():
        pts = []
        with open(arc_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pts.append([float(row['x']), float(row['y'])])
        arc_points = ArcPoints(positions=np.array(pts))
    
    return ObservablesBundle(
        name=name,
        image_sets=[image_set],
        flux_ratios=flux_ratios,
        time_delays=time_delays,
        arc_points=arc_points,
        metadata=meta
    )


def list_available_systems(base_dir: str = "data/observations") -> list:
    """List available observation packs."""
    base = Path(base_dir)
    if not base.exists():
        return []
    return [d.name for d in base.iterdir() if d.is_dir()]
