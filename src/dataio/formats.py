"""
Data Format I/O for Lens Inversion

Supports:
- JSON format for image positions and parameters
- CSV format for tabular data
- Plain text for simple lists

No external dependencies beyond standard library.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def load_json(filepath: str) -> Dict:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, filepath: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=_json_serializer)


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_image_positions(filepath: str) -> np.ndarray:
    """
    Load image positions from file.
    
    Supported formats:
    - JSON with 'images' key containing [[x1,y1], [x2,y2], ...]
    - CSV with columns x, y
    - Plain text with 'x y' per line
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.json':
        data = load_json(filepath)
        if 'images' in data:
            return np.array(data['images'])
        elif 'points' in data:
            return np.array(data['points'])
        else:
            raise ValueError("JSON must have 'images' or 'points' key")
    
    elif ext == '.csv':
        return load_csv_positions(filepath)
    
    else:
        return load_text_positions(filepath)


def load_csv_positions(filepath: str) -> np.ndarray:
    """Load positions from CSV file."""
    positions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().strip().lower()
        has_header = 'x' in header or 'pos' in header
        
        if not has_header:
            f.seek(0)
        
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.replace(',', ' ').split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    positions.append([x, y])
                except ValueError:
                    continue
    
    return np.array(positions)


def load_text_positions(filepath: str) -> np.ndarray:
    """Load positions from plain text file."""
    positions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    positions.append([x, y])
                except ValueError:
                    continue
    
    return np.array(positions)


def save_image_positions(
    positions: np.ndarray,
    filepath: str,
    format: str = 'json',
    metadata: Optional[Dict] = None
) -> None:
    """
    Save image positions to file.
    
    Parameters
    ----------
    positions : ndarray
        Image positions, shape (n, 2)
    filepath : str
        Output file path
    format : str
        'json', 'csv', or 'txt'
    metadata : dict, optional
        Additional metadata to include
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    if format == 'json':
        data = {'images': positions.tolist()}
        if metadata:
            data.update(metadata)
        save_json(data, filepath)
    
    elif format == 'csv':
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("x,y\n")
            for pos in positions:
                f.write(f"{pos[0]},{pos[1]}\n")
    
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            for pos in positions:
                f.write(f"{pos[0]} {pos[1]}\n")


def save_solution(
    params: Dict[str, float],
    filepath: str,
    residuals: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None
) -> None:
    """Save inversion solution to JSON file."""
    data = {
        'parameters': params,
        'method': 'no-fit exact inversion'
    }
    
    if residuals is not None:
        data['residuals'] = {
            'values': residuals.tolist(),
            'max_abs': float(np.max(np.abs(residuals))),
            'rms': float(np.sqrt(np.mean(residuals**2)))
        }
    
    if metadata:
        data['metadata'] = metadata
    
    save_json(data, filepath)


def load_solution(filepath: str) -> Dict:
    """Load solution from JSON file."""
    data = load_json(filepath)
    
    result = {
        'params': data.get('parameters', {}),
        'method': data.get('method', 'unknown')
    }
    
    if 'residuals' in data:
        result['residuals'] = np.array(data['residuals'].get('values', []))
        result['max_residual'] = data['residuals'].get('max_abs', 0)
        result['rms_residual'] = data['residuals'].get('rms', 0)
    
    if 'metadata' in data:
        result['metadata'] = data['metadata']
    
    return result


def format_params_table(params: Dict[str, float]) -> str:
    """Format parameters as aligned table string."""
    lines = []
    max_key_len = max(len(k) for k in params.keys())
    
    for key, val in sorted(params.items()):
        if 'phi' in key:
            val_str = f"{np.degrees(val):+8.3f} deg"
        else:
            val_str = f"{val:+12.6f}"
        lines.append(f"  {key:<{max_key_len}} : {val_str}")
    
    return "\n".join(lines)


def export_for_plotting(
    images: np.ndarray,
    params: Dict[str, float],
    filepath: str
) -> None:
    """
    Export data in format suitable for plotting tools.
    
    Creates a comprehensive JSON with all data needed for visualization.
    """
    data = {
        'image_positions': {
            'x': images[:, 0].tolist(),
            'y': images[:, 1].tolist(),
            'r': np.sqrt(images[:, 0]**2 + images[:, 1]**2).tolist(),
            'phi': np.arctan2(images[:, 1], images[:, 0]).tolist()
        },
        'parameters': params,
        'derived': {}
    }
    
    if 'theta_E' in params:
        data['derived']['einstein_radius'] = params['theta_E']
    
    if 'beta_x' in params and 'beta_y' in params:
        beta = np.sqrt(params['beta_x']**2 + params['beta_y']**2)
        data['derived']['source_offset'] = beta
        data['derived']['source_angle'] = np.arctan2(
            params['beta_y'], params['beta_x']
        )
    
    save_json(data, filepath)
