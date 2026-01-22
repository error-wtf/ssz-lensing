"""
Data I/O for Gauge Gravitational Lensing

Modules:
- formats: JSON, CSV loaders/savers
- datasets: Standard dataset generators and loaders
"""

from .formats import (
    load_json, save_json,
    load_image_positions, save_image_positions,
    save_solution, load_solution
)
from .datasets import (
    generate_cross_images,
    generate_ring_images,
    generate_multipole_images,
    einstein_cross_q2237,
    standard_test_cases
)

__all__ = [
    'load_json', 'save_json',
    'load_image_positions', 'save_image_positions',
    'save_solution', 'load_solution',
    'generate_cross_images', 'generate_ring_images',
    'generate_multipole_images', 'einstein_cross_q2237',
    'standard_test_cases'
]
