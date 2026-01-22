"""
Geometry Module - 3D Scene representation for lensing.

Add-only: Makes derivation from 3D geometry to 2D lensing visible.
"""

from .triad_scene import TriadScene, Position3D, LensPlaneSetup
from .projection import project_to_lens_plane

__all__ = ['TriadScene', 'Position3D', 'LensPlaneSetup', 'project_to_lens_plane']
