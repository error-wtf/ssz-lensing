"""
Model Zoo Module - Add-only wrapper for parallel model comparison.

This module WRAPS existing solvers without modifying them.
"""

from .runner import ModelZooRunner
from .models import ModelFamily

__all__ = ['ModelZooRunner', 'ModelFamily']
