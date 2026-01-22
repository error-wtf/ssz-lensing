"""
Tests for Model Zoo: Parallel model comparison and extended observables.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.model_zoo import (
    ModelType, MODEL_SPECS, LensSystem, RegimeGate, ModelZoo
)


def gen_images(theta_E=1.0, c_2=0.1, s_2=0.05, gamma_1=0.0, gamma_2=0.0,
               c_3=0.0, s_3=0.0, beta=(0.1, -0.05), n=4):
    """Generate synthetic quad images."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + 0.1
    images = []
    for phi in angles:
        r = theta_E + c_2*np.cos(2*phi) + s_2*np.sin(2*phi)
        r += c_3*np.cos(3*phi) + s_3*np.sin(3*phi)
        x = r*np.cos(phi) + gamma_1*r*np.cos(phi) + gamma_2*r*np.sin(phi) + beta[0]
        y = r*np.sin(phi) - gamma_1*r*np.sin(phi) + gamma_2*r*np.cos(phi) + beta[1]
        images.append([x, y])
    return np.array(images)


def test_m2_allowed():
    """m=2 allowed with 4 images."""
    system = LensSystem(name="t", images=[gen_images()])
    status = RegimeGate.check(MODEL_SPECS[ModelType.M2], system)
    assert status.allowed
    assert status.n_constraints == 8


def test_m2_shear_m3_forbidden():
    """m=2+shear+m=3 FORBIDDEN with only 4 images."""
    system = LensSystem(name="t", images=[gen_images()])
    status = RegimeGate.check(MODEL_SPECS[ModelType.M2_SHEAR_M3], system)
    assert not status.allowed
    assert status.regime == "FORBIDDEN"
    assert len(status.missing_info) > 0


def test_arc_points_rescue():
    """Arc points rescue FORBIDDEN model."""
    system = LensSystem(name="t", images=[gen_images()],
                        arc_points=np.array([[1.0, 0.1], [0.9, 0.2], [1.1, 0.15]]))
    status = RegimeGate.check(MODEL_SPECS[ModelType.M2_SHEAR_M3], system)
    assert status.allowed


def test_multi_source_rescue():
    """Two sources rescue FORBIDDEN model."""
    system = LensSystem(name="t", images=[gen_images(), gen_images(beta=(-0.1, 0.1))],
                        n_sources=2)
    status = RegimeGate.check(MODEL_SPECS[ModelType.M2_SHEAR_M3], system)
    assert status.allowed


def test_shear_recovery():
    """Recover shear parameters."""
    images = gen_images(gamma_1=0.03, gamma_2=-0.02)
    system = LensSystem(name="t", images=[images])
    zoo = ModelZoo(system)
    results = zoo.run_all()
    assert results[ModelType.M2_SHEAR].success


def test_m3_recovery():
    """Recover m=3 parameters."""
    images = gen_images(c_3=0.02, s_3=-0.015)
    system = LensSystem(name="t", images=[images])
    zoo = ModelZoo(system)
    results = zoo.run_all()
    assert results[ModelType.M2_M3].success


def test_zoo_comparison():
    """Zoo runs all models and compares."""
    system = LensSystem(name="t", images=[gen_images()])
    zoo = ModelZoo(system)
    results = zoo.run_all()
    report = zoo.compare()
    assert "m=2" in report
    assert "FORBIDDEN" in report
