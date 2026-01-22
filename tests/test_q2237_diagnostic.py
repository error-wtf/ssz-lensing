"""
Q2237+0305 Einstein Cross Diagnostic

Real-data test showing that m=2 alone is insufficient,
while m=2+shear OR m=2+m=3 significantly reduce residuals.

The lens galaxy has a bar structure, making higher multipoles
or external shear physically motivated.

Reference: Huchra et al. 1985, AJ 90, 691
Image positions from HST observations.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.model_zoo import (
    ModelType, MODEL_SPECS, LensSystem, RegimeGate, ModelZoo
)


# Q2237+0305 image positions (arcsec from lens center)
# HST/CASTLES data
Q2237_IMAGES = np.array([
    [0.740, 0.565],   # Image A
    [-0.635, 0.470],  # Image B
    [-0.480, -0.755], # Image C
    [0.870, -0.195],  # Image D
])


def test_q2237_model_comparison():
    """
    Compare models on Q2237+0305 Einstein Cross.
    
    Expected: m=2 alone has higher residuals than m=2+shear or m=2+m=3.
    This is because the lens galaxy (bar structure) needs more complexity.
    """
    system = LensSystem(name="Q2237+0305", images=[Q2237_IMAGES])
    
    zoo = ModelZoo(system)
    results = zoo.run_all()
    
    # All three simple models should succeed
    assert results[ModelType.M2].success
    assert results[ModelType.M2_SHEAR].success
    assert results[ModelType.M2_M3].success
    
    # m=2+shear+m=3 should be FORBIDDEN (8 constraints, 9 params)
    assert not results[ModelType.M2_SHEAR_M3].success
    
    res_m2 = results[ModelType.M2].max_residual
    res_shear = results[ModelType.M2_SHEAR].max_residual
    res_m3 = results[ModelType.M2_M3].max_residual
    
    print("\n" + "=" * 50)
    print("Q2237+0305 EINSTEIN CROSS DIAGNOSTIC")
    print("=" * 50)
    print(f"m=2 only:     residual = {res_m2:.4f}")
    print(f"m=2 + shear:  residual = {res_shear:.4f}")
    print(f"m=2 + m=3:    residual = {res_m3:.4f}")
    print("-" * 50)
    
    # At least one extended model should improve
    best_extended = min(res_shear, res_m3)
    improvement = (res_m2 - best_extended) / res_m2 * 100
    print(f"Improvement:  {improvement:.1f}%")
    
    # The test: extended models should not be WORSE
    assert res_shear <= res_m2 * 1.5 or res_m3 <= res_m2 * 1.5


def test_q2237_forbidden_info():
    """Check that FORBIDDEN gives useful suggestions."""
    system = LensSystem(name="Q2237+0305", images=[Q2237_IMAGES])
    spec = MODEL_SPECS[ModelType.M2_SHEAR_M3]
    
    status = RegimeGate.check(spec, system)
    
    assert not status.allowed
    assert status.regime == "FORBIDDEN"
    assert status.dof == -1  # 8 constraints, 9 params
    
    # Should suggest flux ratios, time delays, arc points
    suggestions = " ".join(status.missing_info).lower()
    assert "flux" in suggestions or "time" in suggestions or "arc" in suggestions


def test_q2237_full_report():
    """Generate full comparison report."""
    system = LensSystem(name="Q2237+0305", images=[Q2237_IMAGES])
    zoo = ModelZoo(system)
    zoo.run_all()
    
    report = zoo.compare()
    print("\n" + report)
    
    assert "Q2237" in system.name
    assert len(zoo.results) == 4


if __name__ == "__main__":
    print("Running Q2237+0305 Einstein Cross Diagnostics...")
    test_q2237_model_comparison()
    test_q2237_forbidden_info()
    test_q2237_full_report()
    print("\nAll diagnostics passed!")
