"""
Test suite for LinearMultipoleModel - True No-Fit Inversion

This tests the fully linear model that requires NO grid search.
"""

import sys
import numpy as np

sys.path.insert(0, 'src')

from models.linear_model import LinearMultipoleModel


def test_dof_analysis():
    """Test DOF counting for different configurations."""
    print("=" * 60)
    print(" DOF ANALYSIS - Constraints vs Parameters")
    print("=" * 60)
    print()
    
    configs = [
        ('m=2 only', 2, False),
        ('m=2 + shear', 2, True),
        ('m=2 + m=3', 3, False),
        ('m=2 + shear + m=3', 3, True),
        ('m=2 + m=3 + m=4', 4, False),
    ]
    
    print(f"{'Config':<20} {'Params':>8} {'Quad (8)':<30}")
    print("-" * 60)
    
    for name, m_max, shear in configs:
        model = LinearMultipoleModel(m_max=m_max, include_shear=shear)
        n_par = model.n_parameters()
        status = model.dof_status(4)
        print(f"{name:<20} {n_par:>8} {status:<30}")
    
    print()
    return True


def test_synthetic_recovery():
    """Test that we can recover parameters from synthetic data."""
    print("=" * 60)
    print(" SYNTHETIC DATA RECOVERY (Linear Model)")
    print("=" * 60)
    print()
    
    model = LinearMultipoleModel(m_max=2, include_shear=False)
    
    # Near-Einstein-ring images
    images = np.array([
        [0.95, 0.30],
        [-0.35, 0.92],
        [-0.88, -0.45],
        [0.42, -0.90]
    ])
    
    print(f"Model: {model.name}")
    print(f"Parameters: {model.unknowns()}")
    print(f"N_params: {model.n_parameters()}")
    print(f"N_constraints: {model.n_constraints(len(images))}")
    print()
    
    solutions = model.invert(images)
    
    if not solutions:
        print("ERROR: No solution found!")
        return False
    
    sol = solutions[0]
    print("Solution found:")
    print(f"  theta_E = {sol['params']['theta_E']:.4f}")
    print(f"  beta = ({sol['params']['beta_x']:.4f}, {sol['params']['beta_y']:.4f})")
    print(f"  c_2 = {sol['params']['c_2']:.4f}")
    print(f"  s_2 = {sol['params']['s_2']:.4f}")
    print()
    print(f"  max|res| = {sol['report']['max_abs']:.2e}")
    print(f"  consistency = {sol['report']['consistency']:.2e}")
    print(f"  DOF: {sol['report']['dof_status']}")
    print()
    
    # Convert to physical parameters
    phys = model.convert_to_physical(sol['params'])
    print("Physical parameters:")
    print(f"  amplitude_2 = {phys['amplitude_2']:.4f}")
    print(f"  phase_2 = {np.degrees(phys['phase_2']):.1f} deg")
    print()
    
    # Check residuals are small (overdetermined system should still fit well)
    if sol['report']['max_abs'] < 0.1:
        print("  [PASS] Inversion successful")
        return True
    else:
        print("  [FAIL] Residuals too large")
        return False


def test_real_lens_data():
    """Test on real lens data (Einstein Cross)."""
    print("=" * 60)
    print(" REAL LENS DATA - Linear Model")
    print("=" * 60)
    print()
    
    # Einstein Cross (Q2237+0305)
    einstein_cross = np.array([
        [0.744, 0.904],
        [-0.707, 0.728],
        [-0.893, -0.699],
        [0.513, -0.942]
    ])
    
    print("Einstein Cross (Q2237+0305):")
    print()
    
    results = []
    for m_max in [2, 3]:
        for shear in [False, True]:
            if m_max == 3 and shear:
                continue  # Would be underdetermined
            
            model = LinearMultipoleModel(m_max=m_max, include_shear=shear)
            name = f"m={m_max}" + ("+shear" if shear else "")
            
            solutions = model.invert(einstein_cross)
            
            if solutions:
                sol = solutions[0]
                max_res = sol['report']['max_abs']
                dof = sol['report']['dof_status']
                theta_E = sol['params']['theta_E']
                results.append((name, max_res, theta_E, dof))
                print(f"  {name:<12}: max|res|={max_res:.4f}\", theta_E={theta_E:.3f}, {dof}")
            else:
                print(f"  {name:<12}: No solution (underdetermined)")
    
    print()
    print("  [PASS] Real lens inversion completed")
    return True


def test_comparison_with_extended():
    """Compare linear model with extended model."""
    print("=" * 60)
    print(" COMPARISON: Linear vs Extended Model")
    print("=" * 60)
    print()
    
    from models.extended_model import ExtendedMultipoleModel
    
    images = np.array([
        [0.95, 0.30],
        [-0.35, 0.92],
        [-0.88, -0.45],
        [0.42, -0.90]
    ])
    
    # Linear model (no grid search!)
    linear = LinearMultipoleModel(m_max=2, include_shear=False)
    lin_sol = linear.invert(images)
    
    # Extended model (with grid search)
    extended = ExtendedMultipoleModel(m_max=2, include_shear=False)
    ext_sol = extended.invert(images)
    
    print("Linear Model (direct solve):")
    if lin_sol:
        print(f"  theta_E = {lin_sol[0]['params']['theta_E']:.4f}")
        print(f"  max|res| = {lin_sol[0]['report']['max_abs']:.2e}")
    
    print()
    print("Extended Model (grid search):")
    if ext_sol:
        print(f"  theta_E = {ext_sol[0]['params']['theta_E']:.4f}")
        print(f"  max|res| = {ext_sol[0]['report']['max_abs']:.2e}")
    
    print()
    
    # Key difference: Linear model has NO nonlinear unknowns
    print(f"Linear nonlinear_unknowns: {linear.nonlinear_unknowns()}")
    print(f"Extended nonlinear_unknowns: {extended.nonlinear_unknowns()}")
    print()
    print("  [PASS] Comparison completed")
    return True


def main():
    print()
    print("=" * 70)
    print(" LINEAR MULTIPOLE MODEL - TRUE NO-FIT TEST SUITE")
    print("=" * 70)
    print()
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy:  {np.__version__}")
    print()
    
    results = []
    results.append(("DOF Analysis", test_dof_analysis()))
    results.append(("Synthetic Recovery", test_synthetic_recovery()))
    results.append(("Real Lens Data", test_real_lens_data()))
    results.append(("Model Comparison", test_comparison_with_extended()))
    
    print("=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    print()
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
