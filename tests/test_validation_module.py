"""
Test SSZ-inspired validation module.
"""
import sys
import numpy as np
sys.path.insert(0, 'src')

from validation.data_model import validate_images, ValidationResult
from validation.regime import dof_analysis, select_model_regime, print_dof_table
from validation.results import InversionResult, compare_models, interpret_residuals


def test_image_validation():
    """Test image data validation."""
    print("=" * 50)
    print("TEST: Image Validation")
    print("=" * 50)
    
    images = np.array([
        [0.9, 0.3],
        [-0.3, 0.9],
        [-0.9, -0.4],
        [0.4, -0.9]
    ])
    
    result = validate_images(images)
    print(result.summary())
    print(f"Valid: {result.valid}")
    print(f"Constraints: {result.n_constraints}")
    
    assert result.valid
    assert result.n_constraints == 8
    print("[PASS]")
    return True


def test_dof_analysis():
    """Test DOF analysis."""
    print("\n" + "=" * 50)
    print("TEST: DOF Analysis")
    print("=" * 50)
    
    print_dof_table(4)
    
    # Test auto-selection
    model, analysis = select_model_regime(4)
    print(f"\nAuto-selected: {model}")
    print(f"Status: {analysis.status}")
    print(f"DOF: {analysis.dof}")
    
    assert model == "m2_only"
    assert analysis.dof == 3
    print("[PASS]")
    return True


def test_result_interpretation():
    """Test result interpretation."""
    print("\n" + "=" * 50)
    print("TEST: Result Interpretation")
    print("=" * 50)
    
    images = np.array([[0.9, 0.3], [-0.3, 0.9], [-0.9, -0.4], [0.4, -0.9]])
    residuals = np.array([[0.01, 0.02], [0.01, 0.01], [0.02, 0.01], [0.01, 0.02]])
    
    result = InversionResult(
        params={'theta_E': 0.98},
        predicted_images=images,
        residuals=residuals,
        max_residual=0.02,
        rms_residual=0.015,
        consistency=0.01,
        n_constraints=8,
        n_parameters=5,
        dof=3,
        dof_status='OVERDETERMINED',
        model_name='Linear m=2',
        solver_converged=True
    )
    
    print(result.summary())
    
    interp = interpret_residuals(result)
    print(f"\nQuality: {interp['quality']}")
    print(f"Residual/Noise: {interp['residual_to_noise']:.1f}x")
    print(f"Interpretation: {interp['interpretation']}")
    print(f"Model adequate: {interp['model_adequate']}")
    
    assert interp['quality'] in ['EXCELLENT', 'GOOD', 'MARGINAL', 'POOR']
    print("[PASS]")
    return True


def test_model_comparison():
    """Test model comparison."""
    print("\n" + "=" * 50)
    print("TEST: Model Comparison")
    print("=" * 50)
    
    images = np.array([[0.9, 0.3], [-0.3, 0.9], [-0.9, -0.4], [0.4, -0.9]])
    
    result1 = InversionResult(
        params={'theta_E': 0.98},
        predicted_images=images,
        residuals=np.zeros((4, 2)),
        max_residual=0.05,
        rms_residual=0.03,
        consistency=0.02,
        n_constraints=8,
        n_parameters=5,
        dof=3,
        dof_status='OVERDETERMINED',
        model_name='m=2 only',
        solver_converged=True
    )
    
    result2 = InversionResult(
        params={'theta_E': 0.99},
        predicted_images=images,
        residuals=np.zeros((4, 2)),
        max_residual=0.02,
        rms_residual=0.01,
        consistency=0.005,
        n_constraints=8,
        n_parameters=7,
        dof=1,
        dof_status='OVERDETERMINED',
        model_name='m=2 + shear',
        solver_converged=True
    )
    
    comparison = compare_models([result1, result2])
    print(comparison.summary())
    
    assert comparison.winner in ['m=2 only', 'm=2 + shear']
    print("[PASS]")
    return True


def main():
    print("\n" + "=" * 60)
    print(" SSZ-INSPIRED VALIDATION MODULE TESTS")
    print("=" * 60)
    
    results = []
    results.append(("Image Validation", test_image_validation()))
    results.append(("DOF Analysis", test_dof_analysis()))
    results.append(("Result Interpretation", test_result_interpretation()))
    results.append(("Model Comparison", test_model_comparison()))
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
