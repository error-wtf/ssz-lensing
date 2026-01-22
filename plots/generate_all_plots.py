#!/usr/bin/env python3
"""
Master Script to Generate All Plots

Generates comprehensive visualizations of the No-Fit Gravitational
Lens Inversion framework from multiple perspectives:

1. Geometry plots - Lens configuration, images, quadrupole effects
2. Rootfinding plots - Consistency function, bisection convergence
3. Residual plots - Model validation and diagnostics
4. Parameter plots - Recovery accuracy and comparisons
5. DoF plots - Degrees of freedom analysis and system structure

Usage:
    python generate_all_plots.py

Outputs PNG and PDF files to the plots/ directory.
"""

import os
import sys

# Ensure we can import the individual plot modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("GENERATING ALL VISUALIZATION PLOTS")
    print("No-Fit Gravitational Lens Inversion Framework")
    print("=" * 60)

    # Check matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        print(f"\nMatplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        return 1

    # Check numpy is available
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("ERROR: numpy not installed. Run: pip install numpy")
        return 1

    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Output directory: {output_dir}\n")

    # Generate each category of plots
    categories = [
        ('Geometry', 'plot_geometry'),
        ('Rootfinding', 'plot_rootfinding'),
        ('Residuals', 'plot_residuals'),
        ('Parameters', 'plot_parameters'),
        ('DoF Analysis', 'plot_dof'),
    ]

    total_success = 0
    total_failed = 0

    for name, module_name in categories:
        print(f"\n{'-' * 40}")
        print(f"[{name}] Generating plots...")
        print(f"{'-' * 40}")

        try:
            module = __import__(module_name)

            # Run all plot functions in the module
            for attr_name in dir(module):
                if attr_name.startswith('plot_'):
                    func = getattr(module, attr_name)
                    if callable(func):
                        try:
                            func()
                            total_success += 1
                        except Exception as e:
                            print(f"  ERROR in {attr_name}: {e}")
                            total_failed += 1

        except Exception as e:
            print(f"  ERROR importing {module_name}: {e}")
            total_failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful plots: {total_success}")
    print(f"Failed plots: {total_failed}")

    # List generated files
    print(f"\nGenerated files in {output_dir}:")
    png_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    pdf_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.pdf')])

    print(f"\nPNG files ({len(png_files)}):")
    for f in png_files:
        print(f"  • {f}")

    print(f"\nPDF files ({len(pdf_files)}):")
    for f in pdf_files:
        print(f"  • {f}")

    print("\n" + "=" * 60)
    print("PLOT CATEGORIES OVERVIEW")
    print("=" * 60)
    print("""
GEOMETRY PLOTS:
   - geometry_einstein_cross.png - Full annotated lens geometry
   - geometry_quadrupole_effect.png - How quadrupole distorts ring
   - geometry_image_formation.png - Images vs source position
   - geometry_polar_view.png - Polar coordinate representation

ROOTFINDING PLOTS:
   - rootfinding_consistency.png - h(phi_gamma)=0 consistency function
   - rootfinding_bisection.png - Bisection convergence
   - rootfinding_uniqueness.png - Root stability analysis
   - rootfinding_algorithm.png - Algorithm flowchart

RESIDUAL PLOTS:
   - residuals_bars.png - Residual per equation
   - residuals_spatial.png - Residuals at image positions
   - residuals_noise_comparison.png - Clean vs noisy data
   - residuals_model_adequacy.png - Model validation

PARAMETER PLOTS:
   - parameters_comparison.png - True vs recovered
   - parameters_accuracy_grid.png - Recovery across configs
   - parameters_error_distribution.png - Error histograms
   - parameters_phase_recovery.png - phi_gamma recovery specifically

DOF ANALYSIS PLOTS:
   - dof_bookkeeping.png - Equation/unknown counting
   - dof_comparison.png - DoF vs multipole order
   - dof_system_diagram.png - Linear system structure
   - dof_conditional_linearity.png - Key concept explanation
""")

    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
