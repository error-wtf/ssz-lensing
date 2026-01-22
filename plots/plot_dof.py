#!/usr/bin/env python3
"""
Degrees of Freedom Analysis Visualization

Shows:
1. DoF bookkeeping for different models
2. Overdetermined vs underdetermined systems
3. Why 4 images → exact solution for m=2
4. Constraint counting for general multipoles
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_dof_bookkeeping():
    """Visualize degrees of freedom counting."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    ax.text(0.5, 0.95, 'DEGREES OF FREEDOM BOOKKEEPING', fontsize=16,
           ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.90, 'Why 4 images give an exact solution for m=2 model',
           fontsize=12, ha='center', style='italic', transform=ax.transAxes)

    y = 0.82

    ax.text(0.02, y, 'EQUATIONS (Constraints):', fontsize=13,
           fontweight='bold', transform=ax.transAxes)
    y -= 0.06
    ax.text(0.05, y, '• Each image gives 2 equations (x and y components)',
           fontsize=11, transform=ax.transAxes)
    y -= 0.05
    ax.text(0.05, y, '• 4 images → 8 equations total',
           fontsize=11, transform=ax.transAxes, color='green')

    y -= 0.08
    ax.text(0.02, y, 'UNKNOWNS (Parameters):', fontsize=13,
           fontweight='bold', transform=ax.transAxes)
    y -= 0.06
    ax.text(0.05, y, '• θ_E (Einstein radius): 1 parameter',
           fontsize=11, transform=ax.transAxes)
    y -= 0.05
    ax.text(0.05, y, '• (a, b) quadrupole amplitudes: 2 parameters',
           fontsize=11, transform=ax.transAxes)
    y -= 0.05
    ax.text(0.05, y, '• φ_γ quadrupole phase: 1 parameter (NONLINEAR)',
           fontsize=11, transform=ax.transAxes, color='red')
    y -= 0.05
    ax.text(0.05, y, '• (β_x, β_y) source position: 2 parameters',
           fontsize=11, transform=ax.transAxes)
    y -= 0.05
    ax.text(0.05, y, '• Total: 6 parameters (5 linear + 1 nonlinear)',
           fontsize=11, transform=ax.transAxes, color='blue')

    y -= 0.08
    ax.text(0.02, y, 'SOLUTION STRATEGY:', fontsize=13,
           fontweight='bold', transform=ax.transAxes)
    y -= 0.06
    ax.text(0.05, y, '1. Fix φ_γ → 5 linear unknowns, 8 equations',
           fontsize=11, transform=ax.transAxes)
    y -= 0.05
    ax.text(0.05, y, '2. Use 5 equations to solve for 5 unknowns exactly',
           fontsize=11, transform=ax.transAxes)
    y -= 0.05
    ax.text(0.05, y, '3. Remaining 3 equations give consistency condition',
           fontsize=11, transform=ax.transAxes)
    y -= 0.05
    ax.text(0.05, y, '4. Find φ_γ where consistency = 0 (rootfinding)',
           fontsize=11, transform=ax.transAxes, color='red')

    y -= 0.08
    bbox = dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
    ax.text(0.5, y, '8 equations - 6 unknowns = 2 redundant equations\n'
           '→ System is OVERDETERMINED → Consistency check possible!',
           fontsize=12, ha='center', transform=ax.transAxes, bbox=bbox)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dof_bookkeeping.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'dof_bookkeeping.pdf'))
    plt.close()
    print("Saved: dof_bookkeeping.png/pdf")


def plot_dof_comparison():
    """Compare DoF for different multipole orders."""
    fig, ax = plt.subplots(figsize=(12, 7))

    multipoles = [2, 3, 4, 5, 6]
    n_images_list = [4, 6, 8, 10, 12]

    data = []
    for m, n_img in zip(multipoles, n_images_list):
        n_eq = 2 * n_img
        n_amp = 2
        n_phase = 1
        n_source = 2
        n_theta_E = 1
        n_unknowns = n_amp + n_phase + n_source + n_theta_E
        redundancy = n_eq - n_unknowns
        data.append({
            'm': m, 'n_images': n_img, 'n_equations': n_eq,
            'n_unknowns': n_unknowns, 'redundancy': redundancy
        })

    x = np.arange(len(multipoles))
    width = 0.25

    ax.bar(x - width, [d['n_equations'] for d in data], width,
          label='Equations', color='steelblue')
    ax.bar(x, [d['n_unknowns'] for d in data], width,
          label='Unknowns', color='coral')
    ax.bar(x + width, [d['redundancy'] for d in data], width,
          label='Redundancy', color='lightgreen')

    ax.set_xlabel('Multipole Order m', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Degrees of Freedom vs Multipole Order\n'
                '(Single multipole model)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'm={m}' for m in multipoles])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for i, d in enumerate(data):
        ax.annotate(f"{d['n_images']} images", (x[i] - width, d['n_equations'] + 0.5),
                   ha='center', fontsize=9)

    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dof_comparison.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'dof_comparison.pdf'))
    plt.close()
    print("Saved: dof_comparison.png/pdf")


def plot_system_diagram():
    """Visual diagram of the linear system structure."""
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.text(0.5, 0.95, 'LINEAR SYSTEM STRUCTURE: Ax = b',
           fontsize=16, ha='center', fontweight='bold', transform=ax.transAxes)

    matrix_x, matrix_y = 0.15, 0.25
    cell_w, cell_h = 0.08, 0.06

    headers = ['β_x', 'β_y', 'θ_E', 'a', 'b']
    row_labels = ['Im1_x', 'Im1_y', 'Im2_x', 'Im2_y',
                  'Im3_x', 'Im3_y', 'Im4_x', 'Im4_y']

    for j, h in enumerate(headers):
        ax.text(matrix_x + (j + 0.5) * cell_w, matrix_y + 8.5 * cell_h, h,
               ha='center', fontsize=10, fontweight='bold')

    for i, label in enumerate(row_labels):
        ax.text(matrix_x - 0.02, matrix_y + (7.5 - i) * cell_h, label,
               ha='right', fontsize=9)

    colors_solve = ['lightblue'] * 5 + ['lightyellow'] * 3
    colors_check = ['lightyellow'] * 5 + ['lightgreen'] * 3

    for i in range(8):
        for j in range(5):
            color = colors_solve[i] if i < 5 else colors_check[i]
            rect = plt.Rectangle((matrix_x + j * cell_w, matrix_y + (7 - i) * cell_h),
                                 cell_w, cell_h, facecolor=color,
                                 edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(matrix_x + (j + 0.5) * cell_w, matrix_y + (7.5 - i) * cell_h,
                   'A', ha='center', va='center', fontsize=8)

    ax.text(matrix_x + 5.5 * cell_w, matrix_y + 4 * cell_h, '×', fontsize=20)

    x_x = matrix_x + 6 * cell_w
    for i, var in enumerate(headers):
        rect = plt.Rectangle((x_x, matrix_y + (4 - i) * cell_h),
                             cell_w * 0.8, cell_h, facecolor='white',
                             edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x_x + cell_w * 0.4, matrix_y + (4.5 - i) * cell_h, var,
               ha='center', va='center', fontsize=9)

    ax.text(x_x + cell_w * 1.5, matrix_y + 4 * cell_h, '=', fontsize=20)

    b_x = x_x + cell_w * 2.2
    for i in range(8):
        color = colors_solve[i] if i < 5 else colors_check[i]
        rect = plt.Rectangle((b_x, matrix_y + (7 - i) * cell_h),
                             cell_w * 0.8, cell_h, facecolor=color,
                             edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(b_x + cell_w * 0.4, matrix_y + (7.5 - i) * cell_h,
               f'b_{i+1}', ha='center', va='center', fontsize=8)

    ax.text(0.15, 0.15, 'Blue rows: Used to SOLVE for parameters',
           fontsize=11, color='steelblue', transform=ax.transAxes)
    ax.text(0.15, 0.10, 'Green rows: Used to CHECK consistency → h(φ_γ) = 0',
           fontsize=11, color='green', transform=ax.transAxes)
    ax.text(0.15, 0.05, 'Yellow rows: Redundant (extra consistency checks)',
           fontsize=11, color='orange', transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dof_system_diagram.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'dof_system_diagram.pdf'))
    plt.close()
    print("Saved: dof_system_diagram.png/pdf")


def plot_conditional_linearity():
    """Explain conditional linearity concept."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.text(0.5, 0.95, 'FULL PROBLEM', fontsize=14, ha='center',
            fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.85, 'Nonlinear in φ_γ', fontsize=12, ha='center',
            color='red', transform=ax1.transAxes)
    ax1.text(0.5, 0.65, 'p = [β_x, β_y, θ_E, a, b, φ_γ]', fontsize=11,
            ha='center', family='monospace', transform=ax1.transAxes)
    ax1.text(0.5, 0.50, 'Matrix coefficients depend\non φ_γ nonlinearly:\n\n'
            'cos(2(φ - φ_γ))\nsin(2(φ - φ_γ))',
            fontsize=11, ha='center', transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax1.annotate('', xy=(0.95, 0.35), xytext=(0.5, 0.35),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                transform=ax1.transAxes)
    ax1.text(0.75, 0.40, 'Fix φ_γ', fontsize=11, ha='center',
            transform=ax1.transAxes)

    ax1.axis('off')

    ax2 = axes[1]
    ax2.text(0.5, 0.95, 'CONDITIONAL PROBLEM', fontsize=14, ha='center',
            fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.85, 'Linear in remaining parameters!', fontsize=12,
            ha='center', color='green', transform=ax2.transAxes)
    ax2.text(0.5, 0.65, 'p_linear = [β_x, β_y, θ_E, a, b]', fontsize=11,
            ha='center', family='monospace', transform=ax2.transAxes)
    ax2.text(0.5, 0.45, 'A(φ_γ) · p_linear = b\n\n'
            'For fixed φ_γ, A is constant!\n→ Solve exactly with linear algebra',
            fontsize=11, ha='center', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax2.text(0.5, 0.15, 'This is CONDITIONAL LINEARITY:\n'
            'Linear in amplitudes when phase is fixed',
            fontsize=11, ha='center', style='italic', transform=ax2.transAxes)

    ax2.axis('off')

    plt.suptitle('Conditional Linearity: The Key to No-Fit Inversion',
                fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dof_conditional_linearity.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'dof_conditional_linearity.pdf'))
    plt.close()
    print("Saved: dof_conditional_linearity.png/pdf")


if __name__ == '__main__':
    print("Generating DoF analysis plots...")
    plot_dof_bookkeeping()
    plot_dof_comparison()
    plot_system_diagram()
    plot_conditional_linearity()
    print("Done!")
