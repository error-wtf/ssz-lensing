#!/usr/bin/env python3
"""
Rootfinding Visualization

Shows:
1. Consistency function h(phi_gamma) with roots
2. Bisection convergence
3. Multiple row combinations
4. Root sensitivity analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_synthetic_cross(theta_E, a, b, beta, phi_beta, phi_gamma):
    """Generate synthetic Einstein Cross image positions."""
    def angular_condition(phi):
        return beta * np.sin(phi - phi_beta) + b * np.sin(2 * (phi - phi_gamma))

    phi_test = np.linspace(0, 2*np.pi, 1000)
    f_vals = [angular_condition(p) for p in phi_test]
    
    roots = []
    for i in range(len(phi_test) - 1):
        if f_vals[i] * f_vals[i+1] < 0:
            lo, hi = phi_test[i], phi_test[i+1]
            for _ in range(50):
                mid = (lo + hi) / 2
                if angular_condition(lo) * angular_condition(mid) < 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
    
    phi_solutions = np.array(roots)
    if len(phi_solutions) == 0:
        return np.array([]).reshape(0, 2), np.array([])
    
    radii = (theta_E + a * np.cos(2 * (phi_solutions - phi_gamma))
             + beta * np.cos(phi_solutions - phi_beta))
    
    points = np.column_stack([
        radii * np.cos(phi_solutions),
        radii * np.sin(phi_solutions)
    ])
    return points, phi_solutions


def build_linear_system(points, phi_gamma):
    """Build the linear system."""
    n = len(points)
    A = np.zeros((2 * n, 5))
    b_vec = np.zeros(2 * n)

    for i, (x, y) in enumerate(points):
        phi = np.arctan2(y, x)
        Delta = phi - phi_gamma
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        cos_2D, sin_2D = np.cos(2 * Delta), np.sin(2 * Delta)

        row_x = 2 * i
        A[row_x, 0] = 1.0
        A[row_x, 2] = cos_phi
        A[row_x, 3] = cos_2D * cos_phi
        A[row_x, 4] = -sin_2D * sin_phi
        b_vec[row_x] = x

        row_y = 2 * i + 1
        A[row_y, 1] = 1.0
        A[row_y, 2] = sin_phi
        A[row_y, 3] = cos_2D * sin_phi
        A[row_y, 4] = sin_2D * cos_phi
        b_vec[row_y] = y

    return A, b_vec


def consistency_residual(phi_gamma, points, row_subset, check_row):
    """Compute consistency residual."""
    A, b_vec = build_linear_system(points, phi_gamma)
    A_sub = A[row_subset, :]
    b_sub = b_vec[row_subset]
    try:
        det = np.linalg.det(A_sub)
        if abs(det) < 1e-14:
            return np.inf
        p = np.linalg.solve(A_sub, b_sub)
        return A[check_row, :] @ p - b_vec[check_row]
    except:
        return np.inf


def plot_consistency_function():
    """Plot h(phi_gamma) showing where roots are."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Generate data
    theta_E, a, b, beta = 1.0, 0.05, 0.15, 0.08
    phi_beta, phi_gamma_true = np.radians(30), np.radians(20)
    
    images, _ = generate_synthetic_cross(theta_E, a, b, beta, phi_beta, phi_gamma_true)
    
    phi_range = np.linspace(0, np.pi/2, 500)
    
    # Different row combinations
    row_combinations = [
        ([0, 1, 2, 3, 4], 5, 'Rows 0-4, check 5'),
        ([0, 1, 2, 3, 5], 4, 'Rows 0-3,5, check 4'),
        ([0, 1, 2, 4, 5], 3, 'Rows 0-2,4,5, check 3'),
        ([0, 1, 3, 4, 5], 2, 'Rows 0,1,3-5, check 2'),
    ]
    
    for ax, (row_subset, check_row, label) in zip(axes.flat, row_combinations):
        h_vals = []
        for phi in phi_range:
            h = consistency_residual(phi, images, row_subset, check_row)
            h_vals.append(h if np.isfinite(h) else np.nan)
        h_vals = np.array(h_vals)
        
        ax.plot(np.degrees(phi_range), h_vals, 'b-', linewidth=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(np.degrees(phi_gamma_true), color='red', linestyle='-',
                  linewidth=2, alpha=0.7, label=f'True φ_γ = {np.degrees(phi_gamma_true):.0f}°')
        
        # Find and mark roots
        for i in range(len(phi_range) - 1):
            if np.isfinite(h_vals[i]) and np.isfinite(h_vals[i+1]):
                if h_vals[i] * h_vals[i+1] < 0:
                    # Linear interpolation for root
                    root = phi_range[i] - h_vals[i] * (phi_range[i+1] - phi_range[i]) / (h_vals[i+1] - h_vals[i])
                    ax.plot(np.degrees(root), 0, 'go', markersize=12, 
                           markeredgecolor='black', markeredgewidth=2)
        
        ax.set_xlabel('φ_γ (degrees)', fontsize=11)
        ax.set_ylabel('h(φ_γ)', fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_xlim(0, 90)
    
    plt.suptitle('Consistency Function h(φ_γ) = 0\nRoots give valid quadrupole orientation', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rootfinding_consistency.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'rootfinding_consistency.pdf'))
    plt.close()
    print("Saved: rootfinding_consistency.png/pdf")


def plot_bisection_convergence():
    """Show bisection algorithm converging to root."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Generate data
    theta_E, a, b, beta = 1.0, 0.05, 0.15, 0.08
    phi_beta, phi_gamma_true = np.radians(30), np.radians(20)
    images, _ = generate_synthetic_cross(theta_E, a, b, beta, phi_beta, phi_gamma_true)
    
    row_subset, check_row = [0, 1, 2, 3, 4], 5
    
    def h(phi):
        return consistency_residual(phi, images, row_subset, check_row)
    
    # Bisection tracking
    a_val, b_val = 0.1, 0.6  # Initial bracket containing root
    iterations = []
    midpoints = []
    h_values = []
    intervals = []
    
    fa = h(a_val)
    for i in range(25):
        mid = (a_val + b_val) / 2
        fm = h(mid)
        
        iterations.append(i)
        midpoints.append(np.degrees(mid))
        h_values.append(fm)
        intervals.append(np.degrees(b_val - a_val))
        
        if fa * fm < 0:
            b_val = mid
        else:
            a_val = mid
            fa = fm
    
    # Left: Convergence of midpoint
    ax1 = axes[0]
    ax1.plot(iterations, midpoints, 'b.-', markersize=8, linewidth=1.5)
    ax1.axhline(np.degrees(phi_gamma_true), color='red', linestyle='--',
               label=f'True value: {np.degrees(phi_gamma_true):.2f}°')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Midpoint φ_γ (degrees)', fontsize=12)
    ax1.set_title('Bisection Convergence', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right: Interval size (log scale)
    ax2 = axes[1]
    ax2.semilogy(iterations, intervals, 'g.-', markersize=8, linewidth=1.5)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Interval size (degrees)', fontsize=12)
    ax2.set_title('Interval Size (Log Scale)\nHalves each iteration', fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add convergence rate annotation
    ax2.annotate('Rate: 2× per iteration\n(linear convergence)',
                xy=(15, intervals[15]), xytext=(18, intervals[10]),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rootfinding_bisection.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'rootfinding_bisection.pdf'))
    plt.close()
    print("Saved: rootfinding_bisection.png/pdf")


def plot_root_uniqueness():
    """Show that there's typically one root in [0, pi/2]."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    theta_E = 1.0
    a = 0.05
    b = 0.15
    phi_gamma_true = np.radians(20)
    
    phi_range = np.linspace(0, np.pi, 500)
    row_subset, check_row = [0, 1, 2, 3, 4], 5
    
    # Different source positions
    configs = [
        (0.05, 0, 'β=0.05'),
        (0.08, 30, 'β=0.08'),
        (0.12, 45, 'β=0.12'),
        (0.15, 60, 'β=0.15'),
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(configs)))
    
    for (beta, phi_beta_deg, label), color in zip(configs, colors):
        phi_beta = np.radians(phi_beta_deg)
        images, _ = generate_synthetic_cross(theta_E, a, b, beta, phi_beta, phi_gamma_true)
        
        if len(images) < 4:
            continue
        
        h_vals = []
        for phi in phi_range:
            h = consistency_residual(phi, images, row_subset, check_row)
            h_vals.append(h if np.isfinite(h) else np.nan)
        
        ax.plot(np.degrees(phi_range), h_vals, '-', color=color, 
               linewidth=2, label=label)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(np.degrees(phi_gamma_true), color='red', linestyle='-',
              linewidth=2, alpha=0.7, label=f'True φ_γ')
    ax.axvline(90, color='orange', linestyle=':', alpha=0.5)
    
    ax.fill_between([0, 90], [-0.3, -0.3], [0.3, 0.3], alpha=0.1, color='green')
    ax.text(45, 0.25, 'Search Region [0, π/2]', fontsize=11, ha='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('φ_γ (degrees)', fontsize=12)
    ax.set_ylabel('h(φ_γ)', fontsize=12)
    ax.set_title('Consistency Function for Different Source Offsets\nRoot location stable near true value', 
                fontsize=13)
    ax.set_xlim(0, 180)
    ax.set_ylim(-0.3, 0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rootfinding_uniqueness.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'rootfinding_uniqueness.pdf'))
    plt.close()
    print("Saved: rootfinding_uniqueness.png/pdf")


def plot_algorithm_flowchart():
    """Create a visual representation of the algorithm."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Box style
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="lightblue", 
                     edgecolor="navy", linewidth=2)
    bbox_decision = dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                        edgecolor="orange", linewidth=2)
    bbox_result = dict(boxstyle="round,pad=0.4", facecolor="lightgreen",
                      edgecolor="darkgreen", linewidth=2)
    
    # Title
    ax.text(6, 9.5, 'NO-FIT INVERSION ALGORITHM', fontsize=16, ha='center',
           fontweight='bold', color='navy')
    
    # Input
    ax.text(6, 8.5, 'Input: 4 image positions (x_i, y_i)', fontsize=12, 
           ha='center', bbox=bbox_props)
    
    # Step 1
    ax.annotate('', xy=(6, 7.8), xytext=(6, 8.2),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(6, 7.3, 'Sample h(φ_γ) over [0, π/2]\nFind sign changes', 
           fontsize=11, ha='center', bbox=bbox_props)
    
    # Step 2
    ax.annotate('', xy=(6, 6.6), xytext=(6, 7.0),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(6, 6.1, 'Bisection rootfinding\non each bracket', 
           fontsize=11, ha='center', bbox=bbox_props)
    
    # Step 3
    ax.annotate('', xy=(6, 5.4), xytext=(6, 5.8),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(6, 4.9, 'For each root φ_γ:\nSolve Ax = b exactly', 
           fontsize=11, ha='center', bbox=bbox_props)
    
    # Decision
    ax.annotate('', xy=(6, 4.2), xytext=(6, 4.6),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(6, 3.7, 'θ_E > 0?', fontsize=11, ha='center', bbox=bbox_decision)
    
    # Branches
    ax.annotate('', xy=(3.5, 3.0), xytext=(5.5, 3.4),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(3.5, 2.5, 'No: Skip', fontsize=10, ha='center', color='red')
    
    ax.annotate('', xy=(8.5, 3.0), xytext=(6.5, 3.4),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(8.5, 2.5, 'Yes: Compute\nresiduals', fontsize=10, ha='center', 
           bbox=bbox_props)
    
    # Final
    ax.annotate('', xy=(6, 1.5), xytext=(8.5, 2.2),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(6, 1.0, 'Output: Parameters with\nsmallest max|residual|', 
           fontsize=11, ha='center', bbox=bbox_result)
    
    # Key insight
    ax.text(0.5, 1.5, 'KEY INSIGHT:\n\n'
           '• h(φ_γ) = 0 is the\n'
           '  nonlinear condition\n\n'
           '• Once φ_γ known,\n'
           '  rest is LINEAR\n\n'
           '• No fitting needed!', 
           fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rootfinding_algorithm.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'rootfinding_algorithm.pdf'))
    plt.close()
    print("Saved: rootfinding_algorithm.png/pdf")


if __name__ == '__main__':
    print("Generating rootfinding plots...")
    plot_consistency_function()
    plot_bisection_convergence()
    plot_root_uniqueness()
    plot_algorithm_flowchart()
    print("Done!")
