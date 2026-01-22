#!/usr/bin/env python3
"""
Residual Analysis Visualization

Shows:
1. Residual distribution
2. Residual vs equation number
3. Residual magnitude comparison
4. Model adequacy diagnostics
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_and_invert():
    """Generate data and perform inversion, return residuals."""
    # Parameters
    theta_E, a, b, beta = 1.0, 0.05, 0.15, 0.08
    phi_beta = np.radians(30)
    phi_gamma_true = np.radians(20)
    
    # Generate images
    def angular_condition(phi):
        return beta * np.sin(phi - phi_beta) + b * np.sin(2 * (phi - phi_gamma_true))
    
    phi_test = np.linspace(0, 2*np.pi, 1000)
    roots = []
    for i in range(len(phi_test) - 1):
        f1 = angular_condition(phi_test[i])
        f2 = angular_condition(phi_test[i+1])
        if f1 * f2 < 0:
            lo, hi = phi_test[i], phi_test[i+1]
            for _ in range(50):
                mid = (lo + hi) / 2
                if angular_condition(lo) * angular_condition(mid) < 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
    
    phi_solutions = np.array(roots)
    radii = (theta_E + a * np.cos(2 * (phi_solutions - phi_gamma_true))
             + beta * np.cos(phi_solutions - phi_beta))
    images = np.column_stack([radii * np.cos(phi_solutions), 
                              radii * np.sin(phi_solutions)])
    
    # Build and solve system
    def build_system(points, phi_gamma):
        n = len(points)
        A = np.zeros((2 * n, 5))
        b_vec = np.zeros(2 * n)
        for i, (x, y) in enumerate(points):
            phi = np.arctan2(y, x)
            Delta = phi - phi_gamma
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)
            cos_2D, sin_2D = np.cos(2 * Delta), np.sin(2 * Delta)
            A[2*i, 0], A[2*i, 2] = 1.0, cos_phi
            A[2*i, 3], A[2*i, 4] = cos_2D * cos_phi, -sin_2D * sin_phi
            b_vec[2*i] = x
            A[2*i+1, 1], A[2*i+1, 2] = 1.0, sin_phi
            A[2*i+1, 3], A[2*i+1, 4] = cos_2D * sin_phi, sin_2D * cos_phi
            b_vec[2*i+1] = y
        return A, b_vec
    
    A, b_vec = build_system(images, phi_gamma_true)
    p = np.linalg.lstsq(A, b_vec, rcond=None)[0]
    residuals = A @ p - b_vec
    
    return residuals, images, p, phi_gamma_true


def plot_residual_bars():
    """Bar chart of residuals per equation."""
    residuals, images, p, _ = generate_and_invert()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Absolute residuals
    ax1 = axes[0]
    eq_labels = []
    for i in range(4):
        eq_labels.extend([f'Img{i+1}_x', f'Img{i+1}_y'])
    
    colors = ['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e',
              '#2ca02c', '#2ca02c', '#d62728', '#d62728']
    
    bars = ax1.bar(range(8), np.abs(residuals), color=colors, edgecolor='black')
    ax1.set_xticks(range(8))
    ax1.set_xticklabels(eq_labels, rotation=45, ha='right')
    ax1.set_ylabel('|Residual|', fontsize=12)
    ax1.set_title('Absolute Residuals per Equation', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(1e-10, color='green', linestyle='--', 
               label='Machine precision', alpha=0.7)
    ax1.legend()
    
    # Right: Signed residuals
    ax2 = axes[1]
    ax2.bar(range(8), residuals, color=colors, edgecolor='black')
    ax2.set_xticks(range(8))
    ax2.set_xticklabels(eq_labels, rotation=45, ha='right')
    ax2.set_ylabel('Residual', fontsize=12)
    ax2.set_title('Signed Residuals', fontsize=12)
    ax2.axhline(0, color='gray', linestyle='-', linewidth=2)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = (f'Max |res|: {np.max(np.abs(residuals)):.2e}\n'
                  f'RMS: {np.sqrt(np.mean(residuals**2)):.2e}\n'
                  f'Sum: {np.sum(residuals):.2e}')
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_bars.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_bars.pdf'))
    plt.close()
    print("Saved: residuals_bars.png/pdf")


def plot_residual_spatial():
    """Show residuals as vectors at image positions."""
    residuals, images, p, phi_gamma = generate_and_invert()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Einstein ring
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'b--', alpha=0.5, linewidth=1.5)
    
    # Images with residual vectors (scaled up for visibility)
    scale = 1e9  # Scale factor for tiny residuals
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (img, c) in enumerate(zip(images, colors)):
        x, y = img
        res_x = residuals[2*i] * scale
        res_y = residuals[2*i + 1] * scale
        
        ax.plot(x, y, 'o', color=c, markersize=15, 
               markeredgecolor='black', markeredgewidth=2)
        
        # Residual vector (if visible)
        if np.sqrt(res_x**2 + res_y**2) > 0.01:
            ax.arrow(x, y, res_x, res_y, head_width=0.03, head_length=0.02,
                    fc='red', ec='red', linewidth=2)
    
    ax.plot(0, 0, 'ko', markersize=10)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x / θ_E', fontsize=12)
    ax.set_ylabel('y / θ_E', fontsize=12)
    ax.set_title(f'Residual Vectors at Image Positions\n(scaled by {scale:.0e} for visibility)',
                fontsize=12)
    
    # Add note
    ax.text(0.02, 0.98, 'Residuals are at\nmachine precision\n(~10⁻¹¹)', 
           transform=ax.transAxes, fontsize=11, va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_spatial.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_spatial.pdf'))
    plt.close()
    print("Saved: residuals_spatial.png/pdf")


def plot_residual_comparison():
    """Compare residuals for exact vs noisy data."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Generate base data
    theta_E, a, b, beta = 1.0, 0.05, 0.15, 0.08
    phi_beta = np.radians(30)
    phi_gamma_true = np.radians(20)
    
    def angular_condition(phi):
        return beta * np.sin(phi - phi_beta) + b * np.sin(2 * (phi - phi_gamma_true))
    
    phi_test = np.linspace(0, 2*np.pi, 1000)
    roots = []
    for i in range(len(phi_test) - 1):
        f1, f2 = angular_condition(phi_test[i]), angular_condition(phi_test[i+1])
        if f1 * f2 < 0:
            lo, hi = phi_test[i], phi_test[i+1]
            for _ in range(50):
                mid = (lo + hi) / 2
                if angular_condition(lo) * angular_condition(mid) < 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
    
    phi_solutions = np.array(roots)
    radii = (theta_E + a * np.cos(2 * (phi_solutions - phi_gamma_true))
             + beta * np.cos(phi_solutions - phi_beta))
    images_clean = np.column_stack([radii * np.cos(phi_solutions),
                                    radii * np.sin(phi_solutions)])
    
    # Noise levels
    noise_levels = [0.0, 0.001, 0.01]
    titles = ['Clean Data\n(no noise)', 'Small Noise\n(σ=0.001)', 'Large Noise\n(σ=0.01)']
    
    for ax, noise, title in zip(axes, noise_levels, titles):
        np.random.seed(42)
        images = images_clean + np.random.normal(0, noise, images_clean.shape)
        
        # Solve
        def build_system(points, phi_gamma):
            n = len(points)
            A = np.zeros((2 * n, 5))
            b_vec = np.zeros(2 * n)
            for i, (x, y) in enumerate(points):
                phi = np.arctan2(y, x)
                Delta = phi - phi_gamma
                cos_phi, sin_phi = np.cos(phi), np.sin(phi)
                cos_2D, sin_2D = np.cos(2 * Delta), np.sin(2 * Delta)
                A[2*i, 0], A[2*i, 2] = 1.0, cos_phi
                A[2*i, 3], A[2*i, 4] = cos_2D * cos_phi, -sin_2D * sin_phi
                b_vec[2*i] = x
                A[2*i+1, 1], A[2*i+1, 2] = 1.0, sin_phi
                A[2*i+1, 3], A[2*i+1, 4] = cos_2D * sin_phi, sin_2D * cos_phi
                b_vec[2*i+1] = y
            return A, b_vec
        
        A, b_vec = build_system(images, phi_gamma_true)
        p = np.linalg.lstsq(A, b_vec, rcond=None)[0]
        residuals = A @ p - b_vec
        
        ax.bar(range(8), np.abs(residuals), color='steelblue', edgecolor='black')
        ax.set_xticks(range(8))
        ax.set_xticklabels([f'{i//2+1}{"xy"[i%2]}' for i in range(8)])
        ax.set_ylabel('|Residual|', fontsize=11)
        ax.set_title(title, fontsize=12)
        
        if noise == 0:
            ax.set_ylim(1e-15, 1e-9)
            ax.set_yscale('log')
        else:
            ax.set_ylim(0, max(np.abs(residuals)) * 1.2)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        max_res = np.max(np.abs(residuals))
        ax.text(0.95, 0.95, f'Max: {max_res:.2e}', transform=ax.transAxes,
               fontsize=10, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Residuals Reveal Data Quality\n'
                'Clean data → machine precision | Noisy data → noise-level residuals',
                fontsize=13, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_noise_comparison.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_noise_comparison.pdf'))
    plt.close()
    print("Saved: residuals_noise_comparison.png/pdf")


def plot_model_adequacy():
    """Show how residuals reveal model inadequacy."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    scenarios = [
        {'m3': 0.0, 'title': 'Quadrupole Only (m=2)\nModel adequate'},
        {'m3': 0.02, 'title': 'Small m=3 term\nSlightly inadequate'},
        {'m3': 0.05, 'title': 'Large m=3 term\nModel inadequate'},
    ]
    
    theta_E, a, b = 1.0, 0.05, 0.15
    phi_gamma = np.radians(20)
    
    for ax, scenario in zip(axes, scenarios):
        m3 = scenario['m3']
        
        # Generate images with possible m=3 term
        # Using fixed angles for simplicity
        phi_imgs = np.array([0.4, 2.2, 3.8, 5.4])
        radii = (theta_E + a * np.cos(2 * (phi_imgs - phi_gamma))
                 + m3 * np.cos(3 * phi_imgs))  # Extra m=3 term
        
        images = np.column_stack([radii * np.cos(phi_imgs),
                                  radii * np.sin(phi_imgs)])
        
        # Solve with m=2 model (ignoring m=3)
        def build_system(points, phi_g):
            n = len(points)
            A = np.zeros((2 * n, 5))
            b_vec = np.zeros(2 * n)
            for i, (x, y) in enumerate(points):
                phi = np.arctan2(y, x)
                Delta = phi - phi_g
                cos_phi, sin_phi = np.cos(phi), np.sin(phi)
                cos_2D, sin_2D = np.cos(2 * Delta), np.sin(2 * Delta)
                A[2*i, 0], A[2*i, 2] = 1.0, cos_phi
                A[2*i, 3], A[2*i, 4] = cos_2D * cos_phi, -sin_2D * sin_phi
                b_vec[2*i] = x
                A[2*i+1, 1], A[2*i+1, 2] = 1.0, sin_phi
                A[2*i+1, 3], A[2*i+1, 4] = cos_2D * sin_phi, sin_2D * cos_phi
                b_vec[2*i+1] = y
            return A, b_vec
        
        A, b_vec = build_system(images, phi_gamma)
        p = np.linalg.lstsq(A, b_vec, rcond=None)[0]
        residuals = A @ p - b_vec
        
        colors = ['#2ca02c' if abs(r) < 0.001 else '#d62728' for r in residuals]
        ax.bar(range(8), np.abs(residuals), color=colors, edgecolor='black')
        ax.set_xticks(range(8))
        ax.set_xticklabels([f'{i//2+1}{"xy"[i%2]}' for i in range(8)])
        ax.set_ylabel('|Residual|', fontsize=11)
        ax.set_title(scenario['title'], fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        max_res = np.max(np.abs(residuals))
        color = 'green' if max_res < 0.001 else 'red'
        ax.text(0.95, 0.95, f'Max: {max_res:.2e}', transform=ax.transAxes,
               fontsize=10, ha='right', va='top', color=color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Residuals Diagnose Model Adequacy\n'
                'Large residuals → model missing physics (e.g., higher multipoles)',
                fontsize=13, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_model_adequacy.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_model_adequacy.pdf'))
    plt.close()
    print("Saved: residuals_model_adequacy.png/pdf")


if __name__ == '__main__':
    print("Generating residual analysis plots...")
    plot_residual_bars()
    plot_residual_spatial()
    plot_residual_comparison()
    plot_model_adequacy()
    print("Done!")
