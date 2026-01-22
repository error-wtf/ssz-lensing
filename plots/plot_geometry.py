#!/usr/bin/env python3
"""
Lens Geometry Visualization

Plots:
1. Einstein ring with image positions
2. Source-lens-image geometry
3. Quadrupole distortion visualization
4. Multiple image configurations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import os

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_synthetic_cross(theta_E, a, b, beta, phi_beta, phi_gamma):
    """Generate synthetic Einstein Cross image positions."""
    def angular_condition(phi):
        return beta * np.sin(phi - phi_beta) + b * np.sin(2 * (phi - phi_gamma))

    # Find roots
    phi_test = np.linspace(0, 2*np.pi, 1000)
    f_vals = [angular_condition(p) for p in phi_test]
    
    roots = []
    for i in range(len(phi_test) - 1):
        if f_vals[i] * f_vals[i+1] < 0:
            # Bisection
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


def plot_einstein_cross_geometry():
    """Plot Einstein Cross with annotated geometry."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Parameters
    theta_E = 1.0
    a = 0.05
    b = 0.15
    beta = 0.08
    phi_beta = np.radians(30)
    phi_gamma = np.radians(20)
    
    # Generate images
    images, phi_sols = generate_synthetic_cross(theta_E, a, b, beta, phi_beta, phi_gamma)
    
    # Einstein ring (perfect circle)
    theta = np.linspace(0, 2*np.pi, 100)
    ring_x = theta_E * np.cos(theta)
    ring_y = theta_E * np.sin(theta)
    ax.plot(ring_x, ring_y, 'b--', linewidth=1.5, alpha=0.5, label='Einstein Ring')
    
    # Distorted ring (with quadrupole)
    r_dist = theta_E + a * np.cos(2 * (theta - phi_gamma))
    dist_x = r_dist * np.cos(theta)
    dist_y = r_dist * np.sin(theta)
    ax.plot(dist_x, dist_y, 'b-', linewidth=2, alpha=0.7, label='Distorted Ring')
    
    # Lens center
    ax.plot(0, 0, 'ko', markersize=12, label='Lens Center')
    ax.plot(0, 0, 'k+', markersize=20, markeredgewidth=2)
    
    # Source position
    source_x = beta * np.cos(phi_beta)
    source_y = beta * np.sin(phi_beta)
    ax.plot(source_x, source_y, 'r*', markersize=15, label='Source', zorder=5)
    
    # Image positions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (x, y) in enumerate(images):
        ax.plot(x, y, 'o', color=colors[i], markersize=12, 
                markeredgecolor='black', markeredgewidth=1.5)
        ax.annotate(f'Image {i+1}', (x, y), xytext=(10, 10),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Quadrupole axis
    axis_len = 1.3
    ax.plot([axis_len * np.cos(phi_gamma), -axis_len * np.cos(phi_gamma)],
            [axis_len * np.sin(phi_gamma), -axis_len * np.sin(phi_gamma)],
            'g--', linewidth=1.5, alpha=0.7, label=f'Quadrupole Axis (φ_γ={np.degrees(phi_gamma):.0f}°)')
    
    # Source offset arrow
    ax.annotate('', xy=(source_x, source_y), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(source_x/2 - 0.05, source_y/2 + 0.05, 'β', fontsize=14, color='red')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (Einstein radii)', fontsize=12)
    ax.set_ylabel('y (Einstein radii)', fontsize=12)
    ax.set_title('Einstein Cross Geometry\nNo-Fit Inversion Model', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    
    # Add parameter box
    params_text = (f'θ_E = {theta_E}\n'
                   f'a = {a}\n'
                   f'b = {b}\n'
                   f'β = {beta}\n'
                   f'φ_β = {np.degrees(phi_beta):.0f}°\n'
                   f'φ_γ = {np.degrees(phi_gamma):.0f}°')
    ax.text(0.02, 0.98, params_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
           family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'geometry_einstein_cross.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'geometry_einstein_cross.pdf'))
    plt.close()
    print("Saved: geometry_einstein_cross.png/pdf")


def plot_quadrupole_effect():
    """Show how quadrupole distorts the ring."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    theta = np.linspace(0, 2*np.pi, 200)
    theta_E = 1.0
    
    configs = [
        {'a': 0.0, 'b': 0.0, 'title': 'Perfect Ring\n(a=0, b=0)'},
        {'a': 0.1, 'b': 0.0, 'title': 'Radial Quadrupole\n(a=0.1, b=0)'},
        {'a': 0.0, 'b': 0.15, 'title': 'Tangential Quadrupole\n(a=0, b=0.15)'},
    ]
    
    phi_gamma = np.radians(30)
    
    for ax, cfg in zip(axes, configs):
        a, b = cfg['a'], cfg['b']
        
        # Perfect ring
        ax.plot(np.cos(theta), np.sin(theta), 'b--', alpha=0.3, linewidth=1)
        
        # Distorted ring
        r = theta_E + a * np.cos(2 * (theta - phi_gamma))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, 'b-', linewidth=2, label='Ring shape')
        
        # Shear effect (tangential)
        if b != 0:
            for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                r0 = theta_E
                x0 = r0 * np.cos(phi)
                y0 = r0 * np.sin(phi)
                # Tangential displacement
                dx = -b * np.sin(2 * (phi - phi_gamma)) * np.sin(phi)
                dy = b * np.sin(2 * (phi - phi_gamma)) * np.cos(phi)
                ax.arrow(x0, y0, dx*0.5, dy*0.5, head_width=0.05,
                        head_length=0.02, fc='red', ec='red', alpha=0.7)
        
        # Quadrupole axis
        ax.plot([1.2*np.cos(phi_gamma), -1.2*np.cos(phi_gamma)],
               [1.2*np.sin(phi_gamma), -1.2*np.sin(phi_gamma)],
               'g--', alpha=0.5, linewidth=1.5)
        
        ax.plot(0, 0, 'ko', markersize=8)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(cfg['title'], fontsize=12)
        ax.set_xlabel('x / θ_E')
        ax.set_ylabel('y / θ_E')
    
    plt.suptitle('Quadrupole Distortion Effects', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'geometry_quadrupole_effect.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'geometry_quadrupole_effect.pdf'))
    plt.close()
    print("Saved: geometry_quadrupole_effect.png/pdf")


def plot_image_formation():
    """Show image formation for different source positions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    theta_E = 1.0
    a = 0.05
    b = 0.15
    phi_gamma = np.radians(20)
    
    # Different source positions
    source_configs = [
        {'beta': 0.0, 'phi_beta': 0, 'title': 'On-axis (β=0)'},
        {'beta': 0.05, 'phi_beta': np.radians(0), 'title': 'β=0.05, φ_β=0°'},
        {'beta': 0.1, 'phi_beta': np.radians(30), 'title': 'β=0.1, φ_β=30°'},
        {'beta': 0.15, 'phi_beta': np.radians(45), 'title': 'β=0.15, φ_β=45°'},
        {'beta': 0.2, 'phi_beta': np.radians(90), 'title': 'β=0.2, φ_β=90°'},
        {'beta': 0.3, 'phi_beta': np.radians(0), 'title': 'β=0.3 (2-image?)'},
    ]
    
    theta = np.linspace(0, 2*np.pi, 200)
    
    for ax, cfg in zip(axes.flat, source_configs):
        beta = cfg['beta']
        phi_beta = cfg['phi_beta']
        
        # Distorted ring
        r = theta_E + a * np.cos(2 * (theta - phi_gamma))
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'b-', linewidth=1.5, alpha=0.5)
        
        # Generate images
        images, _ = generate_synthetic_cross(theta_E, a, b, beta, phi_beta, phi_gamma)
        
        # Lens center
        ax.plot(0, 0, 'ko', markersize=8)
        
        # Source
        sx = beta * np.cos(phi_beta)
        sy = beta * np.sin(phi_beta)
        if beta > 0:
            ax.plot(sx, sy, 'r*', markersize=12)
        
        # Images
        colors = plt.cm.tab10(np.linspace(0, 1, len(images)))
        for i, ((x, y), c) in enumerate(zip(images, colors)):
            ax.plot(x, y, 'o', color=c, markersize=10, markeredgecolor='k')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{cfg["title"]}\n({len(images)} images)', fontsize=11)
    
    plt.suptitle('Image Formation vs Source Position', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'geometry_image_formation.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'geometry_image_formation.pdf'))
    plt.close()
    print("Saved: geometry_image_formation.png/pdf")


def plot_polar_view():
    """Show images in polar coordinates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    theta_E = 1.0
    a = 0.05
    b = 0.15
    beta = 0.08
    phi_beta = np.radians(30)
    phi_gamma = np.radians(20)
    
    images, phi_sols = generate_synthetic_cross(theta_E, a, b, beta, phi_beta, phi_gamma)
    
    # Left: Polar plot
    ax1 = plt.subplot(121, projection='polar')
    
    # Ring
    theta = np.linspace(0, 2*np.pi, 200)
    r_ring = theta_E + a * np.cos(2 * (theta - phi_gamma))
    ax1.plot(theta, r_ring, 'b-', linewidth=2, label='Distorted ring')
    ax1.plot(theta, np.ones_like(theta) * theta_E, 'b--', alpha=0.3, label='Einstein ring')
    
    # Images
    radii = np.sqrt(images[:, 0]**2 + images[:, 1]**2)
    angles = np.arctan2(images[:, 1], images[:, 0])
    ax1.scatter(angles, radii, s=100, c='red', zorder=5, edgecolors='black', linewidth=1.5)
    
    ax1.set_title('Polar View', fontsize=12)
    ax1.set_ylim(0, 1.5)
    ax1.legend(loc='upper right')
    
    # Right: r vs phi
    ax2 = axes[1]
    ax2.plot(np.degrees(theta), r_ring, 'b-', linewidth=2, label='Ring r(φ)')
    ax2.scatter(np.degrees(angles), radii, s=100, c='red', zorder=5,
               edgecolors='black', linewidth=1.5, label='Images')
    ax2.axhline(theta_E, color='gray', linestyle='--', alpha=0.5)
    
    # Quadrupole axis markers
    ax2.axvline(np.degrees(phi_gamma), color='green', linestyle=':', alpha=0.7)
    ax2.axvline(np.degrees(phi_gamma) + 90, color='green', linestyle=':', alpha=0.7)
    ax2.axvline(np.degrees(phi_gamma) + 180, color='green', linestyle=':', alpha=0.7)
    ax2.axvline(np.degrees(phi_gamma) + 270, color='green', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('φ (degrees)', fontsize=12)
    ax2.set_ylabel('r / θ_E', fontsize=12)
    ax2.set_title('Radius vs Azimuthal Angle', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-180, 180)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'geometry_polar_view.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'geometry_polar_view.pdf'))
    plt.close()
    print("Saved: geometry_polar_view.png/pdf")


if __name__ == '__main__':
    print("Generating geometry plots...")
    plot_einstein_cross_geometry()
    plot_quadrupole_effect()
    plot_image_formation()
    plot_polar_view()
    print("Done!")
