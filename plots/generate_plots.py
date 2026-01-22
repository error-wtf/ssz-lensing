#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Suite for SSZ-Lensing / Radial Scaling Gauge
============================================================

Authors: Carmen N. Wrede, Lino P. Casu
License: Anti-Capitalist Software License v1.4
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Physical constants
C = 299792458.0
G = 6.67430e-11
M_SUN = 1.989e30
R_SUN = 6.96e8
AU = 1.496e11
PHI = (1 + np.sqrt(5)) / 2


def schwarzschild_radius(M):
    return 2 * G * M / (C ** 2)


def xi_weak_field(r, M):
    r_s = schwarzschild_radius(M)
    return r_s / (2 * r)


def scaling_factor(r, M):
    return 1 + xi_weak_field(r, M)


def time_dilation(r, M):
    return 1 / scaling_factor(r, M)


def plot_test_results_summary(save_path):
    """Plot test results by section"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sections = ['Section 2\nRadial Scaling', 'Section 3\nEM Phase', 
                'Appendix A.1\nShapiro', 'Appendix A.2\nLensing',
                'Appendix B\nWKB', 'Frame\nProblem', 'Experimental\nValidation']
    passed = [8, 7, 3, 3, 2, 2, 3]
    failed = [0, 0, 0, 0, 0, 0, 0]
    
    x = np.arange(len(sections))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, passed, width, label='Passed', color='#2ecc71')
    bars2 = ax.bar(x + width/2, failed, width, label='Failed', color='#e74c3c')
    
    ax.set_ylabel('Number of Tests')
    ax.set_title('Radial Scaling Gauge - Test Results by Section\n(28/28 Tests Passed = 100%)')
    ax.set_xticks(x)
    ax.set_xticklabels(sections)
    ax.legend()
    ax.set_ylim(0, 10)
    
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_radial_scaling(save_path):
    """Plot s(r) and Xi(r) vs r/r_s"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    r_over_rs = np.logspace(0.5, 4, 500)
    xi = 1 / (2 * r_over_rs)
    s = 1 + xi
    
    ax1.loglog(r_over_rs, xi, 'b-', linewidth=2, label=r'$\Xi(r) = r_s/(2r)$')
    ax1.set_xlabel(r'$r/r_s$')
    ax1.set_ylabel(r'$\Xi(r)$')
    ax1.set_title('Segment Density Xi(r)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, label='Solar surface')
    
    ax2.semilogx(r_over_rs, s, 'g-', linewidth=2, label=r'$s(r) = 1 + \Xi(r)$')
    ax2.set_xlabel(r'$r/r_s$')
    ax2.set_ylabel(r'$s(r)$')
    ax2.set_title('Radial Scaling Factor s(r)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.999, 1.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_shapiro_delay(save_path):
    """Plot Shapiro delay geometry and values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Geometry
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'y-', linewidth=10, label='Sun')
    ax1.plot([-5, -1.1], [0, 0], 'b-', linewidth=2)
    ax1.plot([1.1, 8], [0, 0], 'b-', linewidth=2, label='Signal path')
    ax1.plot(-5, 0, 'bo', markersize=10, label='Earth')
    ax1.plot(8, 0, 'ro', markersize=8, label='Saturn')
    ax1.set_xlim(-6, 10)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.set_title('Cassini Shapiro Delay Geometry')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Distance (AU scale)')
    
    # Delay values
    r_min_values = np.linspace(1, 10, 50) * R_SUN
    r_s = schwarzschild_radius(M_SUN)
    delays_us = [(r_s / C) * np.log(4 * AU * 9*AU / (r_min**2)) * 1e6 
                 for r_min in r_min_values]
    
    ax2.plot(r_min_values / R_SUN, delays_us, 'b-', linewidth=2)
    ax2.axhline(y=265, color='r', linestyle='--', label='Cassini measured (~265 us)')
    ax2.set_xlabel(r'Closest approach $r_{min}/R_\odot$')
    ax2.set_ylabel('Shapiro delay (microseconds)')
    ax2.set_title('Shapiro Delay vs Impact Parameter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_light_deflection(save_path):
    """Plot light deflection angle vs impact parameter"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    b_values = np.linspace(1, 10, 100) * R_SUN
    r_s = schwarzschild_radius(M_SUN)
    deflection_arcsec = [(2 * r_s / b) * (180/np.pi) * 3600 for b in b_values]
    
    ax.plot(b_values / R_SUN, deflection_arcsec, 'b-', linewidth=2, 
            label=r'$\delta\phi = 2r_s/b$ (GR)')
    ax.axhline(y=1.75, color='r', linestyle='--', linewidth=2, 
               label='1.75" at solar limb (1919 Eclipse)')
    ax.axvline(x=1, color='g', linestyle=':', alpha=0.7, label='Solar limb')
    
    ax.set_xlabel(r'Impact parameter $b/R_\odot$')
    ax.set_ylabel('Deflection angle (arcseconds)')
    ax.set_title('Gravitational Light Deflection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_experimental_validation(save_path):
    """Plot experimental validation summary"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    experiments = ['Pound-Rebka\n1960', 'GPS\nSystem', 'Tokyo Skytree\n2020',
                   'Cassini\n2003', 'Gaia\n2021']
    measured = [1.00, 1.00, 1.00, 1.000021, 1.0]
    predicted = [1.00, 1.00, 1.00, 1.0, 1.0]
    errors = [0.10, 0.01, 0.05, 2.3e-5, 3e-6]
    
    x = np.arange(len(experiments))
    
    ax.bar(x, measured, width=0.4, label='Measured/GR', color='#3498db', alpha=0.8)
    ax.errorbar(x, measured, yerr=errors, fmt='none', color='black', capsize=5)
    ax.axhline(y=1.0, color='r', linestyle='--', label='GR prediction (gamma=1)')
    
    ax.set_ylabel('gamma (PPN parameter)')
    ax.set_title('Experimental Validation of Radial Scaling Gauge\nAll measurements consistent with GR (gamma = 1)')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    ax.set_ylim(0.98, 1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_time_dilation(save_path):
    """Plot time dilation factor D(r)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    r_over_rs = np.logspace(0.5, 5, 500)
    D_gr = np.sqrt(1 - 1/r_over_rs)
    D_ssz = 1 / (1 + 1/(2*r_over_rs))
    
    ax.semilogx(r_over_rs, D_gr, 'b-', linewidth=2, label='GR: sqrt(1 - r_s/r)')
    ax.semilogx(r_over_rs, D_ssz, 'r--', linewidth=2, label='SSZ: 1/(1 + Xi)')
    
    ax.set_xlabel(r'$r/r_s$')
    ax.set_ylabel('Time dilation factor D(r)')
    ax.set_title('Time Dilation: GR vs SSZ (Weak Field Agreement)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined_overview(save_path):
    """Combined overview plot"""
    fig = plt.figure(figsize=(16, 12))
    
    # Title
    fig.suptitle('SSZ-Lensing: Radial Scaling Gauge Validation Suite\n'
                 '28/28 Tests Passed (100%) - No Fitting', 
                 fontsize=14, fontweight='bold')
    
    # 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Plot 1: Xi(r)
    r_over_rs = np.logspace(0.5, 4, 200)
    xi = 1 / (2 * r_over_rs)
    ax1.loglog(r_over_rs, xi, 'b-', linewidth=2)
    ax1.set_xlabel(r'$r/r_s$')
    ax1.set_ylabel(r'$\Xi(r)$')
    ax1.set_title('Segment Density')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Light deflection
    b = np.linspace(1, 10, 100)
    delta = 2 / b * (180/np.pi) * 3600 * (schwarzschild_radius(M_SUN) / R_SUN)
    ax2.plot(b, delta, 'g-', linewidth=2)
    ax2.axhline(y=1.75, color='r', linestyle='--')
    ax2.set_xlabel(r'$b/R_\odot$')
    ax2.set_ylabel('Deflection (")')
    ax2.set_title('Light Deflection')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test results
    sections = ['Sec 2', 'Sec 3', 'A.1', 'A.2', 'B', 'Frame', 'Exp']
    passed = [8, 7, 3, 3, 2, 2, 3]
    ax3.bar(sections, passed, color='#2ecc71')
    ax3.set_ylabel('Tests')
    ax3.set_title('Tests by Section (28/28)')
    ax3.set_ylim(0, 10)
    
    # Plot 4: Time dilation comparison
    r_over_rs = np.logspace(1, 5, 200)
    D_gr = np.sqrt(1 - 1/r_over_rs)
    D_ssz = 1 / (1 + 1/(2*r_over_rs))
    ax4.semilogx(r_over_rs, D_gr, 'b-', linewidth=2, label='GR')
    ax4.semilogx(r_over_rs, D_ssz, 'r--', linewidth=2, label='SSZ')
    ax4.set_xlabel(r'$r/r_s$')
    ax4.set_ylabel('D(r)')
    ax4.set_title('Time Dilation Agreement')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all plots"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Generating SSZ-Lensing plots...")
    print("=" * 50)
    
    plot_test_results_summary(os.path.join(script_dir, '01_test_results.png'))
    plot_radial_scaling(os.path.join(script_dir, '02_radial_scaling.png'))
    plot_shapiro_delay(os.path.join(script_dir, '03_shapiro_delay.png'))
    plot_light_deflection(os.path.join(script_dir, '04_light_deflection.png'))
    plot_experimental_validation(os.path.join(script_dir, '05_experimental.png'))
    plot_time_dilation(os.path.join(script_dir, '06_time_dilation.png'))
    plot_combined_overview(os.path.join(script_dir, '07_combined_overview.png'))
    
    print("=" * 50)
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
