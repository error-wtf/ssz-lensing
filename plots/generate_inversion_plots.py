#!/usr/bin/env python3
"""
Visualization Suite for Dual-Path Lensing Inversion
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from models.regime_classifier import RegimeClassifier, Regime, UnderdeterminedExplorer
from models.dual_path_inversion import AlgebraicSolver, PhaseScanSolver
from dataio.datasets import generate_cross_images

np.random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_regime_overview():
    """Plot 1: All 4 regimes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Regime Classification Overview', fontsize=14, fontweight='bold')
    
    data = [
        ('DETERMINED', np.eye(8) + 0.1*np.random.randn(8,8), 8),
        ('OVERDETERMINED', np.random.randn(12, 8), 8),
        ('UNDERDETERMINED', np.random.randn(6, 10), 10),
        ('ILL-CONDITIONED', np.diag([1,1,1,1,1,1,1,1e-12]), 8),
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for ax, (name, A, n), color in zip(axes.flat, data, colors):
        analysis = RegimeClassifier.classify(A, [f'p{i}' for i in range(n)])
        ax.imshow(np.abs(A), cmap='Blues', aspect='auto')
        ax.set_title(f'{name}\n{analysis.n_constraints}C, {n}P', color=color, fontweight='bold')
        ax.text(0.98, 0.02, f'Rank:{analysis.rank}\nNull:{analysis.nullspace_dim}',
                transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_regime_overview.png'), dpi=150)
    plt.close()
    print("Generated: 01_regime_overview.png")


def plot_nullspace():
    """Plot 2: Nullspace exploration."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Underdetermined: Nullspace Exploration', fontsize=14, fontweight='bold')
    
    A, b = np.random.randn(6, 10), np.random.randn(6)
    params = ['θE', 'a2', 'b2', 'a3', 'b3', 'a4', 'b4', 'g1', 'g2', 'bx']
    
    analysis = RegimeClassifier.classify(A, params)
    explorer = UnderdeterminedExplorer(params)
    result = explorer.explore(A, b, analysis)
    
    # Parameter ranges
    ranges = list(result.parameter_ranges.values())
    axes[0].barh(range(len(params)), [r[1]-r[0] for r in ranges], 
                 left=[r[0] for r in ranges], color='#e74c3c', alpha=0.7)
    axes[0].set_yticks(range(len(params)))
    axes[0].set_yticklabels(params)
    axes[0].set_title('Parameter Ranges (degeneracy)')
    axes[0].axvline(0, color='gray', linestyle='--')
    
    # Nullspace basis
    if analysis.nullspace_basis is not None:
        im = axes[1].imshow(analysis.nullspace_basis.T, cmap='RdBu', aspect='auto')
        axes[1].set_title(f'Nullspace Basis ({analysis.nullspace_dim}D)')
        plt.colorbar(im, ax=axes[1])
    
    # Regularizers
    regs = [s.regularizer_used for s in result.solutions]
    res = [s.residual for s in result.solutions]
    axes[2].bar(regs, res, color=['#2ecc71', '#3498db'][:len(regs)])
    axes[2].set_ylabel('Residual')
    axes[2].set_title('Different Regularizers')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_nullspace.png'), dpi=150)
    plt.close()
    print("Generated: 02_nullspace.png")


def plot_phase_scan():
    """Plot 3: Phase scan residual landscape."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phase Scan: Residual Landscape', fontsize=14, fontweight='bold')
    
    imgs, _ = generate_cross_images(theta_E=1.0, beta=0.1, b=0.15)
    result = PhaseScanSolver(m_max=2).scan_phases_then_solve_linear(
        [imgs], phi_2_range=(0, np.pi, 72))
    
    phases = [c.phases.get('phi_2', 0) for c in result.all_candidates]
    residuals = [c.residual for c in result.all_candidates]
    
    axes[0].plot(phases, residuals, 'b-', lw=2)
    axes[0].axvline(phases[np.argmin(residuals)], color='r', linestyle='--')
    axes[0].set_xlabel('Phase φ₂ (rad)')
    axes[0].set_ylabel('Residual')
    axes[0].set_title('Residual vs Phase')
    
    # Degeneracy
    threshold = min(residuals) * 1.1
    colors = ['#e74c3c' if r < threshold else '#3498db' for r in residuals]
    axes[1].scatter(phases, residuals, c=colors, s=30)
    axes[1].axhline(threshold, color='orange', linestyle='--')
    axes[1].set_title(f'Degeneracy: {sum(r < threshold for r in residuals)} pts within 10%')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_phase_scan.png'), dpi=150)
    plt.close()
    print("Generated: 03_phase_scan.png")


def plot_path_comparison():
    """Plot 4: Path A vs B."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Path A vs Path B Comparison', fontsize=14, fontweight='bold')
    
    imgs, _ = generate_cross_images(theta_E=1.0, beta=0.1, b=0.15)
    result_a = AlgebraicSolver(m_max=2).solve([imgs])
    result_b = PhaseScanSolver(m_max=2).scan_phases_then_solve_linear(
        [imgs], (0, np.pi, 36), result_a)
    
    # Residuals
    res = [result_a.max_residual, 
           result_b.best_candidate.residual if result_b.best_candidate else 0]
    axes[0].bar(['Path A\n(Algebraic)', 'Path B\n(Scan)'], res, color=['#3498db', '#e74c3c'])
    axes[0].set_ylabel('Residual')
    axes[0].set_title('Residual Comparison')
    
    # Phases
    phi_a = result_a.derived_phases.get('phi_2', 0)
    phi_b = result_b.best_candidate.phases.get('phi_2', 0) if result_b.best_candidate else 0
    theta = np.linspace(0, 2*np.pi, 100)
    axes[1].plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)
    axes[1].arrow(0, 0, 0.8*np.cos(phi_a), 0.8*np.sin(phi_a), head_width=0.1, color='#3498db')
    axes[1].arrow(0, 0, 0.6*np.cos(phi_b), 0.6*np.sin(phi_b), head_width=0.1, color='#e74c3c')
    axes[1].set_title(f'Phases: A={phi_a:.2f}, B={phi_b:.2f}')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_path_comparison.png'), dpi=150)
    plt.close()
    print("Generated: 04_path_comparison.png")


def plot_dof_rescue():
    """Plot 5: DOF rescue with multi-source."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('DOF Rescue: Adding Sources', fontsize=14, fontweight='bold')
    
    scenarios = [('1 src, m=2', 8, 5), ('1 src, m=4', 8, 9), ('2 src, m=4', 16, 11)]
    x = np.arange(len(scenarios))
    
    constraints = [s[1] for s in scenarios]
    params = [s[2] for s in scenarios]
    
    ax.bar(x - 0.2, constraints, 0.4, label='Constraints', color='#3498db')
    ax.bar(x + 0.2, params, 0.4, label='Parameters', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in scenarios])
    ax.legend()
    ax.set_ylabel('Count')
    
    for i, (c, p) in enumerate(zip(constraints, params)):
        status = 'OK' if c >= p else 'UNDER'
        ax.text(i, max(c, p) + 0.5, status, ha='center', fontweight='bold',
                color='#2ecc71' if c >= p else '#e74c3c')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_dof_rescue.png'), dpi=150)
    plt.close()
    print("Generated: 05_dof_rescue.png")


def run_all():
    print("Generating plots...")
    plot_regime_overview()
    plot_nullspace()
    plot_phase_scan()
    plot_path_comparison()
    plot_dof_rescue()
    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    run_all()
