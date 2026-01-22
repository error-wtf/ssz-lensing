#!/usr/bin/env python3
"""
Extended Visualization Suite - Framework Overview and Decision Logic
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

from models.regime_classifier import RegimeClassifier, Regime, UnderdeterminedExplorer
from models.dual_path_inversion import AlgebraicSolver, PhaseScanSolver
from dataio.datasets import generate_cross_images

np.random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_framework_overview():
    """Complete framework overview with all three paths."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('RSG Lensing Inversion: Complete Framework', 
                 fontsize=16, fontweight='bold')
    
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Panel 1: Input data
    ax1 = fig.add_subplot(gs[0, 0])
    imgs, _ = generate_cross_images(theta_E=1.0, beta=0.1, b=0.15)
    ax1.scatter(imgs[:, 0], imgs[:, 1], s=100, c=['r', 'g', 'b', 'orange'])
    ax1.scatter(0, 0, s=200, marker='*', c='black')
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('INPUT: Image Positions')
    
    # Panel 2: Regime Classification
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    regime_text = """
    REGIME CLASSIFIER
    ─────────────────────
    
    Input: Design matrix A
    
    ┌─────────────────────┐
    │  SVD Analysis       │
    │  • Rank             │
    │  • Condition        │
    │  • Nullspace        │
    └─────────────────────┘
    
    Output: Regime + Path
    """
    ax2.text(0.5, 0.5, regime_text, transform=ax2.transAxes,
             fontsize=10, va='center', ha='center', family='monospace',
             bbox=dict(facecolor='#e3f2fd', edgecolor='#1976d2', lw=2))
    
    # Panel 3: Output
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    output_text = """
    OUTPUTS
    ─────────────────────
    
    Parameters:
    • θ_E (Einstein radius)
    • (a_m, b_m) multipoles
    • (γ₁, γ₂) shear
    • β source position
    
    Diagnostics:
    • Residuals
    • Identifiability
    • Degeneracy info
    """
    ax3.text(0.5, 0.5, output_text, transform=ax3.transAxes,
             fontsize=10, va='center', ha='center', family='monospace',
             bbox=dict(facecolor='#e8f5e9', edgecolor='#388e3c', lw=2))
    
    # Panel 4: Path A
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    path_a_text = """
    PATH A: ALGEBRAIC
    ═══════════════════════
    
    Parametrization:
      (a_m, b_m) components
    
    Method:
      A·x = b  →  lstsq
    
    Phase:
      φ_m = atan2(b_m, a_m)
      OUTPUT (derived)
    
    Properties:
      ✓ Deterministic
      ✓ Fast
      ✓ Canonical
    """
    ax4.text(0.5, 0.5, path_a_text, transform=ax4.transAxes,
             fontsize=9, va='center', ha='center', family='monospace',
             bbox=dict(facecolor='#bbdefb', edgecolor='#1976d2', lw=2))
    
    # Panel 5: Path B
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    path_b_text = """
    PATH B: PHASE SCAN
    ═══════════════════════
    
    Parametrization:
      (A_m, φ_m) amplitude/phase
    
    Method:
      For each φ: solve linear
      → residual landscape
    
    Phase:
      INPUT (scanned)
    
    Properties:
      ✓ Degeneracy visible
      ✓ Hypothesis test
      ○ Grid-dependent
    """
    ax5.text(0.5, 0.5, path_b_text, transform=ax5.transAxes,
             fontsize=9, va='center', ha='center', family='monospace',
             bbox=dict(facecolor='#ffcdd2', edgecolor='#d32f2f', lw=2))
    
    # Panel 6: Path C
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    path_c_text = """
    PATH C: EXPLORER
    ═══════════════════════
    
    For: UNDERDETERMINED
    
    Method:
      1. Compute nullspace
      2. Particular solution
      3. Generate family
      4. Apply regularizers
    
    Output:
      • Solution space
      • Parameter ranges
      • Non-identifiable
    
    Properties:
      ✓ Never rejects
      ✓ Learns from all
    """
    ax6.text(0.5, 0.5, path_c_text, transform=ax6.transAxes,
             fontsize=9, va='center', ha='center', family='monospace',
             bbox=dict(facecolor='#fff9c4', edgecolor='#f9a825', lw=2))
    
    # Panel 7-9: Decision flowchart
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    flow_text = """
    ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                    DECISION FLOWCHART                                        │
    ├─────────────────────────────────────────────────────────────────────────────────────────────┤
    │                                                                                              │
    │    [Input Images]  →  [Build A, b]  →  [Classify Regime]                                    │
    │                                              │                                               │
    │              ┌───────────────┬───────────────┼───────────────┬───────────────┐              │
    │              ▼               ▼               ▼               ▼               ▼              │
    │         DETERMINED     OVERDETERMINED   UNDERDETERMINED  ILL-CONDITIONED                    │
    │              │               │               │               │                              │
    │              ▼               ▼               ▼               ▼                              │
    │          Path A          Path A          Path C          Path A+B                           │
    │         (unique)      (+ residual      (explore        (with uncertainty)                   │
    │                        check)           nullspace)                                          │
    │                                                                                              │
    └─────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    ax7.text(0.5, 0.5, flow_text, transform=ax7.transAxes,
             fontsize=9, va='center', ha='center', family='monospace',
             bbox=dict(facecolor='#f5f5f5', edgecolor='#424242', lw=2))
    
    plt.savefig(os.path.join(OUTPUT_DIR, '06_framework_overview.png'), dpi=150)
    plt.close()
    print("Generated: 06_framework_overview.png")


def plot_regime_decision_tree():
    """Visual decision tree for regime selection."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Regime Classification Decision Tree', fontsize=14, fontweight='bold')
    
    # Boxes
    boxes = [
        (5, 9, 'INPUT\nn_constraints, n_params', '#e3f2fd'),
        (5, 7, 'rank(A) < min(C,P)?', '#fff3e0'),
        (2, 5, 'ILL-CONDITIONED\ncond > 10¹⁰', '#ffcdd2'),
        (5, 5, 'C > P?', '#fff3e0'),
        (3, 3, 'OVERDETERMINED\nresidual = model check', '#c8e6c9'),
        (7, 3, 'C < P?', '#fff3e0'),
        (5, 1, 'DETERMINED\nunique solution', '#c8e6c9'),
        (9, 1, 'UNDERDETERMINED\nexplore nullspace', '#ffecb3'),
    ]
    
    for x, y, text, color in boxes:
        box = FancyBboxPatch((x-1.2, y-0.6), 2.4, 1.2,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', lw=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9)
    
    # Arrows with labels
    arrows = [
        ((5, 8.4), (5, 7.6), ''),
        ((5, 6.4), (2, 5.6), 'YES'),
        ((5, 6.4), (5, 5.6), 'NO'),
        ((5, 4.4), (3, 3.6), 'YES'),
        ((5, 4.4), (7, 3.6), 'NO'),
        ((7, 2.4), (5, 1.6), 'NO'),
        ((7, 2.4), (9, 1.6), 'YES'),
    ]
    
    for start, end, label in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        if label:
            mid = ((start[0]+end[0])/2, (start[1]+end[1])/2)
            ax.text(mid[0]+0.3, mid[1], label, fontsize=8, color='blue')
    
    plt.savefig(os.path.join(OUTPUT_DIR, '07_decision_tree.png'), dpi=150)
    plt.close()
    print("Generated: 07_decision_tree.png")


def plot_sensitivity_comparison():
    """Compare sensitivity across regimes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Sensitivity Analysis Across Regimes', fontsize=14, fontweight='bold')
    
    noise_levels = np.logspace(-10, -2, 20)
    
    # Well-conditioned
    ax1 = axes[0, 0]
    A_good = np.eye(8) + 0.1 * np.random.randn(8, 8)
    b = np.ones(8)
    x_clean = np.linalg.lstsq(A_good, b, rcond=None)[0]
    
    variations = []
    for noise in noise_levels:
        np.random.seed(42)
        b_noisy = b + noise * np.random.randn(len(b))
        x_noisy = np.linalg.lstsq(A_good, b_noisy, rcond=None)[0]
        variations.append(np.linalg.norm(x_noisy - x_clean))
    
    ax1.loglog(noise_levels, variations, 'g-', lw=2)
    ax1.set_title(f'DETERMINED (cond={np.linalg.cond(A_good):.1f})')
    ax1.set_xlabel('Data noise')
    ax1.set_ylabel('Parameter change')
    ax1.grid(True, alpha=0.3)
    
    # Ill-conditioned
    ax2 = axes[0, 1]
    A_bad = np.eye(8)
    A_bad[7, 7] = 1e-10
    x_clean = np.linalg.lstsq(A_bad, b, rcond=None)[0]
    
    variations = []
    for noise in noise_levels:
        np.random.seed(42)
        b_noisy = b + noise * np.random.randn(len(b))
        x_noisy = np.linalg.lstsq(A_bad, b_noisy, rcond=None)[0]
        variations.append(np.linalg.norm(x_noisy - x_clean))
    
    ax2.loglog(noise_levels, variations, 'r-', lw=2)
    ax2.set_title(f'ILL-CONDITIONED (cond={np.linalg.cond(A_bad):.1e})')
    ax2.set_xlabel('Data noise')
    ax2.set_ylabel('Parameter change')
    ax2.grid(True, alpha=0.3)
    
    # Overdetermined
    ax3 = axes[1, 0]
    A_over = np.random.randn(12, 8)
    b_over = np.random.randn(12)
    x_clean = np.linalg.lstsq(A_over, b_over, rcond=None)[0]
    
    variations = []
    for noise in noise_levels:
        np.random.seed(42)
        b_noisy = b_over + noise * np.random.randn(len(b_over))
        x_noisy = np.linalg.lstsq(A_over, b_noisy, rcond=None)[0]
        variations.append(np.linalg.norm(x_noisy - x_clean))
    
    ax3.loglog(noise_levels, variations, 'b-', lw=2)
    ax3.set_title(f'OVERDETERMINED (12C vs 8P)')
    ax3.set_xlabel('Data noise')
    ax3.set_ylabel('Parameter change')
    ax3.grid(True, alpha=0.3)
    
    # Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary = """
    SENSITIVITY INSIGHTS
    ════════════════════════════════════
    
    DETERMINED (well-cond):
      Linear response to noise
      Stable solutions
    
    ILL-CONDITIONED:
      Exponential amplification!
      Small noise → huge changes
      ⚠️ Results unreliable
    
    OVERDETERMINED:
      Averaging reduces noise
      Most robust regime
    
    UNDERDETERMINED:
      Infinite solutions
      Noise moves within nullspace
      → Use regularization
    """
    ax4.text(0.5, 0.5, summary, transform=ax4.transAxes,
             fontsize=10, va='center', ha='center', family='monospace',
             bbox=dict(facecolor='#f5f5f5', edgecolor='#424242'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '08_sensitivity.png'), dpi=150)
    plt.close()
    print("Generated: 08_sensitivity.png")


def plot_solution_space():
    """Visualize solution space for underdetermined system."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Underdetermined: Solution Space Visualization', 
                 fontsize=14, fontweight='bold')
    
    # 2D example: 1 constraint, 2 params
    ax1 = axes[0]
    x = np.linspace(-2, 2, 100)
    # Constraint: 2*p1 + p2 = 1  →  p2 = 1 - 2*p1
    y = 1 - 2*x
    ax1.plot(x, y, 'b-', lw=2, label='All solutions (line)')
    ax1.scatter([0.2], [0.6], s=100, c='red', zorder=5, label='Min-norm')
    ax1.scatter([0.5], [0], s=100, c='green', zorder=5, label='Sparse')
    ax1.set_xlabel('Parameter 1')
    ax1.set_ylabel('Parameter 2')
    ax1.set_title('1 constraint, 2 params\n(1D solution line)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.axhline(0, color='k', lw=0.5)
    ax1.axvline(0, color='k', lw=0.5)
    
    # 3D example: 1 constraint, 3 params
    ax2 = axes[1]
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    zz = 1 - xx - yy  # Constraint: p1 + p2 + p3 = 1
    ax2.plot_surface(xx, yy, zz, alpha=0.5, color='blue')
    ax2.scatter([1/3], [1/3], [1/3], s=100, c='red', label='Min-norm')
    ax2.set_xlabel('P1')
    ax2.set_ylabel('P2')
    ax2.set_zlabel('P3')
    ax2.set_title('1 constraint, 3 params\n(2D solution plane)')
    
    # Regularizer comparison
    ax3 = axes[2]
    regularizers = ['Min-Norm\n(Occam)', 'Sparse\n(L1)', 'Smooth\n(multipole)']
    values = [0.58, 0.5, 0.65]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax3.bar(regularizers, values, color=colors)
    ax3.set_ylabel('Solution norm')
    ax3.set_title('Same residual,\ndifferent regularizers')
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    for bar, v in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.02, 
                 f'{v:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '09_solution_space.png'), dpi=150)
    plt.close()
    print("Generated: 09_solution_space.png")


def plot_learning_insights():
    """Key learning insights from regime analysis."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    ax.set_title('Key Learning Insights from Regime Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    insights = """
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║                           LEARNING FROM ALL REGIMES                                    ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                        ║
    ║  1. DETERMINED SYSTEM                                                                  ║
    ║     ───────────────────                                                                ║
    ║     • Unique solution exists                                                           ║
    ║     • Residual = numerical noise only                                                  ║
    ║     • USE: Path A (algebraic) - fast, deterministic                                    ║
    ║                                                                                        ║
    ║  2. OVERDETERMINED SYSTEM                                                              ║
    ║     ─────────────────────                                                              ║
    ║     • More constraints than needed                                                     ║
    ║     • Residual = MODEL ADEQUACY CHECK                                                  ║
    ║       - Low residual: model fits well                                                  ║
    ║       - High residual: model insufficient (need higher multipoles?)                    ║
    ║     • USE: Path A, interpret residual as diagnostic                                    ║
    ║                                                                                        ║
    ║  3. UNDERDETERMINED SYSTEM                                                             ║
    ║     ───────────────────────                                                            ║
    ║     • Infinitely many solutions fit equally well                                       ║
    ║     • OLD: "FORBIDDEN" → abort                                                         ║
    ║     • NEW: Explore! What CAN we learn?                                                 ║
    ║       - Which parameters ARE identifiable?                                             ║
    ║       - What is the degeneracy structure?                                              ║
    ║       - What additional data would help?                                               ║
    ║     • USE: Path C (Explorer) with explicit regularizers                                ║
    ║                                                                                        ║
    ║  4. ILL-CONDITIONED SYSTEM                                                             ║
    ║     ───────────────────────                                                            ║
    ║     • Solution exists but is SENSITIVE                                                 ║
    ║     • Small data errors → large parameter errors                                       ║
    ║     • Often: near-degenerate configuration (close to caustic?)                         ║
    ║     • USE: Path A+B with uncertainty quantification                                    ║
    ║                                                                                        ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                        ║
    ║  PARADIGM SHIFT: Don't abort on "bad" regimes - LEARN from them!                      ║
    ║                                                                                        ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, insights, transform=ax.transAxes,
            fontsize=10, va='center', ha='center', family='monospace',
            bbox=dict(facecolor='#fafafa', edgecolor='#333', lw=2))
    
    plt.savefig(os.path.join(OUTPUT_DIR, '10_learning_insights.png'), dpi=150)
    plt.close()
    print("Generated: 10_learning_insights.png")


def run_all():
    print("Generating extended plots...")
    plot_framework_overview()
    plot_regime_decision_tree()
    plot_sensitivity_comparison()
    plot_solution_space()
    plot_learning_insights()
    print(f"\nAll extended plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    run_all()
