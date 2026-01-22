"""
Tests for Ring vs Cross Morphology Classification.

Tests:
R1: Perfect ring detection
R2: Shear ring detection  
R3: Ring -> Cross transition
R4: Quad classification
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.morphology import (
    MorphologyClassifier, Morphology, 
    estimate_ring_center, estimate_ring_radius
)
from models.ring_analysis import RingAnalyzer, generate_ring_points


class TestMorphologyClassifier:
    """Test morphology classification."""

    def test_perfect_ring_classified_as_ring(self):
        """R1: Perfect ring should be classified as RING."""
        # Generate perfect ring (50 points, no perturbation)
        positions = generate_ring_points(theta_E=1.0, n_points=50, noise=0.01)
        
        classifier = MorphologyClassifier()
        result = classifier.classify(positions)
        
        print(f"\nPerfect Ring Analysis:")
        print(f"  Morphology: {result.primary.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Radial scatter: {result.radial_scatter:.4f}")
        print(f"  Azimuthal coverage: {result.azimuthal_coverage:.2f}")
        print(f"  Recommended: {result.recommended_models}")
        
        assert result.primary == Morphology.RING
        assert result.radial_scatter < 0.05
        assert "isotropic" in result.recommended_models

    def test_shear_ring_detected(self):
        """R2: Ring with shear should show m=2 component."""
        # Generate ring with m=2 perturbation (elliptical)
        positions = generate_ring_points(
            theta_E=1.0, n_points=50, 
            c2=0.1, s2=0.05,  # Strong quadrupole
            noise=0.01
        )
        
        classifier = MorphologyClassifier()
        result = classifier.classify(positions)
        
        print(f"\nShear Ring Analysis:")
        print(f"  Morphology: {result.primary.value}")
        print(f"  m2 amplitude: {result.m2_amplitude:.4f}")
        print(f"  Recommended: {result.recommended_models}")
        
        assert result.m2_amplitude > 0.05
        assert any("shear" in m or "m2" in m for m in result.recommended_models)

    def test_quad_classified_as_quad(self):
        """R4: 4 discrete images should be QUAD."""
        # Generate 4 discrete quad images
        angles = np.array([0.3, 1.8, 3.5, 5.2])
        r = 1.0
        positions = np.column_stack([
            r * np.cos(angles),
            r * np.sin(angles)
        ])
        
        classifier = MorphologyClassifier()
        result = classifier.classify(positions)
        
        print(f"\nQuad Analysis:")
        print(f"  Morphology: {result.primary.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Azimuthal coverage: {result.azimuthal_coverage:.2f}")
        print(f"  Recommended: {result.recommended_models}")
        
        assert result.primary == Morphology.QUAD
        assert "m2" in result.recommended_models

    def test_ring_to_cross_transition(self):
        """R3: Increasing m2 should transition from ring to non-ring."""
        print("\nRing -> Cross Transition:")
        
        classifier = MorphologyClassifier()
        
        for c2 in [0.0, 0.05, 0.1, 0.2, 0.3]:
            # Generate ring with increasing quadrupole
            positions = generate_ring_points(
                theta_E=1.0, n_points=20, c2=c2, noise=0.01
            )
            result = classifier.classify(positions)
            
            print(f"  c2={c2:.2f}: {result.primary.value}, "
                  f"scatter={result.radial_scatter:.3f}, "
                  f"m2={result.m2_amplitude:.3f}")


class TestRingAnalyzer:
    """Test ring geometry fitting."""

    def test_perfect_ring_fit(self):
        """Should fit perfect ring accurately."""
        positions = generate_ring_points(theta_E=1.5, n_points=30, noise=0.005)
        
        analyzer = RingAnalyzer()
        result = analyzer.fit_ring(positions)
        
        print(f"\nRing Fit Result:")
        print(f"  Radius: {result.radius:.4f} (expected 1.5)")
        print(f"  Center: ({result.center_x:.4f}, {result.center_y:.4f})")
        print(f"  RMS residual: {result.rms_residual:.4f}")
        print(f"  Perturbation: {result.perturbation_type}")
        
        assert abs(result.radius - 1.5) < 0.1
        assert result.perturbation_type == "isotropic"

    def test_perturbed_ring_detects_m2(self):
        """Should detect quadrupole perturbation."""
        positions = generate_ring_points(
            theta_E=1.0, n_points=50,
            c2=0.1, s2=0.0,  # Pure cos(2φ)
            noise=0.005
        )
        
        analyzer = RingAnalyzer()
        result = analyzer.fit_ring(positions)
        
        print(f"\nPerturbed Ring (m=2):")
        print(f"  m2 amplitude: {result.m2_component[0]:.4f}")
        print(f"  m4 amplitude: {result.m4_component[0]:.4f}")
        print(f"  Perturbation: {result.perturbation_type}")
        
        assert result.m2_component[0] > result.m4_component[0]
        assert "quadrupole" in result.perturbation_type

    def test_m4_perturbation_detected(self):
        """Should detect hexadecapole (m=4) perturbation."""
        positions = generate_ring_points(
            theta_E=1.0, n_points=50,
            c4=0.08, s4=0.0,  # Pure cos(4φ)
            noise=0.005
        )
        
        analyzer = RingAnalyzer()
        result = analyzer.fit_ring(positions)
        
        print(f"\nPerturbed Ring (m=4):")
        print(f"  m2 amplitude: {result.m2_component[0]:.4f}")
        print(f"  m4 amplitude: {result.m4_component[0]:.4f}")
        print(f"  Perturbation: {result.perturbation_type}")
        
        assert result.m4_component[0] > 0.05

    def test_off_center_ring(self):
        """Should find correct center for off-center ring."""
        true_center = (0.2, -0.1)
        positions = generate_ring_points(
            theta_E=1.0, n_points=40,
            center=true_center,
            noise=0.01
        )
        
        analyzer = RingAnalyzer()
        result = analyzer.fit_ring(positions)
        
        print(f"\nOff-Center Ring:")
        print(f"  True center: {true_center}")
        print(f"  Found center: ({result.center_x:.4f}, {result.center_y:.4f})")
        
        assert abs(result.center_x - true_center[0]) < 0.05
        assert abs(result.center_y - true_center[1]) < 0.05


class TestCenterEstimation:
    """Test algebraic center estimation."""

    def test_estimate_ring_center(self):
        """Should estimate center correctly."""
        true_center = (0.1, -0.05)
        positions = generate_ring_points(
            theta_E=1.0, n_points=20, center=true_center, noise=0.005
        )
        
        cx, cy = estimate_ring_center(positions)
        
        print(f"\nCenter Estimation:")
        print(f"  True: {true_center}")
        print(f"  Estimated: ({cx:.4f}, {cy:.4f})")
        
        assert abs(cx - true_center[0]) < 0.02
        assert abs(cy - true_center[1]) < 0.02

    def test_estimate_ring_radius(self):
        """Should estimate radius correctly."""
        true_radius = 1.2
        positions = generate_ring_points(theta_E=true_radius, n_points=30)
        
        radius = estimate_ring_radius(positions)
        
        print(f"\nRadius Estimation:")
        print(f"  True: {true_radius}")
        print(f"  Estimated: {radius:.4f}")
        
        assert abs(radius - true_radius) < 0.05
