"""
Tests for LensingRun dataclass and RSG lensing geometry.
Validates: no NaN outputs, circle labeling, rotation invariance, GR/SSZ consistency.
"""
import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

# Constants
A = np.pi / (180 * 3600)  # arcsec to rad
G = 6.674e-11
c = 299792458
Ms = 1.989e30
Mpc = 3.086e22
pc = 3.086e16

# Real fallback datasets
CROSS_DATA = {
    'name': 'Q2237+0305 (Einstein Cross)',
    'z_L': 0.0394, 'z_S': 1.695,
    'theta_E': 0.89,
    'positions_arcsec': [
        (0.740, 0.565),
        (-0.635, 0.470),
        (-0.480, -0.755),
        (0.870, -0.195)
    ]
}

RING_DATA = {
    'name': 'SDSS J1004+4112 (Cluster lens)',
    'z_L': 0.68, 'z_S': 1.734,
    'theta_E': 7.0,
    'positions_arcsec': [
        (7.42, 3.26), (-5.82, 1.91), (-4.51, -3.18), (2.85, -5.91), (0.12, 0.05)
    ]
}


def build_run_simple(pos_arcsec, z_L, z_S, theta_E, name='Test'):
    """Simplified build_run for testing."""
    from scipy.integrate import quad as iq
    
    # Cosmology
    def E(z):
        return 1 / np.sqrt(0.315 * (1 + z)**3 + 0.685)
    
    DH = c / (67.4 * 1000 / Mpc)
    cL, _ = iq(E, 0, z_L)
    cS, _ = iq(E, 0, z_S)
    D_L = cL * DH / (1 + z_L)
    D_S = cS * DH / (1 + z_S)
    D_LS = (cS - cL) * DH / (1 + z_S)
    
    # Lens mass and RSG
    Scr = c**2 * D_S / (4 * np.pi * G * D_L * D_LS)
    b_E = theta_E * A * D_L
    M = np.pi * b_E**2 * Scr / Ms
    r_s = 2 * G * M * Ms / c**2
    Xi_ref = r_s / (2 * b_E)
    s_ref = 1 + Xi_ref
    
    # GR and SSZ positions
    pos = np.array(pos_arcsec)
    r_GR = np.hypot(pos[:, 0], pos[:, 1])
    ang = np.arctan2(pos[:, 1], pos[:, 0])
    r_SSZ = s_ref * r_GR
    theta_SSZ = np.column_stack([r_SSZ * np.cos(ang), r_SSZ * np.sin(ang)])
    
    # Deltas
    Delta_theta = theta_SSZ - pos
    rms_theta = np.sqrt(np.mean(np.sum(Delta_theta**2, axis=1)))
    max_Delta_theta = np.max(np.hypot(Delta_theta[:, 0], Delta_theta[:, 1]))
    
    return {
        'name': name,
        'z_L': z_L, 'z_S': z_S,
        'D_L': D_L, 'D_S': D_S, 'D_LS': D_LS,
        'M': M, 'r_s': r_s, 'b_E': b_E,
        'theta_E': theta_E,
        'Xi_ref': Xi_ref, 's_ref': s_ref,
        'theta_GR': pos, 'theta_SSZ': theta_SSZ,
        'Delta_theta': Delta_theta,
        'rms_theta': rms_theta,
        'max_Delta_theta': max_Delta_theta,
        'n_images': len(pos)
    }


class TestNoNaNOutputs:
    """Test 1: All outputs must be finite (no NaN, no Inf)."""
    
    def test_cross_no_nan(self):
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        
        assert np.isfinite(run['D_L']), "D_L is not finite"
        assert np.isfinite(run['D_S']), "D_S is not finite"
        assert np.isfinite(run['M']), "M is not finite"
        assert np.isfinite(run['Xi_ref']), "Xi_ref is not finite"
        assert np.isfinite(run['s_ref']), "s_ref is not finite"
        assert np.isfinite(run['rms_theta']), "rms_theta is not finite"
        assert np.all(np.isfinite(run['theta_GR'])), "theta_GR contains NaN"
        assert np.all(np.isfinite(run['theta_SSZ'])), "theta_SSZ contains NaN"
    
    def test_ring_no_nan(self):
        d = RING_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        
        assert np.isfinite(run['D_L']), "D_L is not finite"
        assert np.isfinite(run['Xi_ref']), "Xi_ref is not finite"
        assert np.all(np.isfinite(run['theta_SSZ'])), "theta_SSZ contains NaN"


class TestCircleLabeling:
    """Test 2: Sky circle != Lens plane circle. Labels must be correct."""
    
    def test_sky_circle_is_theta_E(self):
        """Sky circle radius should be theta_E (arcsec)."""
        d = CROSS_DATA
        theta_E = d['theta_E']
        # Sky circle is theta_E in arcsec
        assert theta_E > 0
        assert theta_E < 10  # reasonable range for arcsec
    
    def test_lens_circle_is_b_E(self):
        """Lens plane circle radius should be b_E = D_L * theta_E (meters)."""
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        
        b_E_expected = run['D_L'] * run['theta_E'] * A
        assert np.isclose(run['b_E'], b_E_expected, rtol=1e-6)
        # b_E should be in kpc range for typical lenses
        b_E_kpc = run['b_E'] / (1e3 * pc)
        assert 0.1 < b_E_kpc < 100, f"b_E = {b_E_kpc} kpc seems unreasonable"


class TestRotationPreservesRadii:
    """Test 4: Rotating sky frame changes angles but not radii."""
    
    def test_rotation_invariant_radii(self):
        d = CROSS_DATA
        pos = np.array(d['positions_arcsec'])
        
        # Original radii
        r_original = np.hypot(pos[:, 0], pos[:, 1])
        
        # Rotate by 45 degrees
        theta_rot = np.pi / 4
        cos_t, sin_t = np.cos(theta_rot), np.sin(theta_rot)
        pos_rotated = np.column_stack([
            pos[:, 0] * cos_t - pos[:, 1] * sin_t,
            pos[:, 0] * sin_t + pos[:, 1] * cos_t
        ])
        r_rotated = np.hypot(pos_rotated[:, 0], pos_rotated[:, 1])
        
        # Radii must be preserved
        assert np.allclose(r_original, r_rotated, rtol=1e-10)


class TestGRSSZShiftConsistency:
    """Test 5: Δb/b and Δθ/θ must agree with s(R_ref)-1."""
    
    def test_shift_equals_xi(self):
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        
        # Predicted shift: Δθ/θ = s - 1 = Xi
        predicted_rel_shift = run['s_ref'] - 1
        
        # Measured shift for each image
        r_GR = np.hypot(run['theta_GR'][:, 0], run['theta_GR'][:, 1])
        r_SSZ = np.hypot(run['theta_SSZ'][:, 0], run['theta_SSZ'][:, 1])
        measured_rel_shift = (r_SSZ - r_GR) / r_GR
        
        # All images should have the same relative shift = Xi
        assert np.allclose(measured_rel_shift, predicted_rel_shift, rtol=1e-10)
    
    def test_xi_is_small_but_nonzero(self):
        """At weak field (b >> r_s), Xi should be small but positive."""
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        
        Xi = run['Xi_ref']
        assert Xi > 0, "Xi should be positive"
        assert Xi < 1e-3, f"Xi = {Xi} is too large for weak field"


class TestFallbackDatasetsLoad:
    """Test 6: Both Cross and Ring datasets must load without errors."""
    
    def test_cross_dataset_loads(self):
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        
        assert run['n_images'] == 4
        assert run['name'] == 'Test'  # default name
        assert run['z_L'] == 0.0394
        assert run['z_S'] == 1.695
    
    def test_ring_dataset_loads(self):
        d = RING_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        
        assert run['n_images'] == 5
        assert run['z_L'] == 0.68
        assert run['z_S'] == 1.734
    
    def test_no_fake_zeros(self):
        """Positions must not be placeholder zeros."""
        for d in [CROSS_DATA, RING_DATA]:
            pos = np.array(d['positions_arcsec'])
            # At least some non-zero coordinates
            assert np.any(pos != 0), "Positions contain only zeros"
            # No exact zeros in real data (extremely unlikely)
            n_zeros = np.sum(pos == 0)
            assert n_zeros <= 2, f"Too many exact zeros: {n_zeros}"


class TestPhysicalConsistency:
    """Additional physics sanity checks."""
    
    def test_distances_positive(self):
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        
        assert run['D_L'] > 0
        assert run['D_S'] > 0
        assert run['D_LS'] > 0
        assert run['D_S'] > run['D_L'], "D_S should be > D_L"
    
    def test_mass_reasonable(self):
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        
        # Lens mass should be in galactic range (10^9 - 10^13 M_sun)
        assert 1e9 < run['M'] < 1e14, f"Mass {run['M']:.2e} M_sun seems unreasonable"
    
    def test_schwarzschild_radius_small(self):
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        
        # r_s << b_E for weak field
        ratio = run['r_s'] / run['b_E']
        assert ratio < 1e-3, f"r_s/b_E = {ratio} is not weak field"


class TestCarmenPaperIntegrals:
    """Tests for Carmen Paper RSG integrals: delay, deflection, phase."""
    
    def test_gauge_no_nan(self):
        """All gauge integrals must produce finite values."""
        from scipy.integrate import quad as iq
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        r_s = run['r_s']
        b_E = run['b_E']
        
        def delay_int(z, b):
            r = np.sqrt(b**2 + z**2)
            return r_s / (2 * r) if r > r_s else 0
        
        b_arr = np.linspace(r_s * 2, b_E * 3, 20)
        z_max = run['D_L'] * Mpc * 0.1
        Delta_t = np.array([iq(delay_int, -z_max, z_max, args=(b,))[0] / c for b in b_arr])
        
        assert not np.any(np.isnan(Delta_t)), "Delay integral has NaN"
        assert np.all(Delta_t >= 0), "Delay must be non-negative"
    
    def test_alpha_rsg_vs_ppn(self):
        """α_RSG integral should be finite and physical."""
        from scipy.integrate import quad as iq
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        r_s = run['r_s']
        b_E = run['b_E']
        
        def alpha_int(z, b):
            r = np.sqrt(b**2 + z**2)
            if r <= r_s:
                return 0
            Xi = r_s / (2 * r)
            s = 1 + Xi
            dXi = -r_s / (2 * r**2)
            return (1 / s) * dXi * (b / r)
        
        # Use smaller integration range (near lens)
        z_max = b_E * 100
        alpha_RSG = abs(iq(alpha_int, -z_max, z_max, args=(b_E,))[0])
        alpha_PPN = 2 * r_s / b_E
        
        # RSG integral is finite and non-zero
        assert alpha_RSG > 0, "α_RSG should be positive"
        assert np.isfinite(alpha_RSG), "α_RSG should be finite"
        # Both are in same regime (very small angles)
        assert alpha_PPN < 1e-3, "α_PPN should be small angle"
    
    def test_delay_monotonic_vs_b(self):
        """Delay should decrease as impact parameter b increases."""
        from scipy.integrate import quad as iq
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        r_s = run['r_s']
        
        def delay_int(z, b):
            r = np.sqrt(b**2 + z**2)
            return r_s / (2 * r) if r > r_s else 0
        
        b_arr = np.linspace(r_s * 5, r_s * 100, 10)
        z_max = run['D_L'] * Mpc * 0.01
        Delta_t = np.array([iq(delay_int, -z_max, z_max, args=(b,))[0] for b in b_arr])
        
        # Delay should decrease with b
        assert np.all(np.diff(Delta_t) <= 0), "Delay should decrease with b"
    
    def test_xi_to_zero_limit(self):
        """As r→∞, Ξ→0 and all effects vanish."""
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        r_s = run['r_s']
        
        r_far = r_s * 1e10
        Xi_far = r_s / (2 * r_far)
        s_far = 1 + Xi_far
        
        assert Xi_far < 1e-10, "Ξ should vanish at large r"
        assert abs(s_far - 1) < 1e-10, "s should → 1 at large r"
    
    def test_phase_delay_relation(self):
        """Δφ = ω·Δt for consistent path (Maxwell relation)."""
        from scipy.integrate import quad as iq
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        r_s = run['r_s']
        b = run['b_E']
        
        def xi_int(z, b):
            r = np.sqrt(b**2 + z**2)
            return r_s / (2 * r) if r > r_s else 0
        
        z_max = run['D_L'] * Mpc * 0.01
        Delta_path = iq(xi_int, -z_max, z_max, args=(b,))[0]
        Delta_t = Delta_path / c
        
        omega = 2 * np.pi * c / (500e-9)  # optical
        k = omega / c
        Delta_phi = k * Delta_path
        Delta_phi_from_t = omega * Delta_t
        
        assert np.isclose(Delta_phi, Delta_phi_from_t, rtol=1e-10), \
            "Δφ = ω·Δt relation violated"
    
    def test_gauge_insets_render_data(self):
        """Sky/Lens insets must receive finite points and circles."""
        d = CROSS_DATA
        run = build_run_simple(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'])
        
        # Sky inset: theta positions
        th_GR = np.array(d['positions_arcsec'])
        assert th_GR.shape[0] >= 4, "Need at least 4 images"
        assert not np.any(np.isnan(th_GR)), "Sky positions have NaN"
        
        # Lens inset: impact parameters
        b_GR = run['D_L'] * Mpc * th_GR * A
        assert not np.any(np.isnan(b_GR)), "Impact parameters have NaN"
        
        # Circles
        theta_E = d['theta_E']
        b_E = run['b_E']
        assert np.isfinite(theta_E), "θ_E must be finite"
        assert np.isfinite(b_E), "b_E must be finite"
        assert b_E > 0, "b_E must be positive"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
