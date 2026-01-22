# General Framework for Gauge Gravitational Lensing

## A No-Fit Inversion Theory

**Authors:** Carmen N. Wrede, Lino P. Casu

---

## 1. Problem Statement

### 1.1 The Einstein Ring (Symmetric Case)

When a point source lies exactly behind a spherically symmetric lens:
- Source position: β = 0
- Result: Perfect Einstein ring at radius θ_E
- Geometry: All points on ring satisfy the lens equation identically

This is the **calibration case**: it defines the mass scale.

### 1.2 The General Case

Real lenses break symmetry through:

1. **Source offset**: β ≠ 0 (source not perfectly aligned)
2. **Lens anisotropy**: Ellipticity, external shear, substructure
3. **Multi-plane lensing**: Multiple deflectors along line of sight

Breaking symmetry transforms the ring into discrete images (typically 2 or 4 for galaxy lenses, more for clusters).

### 1.3 The Inversion Problem

**Given:** N image positions θ_i = (x_i, y_i), i = 1, ..., N

**Find:** Source position β and lens parameters (θ_E, multipole coefficients)

**Constraint:** No curve fitting. Exact algebraic solution or rootfinding only.

---

## 2. Gauge-Lensing Fundamental Equations

### 2.1 The Scaling Function

In the Radial Scaling Gauge, the metric deviation is encoded in:

```
s(r, φ) = 1 + Ξ(r, φ)
```

where Ξ → 0 as r → ∞ (asymptotic flatness).

### 2.2 Deflection Angle

The deflection angle arises from transverse gradients of Ξ:

```
α(θ) ≈ (1/π) ∫_{-∞}^{+∞} ∇_⊥ Ξ(r(ℓ), φ(ℓ)) dℓ
```

where the integral is along the unperturbed light path at impact parameter θ.

### 2.3 Lens Equation

The fundamental lens equation relates source and image positions:

```
β = θ - α_red(θ)
```

where α_red = (D_LS / D_S) α is the reduced deflection angle.

### 2.4 Einstein Radius

For a spherically symmetric lens, the Einstein radius satisfies:

```
θ_E = α_red(θ_E)
```

This is the characteristic angular scale of the lens.

---

## 3. Multipole Ansatz

### 3.1 Fourier Expansion of Ξ

We expand Ξ in angular harmonics:

```
Ξ(r, φ) = Ξ_0(r) + Σ_{m=1}^{m_max} [Ξ_{m,c}(r) cos(m(φ - φ_m)) + Ξ_{m,s}(r) sin(m(φ - φ_m))]
```

### 3.2 Physical Interpretation of Multipoles

| m | Name | Physical Origin |
|---|------|-----------------|
| 0 | Monopole | Total mass (sets θ_E) |
| 1 | Dipole | Center offset (usually zero) |
| 2 | Quadrupole | Ellipticity + external shear |
| 3 | Octupole | Triangular distortion |
| 4 | Hexadecapole | Box-shaped distortion |

### 3.3 Local Approximation at Einstein Radius

Near the Einstein ring, we can parametrize the deflection as:

```
α_red(r, φ) ≈ θ_E [1 + a(r/θ_E - 1) + b cos(2(φ - φ_γ)) + ...]
```

where:
- a = radial derivative of monopole (mass profile slope)
- b = quadrupole amplitude
- φ_γ = quadrupole orientation

### 3.4 Deflection Potential

For computational convenience, we can write:

```
α = ∇ψ
```

where ψ is the deflection potential. The multipole expansion then becomes:

```
ψ(r, φ) = ψ_0(r) + Σ_m [A_m(r) cos(m(φ - φ_m)) + B_m(r) sin(m(φ - φ_m))]
```

---

## 4. Image Multiplicity and Caustics

### 4.1 Caustic Structure

Caustics are curves in the source plane where the lens mapping is singular. Crossing a caustic changes the number of images by ±2.

### 4.2 Quadrupole Caustic (m = 2)

For dominant quadrupole anisotropy:
- **Diamond caustic** (astroid shape) in source plane
- Source outside diamond: 2 images
- Source inside diamond: 4 images
- Source on caustic: 3 images (2 merge)

### 4.3 Higher Multipoles

| m | Caustic Shape | Max Images |
|---|---------------|------------|
| 2 | Diamond (4 cusps) | 4 |
| 3 | Triangle (3 cusps) | 6 |
| 4 | Square (4 cusps) | 8 |

In practice, m = 2 dominates for most galaxy-scale lenses.

### 4.4 Cross Regime Condition

For 4 images with m = 2 model:

```
|b| > |β| / 2
```

If violated, only 2 images form.

---

## 5. Inversion Logic Without Fitting

### 5.1 Degree of Freedom Bookkeeping

**Equations from N images:**
- 2N scalar equations (x and y components of lens equation)

**Unknowns for m_max model:**
- β_x, β_y: 2 (source position)
- θ_E: 1 (Einstein radius)
- Per multipole m: amplitude + phase = 2 unknowns
- Total for m = 2: 2 + 1 + 2 = 5 unknowns (plus 1 nonlinear phase)

### 5.2 The Closure Condition

System is **exactly determined** when:

```
2N = number of linear unknowns + number of consistency equations
```

For minimal model (m = 2) with N = 4 images:
- 8 equations
- 5 linear unknowns (β_x, β_y, θ_E, a, b)
- 1 nonlinear unknown (φ_γ)
- Strategy: 5 equations for linear solve, 1 for rootfinding, 2 for validation

### 5.3 Inversion Algorithm

```
1. CHOOSE model order m_max (start small)
2. IDENTIFY nonlinear variables (phases φ_m)
3. FOR each candidate phase configuration:
   a. BUILD linear system for remaining unknowns
   b. SELECT invertible subset of equations
   c. SOLVE exactly (no lstsq)
   d. COMPUTE residuals on ALL equations
4. FILTER solutions by residual threshold
5. IF no valid solution:
   - Residuals structured? → Increase m_max
   - Residuals random? → Check data quality or center
6. RETURN all valid solutions + diagnostics
```

### 5.4 Conditional Linearity

**Key insight:** For fixed phases, the system is LINEAR in amplitudes.

This allows:
- Phases: determined by rootfinding (few variables)
- Amplitudes: determined by exact linear solve
- Validation: residuals check (not minimization)

### 5.5 Root Selection

When multiple roots exist:
1. Compute residuals for each
2. Check physical constraints (θ_E > 0, etc.)
3. Return ALL valid solutions
4. Do NOT pick "best" by minimization

---

## 6. Determinability and Degeneracies

### 6.1 What IS Determinable from Image Positions

| Quantity | Determinable? | Notes |
|----------|---------------|-------|
| Source direction β/|β| | Yes | From image centroid offset |
| Source offset |β| | Yes | From image asymmetry |
| θ_E | Yes | From mean image radius |
| Quadrupole b | Yes | From image spread |
| Quadrupole axis φ_γ | Yes (mod 90°) | From image orientation |
| Radial slope a | Weakly | Degenerate with β |

### 6.2 What is NOT Determinable

| Quantity | Why Not | What Breaks It |
|----------|---------|----------------|
| Absolute distances | Only angles observed | Redshifts + cosmology |
| Mass-sheet transform | λ(1-κ) + κ scaling | Time delays Δt |
| Lens center (if unknown) | Shifts all angles | Lens light centroid |
| Source size | Point source assumed | Extended source arcs |

### 6.3 Mass-Sheet Degeneracy

The transformation:

```
κ → λκ + (1 - λ)
β → λβ
```

leaves all image positions invariant. This is the **gauge/scaling degeneracy**.

**Broken by:**
- Time delays: Δt ∝ (1 - λ)
- Stellar kinematics: σ_v² ∝ M ∝ κ
- Magnification ratios (with absolute flux calibration)

---

## 7. Extension Protocol

### 7.1 Adding Time Delays

Time delay between images i and j:

```
Δt_ij = (1 + z_L) (D_L D_S / D_LS) [½(θ_i - β)² - ½(θ_j - β)² - ψ(θ_i) + ψ(θ_j)]
```

Each time delay adds 1 equation. With 4 images, 3 independent Δt.

### 7.2 Adding Flux Ratios

Magnification:

```
μ_i = 1 / |det(∂β/∂θ)|_{θ_i}
```

Flux ratios add equations but are affected by:
- Microlensing (stellar-scale substructure)
- Dust extinction
- Intrinsic source variability

Use with caution; best as consistency check, not primary constraint.

### 7.3 Adding More Images

Some lenses have 5, 6, or more images (especially clusters). Each image adds 2 equations without adding unknowns.

### 7.4 Extended Sources

Arc geometry provides:
- Curvature → mass profile slope
- Width → source size
- Multiple points along arc → many constraints

---

## 8. Practical Implementation Notes

### 8.1 Numerical Stability

- Use pivoted LU decomposition for linear solves
- Bracketing for rootfinding (guaranteed convergence)
- Check condition number before inverting

### 8.2 Residual Interpretation

| Residual Pattern | Suggests |
|------------------|----------|
| All small (< 10⁻⁸) | Model adequate |
| Correlated with angle | Missing higher m |
| Correlated with radius | Wrong center or profile |
| Random, large | Data error or wrong model class |

### 8.3 Recommended Workflow

```
1. Start with m_max = 2 (minimal model)
2. If residuals show angular pattern → try m_max = 3
3. If still bad → check lens center assumption
4. If still bad → consider multi-plane or exotic models
5. Report ALL valid solutions with degeneracy notes
```

---

## 9. Summary

This framework provides:

1. **Exact inversion** without curve fitting
2. **Transparent degeneracies** instead of error bars
3. **Modular extension** for additional observables
4. **Diagnostic residuals** for model adequacy

The key insight is **conditional linearity**: fixing the few nonlinear phase variables allows exact linear solution for all other parameters.

---

## References

1. Schneider, P., Ehlers, J., Falco, E. E., *Gravitational Lenses*, Springer (1992)
2. Kochanek, C. S., "Strong Gravitational Lensing," in *Dark Matter and Dark Energy*, Springer (2004)
3. Wrede, C. N., Casu, L. P., *Radial Scaling Gauge for Maxwell Fields* (2025)

---

*General Framework document for gauge-gravitationslinse-quadratur*
