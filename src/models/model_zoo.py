"""
Model Zoo: Parallel Model Families for Lensing Inversion

Four models running in parallel, NOT replacing each other:
1. Model_M2: Pure quadrupole (m=2 only)
2. Model_M2_Shear: Quadrupole + external shear
3. Model_M2_M3: Quadrupole + octupole (m=3)
4. Model_M2_Shear_M3: Full model (only legal with extended observables)

All use LINEAR parametrization: (c_m, s_m) not (A_m, φ_m)
                                (γ_1, γ_2) not (γ, φ_γ)

Authors: Carmen N. Wrede, Lino P. Casu
License: Anti-Capitalist Software License v1.4
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ModelType(Enum):
    """Available lens models."""
    M2 = "m2"
    M2_SHEAR = "m2_shear"
    M2_M3 = "m2_m3"
    M2_SHEAR_M3 = "m2_shear_m3"


class ObservableType(Enum):
    """Types of observables that provide constraints."""
    IMAGE_POSITION = "image_position"      # 2 constraints per image
    FLUX_RATIO = "flux_ratio"              # 1 constraint per ratio
    TIME_DELAY = "time_delay"              # 1 constraint per delay
    ARC_POINT = "arc_point"                # 2 constraints per point
    MULTI_SOURCE = "multi_source"          # Adds source params but more constraints


@dataclass
class Observable:
    """A single observable measurement."""
    obs_type: ObservableType
    value: np.ndarray  # Position (x,y), ratio, or delay
    uncertainty: float = 0.0
    source_id: int = 0  # For multi-source: which background source


@dataclass
class LensSystem:
    """Complete lens system with all observables."""
    name: str
    images: List[np.ndarray]  # List of image positions per source
    flux_ratios: Optional[np.ndarray] = None  # Relative flux ratios
    time_delays: Optional[np.ndarray] = None  # Relative time delays
    arc_points: Optional[np.ndarray] = None   # Extended arc point positions
    n_sources: int = 1
    
    def count_constraints(self) -> int:
        """Count total constraints from all observables."""
        n = 0
        # Image positions: 2 per image
        for imgs in self.images:
            n += 2 * len(imgs)
        # Flux ratios: N-1 independent ratios for N images
        if self.flux_ratios is not None:
            n += len(self.flux_ratios)
        # Time delays: N-1 independent delays for N images
        if self.time_delays is not None:
            n += len(self.time_delays)
        # Arc points: 2 per point
        if self.arc_points is not None:
            n += 2 * len(self.arc_points)
        return n


@dataclass
class ModelSpec:
    """Specification of a lens model."""
    model_type: ModelType
    include_m2: bool = True
    include_shear: bool = False
    include_m3: bool = False
    
    def param_names(self, n_sources: int = 1) -> List[str]:
        """Get ordered list of parameter names."""
        params = ['theta_E']
        if self.include_m2:
            params.extend(['c_2', 's_2'])
        if self.include_shear:
            params.extend(['gamma_1', 'gamma_2'])
        if self.include_m3:
            params.extend(['c_3', 's_3'])
        # Source positions
        for k in range(n_sources):
            params.extend([f'beta_x_{k}', f'beta_y_{k}'])
        return params
    
    def n_params(self, n_sources: int = 1) -> int:
        """Count number of parameters."""
        return len(self.param_names(n_sources))
    
    @property
    def label(self) -> str:
        """Human-readable label."""
        parts = []
        if self.include_m2:
            parts.append("m=2")
        if self.include_shear:
            parts.append("shear")
        if self.include_m3:
            parts.append("m=3")
        return " + ".join(parts)


# Pre-defined model specs
MODEL_SPECS = {
    ModelType.M2: ModelSpec(
        model_type=ModelType.M2,
        include_m2=True, include_shear=False, include_m3=False
    ),
    ModelType.M2_SHEAR: ModelSpec(
        model_type=ModelType.M2_SHEAR,
        include_m2=True, include_shear=True, include_m3=False
    ),
    ModelType.M2_M3: ModelSpec(
        model_type=ModelType.M2_M3,
        include_m2=True, include_shear=False, include_m3=True
    ),
    ModelType.M2_SHEAR_M3: ModelSpec(
        model_type=ModelType.M2_SHEAR_M3,
        include_m2=True, include_shear=True, include_m3=True
    ),
}


@dataclass
class RegimeStatus:
    """Result of regime gate check."""
    allowed: bool
    regime: str  # "MINIMAL", "STANDARD", "OVERDETERMINED", "FORBIDDEN"
    n_constraints: int
    n_params: int
    dof: int  # constraints - params
    missing_info: List[str] = field(default_factory=list)
    recommendation: str = ""


class RegimeGate:
    """
    Gate that checks if a model is allowed given available constraints.
    
    FORBIDDEN does NOT mean abort - it means "need more g1 observables".
    """
    
    @classmethod
    def check(
        cls,
        model_spec: ModelSpec,
        lens_system: LensSystem
    ) -> RegimeStatus:
        """
        Check if model is allowed for this lens system.
        
        Returns RegimeStatus with:
        - allowed: True if n_constraints >= n_params
        - regime: Classification string
        - missing_info: What observables would make it legal
        """
        n_constraints = lens_system.count_constraints()
        n_params = model_spec.n_params(lens_system.n_sources)
        dof = n_constraints - n_params
        
        if dof >= 2:
            return RegimeStatus(
                allowed=True,
                regime="OVERDETERMINED",
                n_constraints=n_constraints,
                n_params=n_params,
                dof=dof,
                recommendation="Good redundancy for consistency checks."
            )
        elif dof == 1:
            return RegimeStatus(
                allowed=True,
                regime="STANDARD",
                n_constraints=n_constraints,
                n_params=n_params,
                dof=dof,
                recommendation="Minimal redundancy. Consider adding observables."
            )
        elif dof == 0:
            return RegimeStatus(
                allowed=True,
                regime="MINIMAL",
                n_constraints=n_constraints,
                n_params=n_params,
                dof=dof,
                recommendation="Exactly determined. No consistency check possible."
            )
        else:
            # FORBIDDEN - but tell what's missing
            missing = abs(dof)
            missing_info = cls._suggest_observables(missing, lens_system)
            return RegimeStatus(
                allowed=False,
                regime="FORBIDDEN",
                n_constraints=n_constraints,
                n_params=n_params,
                dof=dof,
                missing_info=missing_info,
                recommendation=f"Need {missing} more constraint(s). See missing_info."
            )
    
    @classmethod
    def _suggest_observables(
        cls,
        n_missing: int,
        lens_system: LensSystem
    ) -> List[str]:
        """Suggest which observables could make the system legal."""
        suggestions = []
        
        # Flux ratios
        n_images = sum(len(imgs) for imgs in lens_system.images)
        if lens_system.flux_ratios is None and n_images >= 4:
            potential = n_images - 1
            suggestions.append(
                f"Add {min(n_missing, potential)} flux ratio(s) "
                f"(up to {potential} available from {n_images} images)"
            )
        
        # Time delays
        if lens_system.time_delays is None and n_images >= 4:
            potential = n_images - 1
            suggestions.append(
                f"Add {min(n_missing, potential)} time delay(s) "
                f"(up to {potential} available)"
            )
        
        # Arc points
        n_arc_needed = (n_missing + 1) // 2  # Each arc point gives 2 constraints
        suggestions.append(
            f"Add {n_arc_needed} arc point(s) from extended emission "
            f"(+{2*n_arc_needed} constraints)"
        )
        
        # Multi-source
        if lens_system.n_sources == 1:
            suggestions.append(
                f"Add second background source "
                f"(if 4 images: +8 constraints, -2 params = +6 net)"
            )
        
        return suggestions


@dataclass
class ZooResult:
    """Result from one model in the zoo."""
    model_type: ModelType
    regime_status: RegimeStatus
    params: Optional[Dict[str, float]] = None
    residuals: Optional[np.ndarray] = None
    max_residual: float = float('inf')
    condition_number: float = 0.0
    
    @property
    def success(self) -> bool:
        return self.params is not None and self.max_residual < 1.0


class ModelZoo:
    """
    Run all applicable models in parallel and compare.
    
    Key principle: Models are NOT mutually exclusive.
    Compare residuals to learn which physical effects matter.
    """
    
    def __init__(self, lens_system: LensSystem):
        self.lens_system = lens_system
        self.results: Dict[ModelType, ZooResult] = {}
    
    def run_all(self, include_forbidden: bool = False) -> Dict[ModelType, ZooResult]:
        """
        Run all models and return results.
        
        Args:
            include_forbidden: If True, still try FORBIDDEN models
                               (will fail but shows the constraint gap)
        """
        for model_type, spec in MODEL_SPECS.items():
            status = RegimeGate.check(spec, self.lens_system)
            
            if status.allowed or include_forbidden:
                result = self._solve_model(spec, status)
            else:
                result = ZooResult(
                    model_type=model_type,
                    regime_status=status,
                    params=None,
                    residuals=None,
                    max_residual=float('inf')
                )
            
            self.results[model_type] = result
        
        return self.results
    
    def _solve_model(
        self,
        spec: ModelSpec,
        status: RegimeStatus
    ) -> ZooResult:
        """Solve one model using linear algebra."""
        try:
            A, b = self._build_system(spec)
            
            if A.shape[0] < A.shape[1]:
                # Underdetermined - use pseudoinverse
                x = np.linalg.lstsq(A, b, rcond=None)[0]
            elif A.shape[0] == A.shape[1]:
                x = np.linalg.solve(A, b)
            else:
                # Overdetermined - exact subset solve
                x = np.linalg.lstsq(A, b, rcond=None)[0]
            
            residuals = A @ x - b
            max_res = np.max(np.abs(residuals))
            
            # Condition number
            s = np.linalg.svd(A, compute_uv=False)
            cond = s[0] / s[-1] if s[-1] > 1e-15 else float('inf')
            
            param_names = spec.param_names(self.lens_system.n_sources)
            params = dict(zip(param_names, x))
            
            return ZooResult(
                model_type=spec.model_type,
                regime_status=status,
                params=params,
                residuals=residuals,
                max_residual=max_res,
                condition_number=cond
            )
            
        except np.linalg.LinAlgError as e:
            return ZooResult(
                model_type=spec.model_type,
                regime_status=status,
                params=None,
                residuals=None,
                max_residual=float('inf')
            )
    
    def _build_system(self, spec: ModelSpec) -> Tuple[np.ndarray, np.ndarray]:
        """Build linear system A @ x = b."""
        n_sources = self.lens_system.n_sources
        param_names = spec.param_names(n_sources)
        n_params = len(param_names)
        n_constraints = self.lens_system.count_constraints()
        
        A = np.zeros((n_constraints, n_params))
        b = np.zeros(n_constraints)
        
        row = 0
        
        # Image position constraints
        for source_id, images in enumerate(self.lens_system.images):
            for x, y in images:
                phi = np.arctan2(y, x)
                cos_phi, sin_phi = np.cos(phi), np.sin(phi)
                
                col = 0
                
                # theta_E
                A[row, col] = cos_phi
                A[row + 1, col] = sin_phi
                col += 1
                
                # m=2 (c_2, s_2)
                if spec.include_m2:
                    cos_2, sin_2 = np.cos(2*phi), np.sin(2*phi)
                    A[row, col] = cos_2 * cos_phi
                    A[row, col + 1] = sin_2 * cos_phi
                    A[row + 1, col] = cos_2 * sin_phi
                    A[row + 1, col + 1] = sin_2 * sin_phi
                    col += 2
                
                # Shear (gamma_1, gamma_2)
                if spec.include_shear:
                    A[row, col] = x
                    A[row, col + 1] = y
                    A[row + 1, col] = -y
                    A[row + 1, col + 1] = x
                    col += 2
                
                # m=3 (c_3, s_3)
                if spec.include_m3:
                    cos_3, sin_3 = np.cos(3*phi), np.sin(3*phi)
                    A[row, col] = cos_3 * cos_phi
                    A[row, col + 1] = sin_3 * cos_phi
                    A[row + 1, col] = cos_3 * sin_phi
                    A[row + 1, col + 1] = sin_3 * sin_phi
                    col += 2
                
                # Source position for this source
                beta_col = col + 2 * source_id
                A[row, beta_col] = 1.0
                A[row + 1, beta_col + 1] = 1.0
                
                b[row] = x
                b[row + 1] = y
                row += 2
        
        # Arc point constraints (same as image positions but no source offset)
        if self.lens_system.arc_points is not None:
            for x, y in self.lens_system.arc_points:
                phi = np.arctan2(y, x)
                cos_phi, sin_phi = np.cos(phi), np.sin(phi)
                
                col = 0
                A[row, col] = cos_phi
                A[row + 1, col] = sin_phi
                col += 1
                
                if spec.include_m2:
                    cos_2, sin_2 = np.cos(2*phi), np.sin(2*phi)
                    A[row, col] = cos_2 * cos_phi
                    A[row, col + 1] = sin_2 * cos_phi
                    A[row + 1, col] = cos_2 * sin_phi
                    A[row + 1, col + 1] = sin_2 * sin_phi
                    col += 2
                
                if spec.include_shear:
                    A[row, col] = x
                    A[row, col + 1] = y
                    A[row + 1, col] = -y
                    A[row + 1, col + 1] = x
                    col += 2
                
                if spec.include_m3:
                    cos_3, sin_3 = np.cos(3*phi), np.sin(3*phi)
                    A[row, col] = cos_3 * cos_phi
                    A[row, col + 1] = sin_3 * cos_phi
                    A[row + 1, col] = cos_3 * sin_phi
                    A[row + 1, col + 1] = sin_3 * sin_phi
                    col += 2
                
                # Arc points share source 0 position
                A[row, col] = 1.0
                A[row + 1, col + 1] = 1.0
                
                b[row] = x
                b[row + 1] = y
                row += 2
        
        return A[:row], b[:row]
    
    def compare(self) -> str:
        """Generate comparison report of all models."""
        lines = ["=" * 60]
        lines.append("MODEL ZOO COMPARISON")
        lines.append("=" * 60)
        
        for model_type in ModelType:
            result = self.results.get(model_type)
            if result is None:
                continue
            
            status = result.regime_status
            lines.append(f"\n{MODEL_SPECS[model_type].label}")
            lines.append("-" * 40)
            lines.append(f"  Regime: {status.regime}")
            lines.append(f"  Constraints: {status.n_constraints}, Params: {status.n_params}")
            lines.append(f"  DOF: {status.dof}")
            
            if result.success:
                lines.append(f"  Max Residual: {result.max_residual:.2e}")
                lines.append(f"  Condition: {result.condition_number:.2e}")
            elif status.regime == "FORBIDDEN":
                lines.append("  Status: FORBIDDEN (insufficient constraints)")
                for hint in status.missing_info:
                    lines.append(f"    -> {hint}")
            else:
                lines.append("  Status: FAILED (numerical issue)")
        
        # Ranking
        valid = [(mt, r) for mt, r in self.results.items() if r.success]
        if valid:
            lines.append("\n" + "=" * 60)
            lines.append("RANKING BY RESIDUAL")
            lines.append("=" * 60)
            valid.sort(key=lambda x: x[1].max_residual)
            for i, (mt, r) in enumerate(valid, 1):
                lines.append(f"  {i}. {MODEL_SPECS[mt].label}: {r.max_residual:.2e}")
        
        return "\n".join(lines)
