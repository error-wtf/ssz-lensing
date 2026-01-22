"""
Model Family Definitions - Wraps existing solver configurations.

Add-only: Does NOT modify existing solvers, just defines which to run.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List


class ModelFamily(Enum):
    """Available model families (matches existing solvers)."""
    # Original 4 models
    M2 = "m2"
    M2_SHEAR = "m2_shear"
    M2_M3 = "m2_m3"
    M2_SHEAR_M3 = "m2_shear_m3"
    # Extended with m4 (Add-only: keeping derivation visible)
    M2_M4 = "m2_m4"
    M2_SHEAR_M4 = "m2_shear_m4"
    M2_M3_M4 = "m2_m3_m4"
    M2_SHEAR_M3_M4 = "m2_shear_m3_m4"  # Maximal model


@dataclass
class ModelConfig:
    """Configuration for a model family."""
    family: ModelFamily
    m_max: int
    include_shear: bool
    include_m3: bool = True  # Whether to include m=3 (for m4-only models)
    include_m4: bool = False
    label: str = ""
    n_lens_params: int = 0  # Without source params

    @property
    def param_names(self) -> List[str]:
        """Parameter names for this model."""
        names = ['theta_E', 'c_2', 's_2']
        if self.include_shear:
            names.extend(['gamma_1', 'gamma_2'])
        if self.include_m3 and self.m_max >= 3:
            names.extend(['c_3', 's_3'])
        if self.include_m4 and self.m_max >= 4:
            names.extend(['c_4', 's_4'])
        return names


# Pre-defined configurations matching existing solvers
MODEL_CONFIGS = {
    ModelFamily.M2: ModelConfig(
        family=ModelFamily.M2,
        m_max=2,
        include_shear=False,
        label="m=2 only",
        n_lens_params=3
    ),
    ModelFamily.M2_SHEAR: ModelConfig(
        family=ModelFamily.M2_SHEAR,
        m_max=2,
        include_shear=True,
        label="m=2 + shear",
        n_lens_params=5
    ),
    ModelFamily.M2_M3: ModelConfig(
        family=ModelFamily.M2_M3,
        m_max=3,
        include_shear=False,
        label="m=2 + m=3",
        n_lens_params=5
    ),
    ModelFamily.M2_SHEAR_M3: ModelConfig(
        family=ModelFamily.M2_SHEAR_M3,
        m_max=3,
        include_shear=True,
        label="m=2 + shear + m=3",
        n_lens_params=7
    ),
    # m4 extensions (Add-only: derivation visible)
    ModelFamily.M2_M4: ModelConfig(
        family=ModelFamily.M2_M4,
        m_max=4,
        include_shear=False,
        include_m3=False,
        include_m4=True,
        label="m=2 + m=4",
        n_lens_params=5
    ),
    ModelFamily.M2_SHEAR_M4: ModelConfig(
        family=ModelFamily.M2_SHEAR_M4,
        m_max=4,
        include_shear=True,
        include_m3=False,
        include_m4=True,
        label="m=2 + shear + m=4",
        n_lens_params=7
    ),
    ModelFamily.M2_M3_M4: ModelConfig(
        family=ModelFamily.M2_M3_M4,
        m_max=4,
        include_shear=False,
        include_m3=True,
        include_m4=True,
        label="m=2 + m=3 + m=4",
        n_lens_params=7
    ),
    ModelFamily.M2_SHEAR_M3_M4: ModelConfig(
        family=ModelFamily.M2_SHEAR_M3_M4,
        m_max=4,
        include_shear=True,
        include_m3=True,
        include_m4=True,
        label="m=2 + shear + m=3 + m=4 (maximal)",
        n_lens_params=9
    ),
}


def get_derivation_chain(include_m4: bool = False) -> List[ModelFamily]:
    """
    Return models in derivation order.
    
    Shows stepwise improvement chain:
    m=2 -> +shear -> +m=3 -> +m=4 -> combinations
    
    Args:
        include_m4: If True, include m4 models in chain
    """
    chain = [
        ModelFamily.M2,
        ModelFamily.M2_SHEAR,
        ModelFamily.M2_M3,
        ModelFamily.M2_SHEAR_M3,
    ]
    if include_m4:
        chain.extend([
            ModelFamily.M2_M4,
            ModelFamily.M2_SHEAR_M4,
            ModelFamily.M2_M3_M4,
            ModelFamily.M2_SHEAR_M3_M4,
        ])
    return chain
