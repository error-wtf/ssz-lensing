"""
Base Model Class for Gauge Gravitational Lensing

Abstract interface that all lens models must implement.
Enforces the no-fit policy through the API design.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np


class LensModel(ABC):
    """
    Abstract base class for gravitational lens models.
    
    All models must implement:
    - predict_images: Forward model (source -> images)
    - equations: Residual equations for inversion
    - unknowns: List of unknown parameters
    - observables: List of required observables
    - initial_guess: Heuristic starting point (NOT a fit)
    
    Design principle: No curve fitting, only exact solves and rootfinding.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass
    
    @property
    @abstractmethod
    def m_max(self) -> int:
        """Maximum multipole order in this model."""
        pass
    
    @abstractmethod
    def predict_images(
        self, 
        source_params: Dict[str, float],
        lens_params: Dict[str, float]
    ) -> np.ndarray:
        """
        Forward model: predict image positions from source and lens parameters.
        
        Parameters
        ----------
        source_params : dict
            Source parameters, typically {'beta_x': float, 'beta_y': float}
        lens_params : dict
            Lens parameters (model-specific)
            
        Returns
        -------
        images : ndarray, shape (N, 2)
            Predicted image positions (x, y) for each image
        """
        pass
    
    @abstractmethod
    def equations(
        self,
        images: np.ndarray,
        source_params: Dict[str, float],
        lens_params: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute residual vector: β_predicted - β_input for each image.
        
        For exact solution, all residuals should be zero.
        
        Parameters
        ----------
        images : ndarray, shape (N, 2)
            Observed image positions
        source_params : dict
            Source parameters
        lens_params : dict
            Lens parameters
            
        Returns
        -------
        residuals : ndarray, shape (2*N,)
            Residual for each equation (x and y components interleaved)
        """
        pass
    
    @abstractmethod
    def unknowns(self) -> List[str]:
        """
        List of unknown parameter names.
        
        Returns
        -------
        names : list of str
            Parameter names, e.g., ['beta_x', 'beta_y', 'theta_E', 'a', 'b']
        """
        pass
    
    @abstractmethod
    def nonlinear_unknowns(self) -> List[str]:
        """
        List of unknowns that must be determined by rootfinding.
        
        These are typically phase angles.
        
        Returns
        -------
        names : list of str
            Nonlinear parameter names, e.g., ['phi_gamma']
        """
        pass
    
    @abstractmethod
    def linear_unknowns(self) -> List[str]:
        """
        List of unknowns that can be solved linearly once nonlinear are fixed.
        
        Returns
        -------
        names : list of str
            Linear parameter names
        """
        pass
    
    @abstractmethod
    def observables(self) -> List[str]:
        """
        List of required observable types.
        
        Returns
        -------
        names : list of str
            Observable types, e.g., ['image_positions', 'time_delays']
        """
        pass
    
    @abstractmethod
    def build_linear_system(
        self,
        images: np.ndarray,
        fixed_nonlinear: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build the linear system Ax = b for linear unknowns.
        
        Parameters
        ----------
        images : ndarray, shape (N, 2)
            Observed image positions
        fixed_nonlinear : dict
            Values for nonlinear parameters (phases)
            
        Returns
        -------
        A : ndarray, shape (2*N, n_linear)
            Coefficient matrix
        b : ndarray, shape (2*N,)
            Right-hand side
        param_names : list of str
            Names of parameters in order (columns of A)
        """
        pass
    
    @abstractmethod
    def consistency_function(
        self,
        images: np.ndarray,
        nonlinear_value: float,
        nonlinear_name: str
    ) -> float:
        """
        Function whose zeros give valid nonlinear parameter values.
        
        For rootfinding: find values where this function = 0.
        
        Parameters
        ----------
        images : ndarray, shape (N, 2)
            Observed image positions
        nonlinear_value : float
            Trial value for the nonlinear parameter
        nonlinear_name : str
            Which nonlinear parameter
            
        Returns
        -------
        h : float
            Consistency function value (should be zero for valid solution)
        """
        pass
    
    @abstractmethod
    def initial_guess(self, images: np.ndarray) -> Dict[str, float]:
        """
        Compute heuristic initial guess from image positions.
        
        This is for initialization ONLY, not a final answer.
        Uses moment methods or geometric estimates.
        
        Parameters
        ----------
        images : ndarray, shape (N, 2)
            Observed image positions
            
        Returns
        -------
        guess : dict
            Initial parameter estimates
        """
        pass
    
    def count_equations(self, n_images: int, extra_obs: Optional[Dict] = None) -> int:
        """
        Count number of equations available.
        
        Parameters
        ----------
        n_images : int
            Number of observed images
        extra_obs : dict, optional
            Additional observables {'time_delays': n, 'flux_ratios': n, ...}
            
        Returns
        -------
        n_eq : int
            Total number of equations
        """
        n_eq = 2 * n_images  # x, y for each image
        
        if extra_obs:
            n_eq += extra_obs.get('time_delays', 0)
            n_eq += extra_obs.get('flux_ratios', 0)
            
        return n_eq
    
    def count_unknowns(self) -> int:
        """Count total number of unknown parameters."""
        return len(self.unknowns())
    
    def dof_balance(self, n_images: int, extra_obs: Optional[Dict] = None) -> Dict:
        """
        Check degree of freedom balance.
        
        Returns
        -------
        info : dict
            'n_equations', 'n_unknowns', 'n_linear', 'n_nonlinear',
            'overdetermined', 'underdetermined', 'message'
        """
        n_eq = self.count_equations(n_images, extra_obs)
        n_unk = self.count_unknowns()
        n_lin = len(self.linear_unknowns())
        n_nonlin = len(self.nonlinear_unknowns())
        
        info = {
            'n_equations': n_eq,
            'n_unknowns': n_unk,
            'n_linear': n_lin,
            'n_nonlinear': n_nonlin,
            'overdetermined': n_eq > n_unk,
            'underdetermined': n_eq < n_unk,
        }
        
        if n_eq > n_unk:
            info['message'] = f"Overdetermined by {n_eq - n_unk}. Can validate."
        elif n_eq < n_unk:
            info['message'] = f"Underdetermined by {n_unk - n_eq}. Need more observables or reduce model."
        else:
            info['message'] = "Exactly determined."
            
        return info
    
    def validate_params(self, params: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if parameters are physically valid.
        
        Returns
        -------
        valid : bool
        message : str
        """
        if 'theta_E' in params and params['theta_E'] <= 0:
            return False, "theta_E must be positive"
        return True, "OK"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(m_max={self.m_max})"
