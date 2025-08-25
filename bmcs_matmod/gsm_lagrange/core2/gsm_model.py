"""
GSM Model - High-level Interface

This module provides a high-level interface for GSM material models that combines
symbolic definitions (GSMSymbDef) with numerical execution (GSMEngine) and 
material parameters (MaterialParams).
"""

from typing import Tuple, Optional, List, Any
import numpy as np
import numpy.typing as npt

from bmcs_matmod.gsm_lagrange.core2.gsm_symb_def import GSMSymbDef
from bmcs_matmod.gsm_lagrange.core2.gsm_engine import GSMEngine
from bmcs_matmod.gsm_lagrange.core2.material_params import MaterialParams


class GSMModel:
    """
    High-level GSM material model interface.
    
    This class combines:
    - GSMSymbDef: Symbolic thermodynamic definitions
    - GSMEngine: Numerical execution engine
    - MaterialParams: Parameter repository
    
    It provides a simple interface for material response calculations while
    maintaining separation of concerns between symbolic definitions, numerical
    algorithms, and parameter management.
    """
    
    def __init__(self, 
                 symb_def: GSMSymbDef,
                 material_params: MaterialParams,
                 k_max: int = 10,
                 tol: float = 1e-8,
                 cache_dir: str = ".gsm_cache"):
        """
        Initialize GSM model.
        
        Args:
            symb_def: Symbolic thermodynamic definition
            material_params: Material parameters repository
            k_max: Maximum iterations for Newton-Raphson solver
            tol: Convergence tolerance
            cache_dir: Directory for disk caching
        """
        self.symb_def = symb_def
        self.material_params = material_params
        
        # Validate parameter compatibility
        required_params = symb_def.get_required_parameters()
        is_compatible, missing = material_params.check_model_compatibility(required_params)
        
        if not is_compatible:
            raise ValueError(f"Material parameters missing for model: {missing}")
        
        # Create numerical engine
        self.engine = GSMEngine(symb_def, k_max=k_max, tol=tol, cache_dir=cache_dir)
        
        # Store parameter order for consistent evaluation
        self._param_order = required_params
    
    @property
    def name(self) -> str:
        """Get model name."""
        return f"{self.symb_def.__class__.__name__}_Model"
    
    def get_model_args(self) -> List[float]:
        """
        Get material parameters in the order expected by the engine.
        
        Returns:
            List of parameter values in engine-expected order
        """
        return self.material_params.get_model_params(self._param_order)
    
    def get_response(self, 
                    eps_ta: npt.NDArray[np.float64], 
                    t_t: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], ...]:
        """
        Calculate complete material response for given strain history.
        
        This method performs time integration of the material model to compute
        the complete response including stresses, internal variables, and 
        convergence information.
        
        Args:
            eps_ta: External strain time history array with shape (n_t, n_I, n_eps)
                   where n_t is number of time steps, n_I is number of integration 
                   points, and n_eps is number of strain components
            t_t: Time array with shape (n_t,)
            
        Returns:
            Tuple containing:
            - t_t: Time array (truncated if early termination)
            - eps_ta: External strain history (truncated if early termination)  
            - sig_ta: External stress history
            - Eps_b_t: Internal strain history
            - Sig_b_t: Internal stress history
            - iter_t: Iteration count history
            - lam_t: Lagrange multiplier history
            - increments: Tuple of (d_t_t, d_eps_ta) increments
        """
        args = self.get_model_args()
        
        # Handle input dimensionality
        if eps_ta.ndim == 2:
            eps_ta = eps_ta[:, np.newaxis, :]
        if eps_ta.ndim == 1:
            eps_ta = eps_ta[:, np.newaxis]

        # Calculate increments
        d_eps_ta = np.diff(eps_ta, axis=0)
        d_t_t = np.diff(t_t, axis=0)
        
        # Initialize state arrays
        Eps_b_n1 = np.zeros(eps_ta.shape[1:] + (self.engine.n_Eps_b,), dtype=np.float64)
        Sig_b_n1 = np.zeros_like(Eps_b_n1)
        lam_n1 = np.zeros(eps_ta.shape[1:] + (self.engine.n_lam_phi + self.engine.n_Lam,), dtype=np.float64)
        
        # Initialize history records
        Sig_b_record = [Sig_b_n1]
        Eps_b_record = [Eps_b_n1]
        iter_record = [np.zeros(eps_ta.shape[1:], dtype=np.int32)]
        lam_record = [lam_n1]
        
        # Time integration loop
        for n, dt in enumerate(d_t_t):
            print(f'increment {n+1}', end='\r')
            try:
                sig_a_n1, dsig_deps_aa_n1, Eps_b_n1, Sig_b_n1, lam_n1, k = self.engine.get_state_n1(
                    eps_ta[n], d_eps_ta[n], dt, Eps_b_n1, *args
                )
            except RuntimeError as e:
                print(f'{n+1}({k}) ... {str(e)}', end='\r')
                break
                
            # Store results
            Sig_b_record.append(Sig_b_n1)
            Eps_b_record.append(Eps_b_n1)
            lam_record.append(lam_n1)
            iter_record.append(k)
        
        # Convert records to arrays
        Sig_b_t = np.array(Sig_b_record, dtype=np.float64)
        Eps_b_t = np.array(Eps_b_record, dtype=np.float64)
        iter_t = np.array(iter_record, dtype=np.int32)
        lam_t = np.array(lam_record, dtype=np.float64)
        
        # Adjust arrays to actual computed length
        n_t = len(Eps_b_t)
        eps_ta = eps_ta[:n_t]
        
        # Calculate external stress response
        sig_ta = self._get_sig_a(eps_ta[..., np.newaxis], Eps_b_t, *args)
        
        return (t_t[:n_t], eps_ta, sig_ta, Eps_b_t, Sig_b_t, iter_t, lam_t, (d_t_t[:n_t], d_eps_ta[:n_t]))
        
    def validate_model(self) -> bool:
        """
        Validate the complete model setup.
        
        Returns:
            True if model is valid, False otherwise
        """
        try:
            # Check symbolic definition
            validation = self.symb_def.validate_symbolic_expressions()
            if not validation.is_valid:
                print(f"Symbolic validation failed: {validation.issues}")
                return False
            
            # Check parameter compatibility
            required_params = self.symb_def.get_required_parameters()
            is_compatible, missing = self.material_params.check_model_compatibility(required_params)
            if not is_compatible:
                print(f"Parameter compatibility failed: missing {missing}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Model validation error: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.name}"
    
    def __repr__(self) -> str:
        """Detailed representation of the model."""
        return f"GSMModel(symb_def={self.symb_def.__class__.__name__}, " \
               f"n_params={len(self._param_order)}, " \
               f"engine_tol={self.engine.tol})"


class GSMModelFactory:
    """Factory class for creating GSM models with common parameter sets."""
    
    @staticmethod
    def create_elastic_model(symb_def: GSMSymbDef, 
                           E: float = 210000.0, 
                           nu: float = 0.3,
                           **additional_params: float) -> GSMModel:
        """
        Create a GSM model with elastic material parameters.
        
        Args:
            symb_def: Symbolic definition
            E: Young's modulus
            nu: Poisson's ratio
            **additional_params: Additional model-specific parameters
            
        Returns:
            GSMModel instance
        """
        from bmcs_matmod.gsm_lagrange.core2.material_params import CommonMaterialParams
        
        params = CommonMaterialParams.linear_elastic(E, nu)
        params.update_params(**additional_params)
        
        return GSMModel(symb_def, params)
    
    @staticmethod
    def create_elastoplastic_model(symb_def: GSMSymbDef,
                                 E: float = 210000.0,
                                 nu: float = 0.3,
                                 sig_y: float = 250.0,
                                 H: float = 1000.0,
                                 **additional_params: float) -> GSMModel:
        """
        Create a GSM model with elastic-plastic material parameters.
        
        Args:
            symb_def: Symbolic definition
            E: Young's modulus
            nu: Poisson's ratio
            sig_y: Yield stress
            H: Hardening modulus
            **additional_params: Additional model-specific parameters
            
        Returns:
            GSMModel instance
        """
        from bmcs_matmod.gsm_lagrange.core2.material_params import CommonMaterialParams
        
        params = CommonMaterialParams.elastic_plastic(E, nu, sig_y, H)
        params.update_params(**additional_params)
        
        return GSMModel(symb_def, params)
