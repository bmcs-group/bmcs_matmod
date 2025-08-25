"""
Material Parameters Repository

This module provides a repository class for storing and managing material parameters
that can be shared across multiple GSM models. The repository supports both general
parameters (like elastic moduli) and model-specific parameters.
"""

from typing import Dict, List, Optional, Any, Set
import sympy as sp


class MaterialParams:
    """
    Repository for material parameters that can be shared across multiple GSM models.
    
    This class provides:
    - Storage for both general and model-specific parameters
    - Parameter validation against model requirements
    - Easy parameter access and modification
    - Type checking and value validation
    """
    
    def __init__(self, **initial_params: float):
        """
        Initialize material parameters repository.
        
        Args:
            **initial_params: Initial parameter values as keyword arguments
        """
        self._params: Dict[str, float] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Add initial parameters
        for name, value in initial_params.items():
            self.set_param(name, value)
    
    def set_param(self, name: str, value: float, 
                  description: Optional[str] = None, 
                  units: Optional[str] = None,
                  bounds: Optional[tuple] = None) -> None:
        """
        Set a material parameter value with optional metadata.
        
        Args:
            name: Parameter name (codename)
            value: Parameter value
            description: Optional parameter description
            units: Optional parameter units
            bounds: Optional (min, max) bounds for the parameter
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Parameter value must be numeric, got {type(value)}")
        
        if bounds is not None:
            min_val, max_val = bounds
            if not (min_val <= value <= max_val):
                raise ValueError(f"Parameter {name} = {value} is outside bounds {bounds}")
        
        self._params[name] = float(value)
        self._metadata[name] = {
            'description': description,
            'units': units,
            'bounds': bounds
        }
    
    def get_param(self, name: str) -> float:
        """
        Get a parameter value by name.
        
        Args:
            name: Parameter name (codename)
            
        Returns:
            Parameter value
            
        Raises:
            KeyError: If parameter is not found
        """
        if name not in self._params:
            raise KeyError(f"Parameter '{name}' not found in repository")
        return self._params[name]
    
    def has_param(self, name: str) -> bool:
        """
        Check if a parameter exists in the repository.
        
        Args:
            name: Parameter name (codename)
            
        Returns:
            True if parameter exists, False otherwise
        """
        return name in self._params
    
    def get_all_params(self) -> Dict[str, float]:
        """
        Get all parameters as a dictionary.
        
        Returns:
            Dictionary mapping parameter names to values
        """
        return self._params.copy()
    
    def get_param_names(self) -> List[str]:
        """
        Get list of all parameter names.
        
        Returns:
            List of parameter names
        """
        return list(self._params.keys())
    
    def update_params(self, **params: float) -> None:
        """
        Update multiple parameters at once.
        
        Args:
            **params: Parameter updates as keyword arguments
        """
        for name, value in params.items():
            if name in self._params:
                # Use existing metadata and bounds checking
                metadata = self._metadata.get(name, {})
                self.set_param(name, value, 
                             metadata.get('description'),
                             metadata.get('units'),
                             metadata.get('bounds'))
            else:
                self.set_param(name, value)
    
    def check_model_compatibility(self, required_params: List[str]) -> tuple[bool, Set[str]]:
        """
        Check if the repository contains all required parameters for a model.
        
        Args:
            required_params: List of required parameter names
            
        Returns:
            Tuple of (is_compatible, missing_params)
        """
        required_set = set(required_params)
        available_set = set(self._params.keys())
        missing_params = required_set - available_set
        
        return len(missing_params) == 0, missing_params
    
    def get_model_params(self, required_params: List[str]) -> List[float]:
        """
        Get parameter values for a specific model in the required order.
        
        Args:
            required_params: List of required parameter names in order
            
        Returns:
            List of parameter values in the same order
            
        Raises:
            KeyError: If any required parameter is missing
        """
        is_compatible, missing = self.check_model_compatibility(required_params)
        if not is_compatible:
            raise KeyError(f"Missing required parameters: {missing}")
        
        return [self._params[name] for name in required_params]
    
    def get_param_info(self, name: str) -> Dict[str, Any]:
        """
        Get parameter information including metadata.
        
        Args:
            name: Parameter name
            
        Returns:
            Dictionary with parameter value and metadata
        """
        if name not in self._params:
            raise KeyError(f"Parameter '{name}' not found")
        
        info = {
            'value': self._params[name],
            **self._metadata[name]
        }
        return info
    
    def remove_param(self, name: str) -> None:
        """
        Remove a parameter from the repository.
        
        Args:
            name: Parameter name to remove
        """
        if name in self._params:
            del self._params[name]
            del self._metadata[name]
    
    def __contains__(self, name: str) -> bool:
        """Support 'in' operator for parameter checking."""
        return name in self._params
    
    def __getitem__(self, name: str) -> float:
        """Support dictionary-style access."""
        return self.get_param(name)
    
    def __setitem__(self, name: str, value: float) -> None:
        """Support dictionary-style assignment."""
        self.set_param(name, value)
    
    def __str__(self) -> str:
        """String representation of the parameter repository."""
        if not self._params:
            return "MaterialParams(empty)"
        
        param_strs = []
        for name, value in self._params.items():
            metadata = self._metadata.get(name, {})
            units = metadata.get('units', '')
            if units:
                param_strs.append(f"{name}={value} {units}")
            else:
                param_strs.append(f"{name}={value}")
        
        return f"MaterialParams({', '.join(param_strs)})"
    
    def __repr__(self) -> str:
        """Detailed representation of the parameter repository."""
        return self.__str__()


# Predefined parameter sets for common materials
class CommonMaterialParams:
    """Factory class for creating common material parameter sets."""
    
    @staticmethod
    def linear_elastic(E: float = 210000.0, nu: float = 0.3) -> MaterialParams:
        """
        Create linear elastic material parameters.
        
        Args:
            E: Young's modulus [MPa]
            nu: Poisson's ratio [-]
            
        Returns:
            MaterialParams instance with elastic parameters
        """
        params = MaterialParams()
        params.set_param('E', E, 'Young\'s modulus', 'MPa', (0.1, 1e6))
        params.set_param('nu', nu, 'Poisson\'s ratio', '-', (0.0, 0.5))
        return params
    
    @staticmethod
    def elastic_plastic(E: float = 210000.0, nu: float = 0.3, 
                       sig_y: float = 250.0, H: float = 1000.0) -> MaterialParams:
        """
        Create elastic-plastic material parameters.
        
        Args:
            E: Young's modulus [MPa]
            nu: Poisson's ratio [-]
            sig_y: Yield stress [MPa] 
            H: Hardening modulus [MPa]
            
        Returns:
            MaterialParams instance with elastic-plastic parameters
        """
        params = MaterialParams()
        params.set_param('E', E, 'Young\'s modulus', 'MPa', (0.1, 1e6))
        params.set_param('nu', nu, 'Poisson\'s ratio', '-', (0.0, 0.5))
        params.set_param('sig_y', sig_y, 'Yield stress', 'MPa', (0.1, 1e4))
        params.set_param('H', H, 'Hardening modulus', 'MPa', (0.0, 1e5))
        return params
