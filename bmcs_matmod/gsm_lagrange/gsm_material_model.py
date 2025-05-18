import traits.api as tr
import bmcs_utils.api as bu
import sympy as sp
import numpy as np
from typing import Type, Dict, List, Union, Any, Tuple
import inspect

from .gsm_base import GSMBase
from .gsm_mpdp import GSMMPDP

class GSMMaterialModel(bu.Model):
    """
    A base class for creating executable material models from GSMBase subclasses.
    
    This class serves as a bridge between the symbolic definitions in GSMBase
    and executable models with specific parameter values. It dynamically creates
    traits for material parameters defined in the GSMBase subclass, allowing
    for instance-level control over these parameters.
    
    Attributes:
        gsm_model_type (Type[GSMBase]): The GSMBase subclass that defines the model's 
                                        symbolic structure.
        model_instance (GSMBase): Instance of the GSMBase subclass with initialized engines.
        trace_model_params (Dict): Dictionary mapping from parameter symbols to trait names.
    
    Example:
        ```python
        class MyElasticPlasticModel(GSMBase):
            # Define symbolic variables and expressions
            eps = sp.Symbol(r'\varepsilon', real=True)
            E = sp.Symbol('E', positive=True)
            sigma_y = sp.Symbol(r'\sigma_y', positive=True)
            
            # Define Helmholtz free energy
            F_expr = 0.5 * E * eps**2
            
            # Create engine
            F_engine = GSMMPDP(...)
            
        # Create an executable model with specific parameter values
        material = GSMMaterialModel(
            gsm_model_type=MyElasticPlasticModel, 
            E=210e3, 
            sigma_y=240.0
        )
        
        # Use the model to calculate stresses, etc.
        strain = np.linspace(0, 0.01, 100)
        stress = material.get_sig(strain)
        ```
    """
    
    gsm_model_type = tr.Type(GSMBase)
    """The GSMBase subclass defining the symbolic structure of the material model."""
    
    model_instance = tr.Property(tr.Instance(GSMBase), depends_on='gsm_model_type,+params')
    """Instance of the GSMBase subclass with initialized engines and parameters."""
    
    trait_model_params = tr.Dict
    """Dictionary mapping parameter symbols to trait names."""
    
    def __new__(cls, gsm_model_type=None, **traits):
        """
        Create a new class for the specified GSMBase subclass with traits for model parameters.
        
        This method dynamically creates traits for all symbolic parameters defined in the 
        specified GSMBase subclass. Each parameter becomes a Float trait with appropriate
        metadata (description, min/max values, etc.) based on the symbolic assumptions.
        
        Args:
            gsm_model_type (Type[GSMBase]): The GSMBase subclass to use for the model.
            **traits: Parameter values and other traits to set on the instance.
            
        Returns:
            GSMMaterialModel: A new instance with traits corresponding to material parameters.
        """
        # If gsm_model_type is not specified, create a standard instance
        if gsm_model_type is None:
            return super().__new__(cls)
        
        # Create a new subclass for this specific material model
        model_class_name = f"{gsm_model_type.__name__}Model"
        
        # If the class already exists, return it directly
        if model_class_name in globals():
            return globals()[model_class_name].__new__(globals()[model_class_name])
        
        # Get parameter symbols from the GSMBase subclass
        # This assumes the F_engine is defined at the class level
        if not hasattr(gsm_model_type, 'F_engine'):
            raise ValueError(f"The class {gsm_model_type.__name__} must define F_engine at the class level.")
        
        # Create a temporary instance to access parameter information
        # This is needed because we need to access instance methods/properties
        temp_instance = gsm_model_type()
        param_symbols = temp_instance.F_engine.m_params
        
        # Create traits dictionary for the new class
        traits_dict = {
            '__doc__': f"Executable material model based on {gsm_model_type.__name__}",
            'gsm_model_type': gsm_model_type,
            'trait_model_params': {}
        }
        
        # Add traits for each parameter symbol
        for param_sym in param_symbols:
            # Get the parameter name from the symbol
            param_name = param_sym.name
            
            # Convert from LaTeX name if needed
            trait_name = param_name
            if '\\' in param_name or '{' in param_name or '}' in param_name:
                # Try to get codename from gsm_model_type.param_codenames
                if hasattr(temp_instance, 'param_codenames') and param_sym in temp_instance.param_codenames:
                    trait_name = temp_instance.param_codenames[param_sym]
                else:
                    # Remove LaTeX-specific characters
                    trait_name = param_name.replace('\\', '').replace('{', '').replace('}', '')
            
            # Extract symbolic assumptions for min/max values
            trait_metadata = {}
            
            # Add description based on parameter name
            trait_metadata['desc'] = f"Material parameter {param_name}"
            
            # Add assumptions from the symbol
            if param_sym.is_positive:
                trait_metadata['min'] = 0.0
            
            # Add the trait to the dictionary
            traits_dict[trait_name] = tr.Float(1.0, **trait_metadata)
            
            # Map the trait name back to the parameter symbol
            traits_dict['trait_model_params'][param_sym] = trait_name
        
        # Create the new class
        model_class = type(model_class_name, (cls,), traits_dict)
        
        # Register the class in globals so it can be re-used
        globals()[model_class_name] = model_class
        
        # Create and return an instance of the new class
        instance = model_class.__new__(model_class)
        return instance
    
    def __init__(self, gsm_model_type=None, **traits):
        """
        Initialize with the specified traits.
        
        Args:
            gsm_model_type (Type[GSMBase]): The GSMBase subclass to use for the model.
            **traits: Parameter values and other traits to set on the instance.
        """
        # Set gsm_model_type first if provided
        if gsm_model_type is not None:
            traits['gsm_model_type'] = gsm_model_type
            
        # Initialize using the standard trait initialization
        super().__init__(**traits)
    
    @tr.cached_property
    def _get_model_instance(self):
        """
        Create and initialize an instance of the GSMBase subclass with parameter values.
        
        This property lazy-creates the GSMBase instance with parameter values taken from
        the corresponding traits of this model instance.
        
        Returns:
            GSMBase: An initialized instance of the GSMBase subclass.
        """
        # Create an instance of the GSMBase subclass
        model = self.gsm_model_type()
        
        # Update the model_instance when trait values change
        # This ensures the symbolic engines get updated parameter values
        return model
    
    def get_param_dict(self):
        """
        Get a dictionary of parameter values for use with the model's engines.
        
        This method maps from the trait values of this model to the parameter
        dictionary expected by the GSMBase engines.
        
        Returns:
            Dict: A dictionary mapping from parameter names to their values.
        """
        param_dict = {}
        for sym, trait_name in self.trait_model_params.items():
            param_dict[trait_name] = getattr(self, trait_name)
        return param_dict
    
    def get_args(self):
        """
        Get the parameter values as arguments for the model's engines.
        
        Returns:
            Tuple: Parameter values in the order expected by the engines.
        """
        return self.model_instance.get_args(**self.get_param_dict())
    
    # Delegate the key methods to the model_instance
    
    def get_sig(self, eps, Eps=None):
        """
        Calculate stress for given strain and internal variables.
        
        Args:
            eps: External strain.
            Eps: Internal state variables.
            
        Returns:
            Stress tensor.
        """
        if Eps is None:
            # Initialize Eps to zeros with the correct shape
            Eps = np.zeros(np.shape(eps) + (self.model_instance.F_engine.n_Eps,))
        
        return self.model_instance.get_sig(eps, Eps, *self.get_args())
    
    def get_response(self, eps_ta, t_t=None):
        """
        Calculate time-dependent material response.
        
        Args:
            eps_ta: Time series of strain values.
            t_t: Time points (if None, uniform time steps are assumed).
            
        Returns:
            Tuple of results including stresses, internal variables, etc.
        """
        # Create default time array if not provided
        if t_t is None:
            t_t = np.arange(len(eps_ta))
        
        return self.model_instance.get_response(eps_ta, t_t, *self.get_args())
    
    def get_Sig(self, eps, Eps):
        """
        Calculate thermodynamic forces for given strain and internal variables.
        
        Args:
            eps: External strain.
            Eps: Internal state variables.
            
        Returns:
            Thermodynamic forces.
        """
        return self.model_instance.get_Sig(eps, Eps, *self.get_args())
    
    # Methods for Gibbs free energy based calculations
    
    def get_G_eps(self, sig, Eps=None):
        """
        Calculate strain for given stress and internal variables (Gibbs formulation).
        
        Args:
            sig: External stress.
            Eps: Internal state variables.
            
        Returns:
            Strain tensor.
        """
        if Eps is None:
            # Initialize Eps to zeros with the correct shape
            Eps = np.zeros(np.shape(sig) + (self.model_instance.G_engine.n_Eps,))
        
        return self.model_instance.get_G_eps(sig, Eps, *self.get_args())
    
    def get_G_response(self, sig_ta, t_t=None):
        """
        Calculate time-dependent material response for stress-driven scenario.
        
        Args:
            sig_ta: Time series of stress values.
            t_t: Time points (if None, uniform time steps are assumed).
            
        Returns:
            Tuple of results including strains, internal variables, etc.
        """
        # Create default time array if not provided
        if t_t is None:
            t_t = np.arange(len(sig_ta))
        
        return self.model_instance.get_G_response(sig_ta, t_t, *self.get_args())
    
    def get_G_Sig(self, sig, Eps):
        """
        Calculate thermodynamic forces for given stress and internal variables.
        
        Args:
            sig: External stress.
            Eps: Internal state variables.
            
        Returns:
            Thermodynamic forces (Gibbs formulation).
        """
        return self.model_instance.get_G_Sig(sig, Eps, *self.get_args())
    
    # Visualization and inspection methods
    
    def print_potentials(self):
        """Print the free energy potentials (Helmholtz and Gibbs) and related expressions."""
        self.model_instance.print_potentials()
    
    def markdown(self):
        """Return a markdown representation of the material model."""
        return self.model_instance.markdown()
    
    def get_param_values(self):
        """
        Get a dictionary of parameter values.
        
        Returns:
            Dict: A dictionary mapping from parameter symbols to their values.
        """
        return {sym: getattr(self, trait_name) 
                for sym, trait_name in self.trait_model_params.items()}
    
    def __str__(self):
        """String representation of the material model."""
        params_str = ", ".join(f"{name}={getattr(self, name)}" 
                              for name in self.trait_model_params.values())
        return f"{self.__class__.__name__}({params_str})"
