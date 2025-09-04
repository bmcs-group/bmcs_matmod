"""
GSM Thermodynamic Box

This module provides the foundational thermodynamic framework that defines the static structure
of the thermodynamic square with four corners representing the fundamental state functions.

The GSMThermodynBox serves as the "playground" that:
1. Defines the variable structure (eps_vars, sig_vars, Eps_vars, Sig_vars, T_var, S_var)
2. Establishes the thermodynamic relationships and natural variable mappings
3. Provides the transformation coefficients for Legendre transformations
4. Creates state function accessors (GSMStateFn) for specific computations

This is the static layer that doesn't change during computation - it defines the framework
within which state functions operate.
"""

import sympy as sp
from typing import Dict, Tuple, List, Optional, Union
from enum import Enum


class StateFunctionTag(Enum):
    """
    Tags for the four fundamental thermodynamic state functions.
    
    Renamed from StateFunction to clearly indicate this is a helper enum
    rather than a top-level computational class.
    """
    INTERNAL_ENERGY = "U"      # U(S, ε, Ɛ) 
    HELMHOLTZ = "F"            # F(T, ε, Ɛ)
    ENTHALPY = "H"             # H(S, σ, Ɛ)
    GIBBS = "G"                # G(T, σ, Ɛ)


class GSMThermodynBox:
    """
    Thermodynamic framework defining the static structure of the thermodynamic square.
    
    This class establishes the "playground" for thermodynamic computations by defining:
    
    ## Variable Structure
    - **Corner variables**: T_var, S_var (always present), eps_vars, sig_vars (configurable)
    - **Internal variables**: Eps_vars (extensive), Sig_vars (intensive forces)
    
    ## State Function Corners
    The box defines four corners of the thermodynamic square:
    - U(S, ε, Ɛ): Internal Energy 
    - F(T, ε, Ɛ): Helmholtz Free Energy
    - H(S, σ, Ɛ): Enthalpy  
    - G(T, σ, Ɛ): Gibbs Free Energy
    
    ## Design Philosophy
    The box is the static framework - it doesn't perform computations but provides
    the structure and rules. Actual state function computations are handled by
    GSMStateFn accessor objects created via create_state_fn().
    
    ## Natural Variables Mapping
    Each state function has "natural variables" that make it well-defined and convex:
    - **Extensive variables**: Scale with system size (S, ε, Ɛ)
    - **Intensive variables**: Independent of system size (T, σ)
    
    ## Engineering Significance
    - **Helmholtz F(T,ε,Ɛ)**: Natural for displacement-controlled experiments
    - **Gibbs G(T,σ,Ɛ)**: Natural for load-controlled experiments
    - **Internal Energy U(S,ε,Ɛ)**: Natural for isolated systems
    - **Enthalpy H(S,σ,Ɛ)**: Natural for isentropic processes under load
    """
    
    # Class-level constant mapping of natural and conjugate variables
    NATURAL_VARIABLES_MAPPING: Dict[StateFunctionTag, Tuple[List[str], List[str]]] = {
        # State function: (natural variables, conjugate variables)
        StateFunctionTag.INTERNAL_ENERGY: (['S', 'eps', 'Eps'], ['T', 'sig', 'Sig']),  # U(S,ε,Ɛ)
        StateFunctionTag.HELMHOLTZ:       (['T', 'eps', 'Eps'], ['S', 'sig', 'Sig']),  # F(T,ε,Ɛ)  
        StateFunctionTag.ENTHALPY:        (['S', 'sig', 'Eps'], ['T', 'eps', 'Sig']),  # H(S,σ,Ɛ)
        StateFunctionTag.GIBBS:           (['T', 'sig', 'Eps'], ['S', 'eps', 'Sig']),  # G(T,σ,Ɛ)
    }
    
    # Class-level constant mapping for all possible Legendre transformations
    TRANSFORMATION_MAPPING: Dict[Tuple[StateFunctionTag, StateFunctionTag], Tuple[int, int]] = {
        # Direct (adjacent) transformations
        (StateFunctionTag.INTERNAL_ENERGY, StateFunctionTag.HELMHOLTZ):   (-1,  0),  # U → F: F = U - TS
        (StateFunctionTag.HELMHOLTZ, StateFunctionTag.INTERNAL_ENERGY):   ( 1,  0),  # F → U: U = F + TS
        (StateFunctionTag.INTERNAL_ENERGY, StateFunctionTag.ENTHALPY):    ( 0,  1),  # U → H: H = U + σε
        (StateFunctionTag.ENTHALPY, StateFunctionTag.INTERNAL_ENERGY):    ( 0, -1),  # H → U: U = H - σε
        (StateFunctionTag.HELMHOLTZ, StateFunctionTag.GIBBS):             ( 0, -1),  # F → G: G = F - εσ
        (StateFunctionTag.GIBBS, StateFunctionTag.HELMHOLTZ):             ( 0,  1),  # G → F: F = G + εσ
        (StateFunctionTag.ENTHALPY, StateFunctionTag.GIBBS):              (-1,  0),  # H → G: G = H - TS
        (StateFunctionTag.GIBBS, StateFunctionTag.ENTHALPY):              ( 1,  0),  # G → H: H = G + TS
        
        # Diagonal (two-step) transformations
        (StateFunctionTag.INTERNAL_ENERGY, StateFunctionTag.GIBBS):       (-1, -1),  # U → G: G = U - TS - εσ
        (StateFunctionTag.GIBBS, StateFunctionTag.INTERNAL_ENERGY):       ( 1,  1),  # G → U: U = G + TS + εσ
        (StateFunctionTag.HELMHOLTZ, StateFunctionTag.ENTHALPY):          ( 1,  1),  # F → H: H = F + TS + σε
        (StateFunctionTag.ENTHALPY, StateFunctionTag.HELMHOLTZ):          (-1, -1),  # H → F: F = H - TS - σε
    }
    
    def __init__(self, 
                 T_var: sp.Symbol = None,
                 S_var: sp.Symbol = None,
                 eps_vars: Tuple[sp.Symbol, ...] = (),
                 sig_vars: Tuple[sp.Symbol, ...] = (),
                 Eps_vars: Tuple[sp.Symbol, ...] = (),
                 Sig_vars: Tuple[sp.Symbol, ...] = (),
                 m_params: Tuple[sp.Symbol, ...] = ()):
        """
        Initialize the thermodynamic box framework.
        
        This sets up the static structure - the variable definitions and relationships
        that define the thermodynamic playground. No computations are performed here.
        
        Args:
            T_var: Temperature symbol (default: T)
            S_var: Entropy symbol (default: S)
            eps_vars: External eps variables tuple (configurable corner)
            sig_vars: External sig variables tuple (configurable corner)  
            Eps_vars: Internal Eps variables tuple (inside the box)
            Sig_vars: Internal Sig variables tuple (inside the box)
            m_params: Material parameters tuple
        """
        # Set thermal variables with defaults (always present corners)
        self.T_var = T_var if T_var is not None else sp.Symbol('T', real=True)
        self.S_var = S_var if S_var is not None else sp.Symbol('S', real=True)
        
        # Set configurable corner variables
        self.eps_vars = eps_vars  # External eps variables (ε) - extensive  
        self.sig_vars = sig_vars  # External sig variables (σ) - intensive
        
        # Set internal variables (inside the box)
        self.Eps_vars = Eps_vars  # Internal Eps variables (Ɛ) - extensive
        self.Sig_vars = Sig_vars  # Internal Sig variables (𝒮) - intensive
        
        # Material parameters
        self.m_params = m_params
        
        # Define symbolic state function templates
        self.state_function_templates = self._define_symbolic_functions()
    
    def _define_symbolic_functions(self) -> Dict[StateFunctionTag, sp.Function]:
        """
        Define symbolic state function templates with their natural variables.
        
        Creates function templates (not expressions) that show which variables
        each state function depends on according to thermodynamic theory.
        
        Returns:
            Dictionary mapping state function tags to symbolic function templates
        """
        T, S = self.T_var, self.S_var
        
        # Convert variables to appropriate form for function arguments
        eps_flat = self._flatten_variable_tuple(self.eps_vars)
        sig_flat = self._flatten_variable_tuple(self.sig_vars)
        Eps_flat = self._flatten_variable_tuple(self.Eps_vars)
        
        # Define the four state function templates with their natural variables
        return {
            StateFunctionTag.INTERNAL_ENERGY: sp.Function('U')(
                S, *eps_flat, *Eps_flat    # U depends on extensive variables: S, ε, Ɛ
            ),
            StateFunctionTag.HELMHOLTZ: sp.Function('F')(
                T, *eps_flat, *Eps_flat    # F depends on: T (intensive), ε, Ɛ (extensive)
            ),
            StateFunctionTag.ENTHALPY: sp.Function('H')(
                S, *sig_flat, *Eps_flat    # H depends on: S, Ɛ (extensive), σ (intensive as args)
            ),
            StateFunctionTag.GIBBS: sp.Function('G')(
                T, *sig_flat, *Eps_flat    # G depends on: Ɛ (extensive), T, σ (intensive as args)
            )
        }
    
    def _flatten_variable_tuple(self, vars_tuple: Tuple[sp.Symbol, ...]) -> List[sp.Symbol]:
        """Flatten a tuple of variables into a list of symbols."""
        flattened = []
        for var in vars_tuple:
            flattened.extend(self._flatten_variable(var))
        return flattened
    
    def _flatten_variable(self, var: Union[sp.Symbol, sp.Matrix]) -> List[sp.Symbol]:
        """Flatten a variable (symbol or matrix) into a list of symbols."""
        if isinstance(var, sp.Matrix):
            return [var[i] for i in range(var.shape[0] * var.shape[1])]
        else:
            return [var]
    
    def get_natural_variables(self, state_fn_tag: StateFunctionTag) -> Tuple[List[sp.Symbol], List[sp.Symbol]]:
        """
        Get the natural variables for a given state function.
        
        Args:
            state_fn_tag: State function tag to get variables for
            
        Returns:
            Tuple of (natural_vars, conjugate_vars) for this state function
        """
        # Get variable mapping for this state function
        mapping = self.NATURAL_VARIABLES_MAPPING
        if state_fn_tag not in mapping:
            raise ValueError(f"Unknown state function: {state_fn_tag}")
        
        natural_types, conjugate_types = mapping[state_fn_tag]
        
        # Variable type to actual symbols mapping
        var_lookup = {
            'T': self.T_var,
            'S': self.S_var,
            'eps': self._flatten_variable_tuple(self.eps_vars),
            'sig': self._flatten_variable_tuple(self.sig_vars),
            'Eps': self._flatten_variable_tuple(self.Eps_vars),
            'Sig': self._flatten_variable_tuple(self.Sig_vars),
        }
        
        # Build variable lists from mapping
        natural_vars = []
        conjugate_vars = []
        
        for var_type in natural_types:
            if var_type in ['T', 'S']:
                natural_vars.append(var_lookup[var_type])
            else:  # List variables (eps, sig, Eps, Sig)
                natural_vars.extend(var_lookup[var_type])
        
        for var_type in conjugate_types:
            if var_type in ['T', 'S']:
                conjugate_vars.append(var_lookup[var_type])
            else:  # List variables (eps, sig, Eps, Sig)
                conjugate_vars.extend(var_lookup[var_type])
        
        return (natural_vars, conjugate_vars)
    
    def get_conjugate_pairs(self) -> List[Tuple[sp.Symbol, sp.Symbol]]:
        """Get all conjugate variable pairs in the thermodynamic system."""
        T, S = self.T_var, self.S_var
        eps_flat = self._flatten_variable_tuple(self.eps_vars)
        sig_flat = self._flatten_variable_tuple(self.sig_vars)
        Eps_flat = self._flatten_variable_tuple(self.Eps_vars)
        Sig_flat = self._flatten_variable_tuple(self.Sig_vars)
        
        pairs = [(T, S)]  # Temperature-entropy pair
        
        # Mechanical sig-eps pairs
        for eps_i, sig_i in zip(eps_flat, sig_flat):
            pairs.append((sig_i, eps_i))
        
        # Internal Sig-Eps pairs  
        for Eps_i, Sig_i in zip(Eps_flat, Sig_flat):
            pairs.append((Sig_i, Eps_i))
            
        return pairs
    
    def get_transformation_graph(self) -> Dict[StateFunctionTag, List[StateFunctionTag]]:
        """
        Get the transformation graph showing possible direct transformations.
        
        Returns:
            Dictionary mapping each state function to its directly accessible neighbors
        """
        return {
            StateFunctionTag.INTERNAL_ENERGY: [StateFunctionTag.HELMHOLTZ, StateFunctionTag.ENTHALPY],
            StateFunctionTag.HELMHOLTZ: [StateFunctionTag.INTERNAL_ENERGY, StateFunctionTag.GIBBS],
            StateFunctionTag.ENTHALPY: [StateFunctionTag.INTERNAL_ENERGY, StateFunctionTag.GIBBS],
            StateFunctionTag.GIBBS: [StateFunctionTag.HELMHOLTZ, StateFunctionTag.ENTHALPY]
        }
    
    def create_state_fn(self, state_fn_tag: StateFunctionTag = StateFunctionTag.INTERNAL_ENERGY, initial_expression: sp.Expr = None):
        """
        Create a state function accessor for the specified state function.
        
        This is the primary method for creating computational objects that work
        with specific state functions within this thermodynamic framework.
        
        Args:
            state_fn_tag: Which state function to create accessor for
            initial_expression: Optional initial expression for the state function
            
        Returns:
            GSMStateFn object configured for this thermodynamic box
        """
        from gsm_state_fn import GSMStateFn
        return GSMStateFn(self, state_fn_tag, initial_expression)
    
    def print_box_overview(self) -> None:
        """
        Print an overview of the thermodynamic box configuration.
        
        Shows the variable structure, state function templates, and available transformations.
        """
        print("GSM Thermodynamic Box Configuration")
        print("=" * 40)
        print()
        
        print("Corner Variables:")
        print(f"  Temperature (T): {self.T_var}")
        print(f"  Entropy (S): {self.S_var}")
        print(f"  External eps: {self.eps_vars}")
        print(f"  External sig: {self.sig_vars}")
        print()
        
        print("Internal Variables:")
        print(f"  Internal Eps: {self.Eps_vars}")
        print(f"  Internal Sig: {self.Sig_vars}")
        print()
        
        print("Material Parameters:")
        print(f"  Parameters: {self.m_params}")
        print()
        
        print("State Function Templates:")
        for state_fn_tag, template in self.state_function_templates.items():
            natural_vars, conjugate_vars = self.get_natural_variables(state_fn_tag)
            natural_str = ", ".join([str(v) for v in natural_vars])
            conjugate_str = ", ".join([str(v) for v in conjugate_vars])
            print(f"  {state_fn_tag.value}: {template}")
            print(f"    Natural: [{natural_str}]")
            print(f"    Conjugate: [{conjugate_str}]")
        print()
    
    def print_transformation_table(self) -> None:
        """Print a formatted table of all possible Legendre transformations."""
        mapping = self.TRANSFORMATION_MAPPING
        
        print("Legendre Transformation Mapping")
        print("=" * 50)
        print("Format: Target = Source + thermal_coeff*(T*S) + work_coeff*(σε)")
        print()
        print(f"{'From':<8} {'To':<8} {'T*S Coeff':<10} {'σε Coeff':<10} {'Transformation'}")
        print("-" * 70)
        
        for (from_fn, to_fn), (thermal_coeff, work_coeff) in mapping.items():
            thermal_str = f"{thermal_coeff:+d}" if thermal_coeff != 0 else " 0"
            work_str = f"{work_coeff:+d}" if work_coeff != 0 else " 0"
            
            # Build transformation string
            terms = []
            if thermal_coeff != 0:
                terms.append(f"{thermal_coeff:+d}*T*S")
            if work_coeff != 0:
                terms.append(f"{work_coeff:+d}*σε")
            
            if terms:
                transform_str = f"{to_fn.value} = {from_fn.value} " + " ".join(terms)
            else:
                transform_str = f"{to_fn.value} = {from_fn.value}"
                
            print(f"{from_fn.value:<8} {to_fn.value:<8} {thermal_str:<10} {work_str:<10} {transform_str}")
    
    def print_natural_variables_table(self) -> None:
        """Print a formatted table showing natural and conjugate variables for each state function."""
        mapping = self.NATURAL_VARIABLES_MAPPING
        
        print("Natural Variables Mapping")
        print("=" * 60)
        print("Each state function has natural (independent) and conjugate (derivative) variables")
        print()
        print(f"{'Function':<12} {'Natural Variables':<25} {'Conjugate Variables':<25}")
        print("-" * 62)
        
        for state_fn, (natural_types, conjugate_types) in mapping.items():
            natural_str = ", ".join(natural_types)
            conjugate_str = ", ".join(conjugate_types)
            fn_name = f"{state_fn.value}(...)"
            
            print(f"{fn_name:<12} {natural_str:<25} {conjugate_str:<25}")
        
        print()
        print("Legend:")
        print("  T: Temperature, S: Entropy")
        print("  eps: External strain, sig: External stress") 
        print("  Eps: Internal strain, Sig: Internal stress")
    
    def validate_configuration(self) -> bool:
        """
        Validate the thermodynamic box configuration.
        
        Checks that the variable configuration is thermodynamically consistent.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Check that eps_vars and sig_vars have same length
        if len(self.eps_vars) != len(self.sig_vars):
            print(f"Error: eps_vars ({len(self.eps_vars)}) and sig_vars ({len(self.sig_vars)}) must have same length")
            return False
        
        # Check that Eps_vars and Sig_vars have same length
        if len(self.Eps_vars) != len(self.Sig_vars):
            print(f"Error: Eps_vars ({len(self.Eps_vars)}) and Sig_vars ({len(self.Sig_vars)}) must have same length")
            return False
        
        # Check for symbol name conflicts
        all_symbols = (
            [self.T_var, self.S_var] + 
            list(self.eps_vars) + list(self.sig_vars) + 
            list(self.Eps_vars) + list(self.Sig_vars) + 
            list(self.m_params)
        )
        
        symbol_names = [str(sym) for sym in all_symbols]
        if len(symbol_names) != len(set(symbol_names)):
            print("Error: Duplicate symbol names detected")
            return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation of the thermodynamic box."""
        return (f"GSMThermodynBox("
                f"eps_vars={len(self.eps_vars)}, "
                f"sig_vars={len(self.sig_vars)}, "
                f"Eps_vars={len(self.Eps_vars)}, "
                f"Sig_vars={len(self.Sig_vars)})")
