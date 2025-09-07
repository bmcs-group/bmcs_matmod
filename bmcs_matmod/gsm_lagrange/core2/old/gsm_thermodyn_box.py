"""
GSM Thermodynamic State Function Box

This module provides a comprehensive thermodynamic framework that implements all four
fundamental state functions and their Legendre transformations:

1. Internal Energy U(S, ε, Ɛ) - extensive variables: entropy, eps, internal variables
2. Helmholtz Free Energy F(T, ε, Ɛ) - temperature, eps, internal variables  
3. Enthalpy H(S, σ, Ɛ) - entropy, sig, internal variables
4. Gibbs Free Energy G(T, σ, Ɛ) - temperature, sig, internal variables

The box allows starting with any state function and performing Legendre transformations
to obtain neighboring state functions, maintaining thermodynamic consistency.

## Natural Variables Concept

Each state function has "natural variables" - the specific set of independent variables
that make the function well-defined, convex, and suitable for minimization principles:

- **U(S, ε, Ɛ)**: All extensive variables (entropy, strain, internal strains)
- **F(T, ε, Ɛ)**: Mixed variables (temperature intensive, strains extensive)
- **H(S, σ, Ɛ)**: Mixed variables (entropy extensive, stress intensive, internal extensive)  
- **G(T, σ, Ɛ)**: Mostly intensive (temperature & stress intensive, internal extensive)

The choice of natural variables determines:
1. Which variables can be independently controlled in experiments
2. Which variables are obtained as derivatives (constitutive relations)
3. The mathematical properties (convexity) of the state function
4. The appropriate minimization principle for equilibrium

## Legendre Transformations

Transformations between state functions exchange extensive ↔ intensive variable pairs:
- Thermal: (S, T) ↔ entropy and temperature  
- Mechanical: (ε, σ) ↔ strain and stress
- Internal variables (Ɛ, 𝒮) remain extensive in GSM framework
"""

import sympy as sp
from typing import Dict, Tuple, List, Set, Optional, Union
from enum import Enum
import signal
import warnings


class StateFunction(Enum):
    """Enumeration of the four fundamental thermodynamic state functions."""
    INTERNAL_ENERGY = "U"      # U(S, ε, Ɛ) 
    HELMHOLTZ = "F"            # F(T, ε, Ɛ)
    ENTHALPY = "H"             # H(S, σ, Ɛ)
    GIBBS = "G"                # G(T, σ, Ɛ)


class TimeoutError(Exception):
    """Custom exception for timeout during simplification."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout during simplification."""
    raise TimeoutError("Simplification timed out")


class GSMThermodynBox:
    """
    Thermodynamic state function box implementing all four fundamental state functions
    and their Legendre transformations.
    
    The box defines the thermodynamic square with four corners:
    - U(S, ε, Ɛ): Internal Energy 
    - F(T, ε, Ɛ): Helmholtz Free Energy
    - H(S, σ, Ɛ): Enthalpy  
    - G(T, σ, Ɛ): Gibbs Free Energy
    
    ## Natural Variables and Their Significance
    
    Each state function is defined in terms of its "natural variables" - the specific
    combination of extensive and intensive variables that make the function:
    1. Well-defined and single-valued
    2. Convex in its natural variables  
    3. Suitable for variational principles (minimization for equilibrium)
    
    **Physical Interpretation:**
    - **Extensive variables**: Scale with system size (mass, volume, strain, entropy)
    - **Intensive variables**: Independent of system size (temperature, pressure, stress)
    - **Natural choice**: Determines which variables you control vs. which respond
    
    **Engineering Significance:**
    - **Helmholtz F(T,ε,Ɛ)**: Natural for displacement-controlled experiments
    - **Gibbs G(T,σ,Ɛ)**: Natural for load-controlled experiments
    - **Internal Energy U(S,ε,Ɛ)**: Natural for isolated systems
    - **Enthalpy H(S,σ,Ɛ)**: Natural for isentropic processes under load
    
    Internal variables Ɛ remain extensive (control variables) in all transformations,
    following the GSM framework convention.
    """
    
    # Class-level constant mapping of natural and conjugate variables
    NATURAL_VARIABLES_MAPPING: Dict[StateFunction, Tuple[List[str], List[str]]] = {
        # State function: (natural variables, conjugate variables)
        StateFunction.INTERNAL_ENERGY: (['S', 'eps', 'Eps'], ['T', 'sig', 'Sig']),  # U(S,ε,Ɛ)
        StateFunction.HELMHOLTZ:       (['T', 'eps', 'Eps'], ['S', 'sig', 'Sig']),  # F(T,ε,Ɛ)  
        StateFunction.ENTHALPY:        (['S', 'sig', 'Eps'], ['T', 'eps', 'Sig']),  # H(S,σ,Ɛ)
        StateFunction.GIBBS:           (['T', 'sig', 'Eps'], ['S', 'eps', 'Sig']),  # G(T,σ,Ɛ)
    }
    
    # Class-level constant mapping for all possible Legendre transformations
    TRANSFORMATION_MAPPING: Dict[Tuple[StateFunction, StateFunction], Tuple[int, int]] = {
        # Direct (adjacent) transformations
        (StateFunction.INTERNAL_ENERGY, StateFunction.HELMHOLTZ):   (-1,  0),  # U → F: F = U - TS
        (StateFunction.HELMHOLTZ, StateFunction.INTERNAL_ENERGY):   ( 1,  0),  # F → U: U = F + TS
        (StateFunction.INTERNAL_ENERGY, StateFunction.ENTHALPY):    ( 0,  1),  # U → H: H = U + σε
        (StateFunction.ENTHALPY, StateFunction.INTERNAL_ENERGY):    ( 0, -1),  # H → U: U = H - σε
        (StateFunction.HELMHOLTZ, StateFunction.GIBBS):             ( 0, -1),  # F → G: G = F - εσ
        (StateFunction.GIBBS, StateFunction.HELMHOLTZ):             ( 0,  1),  # G → F: F = G + εσ
        (StateFunction.ENTHALPY, StateFunction.GIBBS):              (-1,  0),  # H → G: G = H - TS
        (StateFunction.GIBBS, StateFunction.ENTHALPY):              ( 1,  0),  # G → H: H = G + TS
        
        # Diagonal (two-step) transformations
        (StateFunction.INTERNAL_ENERGY, StateFunction.GIBBS):       (-1, -1),  # U → G: G = U - TS - εσ
        (StateFunction.GIBBS, StateFunction.INTERNAL_ENERGY):       ( 1,  1),  # G → U: U = G + TS + εσ
        (StateFunction.HELMHOLTZ, StateFunction.ENTHALPY):          ( 1,  1),  # F → H: H = F + TS + σε
        (StateFunction.ENTHALPY, StateFunction.HELMHOLTZ):          (-1, -1),  # H → F: F = H - TS - σε
    }
    
    # Thermal variables (following VARIABLE_NAMING.md convention)
    T_var: sp.Symbol = sp.Symbol('T', real=True)          # Temperature - intensive
    S_var: sp.Symbol = sp.Symbol('S', real=True)          # Entropy - extensive
    
    # External mechanical variables (following GSMSymbDef pattern)
    eps_vars: Tuple[sp.Symbol, ...] = ()                  # External eps variables (ε) - extensive  
    sig_vars: Tuple[sp.Symbol, ...] = ()                  # External sig variables (σ) - intensive
    
    # Internal variables (following GSMSymbDef pattern)
    Eps_vars: Tuple[sp.Symbol, ...] = ()                  # Internal Eps variables (Ɛ) - extensive
    Sig_vars: Tuple[sp.Symbol, ...] = ()                  # Internal Sig variables (𝒮) - intensive
    
    # Material parameters
    m_params: Tuple[sp.Symbol, ...] = ()                  # Material parameters
    
    # State function management
    state_expressions: Dict[StateFunction, sp.Function] = {}     # Template functions showing variable dependencies (e.g., F(T,ε,Ɛ))
    explicit_expressions: Dict[StateFunction, sp.Expr] = {}      # Cache of computed mathematical expressions (e.g., ½(1-ω)Eε²)
    current_state_fn: StateFunction = StateFunction.HELMHOLTZ   # Active state function for transformations
    
    def __init__(self, 
                 *,
                 initial_state_fn: StateFunction,
                 T_var: sp.Symbol = None,
                 S_var: sp.Symbol = None,
                 eps_vars: Tuple[sp.Symbol, ...] = (),
                 sig_vars: Tuple[sp.Symbol, ...] = (),
                 Eps_vars: Tuple[sp.Symbol, ...] = (),
                 Sig_vars: Tuple[sp.Symbol, ...] = (),
                 m_params: Tuple[sp.Symbol, ...] = (),
                 initial_expression: sp.Expr = None,
                 auto_simplify: bool = True,
                 simplify_timeout: float = 10.0):
        """
        Initialize the thermodynamic state function box.
        
        Args:
            initial_state_fn: Starting state function (required - defines the base for transformations)
            T_var: Temperature symbol (default: T)
            S_var: Entropy symbol (default: S)
            eps_vars: External eps variables tuple
            sig_vars: External sig variables tuple  
            Eps_vars: Internal Eps variables tuple
            Sig_vars: Internal Sig variables tuple
            m_params: Material parameters tuple
            initial_expression: Expression for the initial state function
            auto_simplify: Whether to automatically simplify expressions (default: True)
            simplify_timeout: Maximum time in seconds for simplification (default: 10.0)
        """
        # Set thermal variables with defaults
        self.T_var = T_var if T_var is not None else sp.Symbol('T', real=True)
        self.S_var = S_var if S_var is not None else sp.Symbol('S', real=True)
        
        # Set mechanical and internal variables
        self.eps_vars = eps_vars
        self.sig_vars = sig_vars
        self.Eps_vars = Eps_vars
        self.Sig_vars = Sig_vars
        self.m_params = m_params
        
        # Set simplification options
        self.auto_simplify = auto_simplify
        self.simplify_timeout = simplify_timeout
        
        # Initialize state tracking
        self.current_state_fn = initial_state_fn              # Set the active state function
        self.state_expressions = {}                           # Will store function templates like F(T,ε,Ɛ)
        self.explicit_expressions = {}                        # Will store actual expressions like ½(1-ω)Eε²
        
        # Define symbolic state functions with their natural variables
        self._define_symbolic_functions()
        
        # Set initial expression if provided
        if initial_expression is not None:
            # Apply simplification to initial expression if enabled
            if self.auto_simplify:
                initial_expression = self.simplify_with_timeout(initial_expression)
            self.explicit_expressions[initial_state_fn] = initial_expression
    
    def simplify_with_timeout(self, expr: sp.Expr) -> sp.Expr:
        """
        Simplify a SymPy expression with a timeout.
        
        Args:
            expr: SymPy expression to simplify
            
        Returns:
            Simplified expression, or original expression if timeout occurs
        """
        if self.simplify_timeout <= 0:
            return expr
            
        try:
            # Set up the signal handler for timeout
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(self.simplify_timeout))
            
            try:
                # Attempt simplification
                simplified = sp.simplify(expr)
                return simplified
            except TimeoutError:
                warnings.warn(f"Expression simplification timed out after {self.simplify_timeout} seconds. "
                             f"Returning unsimplified expression.", UserWarning)
                return expr
            finally:
                # Clean up the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        except (OSError, ValueError):
            # Signal handling not available (e.g., on Windows or in Jupyter)
            # Fall back to direct simplification without timeout
            warnings.warn("Timeout functionality not available on this platform. "
                         "Performing simplification without timeout.", UserWarning)
            try:
                return sp.simplify(expr)
            except Exception as e:
                warnings.warn(f"Simplification failed: {e}. Returning unsimplified expression.", 
                             UserWarning)
                return expr
    
    def _define_symbolic_functions(self) -> None:
        """
        Define symbolic state functions with their natural variables.
        
        Creates function templates (not expressions) that show which variables
        each state function depends on according to thermodynamic theory:
        - U(S, ε, Ɛ): Internal Energy depends on entropy, strain, internal vars
        - F(T, ε, Ɛ): Helmholtz depends on temperature, strain, internal vars  
        - H(S, σ, Ɛ): Enthalpy depends on entropy, stress, internal vars
        - G(T, σ, Ɛ): Gibbs depends on temperature, stress, internal vars
        """
        T, S = self.T_var, self.S_var
        
        # Convert variables to appropriate form for function arguments
        eps_flat = self._flatten_variable_tuple(self.eps_vars)
        sig_flat = self._flatten_variable_tuple(self.sig_vars)
        Eps_flat = self._flatten_variable_tuple(self.Eps_vars)
        
        # Define the four state functions with their natural variables
        # These are symbolic function templates, not actual mathematical expressions
        self.state_expressions[StateFunction.INTERNAL_ENERGY] = sp.Function('U')(
            S, *eps_flat, *Eps_flat    # U depends on extensive variables: S, ε, Ɛ
        )
        self.state_expressions[StateFunction.HELMHOLTZ] = sp.Function('F')(
            T, *eps_flat, *Eps_flat    # F depends on: T (intensive), ε, Ɛ (extensive)
        )
        self.state_expressions[StateFunction.ENTHALPY] = sp.Function('H')(
            S, *sig_flat, *Eps_flat    # H depends on: S, Ɛ (extensive), σ (intensive as args)
        )
        self.state_expressions[StateFunction.GIBBS] = sp.Function('G')(
            T, *sig_flat, *Eps_flat    # G depends on: Ɛ (extensive), T, σ (intensive as args)
        )
    
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
    
    def get_natural_variables(self, state_fn: StateFunction) -> Tuple[List[sp.Symbol], List[sp.Symbol]]:
        """
        Get the natural variables for a given state function using dictionary mapping.
        
        Natural variables are the thermodynamic variables that each state function
        "naturally" depends on to be well-defined and convex. Each state function
        has a specific set of independent variables:
        
        - U(S, ε, Ɛ): Internal Energy - all extensive variables (entropy, strain, internal)
        - F(T, ε, Ɛ): Helmholtz - temperature (intensive), strain & internal (extensive)  
        - H(S, σ, Ɛ): Enthalpy - entropy & internal (extensive), stress (intensive)
        - G(T, σ, Ɛ): Gibbs - temperature & stress (intensive), internal (extensive)
        
        The choice of natural variables determines:
        1. Which variables are independent (can be controlled)
        2. Which variables are dependent (determined by derivatives)
        3. The convexity properties of the state function
        
        Returns:
            Tuple of (all_natural_vars, conjugate_vars) for this state function
            - all_natural_vars: The independent variables the function depends on
            - conjugate_vars: Variables obtained as derivatives (not independent)
        """
        # Get variable mapping for this state function
        mapping = self.NATURAL_VARIABLES_MAPPING
        if state_fn not in mapping:
            raise ValueError(f"Unknown state function: {state_fn}")
        
        natural_types, conjugate_types = mapping[state_fn]
        
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
        """Get all conjugate variable pairs in the system."""
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
    
    def legendre_transform(self, target_state_fn: StateFunction) -> sp.Expr:
        """
        Perform Legendre transformation from current state function to target.
        
        Args:
            target_state_fn: Target state function to transform to
            
        Returns:
            Expression for the target state function
        """
        # If requesting the current state function, return it directly
        if target_state_fn == self.current_state_fn:
            existing_expr = self.explicit_expressions.get(target_state_fn, 
                           self.state_expressions[target_state_fn])
            # Apply simplification if enabled
            if self.auto_simplify:
                return self.simplify_with_timeout(existing_expr)
            return existing_expr
        
        # Get current explicit expression (must exist for transformation)
        current_expr = self.explicit_expressions.get(self.current_state_fn)
        if current_expr is None:
            raise ValueError(f"No explicit expression available for {self.current_state_fn}")
        
        # Perform the Legendre transformation
        transform_expr = self._compute_legendre_transform(
            self.current_state_fn, target_state_fn, current_expr
        )
        
        # Apply simplification if enabled
        if self.auto_simplify:
            transform_expr = self.simplify_with_timeout(transform_expr)
        
        # Cache the transformed expression for future use
        self.explicit_expressions[target_state_fn] = transform_expr
        
        return transform_expr
    
    def _compute_legendre_transform(self, 
                                   from_fn: StateFunction, 
                                   to_fn: StateFunction,
                                   current_expr: sp.Expr) -> sp.Expr:
        """
        Compute the complete Legendre transformation between state functions.
        
        This includes:
        1. Add/subtract conjugate terms (thermal_coeff*T*S + work_coeff*σ*ε)
        2. Compute constitutive relations by differentiation
        3. Invert relations to express natural variables of 'from' in terms of 'to'
        4. Substitute inverted relations into the transformed expression
        
        This follows the complete procedure from gsm_def.py for proper thermodynamic transformation.
        """
        T, S = self.T_var, self.S_var
        
        # Get flat variables for computation
        eps_flat = self._flatten_variable_tuple(self.eps_vars)
        sig_flat = self._flatten_variable_tuple(self.sig_vars)
        
        # Define the fundamental work terms
        if len(eps_flat) == 1 and len(sig_flat) == 1:
            work_term = sig_flat[0] * eps_flat[0]
        else:
            work_term = sum(sig_flat[i] * eps_flat[i] for i in range(len(eps_flat)))
        
        thermal_term = T * S
        
        # Get transformation mapping (class attribute with all Legendre transformation coefficients)
        transformation_map = self.TRANSFORMATION_MAPPING
        
        # Look up transformation coefficients
        transformation_key = (from_fn, to_fn)
        if transformation_key not in transformation_map:
            raise ValueError(f"Transformation from {from_fn} to {to_fn} not implemented")
        
        thermal_coeff, work_coeff = transformation_map[transformation_key]
        
        # Step 1: Apply basic Legendre transformation (add/subtract conjugate terms)
        intermediate_expr = current_expr + thermal_coeff * thermal_term + work_coeff * work_term
        
        # Step 2: Complete transformation with constitutive relation substitutions
        return self._apply_constitutive_substitutions(from_fn, to_fn, intermediate_expr, current_expr)
    
    def _apply_constitutive_substitutions(self, 
                                        from_fn: StateFunction, 
                                        to_fn: StateFunction,
                                        intermediate_expr: sp.Expr,
                                        original_expr: sp.Expr) -> sp.Expr:
        """
        Apply constitutive relation substitutions to complete the Legendre transformation.
        
        This method:
        1. Identifies which variables need to be exchanged between from_fn and to_fn
        2. Computes constitutive relations by differentiating the original expression
        3. Inverts these relations using sympy.solve
        4. Substitutes the inverted relations into the intermediate expression
        
        Following the pattern from gsm_def.py._calculate_symbolic_expressions and _initialize_gibbs_engine
        """
        # Get thermal and mechanical substitution relations separately
        thermal_substitutions = self._get_thermal_constitutive_substitutions(from_fn, to_fn, original_expr)
        mechanical_substitutions = self._get_mechanical_constitutive_substitutions(from_fn, to_fn, original_expr)
        
        # Combine all substitutions
        all_substitutions = {**thermal_substitutions, **mechanical_substitutions}
        
        # Apply all substitutions to get the final transformed expression
        final_expr = intermediate_expr
        if all_substitutions:
            try:
                final_expr = sp.simplify(intermediate_expr.subs(all_substitutions))
            except Exception:
                # If substitution fails, return the intermediate result
                final_expr = intermediate_expr
        
        return final_expr
    
    def _get_thermal_constitutive_substitutions(self, 
                                              from_fn: StateFunction, 
                                              to_fn: StateFunction,
                                              original_expr: sp.Expr) -> Dict[sp.Symbol, sp.Expr]:
        """
        Get thermal constitutive relation substitutions (T ↔ S transformations).
        
        Computes the thermal constitutive relations by differentiation:
        - S = ∂F/∂T (entropy relation from Helmholtz/Gibbs)  
        - T = ∂U/∂S (temperature relation from Internal Energy/Enthalpy)
        
        Then inverts these relations to express one thermal variable in terms of the other.
        
        Args:
            from_fn: Source state function
            to_fn: Target state function  
            original_expr: Original expression to differentiate
            
        Returns:
            Dictionary mapping thermal variables to their substitution expressions (simplified)
        """
        T, S = self.T_var, self.S_var
        
        # Get natural variables for both state functions
        from_natural, from_conjugate = self.get_natural_variables(from_fn)
        to_natural, to_conjugate = self.get_natural_variables(to_fn)
        
        substitution_relations = {}
        
        # Check thermal variables (T ↔ S)
        if T in from_natural and T not in to_natural:
            # Need to substitute T in terms of S
            # Compute: S = ∂F/∂T, then solve for T = f(S)
            entropy_relation = sp.diff(original_expr, T)
            try:
                T_expr = sp.solve(sp.Eq(S, entropy_relation), T)
                if T_expr:
                    # Apply simplification to the substitution expression
                    T_simplified = self.simplify_with_timeout(T_expr[0])
                    substitution_relations[T] = T_simplified
            except (sp.SolvableError, NotImplementedError):
                # If symbolic inversion fails, skip this substitution
                pass
        elif S in from_natural and S not in to_natural:
            # Need to substitute S in terms of T
            # Compute: T = ∂U/∂S, then solve for S = f(T)  
            temperature_relation = sp.diff(original_expr, S)
            try:
                S_expr = sp.solve(sp.Eq(T, temperature_relation), S)
                if S_expr:
                    # Apply simplification to the substitution expression
                    S_simplified = self.simplify_with_timeout(S_expr[0])
                    substitution_relations[S] = S_simplified
            except (sp.SolvableError, NotImplementedError):
                pass
        
        return substitution_relations
    
    def _get_mechanical_constitutive_substitutions(self, 
                                                 from_fn: StateFunction, 
                                                 to_fn: StateFunction,
                                                 original_expr: sp.Expr) -> Dict[sp.Symbol, sp.Expr]:
        """
        Get mechanical constitutive relation substitutions (ε ↔ σ transformations).
        
        Computes the mechanical constitutive relations by differentiation:
        - σ = ∂F/∂ε (stress relation from strain-based functions: F, U)
        - ε = -∂G/∂σ (strain relation from stress-based functions: G, H)
        
        Then inverts these relations to express one mechanical variable in terms of the other.
        
        Args:
            from_fn: Source state function
            to_fn: Target state function
            original_expr: Original expression to differentiate
            
        Returns:
            Dictionary mapping mechanical variables to their substitution expressions (simplified)
        """
        eps_flat = self._flatten_variable_tuple(self.eps_vars)
        sig_flat = self._flatten_variable_tuple(self.sig_vars)
        
        # Get natural variables for both state functions
        from_natural, from_conjugate = self.get_natural_variables(from_fn)
        to_natural, to_conjugate = self.get_natural_variables(to_fn)
        
        substitution_relations = {}
        
        # Check mechanical variables (ε ↔ σ)
        for i, (eps_i, sig_i) in enumerate(zip(eps_flat, sig_flat)):
            if eps_i in from_natural and eps_i not in to_natural:
                # Need to substitute eps in terms of sig
                # Compute: σ = ∂F/∂ε, then solve for ε = f(σ)
                stress_relation = sp.diff(original_expr, eps_i)
                try:
                    eps_expr = sp.solve(sp.Eq(sig_i, stress_relation), eps_i)
                    if eps_expr:
                        # Apply simplification to the substitution expression
                        eps_simplified = self.simplify_with_timeout(eps_expr[0])
                        substitution_relations[eps_i] = eps_simplified
                except (sp.SolvableError, NotImplementedError):
                    pass
            elif sig_i in from_natural and sig_i not in to_natural:
                # Need to substitute sig in terms of eps
                # Compute: ε = -∂G/∂σ, then solve for σ = f(ε)
                # Use original_expr (not intermediate) because we're working from the current state function
                strain_relation = -sp.diff(original_expr, sig_i)
                try:
                    sig_expr = sp.solve(sp.Eq(eps_i, strain_relation), sig_i)
                    if sig_expr:
                        # Apply simplification to the substitution expression
                        sig_simplified = self.simplify_with_timeout(sig_expr[0])
                        substitution_relations[sig_i] = sig_simplified
                except (sp.SolvableError, NotImplementedError):
                    pass
        
        return substitution_relations
    
    def get_thermal_constitutive_relations(self, state_fn: StateFunction = None) -> Dict[str, sp.Expr]:
        """
        Get thermal constitutive relations for display and analysis.
        
        Computes the thermal constitutive equations by differentiation:
        - Entropy: S = ∂F/∂T (for Helmholtz/Gibbs based functions)
        - Temperature: T = ∂U/∂S (for Internal Energy/Enthalpy based functions)
        
        Args:
            state_fn: State function to compute relations for (default: current)
            
        Returns:
            Dictionary with 'entropy' and/or 'temperature' relations (simplified)
        """
        if state_fn is None:
            state_fn = self.current_state_fn
            
        # Must have explicit expression to compute derivatives
        expr = self.explicit_expressions.get(state_fn)
        if expr is None:
            return {}
        
        T, S = self.T_var, self.S_var
        natural_vars, conjugate_vars = self.get_natural_variables(state_fn)
        
        thermal_relations = {}
        
        # If T is natural variable, compute entropy relation: S = ∂/∂T
        if T in natural_vars:
            entropy_expr = sp.diff(expr, T)
            # Apply simplification with timeout
            entropy_simplified = self.simplify_with_timeout(entropy_expr)
            thermal_relations['entropy'] = entropy_simplified
            
        # If S is natural variable, compute temperature relation: T = ∂/∂S  
        if S in natural_vars:
            temperature_expr = sp.diff(expr, S)
            # Apply simplification with timeout
            temperature_simplified = self.simplify_with_timeout(temperature_expr)
            thermal_relations['temperature'] = temperature_simplified
            
        return thermal_relations
    
    def get_mechanical_constitutive_relations(self, state_fn: StateFunction = None) -> Dict[str, sp.Expr]:
        """
        Get mechanical constitutive relations for display and analysis.
        
        Computes the mechanical constitutive equations by differentiation:
        - Stress: σ = ∂F/∂ε (for strain-based functions: F, U)
        - Strain: ε = -∂G/∂σ (for stress-based functions: G, H)
        
        Args:
            state_fn: State function to compute relations for (default: current)
            
        Returns:
            Dictionary with 'stress' and/or 'strain' relations (simplified)
        """
        if state_fn is None:
            state_fn = self.current_state_fn
            
        # Must have explicit expression to compute derivatives
        expr = self.explicit_expressions.get(state_fn)
        if expr is None:
            return {}
        
        eps_flat = self._flatten_variable_tuple(self.eps_vars)
        sig_flat = self._flatten_variable_tuple(self.sig_vars)
        natural_vars, conjugate_vars = self.get_natural_variables(state_fn)
        
        mechanical_relations = {}
        
        # Process each mechanical variable pair
        for i, (eps_i, sig_i) in enumerate(zip(eps_flat, sig_flat)):
            # If ε is natural variable, compute stress relation: σ = ∂/∂ε
            if eps_i in natural_vars:
                stress_expr = sp.diff(expr, eps_i)
                # Apply simplification with timeout
                stress_simplified = self.simplify_with_timeout(stress_expr)
                mechanical_relations[f'stress_{i}'] = stress_simplified
                
            # If σ is natural variable, compute strain relation: ε = -∂/∂σ
            if sig_i in natural_vars:
                strain_expr = -sp.diff(expr, sig_i)
                # Apply simplification with timeout
                strain_simplified = self.simplify_with_timeout(strain_expr)
                mechanical_relations[f'strain_{i}'] = strain_simplified
                
        return mechanical_relations
    
    def set_current_state_function(self, state_fn: StateFunction, expression: sp.Expr = None):
        """
        Set the current active state function.
        
        Args:
            state_fn: State function to set as current
            expression: Optional explicit expression for the state function
        """
        self.current_state_fn = state_fn
        if expression is not None:
            self.explicit_expressions[state_fn] = expression
    
    def get_current_expression(self) -> sp.Expr:
        """
        Get the explicit expression for the current state function.
        
        Returns the cached mathematical expression if available,
        otherwise falls back to the symbolic function template.
        """
        return self.explicit_expressions.get(self.current_state_fn,
               self.state_expressions[self.current_state_fn])
    
    def get_all_expressions(self) -> Dict[StateFunction, sp.Expr]:
        """
        Get all available explicit expressions.
        
        Returns only the computed/cached mathematical expressions,
        not the symbolic function templates.
        """
        return self.explicit_expressions.copy()
    
    def get_function_templates(self) -> Dict[StateFunction, sp.Function]:
        """
        Get all symbolic function templates showing variable dependencies.
        
        These show which variables each state function depends on (e.g., F(T,ε,Ɛ))
        but do not contain actual mathematical expressions.
        """
        return self.state_expressions.copy()
    
    def print_state_overview(self) -> None:
        """
        Print an overview of the current state of the thermodynamic box.
        
        Shows function templates, available explicit expressions, and natural variables.
        """
        print("GSMSymbBox State Overview")
        print("=" * 30)
        print(f"Current active state function: {self.current_state_fn.value}")
        print()
        
        print("Function Templates (show variable dependencies):")
        for state_fn, template in self.state_expressions.items():
            natural_vars, conjugate_vars = self.get_natural_variables(state_fn)
            natural_str = ", ".join([str(v) for v in natural_vars])
            conjugate_str = ", ".join([str(v) for v in conjugate_vars])
            print(f"  {state_fn.value}: {template}")
            print(f"    Natural (independent): [{natural_str}], Conjugate (derivatives): [{conjugate_str}]")
        print()
        
        print("Explicit Expressions (computed mathematical forms):")
        if self.explicit_expressions:
            for state_fn, expr in self.explicit_expressions.items():
                print(f"  {state_fn.value}: {expr}")
        else:
            print("  None computed yet")
        print()
    
    def compute_constitutive_relations(self, state_fn: StateFunction = None) -> Dict[sp.Symbol, sp.Expr]:
        """
        Compute constitutive relations (derivatives) for the given state function.
        
        For each natural variable, computes the partial derivative of the state function
        to get the corresponding conjugate variable (constitutive relation).
        
        Args:
            state_fn: State function to compute relations for (default: current)
            
        Returns:
            Dictionary mapping conjugate variables to their constitutive expressions
        """
        if state_fn is None:
            state_fn = self.current_state_fn
            
        # Must have explicit expression to compute derivatives
        expr = self.explicit_expressions.get(state_fn)
        if expr is None:
            raise ValueError(f"No explicit expression available for {state_fn}")
        
        natural_vars, conjugate_vars = self.get_natural_variables(state_fn)
        constitutive_relations = {}
        
        # Compute partial derivatives: conjugate = ∂(state_function)/∂(natural)
        for i, natural_var in enumerate(natural_vars):
            if i < len(conjugate_vars):
                conjugate_var = conjugate_vars[i]
                constitutive_relations[conjugate_var] = sp.diff(expr, natural_var)
        
        return constitutive_relations
    
    def get_substitution_relations(self, from_fn: StateFunction, to_fn: StateFunction) -> Dict[sp.Symbol, sp.Expr]:
        """
        Get the substitution relations needed for Legendre transformation between state functions.
        
        Args:
            from_fn: Source state function
            to_fn: Target state function
            
        Returns:
            Dictionary mapping variables to their substitution expressions
        """
        if from_fn not in self.explicit_expressions:
            raise ValueError(f"No explicit expression available for {from_fn}")
            
        original_expr = self.explicit_expressions[from_fn]
        T, S = self.T_var, self.S_var
        eps_flat = self._flatten_variable_tuple(self.eps_vars)
        sig_flat = self._flatten_variable_tuple(self.sig_vars)
        
        # Get natural variables for both state functions
        from_natural, from_conjugate = self.get_natural_variables(from_fn)
        to_natural, to_conjugate = self.get_natural_variables(to_fn)
        
        substitution_relations = {}
        
        # Check thermal variables (T ↔ S)
        if T in from_natural and T not in to_natural:
            # T = f(S): solve ∂F/∂T = S for T
            entropy_relation = sp.diff(original_expr, T)
            try:
                T_expr = sp.solve(sp.Eq(S, entropy_relation), T)
                if T_expr:
                    substitution_relations[T] = T_expr[0]
            except (sp.SolvableError, NotImplementedError):
                pass
        elif S in from_natural and S not in to_natural:
            # S = f(T): solve ∂F/∂S = T for S  
            temperature_relation = sp.diff(original_expr, S)
            try:
                S_expr = sp.solve(sp.Eq(T, temperature_relation), S)
                if S_expr:
                    substitution_relations[S] = S_expr[0]
            except (sp.SolvableError, NotImplementedError):
                pass
        
        # Check mechanical variables (ε ↔ σ)
        for eps_i, sig_i in zip(eps_flat, sig_flat):
            if eps_i in from_natural and eps_i not in to_natural:
                # ε = f(σ): solve ∂F/∂ε = σ for ε
                stress_relation = sp.diff(original_expr, eps_i)
                try:
                    eps_expr = sp.solve(sp.Eq(sig_i, stress_relation), eps_i)
                    if eps_expr:
                        substitution_relations[eps_i] = eps_expr[0]
                except (sp.SolvableError, NotImplementedError):
                    pass
        
        return substitution_relations
    
    def validate_thermodynamic_consistency(self) -> bool:
        """
        Validate thermodynamic consistency across all available expressions.
        
        Returns:
            True if all expressions are consistent, False otherwise
        """
        # Check if we have at least two expressions to compare
        if len(self.explicit_expressions) < 2:
            return True
        
        # For validation, we can check if round-trip transformations preserve expressions
        available_functions = list(self.explicit_expressions.keys())
        
        for i in range(len(available_functions)):
            for j in range(i + 1, len(available_functions)):
                fn1, fn2 = available_functions[i], available_functions[j]
                
                # Store current state
                original_fn = self.current_state_fn
                
                try:
                    # Transform fn1 → fn2 → fn1 and check consistency
                    self.current_state_fn = fn1
                    expr_fn2 = self.legendre_transform(fn2)
                    
                    self.current_state_fn = fn2
                    expr_fn1_back = self.legendre_transform(fn1)
                    
                    # Check if we get back the original expression (simplified)
                    original_expr = self.explicit_expressions[fn1]
                    if not sp.simplify(expr_fn1_back - original_expr).equals(0):
                        return False
                        
                finally:
                    # Restore original state
                    self.current_state_fn = original_fn
        
        return True
    
    def get_transformation_graph(self) -> Dict[StateFunction, List[StateFunction]]:
        """
        Get the transformation graph showing possible direct transformations.
        
        Returns:
            Dictionary mapping each state function to its directly accessible neighbors
        """
        return {
            StateFunction.INTERNAL_ENERGY: [StateFunction.HELMHOLTZ, StateFunction.ENTHALPY],
            StateFunction.HELMHOLTZ: [StateFunction.INTERNAL_ENERGY, StateFunction.GIBBS],
            StateFunction.ENTHALPY: [StateFunction.INTERNAL_ENERGY, StateFunction.GIBBS],
            StateFunction.GIBBS: [StateFunction.HELMHOLTZ, StateFunction.ENTHALPY]
        }
    
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
    
    def print_transformation_details(self, from_fn: StateFunction, to_fn: StateFunction) -> None:
        """
        Print detailed information about a specific Legendre transformation.
        
        Shows:
        1. The basic transformation formula
        2. Constitutive relations (derivatives)
        3. Substitution relations (inversions)
        4. Complete transformation result
        """
        if from_fn not in self.explicit_expressions:
            print(f"No explicit expression available for {from_fn}")
            return
            
        print(f"Legendre Transformation: {from_fn.value} → {to_fn.value}")
        print("=" * 50)
        
        # Get transformation coefficients
        transformation_map = self.TRANSFORMATION_MAPPING
        thermal_coeff, work_coeff = transformation_map.get((from_fn, to_fn), (0, 0))
        
        # Show basic transformation
        terms = []
        if thermal_coeff != 0:
            terms.append(f"{thermal_coeff:+d}*T*S")
        if work_coeff != 0:
            terms.append(f"{work_coeff:+d}*σ*ε")
        
        if terms:
            transform_str = f"{to_fn.value} = {from_fn.value} " + " ".join(terms)
        else:
            transform_str = f"{to_fn.value} = {from_fn.value}"
        
        print(f"1. Basic transformation: {transform_str}")
        
        # Show constitutive relations
        print(f"\n2. Constitutive relations from {from_fn.value}:")
        try:
            const_relations = self.compute_constitutive_relations(from_fn)
            for conjugate, relation in const_relations.items():
                print(f"   {conjugate} = ∂{from_fn.value}/∂(...) = {relation}")
        except Exception as e:
            print(f"   Could not compute: {e}")
        
        # Show substitution relations
        print(f"\n3. Substitution relations for {from_fn.value} → {to_fn.value}:")
        try:
            subs_relations = self.get_substitution_relations(from_fn, to_fn)
            if subs_relations:
                for var, expr in subs_relations.items():
                    print(f"   {var} = {expr}")
            else:
                print("   No substitutions needed (variables match)")
        except Exception as e:
            print(f"   Could not compute substitutions: {e}")
        
        # Show final result
        print(f"\n4. Final transformed expression:")
        try:
            self.set_current_state_function(from_fn)
            result = self.legendre_transform(to_fn)
            print(f"   {to_fn.value} = {result}")
        except Exception as e:
            print(f"   Transformation failed: {e}")
        
        print()
    
    # Properties for convenient access to state functions
    @property
    def U(self) -> sp.Expr:
        """
        Internal Energy U(S, ε, Ɛ).
        
        Returns the explicit expression if computed, otherwise computes it via 
        Legendre transformation from the current state function.
        """
        if StateFunction.INTERNAL_ENERGY in self.explicit_expressions:
            expr = self.explicit_expressions[StateFunction.INTERNAL_ENERGY]
            # Apply simplification if enabled
            if self.auto_simplify:
                return self.simplify_with_timeout(expr)
            return expr
        else:
            return self.legendre_transform(StateFunction.INTERNAL_ENERGY)
    
    @property 
    def F(self) -> sp.Expr:
        """
        Helmholtz Free Energy F(T, ε, Ɛ).
        
        Returns the explicit expression if computed, otherwise computes it via 
        Legendre transformation from the current state function.
        """
        if StateFunction.HELMHOLTZ in self.explicit_expressions:
            expr = self.explicit_expressions[StateFunction.HELMHOLTZ]
            # Apply simplification if enabled
            if self.auto_simplify:
                return self.simplify_with_timeout(expr)
            return expr
        else:
            return self.legendre_transform(StateFunction.HELMHOLTZ)
    
    @property
    def H(self) -> sp.Expr:
        """
        Enthalpy H(S, σ, Ɛ).
        
        Returns the explicit expression if computed, otherwise computes it via 
        Legendre transformation from the current state function.
        """
        if StateFunction.ENTHALPY in self.explicit_expressions:
            expr = self.explicit_expressions[StateFunction.ENTHALPY]
            # Apply simplification if enabled
            if self.auto_simplify:
                return self.simplify_with_timeout(expr)
            return expr
        else:
            return self.legendre_transform(StateFunction.ENTHALPY)
    
    @property
    def G(self) -> sp.Expr:
        """
        Gibbs Free Energy G(T, σ, Ɛ).
        
        Returns the explicit expression if computed, otherwise computes it via 
        Legendre transformation from the current state function.
        """
        if StateFunction.GIBBS in self.explicit_expressions:
            expr = self.explicit_expressions[StateFunction.GIBBS]
            # Apply simplification if enabled
            if self.auto_simplify:
                return self.simplify_with_timeout(expr)
            return expr
        else:
            return self.legendre_transform(StateFunction.GIBBS)
    
    def __repr__(self) -> str:
        """String representation of the thermodynamic box."""
        available = list(self.explicit_expressions.keys())
        return (f"GSMSymbBox(current={self.current_state_fn.value}, "
                f"available={[fn.value for fn in available]})")
