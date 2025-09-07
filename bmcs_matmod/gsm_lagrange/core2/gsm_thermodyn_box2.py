"""
GSM Thermodynamic State Function Box 2 - Interface-based Implementation

This module provides a thermodynamic framework based on the GSMStateFnIfc interface,
where each state function is represented as an instance rather than just expressions.

The box manages four fundamental state functions:
1. Internal Energy U(S, ε, Ɛ) - extensive variables: entropy, strain, internal variables
2. Helmholtz Free Energy F(T, ε, Ɛ) - temperature, strain, internal variables  
3. Enthalpy H(S, σ, Ɛ) - entropy, stress, internal variables
4. Gibbs Free Energy G(T, σ, Ɛ) - temperature, stress, internal variables

Each state function is represented as an instance implementing GSMStateFnIfc,
making the framework more modular and extensible.
"""

import sympy as sp
from typing import Dict, List, Optional, Tuple, Union
from .gsm_state_fn_ifc import GSMStateFnIfc
from .gsm_state_fn import GSMStateFn, StateFunction, NATURAL_VARIABLES_MAPPING, TRANSFORMATION_MAPPING


class GSMThermodynBox2:
    """
    Interface-based thermodynamic state function box.
    
    This implementation uses GSMStateFnIfc instances to represent each of the four
    fundamental state functions. The box manages:
    - State function instances as nodes/edges in the thermodynamic graph
    - Legendre transformations between state functions
    - Variable consistency across transformations
    - Constitutive relation computation
    
    ## Design Philosophy
    
    Instead of storing expressions directly, this box stores state function
    instances that encapsulate:
    1. The mathematical expression
    2. Variable organization (natural vs conjugate)
    3. Constitutive relation computation
    4. Transformation logic
    
    This makes the framework more modular and allows for different state
    function implementations (e.g., numerical, symbolic, hybrid).
    
    Variable mappings are now managed by the GSMStateFn class for better
    encapsulation and intelligence.
    """
    
    def __init__(self, 
                 initial_state_fn: StateFunction,
                 initial_state_instance: GSMStateFnIfc,
                 T_var: sp.Symbol = None,
                 S_var: sp.Symbol = None):
        """
        Initialize the thermodynamic state function box.
        
        Args:
            initial_state_fn: Starting state function type
            initial_state_instance: State function instance implementing GSMStateFnIfc
            T_var: Temperature symbol (default: T)
            S_var: Entropy symbol (default: S)
        """
        # Set thermal variables with defaults
        self.T_var = T_var if T_var is not None else sp.Symbol('T', real=True)
        self.S_var = S_var if S_var is not None else sp.Symbol('S', real=True)
        
        # Initialize state function storage
        self.state_functions: Dict[StateFunction, GSMStateFnIfc] = {}
        self.current_state_fn = initial_state_fn
        
        # Store the initial state function
        self.state_functions[initial_state_fn] = initial_state_instance
        
        # Validate the initial state function
        self._validate_state_function(initial_state_fn, initial_state_instance)
    
    def _validate_state_function(self, state_fn: StateFunction, instance: GSMStateFnIfc) -> None:
        """Validate that a state function instance matches expected variable organization."""
        expected_natural, expected_conjugate = NATURAL_VARIABLES_MAPPING[state_fn]
        
        # Get actual variables from instance
        natural_vars = instance.get_natural_variables()
        conjugate_vars = instance.get_conjugate_variables()
        
        # Basic validation - ensure we have the right number and types of variables
        if len(natural_vars) == 0:
            raise ValueError(f"State function {state_fn} has no natural variables")
        if len(conjugate_vars) == 0:
            raise ValueError(f"State function {state_fn} has no conjugate variables")
            
        # Validate that the instance has the correct type
        if isinstance(instance, GSMStateFn):
            if instance.state_function_type != state_fn:
                raise ValueError(f"State function instance type {instance.state_function_type} "
                               f"does not match expected type {state_fn}")
    
    def set_state_function(self, state_fn: StateFunction, instance: GSMStateFnIfc) -> None:
        """Set a specific state function instance."""
        self._validate_state_function(state_fn, instance)
        self.state_functions[state_fn] = instance
    
    def get_state_function(self, state_fn: StateFunction) -> Optional[GSMStateFnIfc]:
        """Get a specific state function instance."""
        return self.state_functions.get(state_fn)
    
    def get_current_state_function(self) -> GSMStateFnIfc:
        """Get the currently active state function instance."""
        return self.state_functions[self.current_state_fn]
    
    def set_current_state_function(self, state_fn: StateFunction) -> None:
        """Set the currently active state function."""
        if state_fn not in self.state_functions:
            raise ValueError(f"State function {state_fn} not available. "
                           f"Available: {list(self.state_functions.keys())}")
        self.current_state_fn = state_fn
    
    def legendre_transform(self, target_state_fn: StateFunction) -> GSMStateFnIfc:
        """
        Perform Legendre transformation from current to target state function.
        
        Args:
            target_state_fn: Target state function to transform to
            
        Returns:
            State function instance for the target function
        """
        # If target already exists, return it
        if target_state_fn in self.state_functions:
            return self.state_functions[target_state_fn]
        
        # Get current state function
        current_instance = self.get_current_state_function()
        if current_instance is None:
            raise ValueError(f"No current state function set")
        
        # Perform transformation
        transform_key = (self.current_state_fn, target_state_fn)
        if transform_key not in TRANSFORMATION_MAPPING:
            raise ValueError(f"No transformation defined from {self.current_state_fn} to {target_state_fn}")
        
        # Compute transformed expression
        thermal_coeff, work_coeff = TRANSFORMATION_MAPPING[transform_key]
        transformed_expr = self._compute_legendre_transform(
            current_instance, thermal_coeff, work_coeff
        )
        
        # Create new state function instance with proper variable organization
        target_instance = self._create_target_state_function(
            target_state_fn, transformed_expr, current_instance
        )
        
        # Store and return the transformed state function
        self.state_functions[target_state_fn] = target_instance
        return target_instance
    
    def _compute_legendre_transform(self, 
                                  source_instance: GSMStateFnIfc,
                                  thermal_coeff: int,
                                  work_coeff: int) -> sp.Expr:
        """Compute the Legendre transformation expression with constitutive relation substitutions."""
        expr = source_instance.fn_expr
        
        # Get variables from source
        T = source_instance.th_x_var if hasattr(source_instance, 'th_x_var') else self.T_var
        S = source_instance.th_y_var if hasattr(source_instance, 'th_y_var') else self.S_var
        mc_x = source_instance.mc_x_var
        mc_y = source_instance.mc_y_var
        
        # Add/subtract thermal conjugate term: ±T*S
        if thermal_coeff != 0:
            expr += thermal_coeff * T * S
        
        # Add/subtract work conjugate term: ±σ*ε
        if work_coeff != 0:
            expr += work_coeff * mc_x * mc_y
        
        # Apply constitutive relation substitutions to get fully transformed expression
        expr = self._apply_constitutive_substitutions(expr, source_instance, thermal_coeff, work_coeff)
        
        return expr
    
    def _apply_constitutive_substitutions(self, 
                                        expr: sp.Expr,
                                        source_instance: GSMStateFnIfc,
                                        thermal_coeff: int,
                                        work_coeff: int) -> sp.Expr:
        """Apply constitutive relation substitutions to complete the Legendre transformation."""
        substitutions = {}
        
        # Get variables
        T = source_instance.th_x_var if hasattr(source_instance, 'th_x_var') else self.T_var
        S = source_instance.th_y_var if hasattr(source_instance, 'th_y_var') else self.S_var
        mc_x = source_instance.mc_x_var  
        mc_y = source_instance.mc_y_var
        
        # Apply thermal variable substitutions
        if thermal_coeff != 0:
            # Derive constitutive relation for entropy: S = -∂F/∂T
            S_from_source = -sp.diff(source_instance.fn_expr, T)
            
            if S_from_source != 0:
                # If entropy depends on T, solve for T in terms of S
                try:
                    entropy_relation = sp.Eq(S, S_from_source)
                    T_solutions = sp.solve(entropy_relation, T)
                    if T_solutions:
                        substitutions[T] = T_solutions[0]
                except:
                    # Keep T unchanged if inversion fails
                    pass
        
        # Apply mechanical variable substitutions  
        if work_coeff != 0:
            # For F→H: work_coeff = -1, we have -ε*σ term
            # We need to substitute ε with its inverse relation from σ
            
            # Derive stress constitutive relation: σ = ∂F/∂ε
            stress_from_source = sp.diff(source_instance.fn_expr, mc_x)
            
            if stress_from_source != 0:
                try:
                    # Solve σ = ∂F/∂ε for ε in terms of σ
                    stress_relation = sp.Eq(mc_y, stress_from_source)
                    strain_solutions = sp.solve(stress_relation, mc_x)
                    if strain_solutions:
                        substitutions[mc_x] = strain_solutions[0]
                except:
                    # Keep original variables if inversion fails
                    pass
        
        # Apply substitutions if any were found
        if substitutions:
            expr = expr.subs(substitutions)
            # Simplify the result
            expr = sp.simplify(expr)
        
        return expr
    
    def _create_target_state_function(self, 
                                    target_state_fn: StateFunction,
                                    transformed_expr: sp.Expr,
                                    source_instance: GSMStateFnIfc) -> GSMStateFn:
        """Create a new state function instance for the target function."""
        # Get expected variable organization for target
        natural_types, conjugate_types = NATURAL_VARIABLES_MAPPING[target_state_fn]
        
        # Map variable types to actual symbols from source
        T = source_instance.th_x_var if hasattr(source_instance, 'th_x_var') else self.T_var
        S = source_instance.th_y_var if hasattr(source_instance, 'th_y_var') else self.S_var
        
        # Organize variables according to target function's natural variables
        th_x_var, th_y_var = (T, S) if 'T' in natural_types else (S, T)
        
        # Handle mechanical variables - get source variables
        mc_x_source = source_instance.mc_x_var
        mc_y_source = source_instance.mc_y_var
        
        # For mechanical variables, check if eps or sig should be natural
        if 'eps' in natural_types:
            # eps is natural, sig is conjugate
            mc_x_var = mc_x_source if 'eps' in str(mc_x_source) or 'epsilon' in str(mc_x_source) else mc_y_source
            mc_y_var = mc_y_source if 'sig' in str(mc_y_source) or 'sigma' in str(mc_y_source) else mc_x_source
        else:
            # sig is natural, eps is conjugate
            mc_x_var = mc_x_source if 'sig' in str(mc_x_source) or 'sigma' in str(mc_x_source) else mc_y_source
            mc_y_var = mc_y_source if 'eps' in str(mc_y_source) or 'epsilon' in str(mc_y_source) else mc_x_source
        
        # Internal variables remain the same (always extensive in GSM)
        Eps_var = source_instance.Eps_var
        Sig_var = source_instance.Sig_var
        
        return GSMStateFn(
            fn_expr=transformed_expr,
            th_x_var=th_x_var,
            th_y_var=th_y_var,
            mc_x_var=mc_x_var,
            mc_y_var=mc_y_var,
            Eps_var=Eps_var,
            Sig_var=Sig_var,
            state_function_type=target_state_fn
        )
    
    def get_available_state_functions(self) -> List[StateFunction]:
        """Get list of currently available state functions."""
        return list(self.state_functions.keys())
    
    def compute_all_state_functions(self) -> Dict[StateFunction, GSMStateFnIfc]:
        """Compute all four state functions by performing necessary transformations."""
        target_functions = [StateFunction.INTERNAL_ENERGY, StateFunction.HELMHOLTZ, 
                          StateFunction.ENTHALPY, StateFunction.GIBBS]
        
        for target_fn in target_functions:
            if target_fn not in self.state_functions:
                self.legendre_transform(target_fn)
        
        return dict(self.state_functions)
    
    def print_overview(self) -> None:
        """Print overview of the thermodynamic box."""
        print("GSM Thermodynamic Box 2 - Overview")
        print("=" * 40)
        print(f"Current state function: {self.current_state_fn.value}")
        print(f"Available state functions: {[sf.value for sf in self.state_functions.keys()]}")
        print()
        
        if self.current_state_fn in self.state_functions:
            print("Current State Function Details:")
            self.state_functions[self.current_state_fn].print_overview()
    
    def print_all_state_functions(self) -> None:
        """Print details of all available state functions."""
        print("All State Functions")
        print("=" * 50)
        
        for state_fn, instance in self.state_functions.items():
            print(f"\n{state_fn.value} - {state_fn.name}")
            print("-" * 30)
            instance.print_overview()
    
    def validate_thermodynamic_consistency(self) -> bool:
        """Validate thermodynamic consistency across available state functions."""
        # Basic validation - ensure we have at least one state function
        if not self.state_functions:
            print("No state functions available for validation")
            return False
        
        # More sophisticated validation could check Maxwell relations, etc.
        print(f"Validation: {len(self.state_functions)} state functions available")
        return True
    
    def get_transformation_graph(self) -> Dict[StateFunction, List[StateFunction]]:
        """Get the graph of possible transformations from each state function."""
        graph = {}
        
        for state_fn in StateFunction:
            graph[state_fn] = []
            for target_fn in StateFunction:
                if state_fn != target_fn and (state_fn, target_fn) in TRANSFORMATION_MAPPING:
                    graph[state_fn].append(target_fn)
        
        return graph
    
    # Properties for convenient access to state functions with on-demand transformation
    @property
    def U(self) -> Optional[GSMStateFnIfc]:
        """
        Internal Energy state function.
        
        Returns the state function instance if available, otherwise computes it via 
        Legendre transformation from the current state function.
        """
        if StateFunction.INTERNAL_ENERGY in self.state_functions:
            return self.state_functions[StateFunction.INTERNAL_ENERGY]
        else:
            # Perform on-demand transformation
            return self.legendre_transform(StateFunction.INTERNAL_ENERGY)
    
    @property
    def F(self) -> Optional[GSMStateFnIfc]:
        """
        Helmholtz Free Energy state function.
        
        Returns the state function instance if available, otherwise computes it via 
        Legendre transformation from the current state function.
        """
        if StateFunction.HELMHOLTZ in self.state_functions:
            return self.state_functions[StateFunction.HELMHOLTZ]
        else:
            # Perform on-demand transformation
            return self.legendre_transform(StateFunction.HELMHOLTZ)
    
    @property
    def H(self) -> Optional[GSMStateFnIfc]:
        """
        Enthalpy state function.
        
        Returns the state function instance if available, otherwise computes it via 
        Legendre transformation from the current state function.
        """
        if StateFunction.ENTHALPY in self.state_functions:
            return self.state_functions[StateFunction.ENTHALPY]
        else:
            # Perform on-demand transformation
            return self.legendre_transform(StateFunction.ENTHALPY)
    
    @property
    def G(self) -> Optional[GSMStateFnIfc]:
        """
        Gibbs Free Energy state function.
        
        Returns the state function instance if available, otherwise computes it via 
        Legendre transformation from the current state function.
        """
        if StateFunction.GIBBS in self.state_functions:
            return self.state_functions[StateFunction.GIBBS]
        else:
            # Perform on-demand transformation
            return self.legendre_transform(StateFunction.GIBBS)
    
    def __repr__(self) -> str:
        """String representation of the thermodynamic box."""
        available = [sf.value for sf in self.state_functions.keys()]
        return f"GSMThermodynBox2(current={self.current_state_fn.value}, available={available})"
