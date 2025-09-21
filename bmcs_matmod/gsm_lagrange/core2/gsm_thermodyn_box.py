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


class GSMThermodynBox:
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
    
    This makes the framework completely generic and allows for different state
    function implementations (e.g., numerical, symbolic, hybrid).
    
    ## Generic Variable Handling
    
    The box works with completely abstract thermal variables:
    - th_x_var: Natural thermal variable (could be T, S, or any other thermal variable)
    - th_y_var: Conjugate thermal variable (the thermodynamic conjugate of th_x_var)
    
    This makes the framework more general than traditional implementations that
    hard-code temperature T and entropy S. The box can work with any thermal
    conjugate pair.
    
    Variable mappings are managed by the GSMStateFn class for better
    encapsulation and intelligence.
    """
    
    def __init__(self, 
                 initial_state_fn: StateFunction,
                 initial_state_instance: GSMStateFnIfc):
        """
        Initialize the thermodynamic state function box.
        
        Args:
            initial_state_fn: Starting state function type
            initial_state_instance: State function instance implementing GSMStateFnIfc
        """
        # Initialize state function storage
        self.state_functions: Dict[StateFunction, GSMStateFnIfc] = {}
        self.current_state_fn = initial_state_fn
        
        # Store the initial state function
        self.state_functions[initial_state_fn] = initial_state_instance
        
        # Extract thermal variables from the initial state function instance
        # This makes the box completely generic - it works with any thermal variables
        self.th_x_var = initial_state_instance.th_x_var  # Natural thermal variable
        self.th_y_var = initial_state_instance.th_y_var  # Conjugate thermal variable
        
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
        
        # Get variables from source - use generic thermal variables
        th_x = source_instance.th_x_var  # Natural thermal variable
        th_y = source_instance.th_y_var  # Conjugate thermal variable
        mc_x = source_instance.mc_x_var  # Natural mechanical variable
        mc_y = source_instance.mc_y_var  # Conjugate mechanical variable
        
        # Add/subtract thermal conjugate term: ±th_x*th_y (e.g., ±T*S)
        if thermal_coeff != 0:
            expr += thermal_coeff * th_x * th_y
        
        # Add/subtract work conjugate term: ±mc_x*mc_y (e.g., ±ε*σ)
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
        
        # Get variables from source instance
        th_x = source_instance.th_x_var  # Natural thermal variable  
        th_y = source_instance.th_y_var  # Conjugate thermal variable
        mc_x = source_instance.mc_x_var  # Natural mechanical variable
        mc_y = source_instance.mc_y_var  # Conjugate mechanical variable
        
        # Apply thermal variable substitutions using the interface methods
        if thermal_coeff != 0:
            # Use the existing interface method to get thermal constitutive relation
            th_y_var, th_y_from_source = source_instance.get_thermal_constitutive_relation()
            
            # Apply sign convention based on the state function type
            if hasattr(source_instance, 'state_function_type'):
                if source_instance.state_function_type in [StateFunction.HELMHOLTZ, StateFunction.GIBBS]:
                    # For F and G: S = -∂F/∂T, S = -∂G/∂T
                    th_y_from_source = -th_y_from_source
                # For U and H: T = ∂U/∂S, T = ∂H/∂S (positive derivative)
            
            if th_y_from_source != 0:
                # If conjugate variable depends on natural variable, solve for natural in terms of conjugate
                try:
                    thermal_relation = sp.Eq(th_y, th_y_from_source)
                    th_x_solutions = sp.solve(thermal_relation, th_x)
                    if th_x_solutions:
                        substitutions[th_x] = th_x_solutions[0]
                except:
                    # Keep original variable if inversion fails
                    pass
        
        # Apply mechanical variable substitutions using the interface methods
        if work_coeff != 0:
            # Use the existing interface method to get mechanical constitutive relations
            mechanical_relations = source_instance.get_mechanical_constitutive_relations()
            
            if mechanical_relations:
                mc_y_var, mc_y_from_source = mechanical_relations[0]  # Get first mechanical relation
                
                if mc_y_from_source != 0:
                    try:
                        # Solve mc_y = ∂f/∂mc_x for mc_x in terms of mc_y
                        mechanical_relation = sp.Eq(mc_y, mc_y_from_source)
                        mc_x_solutions = sp.solve(mechanical_relation, mc_x)
                        if mc_x_solutions:
                            substitutions[mc_x] = mc_x_solutions[0]
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
        
        # Get thermal variables from source - these remain abstract
        source_th_x = source_instance.th_x_var  # Current natural thermal variable
        source_th_y = source_instance.th_y_var  # Current conjugate thermal variable
        
        # Organize thermal variables according to target function's natural variables
        # The key insight: we don't need to know if these are T/S - just natural/conjugate roles
        if natural_types[0] == conjugate_types[0]:  # This shouldn't happen, but handle edge case
            th_x_var, th_y_var = source_th_x, source_th_y
        else:
            # Check if the current natural thermal variable should remain natural in target
            current_natural_type = 'T' if 'T' in str(source_th_x).upper() else 'S'
            if current_natural_type in natural_types:
                # Current natural remains natural in target
                th_x_var, th_y_var = source_th_x, source_th_y
            else:
                # Thermal variables swap roles in target
                th_x_var, th_y_var = source_th_y, source_th_x
        
        # Handle mechanical variables - get source variables
        mc_x_source = source_instance.mc_x_var
        mc_y_source = source_instance.mc_y_var
        
        # For mechanical variables, check if current natural should remain natural
        current_mc_natural = 'eps' if any(term in str(mc_x_source).lower() for term in ['eps', 'epsilon']) else 'sig'
        
        if current_mc_natural in natural_types:
            # Current mechanical natural remains natural
            mc_x_var = mc_x_source
            mc_y_var = mc_y_source
        else:
            # Mechanical variables swap roles
            mc_x_var = mc_y_source
            mc_y_var = mc_x_source
        
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
        return f"GSMThermoDynBox(current={self.current_state_fn.value}, available={available})"
