"""
GSM State Function Accessor

This module provides the GSMStateFn class, which acts as a cursor/accessor for specific
state functions within a GSMThermodynBox framework. It handles the computational aspects
of working with a particular state function perspective.

The GSMStateFn is the active computational layer that:
1. Maintains a cursor pointing to the current state function
2. Performs Legendre transformations between state functions
3. Computes constitutive relations and derivatives
4. Handles expression management and caching
5. Validates thermodynamic consistency

This separation allows the thermodynamic box to remain static while the state function
accessor handles all dynamic computations and transformations.
"""

import sympy as sp
from typing import Dict, Optional, TYPE_CHECKING

# Avoid circular imports
if TYPE_CHECKING:
    from .gsm_thermodyn_box import GSMThermodynBox, StateFunctionTag


class GSMStateFn:
    """
    State function accessor providing computational interface to a specific state function.
    
    This class acts as a "cursor" that points to a particular state function within
    a GSMThermodynBox and provides all the computational methods for working with
    that state function perspective.
    
    ## Design Philosophy
    The GSMStateFn separates the static framework (GSMThermodynBox) from the dynamic
    computational aspects. It maintains:
    - Current state function cursor
    - Expression cache for computed state functions
    - Methods for transformations and derivatives
    - Consistency validation
    
    ## Key Features
    - **Cursor Management**: Tracks which state function is currently active
    - **Legendre Transformations**: Compute other state functions from current one
    - **Constitutive Relations**: Compute derivatives (stress, temperature, etc.)
    - **Expression Caching**: Store computed expressions for efficiency
    - **Consistency Validation**: Check thermodynamic consistency
    
    ## Usage Pattern
    ```python
    box = GSMThermodynBox(eps_vars=(eps,), sig_vars=(sig,))
    helmholtz = box.create_state_fn(StateFunctionTag.HELMHOLTZ, F_expr)
    gibbs_expr = helmholtz.transform_to(StateFunctionTag.GIBBS)
    stress = helmholtz.compute_constitutive_relation('sig')
    ```
    """
    
    def __init__(self, 
                 thermodyn_box: 'GSMThermodynBox',
                 state_fn_tag: 'StateFunctionTag',
                 initial_expression: Optional[sp.Expr] = None):
        """
        Initialize state function accessor.
        
        Args:
            thermodyn_box: The thermodynamic box framework
            state_fn_tag: Which state function this accessor represents
            initial_expression: Optional initial expression for this state function
        """
        self.thermodyn_box = thermodyn_box
        self.current_state_fn_tag = state_fn_tag
        
        # Expression cache - stores computed expressions for each state function
        self.expressions: Dict['StateFunctionTag', sp.Expr] = {}
        
        # Set initial expression if provided
        if initial_expression is not None:
            self.expressions[state_fn_tag] = initial_expression
    
    def set_expression(self, expression: sp.Expr) -> None:
        """
        Set the expression for the current state function.
        
        Args:
            expression: Mathematical expression for the current state function
        """
        self.expressions[self.current_state_fn_tag] = expression
    
    def get_current_expression(self) -> sp.Expr:
        """
        Get the expression for the current state function.
        
        Returns:
            Expression for current state function, or symbolic template if no expression set
        """
        if self.current_state_fn_tag in self.expressions:
            return self.expressions[self.current_state_fn_tag]
        else:
            # Return symbolic template from the box
            return self.thermodyn_box.state_function_templates[self.current_state_fn_tag]
    
    def switch_to(self, state_fn_tag: 'StateFunctionTag') -> None:
        """
        Switch the cursor to a different state function.
        
        Args:
            state_fn_tag: State function to switch to
        """
        self.current_state_fn_tag = state_fn_tag
    
    def transform_to(self, target_state_fn_tag: 'StateFunctionTag') -> sp.Expr:
        """
        Perform Legendre transformation to target state function.
        
        Args:
            target_state_fn_tag: Target state function to transform to
            
        Returns:
            Expression for the target state function
        """
        # If requesting the current state function, return it directly
        if target_state_fn_tag == self.current_state_fn_tag:
            return self.get_current_expression()
        
        # Get current explicit expression (must exist for transformation)
        current_expr = self.expressions.get(self.current_state_fn_tag)
        if current_expr is None:
            raise ValueError(f"No explicit expression available for {self.current_state_fn_tag}")
        
        # Perform the Legendre transformation
        transform_expr = self._compute_legendre_transform(
            self.current_state_fn_tag, target_state_fn_tag, current_expr
        )
        
        # Cache the transformed expression for future use
        self.expressions[target_state_fn_tag] = transform_expr
        
        return transform_expr
    
    def _compute_legendre_transform(self, 
                                   from_fn_tag: 'StateFunctionTag', 
                                   to_fn_tag: 'StateFunctionTag',
                                   current_expr: sp.Expr) -> sp.Expr:
        """
        Compute the complete Legendre transformation between state functions.
        
        Args:
            from_fn_tag: Source state function tag
            to_fn_tag: Target state function tag
            current_expr: Current expression to transform
            
        Returns:
            Transformed expression
        """
        T, S = self.thermodyn_box.T_var, self.thermodyn_box.S_var
        
        # Get flat variables for computation
        eps_flat = self.thermodyn_box._flatten_variable_tuple(self.thermodyn_box.eps_vars)
        sig_flat = self.thermodyn_box._flatten_variable_tuple(self.thermodyn_box.sig_vars)
        
        # Define the fundamental work terms
        if len(eps_flat) == 1 and len(sig_flat) == 1:
            work_term = sig_flat[0] * eps_flat[0]
        else:
            work_term = sum(sig_flat[i] * eps_flat[i] for i in range(len(eps_flat)))
        
        thermal_term = T * S
        
        # Get transformation mapping
        transformation_map = self.thermodyn_box.TRANSFORMATION_MAPPING
        
        # Look up transformation coefficients
        transformation_key = (from_fn_tag, to_fn_tag)
        if transformation_key not in transformation_map:
            raise ValueError(f"Transformation from {from_fn_tag} to {to_fn_tag} not implemented")
        
        thermal_coeff, work_coeff = transformation_map[transformation_key]
        
        # Step 1: Apply basic Legendre transformation (add/subtract conjugate terms)
        intermediate_expr = current_expr + thermal_coeff * thermal_term + work_coeff * work_term
        
        # Step 2: Complete transformation with constitutive relation substitutions
        return self._apply_constitutive_substitutions(from_fn_tag, to_fn_tag, intermediate_expr, current_expr)
    
    def _apply_constitutive_substitutions(self, 
                                        from_fn_tag: 'StateFunctionTag', 
                                        to_fn_tag: 'StateFunctionTag',
                                        intermediate_expr: sp.Expr,
                                        original_expr: sp.Expr) -> sp.Expr:
        """
        Apply constitutive relation substitutions to complete the Legendre transformation.
        
        Args:
            from_fn_tag: Source state function tag
            to_fn_tag: Target state function tag
            intermediate_expr: Intermediate expression after basic transformation
            original_expr: Original expression before transformation
            
        Returns:
            Final transformed expression with substitutions applied
        """
        T, S = self.thermodyn_box.T_var, self.thermodyn_box.S_var
        eps_flat = self.thermodyn_box._flatten_variable_tuple(self.thermodyn_box.eps_vars)
        sig_flat = self.thermodyn_box._flatten_variable_tuple(self.thermodyn_box.sig_vars)
        
        # Get natural variables for both state functions
        from_natural, from_conjugate = self.thermodyn_box.get_natural_variables(from_fn_tag)
        to_natural, to_conjugate = self.thermodyn_box.get_natural_variables(to_fn_tag)
        
        # Identify variables that need to be exchanged
        substitution_relations = {}
        
        # Check thermal variables (T ↔ S)
        if T in from_natural and T not in to_natural:
            # Need to substitute T in terms of S
            entropy_relation = sp.diff(original_expr, T)
            try:
                T_expr = sp.solve(sp.Eq(S, entropy_relation), T)
                if T_expr:
                    substitution_relations[T] = T_expr[0]
            except (sp.SolvableError, NotImplementedError):
                pass
        elif S in from_natural and S not in to_natural:
            # Need to substitute S in terms of T
            temperature_relation = sp.diff(original_expr, S)
            try:
                S_expr = sp.solve(sp.Eq(T, temperature_relation), S)
                if S_expr:
                    substitution_relations[S] = S_expr[0]
            except (sp.SolvableError, NotImplementedError):
                pass
        
        # Check mechanical variables (ε ↔ σ)
        for i, (eps_i, sig_i) in enumerate(zip(eps_flat, sig_flat)):
            if eps_i in from_natural and eps_i not in to_natural:
                # Need to substitute eps in terms of sig
                stress_relation = sp.diff(original_expr, eps_i)
                try:
                    eps_expr = sp.solve(sp.Eq(sig_i, stress_relation), eps_i)
                    if eps_expr:
                        substitution_relations[eps_i] = eps_expr[0]
                except (sp.SolvableError, NotImplementedError):
                    pass
            elif sig_i in from_natural and sig_i not in to_natural:
                # Need to substitute sig in terms of eps
                strain_relation = -sp.diff(original_expr, sig_i)
                try:
                    sig_expr = sp.solve(sp.Eq(eps_i, strain_relation), sig_i)
                    if sig_expr:
                        substitution_relations[sig_i] = sig_expr[0]
                except (sp.SolvableError, NotImplementedError):
                    pass
        
        # Apply all substitutions to get the final transformed expression
        final_expr = intermediate_expr
        if substitution_relations:
            try:
                final_expr = sp.simplify(intermediate_expr.subs(substitution_relations))
            except Exception:
                # If substitution fails, return the intermediate result
                final_expr = intermediate_expr
        
        return final_expr
    
    def compute_constitutive_relations(self) -> Dict[sp.Symbol, sp.Expr]:
        """
        Compute constitutive relations (derivatives) for the current state function.
        
        For each natural variable, computes the partial derivative to get the
        corresponding conjugate variable (constitutive relation).
        
        Returns:
            Dictionary mapping conjugate variables to their constitutive expressions
        """
        # Must have explicit expression to compute derivatives
        expr = self.expressions.get(self.current_state_fn_tag)
        if expr is None:
            raise ValueError(f"No explicit expression available for {self.current_state_fn_tag}")
        
        natural_vars, conjugate_vars = self.thermodyn_box.get_natural_variables(self.current_state_fn_tag)
        constitutive_relations = {}
        
        # Compute partial derivatives: conjugate = ∂(state_function)/∂(natural)
        for i, natural_var in enumerate(natural_vars):
            if i < len(conjugate_vars):
                conjugate_var = conjugate_vars[i]
                constitutive_relations[conjugate_var] = sp.diff(expr, natural_var)
        
        return constitutive_relations
    
    def compute_constitutive_relation(self, var_type: str) -> sp.Expr:
        """
        Compute a specific constitutive relation.
        
        Args:
            var_type: Type of variable to compute ('T', 'S', 'sig', 'eps', 'Sig', 'Eps')
            
        Returns:
            Expression for the constitutive relation
        """
        relations = self.compute_constitutive_relations()
        
        # Find the variable of the requested type
        for var, expr in relations.items():
            var_str = str(var)
            if var_type == 'T' and var_str == str(self.thermodyn_box.T_var):
                return expr
            elif var_type == 'S' and var_str == str(self.thermodyn_box.S_var):
                return expr
            elif var_type == 'sig' and var in self.thermodyn_box._flatten_variable_tuple(self.thermodyn_box.sig_vars):
                return expr
            elif var_type == 'eps' and var in self.thermodyn_box._flatten_variable_tuple(self.thermodyn_box.eps_vars):
                return expr
            elif var_type == 'Sig' and var in self.thermodyn_box._flatten_variable_tuple(self.thermodyn_box.Sig_vars):
                return expr
            elif var_type == 'Eps' and var in self.thermodyn_box._flatten_variable_tuple(self.thermodyn_box.Eps_vars):
                return expr
        
        raise ValueError(f"No constitutive relation found for variable type: {var_type}")
    
    def get_all_expressions(self) -> Dict['StateFunctionTag', sp.Expr]:
        """
        Get all cached expressions.
        
        Returns:
            Dictionary of all computed expressions
        """
        return self.expressions.copy()
    
    def validate_consistency(self) -> bool:
        """
        Validate thermodynamic consistency across computed expressions.
        
        Returns:
            True if expressions are consistent, False otherwise
        """
        if len(self.expressions) < 2:
            return True
        
        # Test round-trip transformations
        available_functions = list(self.expressions.keys())
        original_cursor = self.current_state_fn_tag
        
        try:
            for i in range(len(available_functions)):
                for j in range(i + 1, len(available_functions)):
                    fn1, fn2 = available_functions[i], available_functions[j]
                    
                    # Transform fn1 → fn2 → fn1 and check consistency
                    self.switch_to(fn1)
                    expr_fn2 = self.transform_to(fn2)
                    
                    self.switch_to(fn2)
                    expr_fn1_back = self.transform_to(fn1)
                    
                    # Check if we get back the original expression (simplified)
                    original_expr = self.expressions[fn1]
                    if not sp.simplify(expr_fn1_back - original_expr).equals(0):
                        return False
            
            return True
            
        finally:
            # Restore original cursor
            self.switch_to(original_cursor)
    
    def print_state_overview(self) -> None:
        """
        Print an overview of the current state function and available expressions.
        """
        print(f"GSM State Function: {self.current_state_fn_tag.value}")
        print("=" * 30)
        print(f"Current state function: {self.current_state_fn_tag.value}")
        print()
        
        print("Available expressions:")
        if self.expressions:
            for state_fn_tag, expr in self.expressions.items():
                marker = " (current)" if state_fn_tag == self.current_state_fn_tag else ""
                print(f"  {state_fn_tag.value}: {expr}{marker}")
        else:
            print("  None computed yet")
        print()
        
        # Show natural variables for current state function
        natural_vars, conjugate_vars = self.thermodyn_box.get_natural_variables(self.current_state_fn_tag)
        natural_str = ", ".join([str(v) for v in natural_vars])
        conjugate_str = ", ".join([str(v) for v in conjugate_vars])
        print(f"Natural variables: [{natural_str}]")
        print(f"Conjugate variables: [{conjugate_str}]")
        print()
    
    def print_transformation_path(self, target_state_fn_tag: 'StateFunctionTag') -> None:
        """
        Print information about transformation path to target state function.
        
        Args:
            target_state_fn_tag: Target state function
        """
        print(f"Transformation Path: {self.current_state_fn_tag.value} → {target_state_fn_tag.value}")
        print("=" * 40)
        
        # Get transformation coefficients
        transformation_map = self.thermodyn_box.TRANSFORMATION_MAPPING
        key = (self.current_state_fn_tag, target_state_fn_tag)
        
        if key in transformation_map:
            thermal_coeff, work_coeff = transformation_map[key]
            print(f"Direct transformation available:")
            print(f"  Thermal coefficient: {thermal_coeff}")
            print(f"  Work coefficient: {work_coeff}")
            
            # Build transformation string
            terms = []
            if thermal_coeff != 0:
                terms.append(f"{thermal_coeff:+d}*T*S")
            if work_coeff != 0:
                terms.append(f"{work_coeff:+d}*σε")
            
            if terms:
                transform_str = f"{target_state_fn_tag.value} = {self.current_state_fn_tag.value} " + " ".join(terms)
            else:
                transform_str = f"{target_state_fn_tag.value} = {self.current_state_fn_tag.value}"
            
            print(f"  Formula: {transform_str}")
        else:
            print("No direct transformation available")
        print()
    
    # Properties for convenient access to state functions
    @property
    def U(self) -> sp.Expr:
        """Internal Energy U(S, ε, Ɛ)."""
        from .gsm_thermodyn_box import StateFunctionTag
        if StateFunctionTag.INTERNAL_ENERGY in self.expressions:
            return self.expressions[StateFunctionTag.INTERNAL_ENERGY]
        else:
            return self.transform_to(StateFunctionTag.INTERNAL_ENERGY)
    
    @property 
    def F(self) -> sp.Expr:
        """Helmholtz Free Energy F(T, ε, Ɛ)."""
        from .gsm_thermodyn_box import StateFunctionTag
        if StateFunctionTag.HELMHOLTZ in self.expressions:
            return self.expressions[StateFunctionTag.HELMHOLTZ]
        else:
            return self.transform_to(StateFunctionTag.HELMHOLTZ)
    
    @property
    def H(self) -> sp.Expr:
        """Enthalpy H(S, σ, Ɛ)."""
        from .gsm_thermodyn_box import StateFunctionTag
        if StateFunctionTag.ENTHALPY in self.expressions:
            return self.expressions[StateFunctionTag.ENTHALPY]
        else:
            return self.transform_to(StateFunctionTag.ENTHALPY)
    
    @property
    def G(self) -> sp.Expr:
        """Gibbs Free Energy G(T, σ, Ɛ)."""
        from .gsm_thermodyn_box import StateFunctionTag
        if StateFunctionTag.GIBBS in self.expressions:
            return self.expressions[StateFunctionTag.GIBBS]
        else:
            return self.transform_to(StateFunctionTag.GIBBS)
    
    def __repr__(self) -> str:
        """String representation of the state function accessor."""
        available = list(self.expressions.keys())
        return (f"GSMStateFn(current={self.current_state_fn_tag.value}, "
                f"available={[fn.value for fn in available]})")
