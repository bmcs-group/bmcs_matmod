"""
GSM State Function - Lightweight Implementation

A simplified state function class that accepts a sympy expression and organizes
variables into natural (independent) and conjugate (derivative) categories
across thermal, mechanical, and internal variable domains.
"""

import sympy as sp


class GSMStateFn2:
    """
    Lightweight thermodynamic state function representation.
    
    Accepts a sympy expression and organizes variables into:
    - Natural variables: th_x_var (thermal), mc_x_var (mechanical), Eps_var (internal)  
    - Conjugate variables: th_y_var (thermal), mc_y_var (mechanical), Sig_var (internal)
    
    This provides the foundation for thermodynamic state function manipulation
    without complex transformation logic.
    """
    
    def __init__(self,
                 fn_expr: sp.Expr,
                 th_x_var: sp.Symbol,  # Thermal natural (S or T)
                 th_y_var: sp.Symbol,  # Thermal conjugate (T or S)
                 mc_x_var: sp.Symbol,  # Mechanical natural (ε or σ)
                 mc_y_var: sp.Symbol,  # Mechanical conjugate (σ or ε)
                 Eps_var: sp.Symbol,   # Internal natural (Ɛ)
                 Sig_var: sp.Symbol):  # Internal conjugate (𝒮)
        """
        Initialize state function with expression and variable organization.
        
        Args:
            fn_expr: Sympy expression for the state function
            th_x_var: Thermal natural variable (e.g., S for entropy, T for temperature)
            th_y_var: Thermal conjugate variable (e.g., T for temperature, S for entropy)
            mc_x_var: Mechanical natural variable (e.g., ε for strain, σ for stress)
            mc_y_var: Mechanical conjugate variable (e.g., σ for stress, ε for strain)
            Eps_var: Internal natural variable (internal strain Ɛ)
            Sig_var: Internal conjugate variable (internal stress 𝒮)
        """
        self.fn_expr = fn_expr
        
        # Natural variables (independent)
        self.th_x_var = th_x_var      # Thermal natural
        self.mc_x_var = mc_x_var      # Mechanical natural  
        self.Eps_var = Eps_var        # Internal natural
        
        # Conjugate variables (derivatives)
        self.th_y_var = th_y_var      # Thermal conjugate
        self.mc_y_var = mc_y_var      # Mechanical conjugate
        self.Sig_var = Sig_var        # Internal conjugate
    
    def get_natural_variables(self) -> list[sp.Symbol]:
        """Get all natural (independent) variables."""
        return [self.th_x_var, self.mc_x_var, self.Eps_var]
    
    def get_conjugate_variables(self) -> list[sp.Symbol]:
        """Get all conjugate (derivative) variables."""
        return [self.th_y_var, self.mc_y_var, self.Sig_var]
    
    def get_all_variables(self) -> list[sp.Symbol]:
        """Get all variables (natural + conjugate)."""
        return self.get_natural_variables() + self.get_conjugate_variables()
    
    def compute_constitutive_relations(self) -> dict[sp.Symbol, sp.Expr]:
        """
        Compute constitutive relations as partial derivatives.
        
        Returns:
            Dictionary mapping conjugate variables to their derivative expressions
        """
        relations = {}
        
        # Thermal: conjugate = ∂f/∂(natural)
        relations[self.th_y_var] = sp.diff(self.fn_expr, self.th_x_var)
        
        # Mechanical: conjugate = ∂f/∂(natural)  
        relations[self.mc_y_var] = sp.diff(self.fn_expr, self.mc_x_var)
        
        # Internal: conjugate = ∂f/∂(natural)
        relations[self.Sig_var] = sp.diff(self.fn_expr, self.Eps_var)
        
        return relations
    
    def print_overview(self) -> None:
        """Print a summary of the state function and its variables."""
        print("GSM State Function Overview")
        print("=" * 30)
        print(f"Expression: {self.fn_expr}")
        print()
        
        print("Natural Variables (independent):")
        print(f"  Thermal: {self.th_x_var}")
        print(f"  Mechanical: {self.mc_x_var}")
        print(f"  Internal: {self.Eps_var}")
        print()
        
        print("Conjugate Variables (derivatives):")
        print(f"  Thermal: {self.th_y_var}")
        print(f"  Mechanical: {self.mc_y_var}")
        print(f"  Internal: {self.Sig_var}")
        print()
    
    def print_constitutive_relations(self) -> None:
        """Print the constitutive relations (derivatives)."""
        relations = self.compute_constitutive_relations()
        
        print("Constitutive Relations:")
        print("=" * 25)
        for conjugate_var, derivative in relations.items():
            print(f"{conjugate_var} = ∂f/∂{self._get_natural_for_conjugate(conjugate_var)}")
            print(f"     = {derivative}")
            print()
    
    def _get_natural_for_conjugate(self, conjugate_var: sp.Symbol) -> sp.Symbol:
        """Get the natural variable corresponding to a conjugate variable."""
        if conjugate_var == self.th_y_var:
            return self.th_x_var
        elif conjugate_var == self.mc_y_var:
            return self.mc_x_var
        elif conjugate_var == self.Sig_var:
            return self.Eps_var
        else:
            raise ValueError(f"Unknown conjugate variable: {conjugate_var}")
    
    def __repr__(self) -> str:
        """String representation of the state function."""
        natural_vars = [str(var) for var in self.get_natural_variables()]
        return f"GSMStateFn2(f({', '.join(natural_vars)}) = {self.fn_expr})"
