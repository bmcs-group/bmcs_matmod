"""
GSM State Function Interface

This module defines the interface that state function implementations must satisfy
to be used in the GSMThermoDynBox framework.
"""

from abc import ABC, abstractmethod
import sympy as sp
from typing import List, Dict, Tuple, Union


class GSMStateFnIfc(ABC):
    """
    Interface for thermodynamic state functions in the GSM framework.
    
    This interface defines the contract that all state function implementations
    must satisfy to be used as nodes/edges in the GSMThermoDynBox.
    
    The interface organizes thermodynamic variables into three domains:
    - Thermal: temperature (T) ↔ entropy (S) conjugate pair
    - Mechanical: strain (ε) ↔ stress (σ) conjugate pair  
    - Internal: internal strain (Ɛ) ↔ internal stress (𝒮) conjugate pair
    
    Each state function has specific "natural variables" (independent variables
    it depends on) and corresponding "conjugate variables" (obtained as derivatives).
    """
    
    @property
    @abstractmethod
    def fn_expr(self) -> sp.Expr:
        """The mathematical expression of the state function."""
        pass
    
    @property
    @abstractmethod
    def th_x_var(self) -> sp.Symbol:
        """Thermal natural variable (T or S)."""
        pass
        
    @property
    @abstractmethod
    def th_y_var(self) -> sp.Symbol:
        """Thermal conjugate variable (S or T)."""
        pass
        
    @property
    @abstractmethod
    def mc_x_var(self) -> sp.Symbol:
        """Mechanical natural variable (ε or σ)."""
        pass
        
    @property
    @abstractmethod
    def mc_y_var(self) -> sp.Symbol:
        """Mechanical conjugate variable (σ or ε)."""
        pass
        
    @property
    @abstractmethod
    def Eps_var(self) -> Union[sp.Symbol, Tuple[sp.Symbol, ...]]:
        """Internal natural variable(s) (Ɛ)."""
        pass
        
    @property
    @abstractmethod
    def Sig_var(self) -> Union[sp.Symbol, Tuple[sp.Symbol, ...]]:
        """Internal conjugate variable(s) (𝒮)."""
        pass
    
    @abstractmethod
    def get_natural_variables(self) -> List[sp.Symbol]:
        """
        Get all natural (independent) variables.
        
        Returns:
            List of all independent variables the state function depends on
        """
        pass
    
    @abstractmethod
    def get_conjugate_variables(self) -> List[sp.Symbol]:
        """
        Get all conjugate (derivative) variables.
        
        Returns:
            List of all variables obtained as derivatives of the state function
        """
        pass
    
    @abstractmethod
    def get_all_variables(self) -> List[sp.Symbol]:
        """
        Get all variables (natural + conjugate).
        
        Returns:
            List of all variables involved in this state function
        """
        pass
    
    @abstractmethod
    def compute_constitutive_relations(self) -> Dict[sp.Symbol, sp.Expr]:
        """
        Compute constitutive relations as partial derivatives.
        
        Computes:
        - Thermal conjugate = ∂f/∂(thermal_natural)
        - Mechanical conjugate = ∂f/∂(mechanical_natural)
        - Internal conjugate = ∂f/∂(internal_natural)
        
        Returns:
            Dictionary mapping conjugate variables to their derivative expressions
        """
        pass
    
    @abstractmethod
    def get_thermal_constitutive_relation(self) -> Tuple[sp.Symbol, sp.Expr]:
        """
        Get the thermal constitutive relation.
        
        Returns:
            Tuple of (conjugate_variable, derivative_expression)
        """
        pass
    
    @abstractmethod
    def get_mechanical_constitutive_relations(self) -> List[Tuple[sp.Symbol, sp.Expr]]:
        """
        Get mechanical constitutive relations.
        
        Returns:
            List of tuples (conjugate_variable, derivative_expression) for each mechanical DOF
        """
        pass
    
    @abstractmethod
    def get_internal_constitutive_relations(self) -> List[Tuple[sp.Symbol, sp.Expr]]:
        """
        Get internal constitutive relations.
        
        Returns:
            List of tuples (conjugate_variable, derivative_expression) for each internal DOF
        """
        pass
    
    def get_organized_constitutive_relations(self) -> Dict[str, List[Tuple[sp.Symbol, sp.Expr]]]:
        """
        Get constitutive relations organized by category for display purposes.
        
        Returns:
            Dictionary with keys 'thermal', 'mechanical', 'internal', each containing
            list of (conjugate_variable, derivative_expression) tuples
        """
        organized = {
            'thermal': [self.get_thermal_constitutive_relation()],
            'mechanical': self.get_mechanical_constitutive_relations(),
            'internal': self.get_internal_constitutive_relations()
        }
        return organized
    
    def print_overview(self) -> None:
        """Print a summary of the state function and its variables."""
        print("GSM State Function Overview")
        print("=" * 30)
        print(f"Expression: {self.fn_expr}")
        print()
        
        print("Natural Variables (independent):")
        natural_vars = self.get_natural_variables()
        for var in natural_vars:
            print(f"  {var}")
        print()
        
        print("Conjugate Variables (derivatives):")
        conjugate_vars = self.get_conjugate_variables()
        for var in conjugate_vars:
            print(f"  {var}")
        print()
    
    def print_constitutive_relations(self) -> None:
        """Print the constitutive relations (derivatives)."""
        relations = self.compute_constitutive_relations()
        
        print("Constitutive Relations:")
        print("=" * 25)
        for conjugate_var, derivative in relations.items():
            # Find the corresponding natural variable
            natural_var = self._get_natural_for_conjugate(conjugate_var)
            print(f"{conjugate_var} = ∂f/∂{natural_var}")
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
            # Handle case where internal variables might be tuples/lists
            if hasattr(self.Sig_var, '__iter__') and conjugate_var in self.Sig_var:
                idx = list(self.Sig_var).index(conjugate_var)
                return list(self.Eps_var)[idx]
            raise ValueError(f"Unknown conjugate variable: {conjugate_var}")
    
    def __repr__(self) -> str:
        """String representation of the state function."""
        natural_vars = [str(var) for var in self.get_natural_variables()]
        return f"GSMStateFn(f({', '.join(natural_vars)}) = {self.fn_expr})"
