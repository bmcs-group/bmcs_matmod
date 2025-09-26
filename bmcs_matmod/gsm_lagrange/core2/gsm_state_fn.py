"""
GSM State Function - Lightweight Implementation

A simplified state function class that accepts a sympy expression and organizes
variables into natural (independent) and conjugate (derivative) categories
across thermal, mechanical, and internal variable domains.
"""

import sympy as sp
from typing import List, Dict, Tuple, Union
from enum import Enum
import signal
import warnings
from .gsm_state_fn_ifc import GSMStateFnIfc
from .gsm_vars import markdown_vars_table


class StateFunctionType(Enum):
    """Enumeration of the four fundamental thermodynamic state functions."""
    INTERNAL_ENERGY = "U"      # U(S, ε, Ɛ) 
    HELMHOLTZ = "F"            # F(T, ε, Ɛ)
    ENTHALPY = "H"             # H(S, σ, Ɛ)
    GIBBS = "G"                # G(T, σ, Ɛ)


class VarTableType(Enum):
    """Enumeration of variable table types for markdown generation."""
    NATURAL = "natural"
    CONJUGATE = "conjugate"
    ALL = "all"


# Class-level mapping of natural and conjugate variables for each state function
NATURAL_VARIABLES_MAPPING: Dict[StateFunctionType, Tuple[List[str], List[str]]] = {
    # State function: (natural variables, conjugate variables)
    StateFunctionType.INTERNAL_ENERGY: (['S', 'eps', 'Eps'], ['T', 'sig', 'Sig']),  # U(S,ε,Ɛ)
    StateFunctionType.HELMHOLTZ:       (['T', 'eps', 'Eps'], ['S', 'sig', 'Sig']),  # F(T,ε,Ɛ)  
    StateFunctionType.ENTHALPY:        (['S', 'sig', 'Eps'], ['T', 'eps', 'Sig']),  # H(S,σ,Ɛ)
    StateFunctionType.GIBBS:           (['T', 'sig', 'Eps'], ['S', 'eps', 'Sig']),  # G(T,σ,Ɛ)
}

# Transformation mapping for Legendre transformations
TRANSFORMATION_MAPPING: Dict[Tuple[StateFunctionType, StateFunctionType], Tuple[int, int]] = {
    # Direct (adjacent) transformations
    (StateFunctionType.INTERNAL_ENERGY, StateFunctionType.HELMHOLTZ):   (-1,  0),  # U → F: F = U - TS
    (StateFunctionType.HELMHOLTZ, StateFunctionType.INTERNAL_ENERGY):   ( 1,  0),  # F → U: U = F + TS
    (StateFunctionType.INTERNAL_ENERGY, StateFunctionType.ENTHALPY):    ( 0,  1),  # U → H: H = U + σε
    (StateFunctionType.ENTHALPY, StateFunctionType.INTERNAL_ENERGY):    ( 0, -1),  # H → U: U = H - σε
    (StateFunctionType.HELMHOLTZ, StateFunctionType.GIBBS):             ( 0, -1),  # F → G: G = F - εσ
    (StateFunctionType.GIBBS, StateFunctionType.HELMHOLTZ):             ( 0,  1),  # G → F: F = G + εσ
    (StateFunctionType.ENTHALPY, StateFunctionType.GIBBS):              (-1,  0),  # H → G: G = H - TS
    (StateFunctionType.GIBBS, StateFunctionType.ENTHALPY):              ( 1,  0),  # G → H: H = G + TS
    
    # Diagonal (two-step) transformations
    (StateFunctionType.INTERNAL_ENERGY, StateFunctionType.GIBBS):       (-1, -1),  # U → G: G = U - TS - εσ
    (StateFunctionType.GIBBS, StateFunctionType.INTERNAL_ENERGY):       ( 1,  1),  # G → U: U = G + TS + εσ
    (StateFunctionType.HELMHOLTZ, StateFunctionType.ENTHALPY):          ( 1, -1),  # F → H: H = F + TS - εσ
    (StateFunctionType.ENTHALPY, StateFunctionType.HELMHOLTZ):          (-1,  1),  # H → F: F = H - TS + εσ
}


class GSMStateFn(GSMStateFnIfc):
    """
    Intelligent thermodynamic state function representation.
    
    Accepts a sympy expression and organizes variables into:
    - Natural variables: th_x_var (thermal), mc_x_var (mechanical), Eps_var (internal)  
    - Conjugate variables: th_y_var (thermal), mc_y_var (mechanical), Sig_var (internal)
    
    The class is aware of its state function type (U, F, H, G) and knows the expected
    variable organization for each type. This provides intelligent behavior for:
    - Variable validation against expected patterns
    - Automatic variable labeling and organization
    - Context awareness for transformations
    """
    
    # Global simplification timeout - can be modified to control simplification behavior
    SIMPLIFICATION_TIMEOUT = 10.0  # seconds - timeout for automatic expression simplification
    
    def __init__(self,
                 fn_expr: sp.Expr,
                 th_x_var: sp.Symbol,  # Thermal natural (S or T)
                 th_y_var: sp.Symbol,  # Thermal conjugate (T or S)
                 mc_x_var: sp.Symbol,  # Mechanical natural (ε or σ)
                 mc_y_var: sp.Symbol,  # Mechanical conjugate (σ or ε)
                 Eps_var: Union[sp.Symbol, Tuple[sp.Symbol, ...]],   # Internal natural (Ɛ)
                 Sig_var: Union[sp.Symbol, Tuple[sp.Symbol, ...]],   # Internal conjugate (𝒮)
                 state_function_type: StateFunctionType):
        """
        Initialize state function with expression and variable organization.
        
        Args:
            fn_expr: Sympy expression for the state function
            th_x_var: Thermal natural variable (e.g., S for entropy, T for temperature)
            th_y_var: Thermal conjugate variable (e.g., T for temperature, S for entropy)
            mc_x_var: Mechanical natural variable (e.g., ε for strain, σ for stress)
            mc_y_var: Mechanical conjugate variable (e.g., σ for stress, ε for strain)
            Eps_var: Internal natural variable(s) (internal strain Ɛ)
            Sig_var: Internal conjugate variable(s) (internal stress 𝒮)
            state_function_type: Type of state function (U, F, H, G) - REQUIRED
        """
        self._fn_expr = fn_expr
        
        # Natural variables (independent)
        self._th_x_var = th_x_var      # Thermal natural
        self._mc_x_var = mc_x_var      # Mechanical natural  
        self._Eps_var = Eps_var        # Internal natural
        
        # Conjugate variables (derivatives)
        self._th_y_var = th_y_var      # Thermal conjugate
        self._mc_y_var = mc_y_var      # Mechanical conjugate
        self._Sig_var = Sig_var        # Internal conjugate
        
        # State function type and intelligence - REQUIRED
        self._state_function_type = state_function_type
        
        # Validate variable organization against expected pattern
        self._validate_variable_organization()
    
    @property
    def state_function_type(self) -> StateFunctionType:
        """The type of state function (U, F, H, G)."""
        return self._state_function_type
    
    @property
    def state_function_name(self) -> str:
        """The name of the state function."""
        return self._state_function_type.value
    
    @property
    def fn_expr(self) -> sp.Expr:
        """The mathematical expression of the state function."""
        return self._fn_expr
    
    @property
    def th_x_var(self) -> sp.Symbol:
        """Thermal natural variable (T or S)."""
        return self._th_x_var
        
    @property
    def th_y_var(self) -> sp.Symbol:
        """Thermal conjugate variable (S or T)."""
        return self._th_y_var
        
    @property
    def mc_x_var(self) -> sp.Symbol:
        """Mechanical natural variable (ε or σ)."""
        return self._mc_x_var
        
    @property
    def mc_y_var(self) -> sp.Symbol:
        """Mechanical conjugate variable (σ or ε)."""
        return self._mc_y_var
        
    @property
    def Eps_var(self) -> Union[sp.Symbol, Tuple[sp.Symbol, ...]]:
        """Internal natural variable(s) (Ɛ)."""
        return self._Eps_var
        
    @property
    def Sig_var(self) -> Union[sp.Symbol, Tuple[sp.Symbol, ...]]:
        """Internal conjugate variable(s) (𝒮)."""
        return self._Sig_var
    
    def _validate_variable_organization(self) -> None:
        """Validate that variable organization matches expected pattern for state function type."""
        expected_natural, expected_conjugate = NATURAL_VARIABLES_MAPPING[self._state_function_type]
        
        def _get_var_id(var):
            """Get variable identifier, preferring codename over string representation."""
            return getattr(var, 'codename', str(var))
        
        # Create mapping from variable type to actual variables using enhanced identification
        actual_vars = {
            'T': self._th_x_var if _get_var_id(self._th_x_var) == 'T' else self._th_y_var,
            'S': self._th_x_var if _get_var_id(self._th_x_var) == 'S' else self._th_y_var,
            'eps': self._mc_x_var if _get_var_id(self._mc_x_var) == 'eps' else self._mc_y_var,
            'sig': self._mc_x_var if _get_var_id(self._mc_x_var) == 'sig' else self._mc_y_var,
            'Eps': self._Eps_var,
            'Sig': self._Sig_var
        }
        
        # Validate thermal variable assignment
        if 'T' in expected_natural and actual_vars['T'] != self._th_x_var:
            print(f"Warning: Expected T as thermal natural variable for {self._state_function_type.value}")
        if 'S' in expected_natural and actual_vars['S'] != self._th_x_var:
            print(f"Warning: Expected S as thermal natural variable for {self._state_function_type.value}")
            
        # Validate mechanical variable assignment
        if 'eps' in expected_natural and actual_vars['eps'] != self._mc_x_var:
            print(f"Warning: Expected eps as mechanical natural variable for {self._state_function_type.value}")
        if 'sig' in expected_natural and actual_vars['sig'] != self._mc_x_var:
            print(f"Warning: Expected sig as mechanical natural variable for {self._state_function_type.value}")
    
    # ========================================================================
    # Internal Expression Simplification (Private)
    # ========================================================================
    
    @classmethod
    def _simplify_with_timeout(cls, expr: sp.Expr) -> sp.Expr:
        """
        Internal method to safely simplify expressions with timeout protection.
        
        Args:
            expr: SymPy expression to simplify
            
        Returns:
            Simplified expression, or original expression if simplification fails/times out
        """
        try:
            # Set up signal handler for timeout (Unix-like systems only)
            if hasattr(signal, 'SIGALRM'):
                def timeout_handler(signum, frame):
                    raise TimeoutError("Simplification timed out")
                    
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, cls.SIMPLIFICATION_TIMEOUT)
                
                try:
                    simplified_expr = sp.simplify(expr)
                    signal.alarm(0)  # Cancel the alarm
                    return simplified_expr
                except TimeoutError:
                    return expr  # Return original if timeout
                except Exception:
                    return expr  # Return original if simplification fails
                finally:
                    signal.alarm(0)  # Ensure alarm is cancelled
                    signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
            else:
                # Fallback for systems without signal support (e.g., Windows)
                try:
                    return sp.simplify(expr)
                except Exception:
                    return expr  # Return original if simplification fails
                    
        except Exception:
            return expr  # Return original if setup fails
    
    def get_expected_natural_variables(self) -> List[str]:
        """Get the expected natural variable names for this state function type."""
        return NATURAL_VARIABLES_MAPPING[self._state_function_type][0]
    
    def get_expected_conjugate_variables(self) -> List[str]:
        """Get the expected conjugate variable names for this state function type."""
        return NATURAL_VARIABLES_MAPPING[self._state_function_type][1]
    
    def is_thermally_intensive(self) -> bool:
        """Check if thermal natural variable is intensive (T) or extensive (S)."""
        expected_natural, _ = NATURAL_VARIABLES_MAPPING[self._state_function_type]
        return 'T' in expected_natural
    
    def is_mechanically_intensive(self) -> bool:
        """Check if mechanical natural variable is intensive (σ) or extensive (ε)."""
        expected_natural, _ = NATURAL_VARIABLES_MAPPING[self._state_function_type]
        return 'sig' in expected_natural
    
    def get_legendre_transformation_targets(self) -> List[StateFunctionType]:
        """Get possible Legendre transformation targets from this state function."""
        targets = []
        for (source, target), _ in TRANSFORMATION_MAPPING.items():
            if source == self._state_function_type:
                targets.append(target)
        return targets
    
    @classmethod
    def create_with_auto_type_detection(cls,
                                      fn_expr: sp.Expr,
                                      th_x_var: sp.Symbol,
                                      th_y_var: sp.Symbol,
                                      mc_x_var: sp.Symbol,
                                      mc_y_var: sp.Symbol,
                                      Eps_var: Union[sp.Symbol, Tuple[sp.Symbol, ...]],
                                      Sig_var: Union[sp.Symbol, Tuple[sp.Symbol, ...]]) -> 'GSMStateFn':
        """Create GSMStateFn with automatic state function type detection."""
        
        # Attempt to detect state function type based on variable patterns
        th_x_name = str(th_x_var).lower()
        mc_x_name = str(mc_x_var).lower()
        
        detected_type = None
        
        # Pattern matching for state function detection
        if 's' in th_x_name and ('eps' in mc_x_name or 'epsilon' in mc_x_name):
            detected_type = StateFunctionType.INTERNAL_ENERGY  # U(S, ε, Ɛ)
        elif 't' in th_x_name and ('eps' in mc_x_name or 'epsilon' in mc_x_name):
            detected_type = StateFunctionType.HELMHOLTZ       # F(T, ε, Ɛ)
        elif 's' in th_x_name and ('sig' in mc_x_name or 'sigma' in mc_x_name):
            detected_type = StateFunctionType.ENTHALPY        # H(S, σ, Ɛ)
        elif 't' in th_x_name and ('sig' in mc_x_name or 'sigma' in mc_x_name):
            detected_type = StateFunctionType.GIBBS           # G(T, σ, Ɛ)
        
        return cls(fn_expr, th_x_var, th_y_var, mc_x_var, mc_y_var, 
                  Eps_var, Sig_var, detected_type)
    
    def get_natural_variables(self) -> List[sp.Symbol]:
        """Get all natural (independent) variables."""
        natural_vars = [self.th_x_var, self.mc_x_var]
            
        # Handle internal variables (could be single or tuple)
        if hasattr(self.Eps_var, '__iter__'):
            natural_vars.extend(self.Eps_var)
        else:
            natural_vars.append(self.Eps_var)
            
        return natural_vars
    
    def get_conjugate_variables(self) -> List[sp.Symbol]:
        """Get all conjugate (derivative) variables."""
        conjugate_vars = [self.th_y_var, self.mc_y_var]
            
        # Handle internal variables (could be single or tuple)
        if hasattr(self.Sig_var, '__iter__'):
            conjugate_vars.extend(self.Sig_var)
        else:
            conjugate_vars.append(self.Sig_var)
            
        return conjugate_vars
    
    def get_all_variables(self) -> List[sp.Symbol]:
        """Get all variables (natural + conjugate)."""
        return self.get_natural_variables() + self.get_conjugate_variables()
    
    def compute_constitutive_relations(self) -> Dict[sp.Symbol, sp.Expr]:
        """
        Compute constitutive relations as partial derivatives with automatic simplification.
        
        Returns:
            Dictionary mapping conjugate variables to their simplified derivative expressions
        """
        relations = {}
        
        # Thermal: conjugate = ∂f/∂(natural)
        thermal_derivative = sp.diff(self.fn_expr, self.th_x_var)
        relations[self.th_y_var] = self._simplify_with_timeout(thermal_derivative)
        
        # Mechanical: conjugate = ∂f/∂(natural)
        mechanical_derivative = sp.diff(self.fn_expr, self.mc_x_var)
        relations[self.mc_y_var] = self._simplify_with_timeout(mechanical_derivative)
        
        # Internal: conjugate = ∂f/∂(natural)
        if hasattr(self.Eps_var, '__iter__'):
            for Eps, Sig in zip(self.Eps_var, self.Sig_var):
                internal_derivative = sp.diff(self.fn_expr, Eps)
                relations[Sig] = self._simplify_with_timeout(internal_derivative)
        else:
            internal_derivative = sp.diff(self.fn_expr, self.Eps_var)
            relations[self.Sig_var] = self._simplify_with_timeout(internal_derivative)
        
        return relations
    
    def get_thermal_constitutive_relation(self) -> Tuple[sp.Symbol, sp.Expr]:
        """Get the thermal constitutive relation with automatic simplification."""
        thermal_derivative = sp.diff(self.fn_expr, self.th_x_var)
        simplified_derivative = self._simplify_with_timeout(thermal_derivative)
        return (self.th_y_var, simplified_derivative)
    
    def get_mechanical_constitutive_relations(self) -> List[Tuple[sp.Symbol, sp.Expr]]:
        """Get mechanical constitutive relations with automatic simplification."""
        derivative = sp.diff(self.fn_expr, self.mc_x_var)
        simplified_derivative = self._simplify_with_timeout(derivative)
        return [(self.mc_y_var, simplified_derivative)]
    
    def get_internal_constitutive_relations(self) -> List[Tuple[sp.Symbol, sp.Expr]]:
        """Get internal constitutive relations with automatic simplification."""
        relations = []
        
        if hasattr(self.Eps_var, '__iter__'):
            for Eps, Sig in zip(self.Eps_var, self.Sig_var):
                derivative = sp.diff(self.fn_expr, Eps)
                simplified_derivative = self._simplify_with_timeout(derivative)
                relations.append((Sig, simplified_derivative))
        else:
            derivative = sp.diff(self.fn_expr, self.Eps_var)
            simplified_derivative = self._simplify_with_timeout(derivative)
            relations.append((self.Sig_var, simplified_derivative))
            
        return relations
    
    def markdown_overview(self) -> str:
        """
        Get a markdown-formatted overview of the state function and its variables.
        
        Returns:
            Markdown string containing complete overview suitable for display(Markdown(...)).
        """
        lines = []
        
        # Title and basic information
        lines.append("# GSM State Function Overview")
        lines.append("")
        lines.append(f"${self._state_function_type.value} = {sp.latex(self.fn_expr)}$")
        lines.append("")
        
        # Natural variables section
        lines.append("## Natural Variables (independent)")
        lines.append(self.markdown_natural_vars_table())
        lines.append("")
        
        # Conjugate variables section  
        lines.append("## Conjugate Variables (derivatives)")
        lines.append(self.markdown_conjugate_vars_table())
        lines.append("")
        
        # Constitutive relations  
        lines.append("## Constitutive relations")
        lines.append(self.markdown_constitutive_relations())
        lines.append("")
        
        # Expected variable organization
        expected_natural = self.get_expected_natural_variables()
        expected_conjugate = self.get_expected_conjugate_variables()
        lines.append("## Expected Variable Organization")
        lines.append(f"- **Natural:** {expected_natural}")
        lines.append(f"- **Conjugate:** {expected_conjugate}")
        lines.append(f"- **Thermally Intensive:** {self.is_thermally_intensive()}")
        lines.append(f"- **Mechanically Intensive:** {self.is_mechanically_intensive()}")
        
        # Transformation targets if available
        targets = self.get_legendre_transformation_targets()
        if targets:
            target_names = [t.value for t in targets]
            lines.append(f"- **Transformation Targets:** {target_names}")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def markdown_constitutive_relations(self, separate_blocks: bool = True) -> str:
        """
        Get constitutive relations in markdown with LaTeX formatting.

        Args:
            separate_blocks: If True, each relation is its own display math block (recommended).
                             If False, relations are returned as a bullet list with inline math.

        Returns:
            Markdown string containing LaTeX-formatted constitutive relations suitable
            for rendering in a Jupyter notebook via display(Markdown(...)).

        Format (separate_blocks=True):
            $$ S = \frac{\partial f}{\partial T} = ... $$
            $$ \sigma = \frac{\partial f}{\partial \varepsilon} = ... $$

        Format (separate_blocks=False):
            - $ S = \frac{\partial f}{\partial T} = ... $
            - $ \sigma = \frac{\partial f}{\partial \varepsilon} = ... $
        """
        relations = self.compute_constitutive_relations()
        lines: List[str] = []

        for conjugate_var, derivative in relations.items():
            natural_var = self._get_natural_for_conjugate(conjugate_var)
            conj_ltx = sp.latex(conjugate_var)
            nat_ltx = sp.latex(natural_var)
            deriv_ltx = sp.latex(derivative)
            relation_expr = f"{conj_ltx} = \\frac{{\\partial f}}{{\\partial {nat_ltx}}} = {deriv_ltx}"
            if separate_blocks:
                lines.append(f"$$ {relation_expr} $$")
            else:
                lines.append(f"- $ {relation_expr} $")

        return "\n".join(lines)
    
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
    
    def markdown_vars_table(self, table_type: VarTableType) -> str:
        """
        Generate markdown table for variables based on type.
        
        Args:
            table_type: Type of variables to include in table
            
        Returns:
            Markdown-formatted table string
            
        Usage:
            >>> helmholtz_fn.get_md_vars_table(VarTableType.NATURAL)
            >>> helmholtz_fn.get_md_vars_table(VarTableType.CONJUGATE)
            >>> helmholtz_fn.get_md_vars_table(VarTableType.ALL)
        """
        if table_type == VarTableType.NATURAL:
            return markdown_vars_table(self.get_natural_variables())
        elif table_type == VarTableType.CONJUGATE:
            return markdown_vars_table(self.get_conjugate_variables())
        elif table_type == VarTableType.ALL:
            return markdown_vars_table(self.get_all_variables())
        else:
            raise ValueError(f"Unknown table type: {table_type}")
    
    def markdown_natural_vars_table(self) -> str:
        """Convenience method to get markdown table for natural variables."""
        return self.markdown_vars_table(VarTableType.NATURAL)
    
    def markdown_conjugate_vars_table(self) -> str:
        """Convenience method to get markdown table for conjugate variables."""
        return self.markdown_vars_table(VarTableType.CONJUGATE)
    
    def markdown_all_vars_table(self) -> str:
        """Convenience method to get markdown table for all variables."""
        return self.markdown_vars_table(VarTableType.ALL)
    
    def __repr__(self) -> str:
        """String representation of the state function."""
        natural_vars = [str(var) for var in self.get_natural_variables()]
        type_info = self._state_function_type.value
        return f"GSMStateFn[{type_info}](f({', '.join(natural_vars)}) = {self.fn_expr})"


# ============================================================================
# Thermodynamic State Function Subclasses
# ============================================================================

class GSMHelmholtzFn(GSMStateFn):
    """
    Helmholtz Free Energy state function: F(T, ε, Ɛ)
    
    Natural variables: T (temperature), eps (strain), Eps (internal strain)
    Conjugate variables: S (entropy), sig (stress), Sig (internal stress)
    
    This subclass provides explicit variable attributes and establishes the
    correct mapping to the general framework variables.
    """
    
    def __init__(self,
                 fn_expr: sp.Expr,
                 T: sp.Symbol,         # Temperature (thermal natural)
                 S: sp.Symbol,         # Entropy (thermal conjugate) 
                 eps: sp.Symbol,       # Strain (mechanical natural)
                 sig: sp.Symbol,       # Stress (mechanical conjugate)
                 Eps: Union[sp.Symbol, Tuple[sp.Symbol, ...]] = None,  # Internal natural
                 Sig: Union[sp.Symbol, Tuple[sp.Symbol, ...]] = None): # Internal conjugate
        """
        Initialize Helmholtz free energy state function.
        
        Args:
            fn_expr: Sympy expression F(T, ε, Ɛ)
            T: Temperature symbol (natural thermal variable)
            S: Entropy symbol (conjugate thermal variable)
            eps: Strain symbol (natural mechanical variable) 
            sig: Stress symbol (conjugate mechanical variable)
            Eps: Internal strain symbol(s) (natural internal variables)
            Sig: Internal stress symbol(s) (conjugate internal variables)
        """
        # Store explicit variable attributes for type safety and clarity
        self.T = T
        self.S = S  
        self.eps = eps
        self.sig = sig
        self.Eps = Eps
        self.Sig = Sig
        
        # Initialize base class with correct mapping: F(T, ε, Ɛ)
        super().__init__(
            fn_expr=fn_expr,
            th_x_var=T,      # Temperature is natural
            th_y_var=S,      # Entropy is conjugate  
            mc_x_var=eps,    # Strain is natural
            mc_y_var=sig,    # Stress is conjugate
            Eps_var=Eps,     # Internal strain (natural)
            Sig_var=Sig,     # Internal stress (conjugate)
            state_function_type=StateFunctionType.HELMHOLTZ
        )


class GSMGibbsFn(GSMStateFn):
    """
    Gibbs Free Energy state function: G(T, σ, Ɛ)
    
    Natural variables: T (temperature), sig (stress), Eps (internal strain)
    Conjugate variables: S (entropy), eps (strain), Sig (internal stress)
    
    This subclass provides explicit variable attributes and establishes the
    correct mapping to the general framework variables.
    """
    
    def __init__(self,
                 fn_expr: sp.Expr,
                 T: sp.Symbol,         # Temperature (thermal natural)
                 S: sp.Symbol,         # Entropy (thermal conjugate)
                 sig: sp.Symbol,       # Stress (mechanical natural)
                 eps: sp.Symbol,       # Strain (mechanical conjugate)
                 Eps: Union[sp.Symbol, Tuple[sp.Symbol, ...]] = None,  # Internal natural
                 Sig: Union[sp.Symbol, Tuple[sp.Symbol, ...]] = None): # Internal conjugate
        """
        Initialize Gibbs free energy state function.
        
        Args:
            fn_expr: Sympy expression G(T, σ, Ɛ)
            T: Temperature symbol (natural thermal variable)
            S: Entropy symbol (conjugate thermal variable)
            sig: Stress symbol (natural mechanical variable)
            eps: Strain symbol (conjugate mechanical variable)
            Eps: Internal strain symbol(s) (natural internal variables)
            Sig: Internal stress symbol(s) (conjugate internal variables)
        """
        # Store explicit variable attributes for type safety and clarity
        self.T = T
        self.S = S
        self.sig = sig
        self.eps = eps
        self.Eps = Eps
        self.Sig = Sig
        
        # Initialize base class with correct mapping: G(T, σ, Ɛ)
        super().__init__(
            fn_expr=fn_expr,
            th_x_var=T,      # Temperature is natural
            th_y_var=S,      # Entropy is conjugate
            mc_x_var=sig,    # Stress is natural
            mc_y_var=eps,    # Strain is conjugate
            Eps_var=Eps,     # Internal strain (natural)
            Sig_var=Sig,     # Internal stress (conjugate)
            state_function_type=StateFunctionType.GIBBS
        )


class GSMEnthalpyFn(GSMStateFn):
    """
    Enthalpy state function: H(S, σ, Ɛ)
    
    Natural variables: S (entropy), sig (stress), Eps (internal strain)
    Conjugate variables: T (temperature), eps (strain), Sig (internal stress)
    
    This subclass provides explicit variable attributes and establishes the
    correct mapping to the general framework variables.
    """
    
    def __init__(self,
                 fn_expr: sp.Expr,
                 S: sp.Symbol,         # Entropy (thermal natural)
                 T: sp.Symbol,         # Temperature (thermal conjugate)
                 sig: sp.Symbol,       # Stress (mechanical natural)
                 eps: sp.Symbol,       # Strain (mechanical conjugate)
                 Eps: Union[sp.Symbol, Tuple[sp.Symbol, ...]] = None,  # Internal natural
                 Sig: Union[sp.Symbol, Tuple[sp.Symbol, ...]] = None): # Internal conjugate
        """
        Initialize enthalpy state function.
        
        Args:
            fn_expr: Sympy expression H(S, σ, Ɛ)
            S: Entropy symbol (natural thermal variable)
            T: Temperature symbol (conjugate thermal variable)
            sig: Stress symbol (natural mechanical variable)
            eps: Strain symbol (conjugate mechanical variable)
            Eps: Internal strain symbol(s) (natural internal variables)
            Sig: Internal stress symbol(s) (conjugate internal variables)
        """
        # Store explicit variable attributes for type safety and clarity
        self.S = S
        self.T = T
        self.sig = sig
        self.eps = eps
        self.Eps = Eps
        self.Sig = Sig
        
        # Initialize base class with correct mapping: H(S, σ, Ɛ)
        super().__init__(
            fn_expr=fn_expr,
            th_x_var=S,      # Entropy is natural
            th_y_var=T,      # Temperature is conjugate
            mc_x_var=sig,    # Stress is natural
            mc_y_var=eps,    # Strain is conjugate
            Eps_var=Eps,     # Internal strain (natural)
            Sig_var=Sig,     # Internal stress (conjugate)
            state_function_type=StateFunctionType.ENTHALPY
        )


class GSMInternalEnergyFn(GSMStateFn):
    """
    Internal Energy state function: U(S, ε, Ɛ)
    
    Natural variables: S (entropy), eps (strain), Eps (internal strain)
    Conjugate variables: T (temperature), sig (stress), Sig (internal stress)
    
    This subclass provides explicit variable attributes and establishes the
    correct mapping to the general framework variables.
    """
    
    def __init__(self,
                 fn_expr: sp.Expr,
                 S: sp.Symbol,         # Entropy (thermal natural)
                 T: sp.Symbol,         # Temperature (thermal conjugate)
                 eps: sp.Symbol,       # Strain (mechanical natural)
                 sig: sp.Symbol,       # Stress (mechanical conjugate)
                 Eps: Union[sp.Symbol, Tuple[sp.Symbol, ...]] = None,  # Internal natural
                 Sig: Union[sp.Symbol, Tuple[sp.Symbol, ...]] = None): # Internal conjugate
        """
        Initialize internal energy state function.
        
        Args:
            fn_expr: Sympy expression U(S, ε, Ɛ)
            S: Entropy symbol (natural thermal variable)
            T: Temperature symbol (conjugate thermal variable)
            eps: Strain symbol (natural mechanical variable)
            sig: Stress symbol (conjugate mechanical variable)
            Eps: Internal strain symbol(s) (natural internal variables)
            Sig: Internal stress symbol(s) (conjugate internal variables)
        """
        # Store explicit variable attributes for type safety and clarity
        self.S = S
        self.T = T
        self.eps = eps
        self.sig = sig
        self.Eps = Eps
        self.Sig = Sig
        
        # Initialize base class with correct mapping: U(S, ε, Ɛ)
        super().__init__(
            fn_expr=fn_expr,
            th_x_var=S,      # Entropy is natural
            th_y_var=T,      # Temperature is conjugate
            mc_x_var=eps,    # Strain is natural
            mc_y_var=sig,    # Stress is conjugate
            Eps_var=Eps,     # Internal strain (natural)
            Sig_var=Sig,     # Internal stress (conjugate)
            state_function_type=StateFunctionType.INTERNAL_ENERGY
        )
