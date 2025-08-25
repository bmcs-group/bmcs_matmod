"""
Pure GSMSymbDef Implementation - Phase 1

This module provides a pure symbolic thermodynamic definition class that separates
symbolic expressions from numerical execution engines.

Key Features:
- Contains only symbolic thermodynamic definitions
- No embedded engines
- Supports model specialization
- Provides expression validation and rendering
- Includes codename mappings for user-friendly display
"""

import sympy as sp
import keyword
from typing import Dict, Tuple, Any, Optional, List, NamedTuple
from dataclasses import dataclass
from functools import cached_property


def is_valid_variable_name(name: str) -> bool:
    """Check if the given name is a valid Python variable name."""
    if not name.isidentifier():
        return False
    if keyword.iskeyword(name):
        return False
    return True


@dataclass
class ValidationResult:
    """Result of symbolic expression validation."""
    is_valid: bool
    issues: List[str]


class GSMSymbDef:
    """
    Pure symbolic thermodynamic definition for GSM framework.
    
    This class contains only symbolic expressions and variable definitions
    without any embedded numerical execution engines. This is the non-discretized
    part of the model formulation.
    """
    
    # Primary variables (always Helmholtz-based)
    eps_vars: Tuple[sp.Symbol, ...] = ()      # External strain variables (ε)
    sig_vars: Tuple[sp.Symbol, ...] = ()      # External stress variables (σ) - conjugate to eps_vars
    Eps_vars: Tuple[sp.Symbol, ...] = ()      # Internal strain variables (ℰ)  
    Sig_vars: Tuple[sp.Symbol, ...] = ()      # Thermodynamic force variables (𝒮) - conjugate to Eps_vars
    m_params: Tuple[sp.Symbol, ...] = ()      # Material parameters
    
    # Thermodynamic expressions
    F_expr: sp.Expr = sp.S.Zero               # Free energy (state potential)
    f_expr: sp.Expr = sp.S.Zero               # Dissipation potential (inequality constraint)
    phi_ext_expr: sp.Expr = sp.S.Zero         # External potential
    h_k: List[sp.Expr] = []                   # Equality constraints
    
    # Variable properties
    Sig_signs: Tuple[int, ...] = ()           # Signs for internal stress variables (𝒮)
    gamma_mech_sign: int = -1                 # Sign for mechanical dissipation (Helmholtz: -1, Gibbs: +1)
    
    # Display mappings
    eps_codenames: Dict[sp.Symbol, str] = {}   # Strain variable codenames (ε)
    sig_codenames: Dict[sp.Symbol, str] = {}   # Stress variable codenames (σ)
    Eps_codenames: Dict[sp.Symbol, str] = {}   # Internal strain codenames (ℰ)
    Sig_codenames: Dict[sp.Symbol, str] = {}   # Thermodynamic force codenames (𝒮)
    param_codenames: Dict[sp.Symbol, str] = {} # Parameter codenames

    def __init__(self):
        """Initialize GSMSymbDef instance."""
        # Initialize empty constraint list if not defined
        if not hasattr(self, 'h_k'):
            self.h_k = []
            
        # Validate the model on initialization
        validation = self.validate_symbolic_expressions()
        if not validation.is_valid:
            print(f"⚠️ Warning: Model validation issues in {self.__class__.__name__}:")
            for issue in validation.issues:
                print(f"  - {issue}")

    @classmethod
    def get_required_parameters(cls) -> List[str]:
        """
        Return list of required parameter codenames.
        
        Returns:
            List of parameter names (codenames) required for this model
        """
        param_names = []
        for param in cls.m_params:
            if param in cls.param_codenames:
                param_names.append(cls.param_codenames[param])
            elif is_valid_variable_name(param.name):
                param_names.append(param.name)
            else:
                param_names.append(str(param))
        return param_names
    
    @classmethod  
    def get_parameter_descriptions(cls) -> Dict[str, str]:
        """
        Return parameter descriptions for documentation.
        
        Returns:
            Dictionary mapping parameter names to descriptions
        """
        descriptions = {}
        for param in cls.m_params:
            param_name = cls.param_codenames.get(param, param.name)
            # Extract description from symbol assumptions or provide default
            if hasattr(param, '_description'):
                descriptions[param_name] = param._description
            else:
                # Generate basic description from symbol properties
                desc_parts = []
                if param.is_positive:
                    desc_parts.append("positive")
                if param.is_real:
                    desc_parts.append("real")
                if desc_parts:
                    descriptions[param_name] = f"Material parameter ({', '.join(desc_parts)})"
                else:
                    descriptions[param_name] = "Material parameter"
        return descriptions
    
    @classmethod
    def validate_symbolic_expressions(cls) -> ValidationResult:
        """
        Validate consistency of symbolic expressions.
        
        Returns:
            ValidationResult indicating if model is valid and any issues found
        """
        issues = []
        
        # Check if expressions are defined
        if cls.F_expr == sp.S.Zero:
            issues.append("F_expr (free energy) is not defined")
            
        # Get all symbols used in expressions
        F_symbols = cls.F_expr.free_symbols if cls.F_expr != sp.S.Zero else set()
        f_symbols = cls.f_expr.free_symbols if cls.f_expr != sp.S.Zero else set()
        phi_symbols = cls.phi_ext_expr.free_symbols if cls.phi_ext_expr != sp.S.Zero else set()
        
        # Collect constraint symbols
        h_symbols = set()
        for constraint in cls.h_k:
            h_symbols.update(constraint.free_symbols)
        
        # All defined symbols
        all_defined_symbols = set(cls.eps_vars + cls.sig_vars + cls.Eps_vars + cls.Sig_vars + cls.m_params)
        
        # Check for undefined symbols
        all_used_symbols = F_symbols | f_symbols | phi_symbols | h_symbols
        undefined_symbols = all_used_symbols - all_defined_symbols
        
        if undefined_symbols:
            issues.append(f"Undefined symbols: {undefined_symbols}")
            
        # Check codename completeness
        missing_codenames = []
        for symbol_group, codename_dict, group_name in [
            (cls.eps_vars, cls.eps_codenames, "strain variables"),
            (cls.sig_vars, cls.sig_codenames, "stress variables"),
            (cls.Eps_vars, cls.Eps_codenames, "internal variables"),
            (cls.Sig_vars, cls.Sig_codenames, "thermodynamic forces"),
            (cls.m_params, cls.param_codenames, "parameters")
        ]:
            for symbol in symbol_group:
                if symbol not in codename_dict and not is_valid_variable_name(symbol.name):
                    missing_codenames.append(f"{symbol} in {group_name}")
        
        if missing_codenames:
            issues.append(f"Missing codenames for: {', '.join(missing_codenames)}")
        
        # Check consistency between eps_vars and sig_vars
        if len(cls.eps_vars) != len(cls.sig_vars):
            issues.append(f"Mismatch: {len(cls.eps_vars)} external strain variables but {len(cls.sig_vars)} external stress variables")
            
        # Check sign definitions for internal variables (only if Sig_signs is defined)
        if hasattr(cls, 'Sig_signs') and cls.Sig_signs:
            if len(cls.Eps_vars) != len(cls.Sig_signs):
                issues.append(f"Mismatch: {len(cls.Eps_vars)} internal variables but {len(cls.Sig_signs)} signs")
        else:
            if cls.Eps_vars:  # Only warn if there are internal variables
                issues.append("Sig_signs not defined for internal variables")
            
        # Check consistency between Eps_vars and Sig_vars (only if both are defined)
        if cls.Eps_vars and cls.Sig_vars:
            if len(cls.Eps_vars) != len(cls.Sig_vars):
                issues.append(f"Mismatch: {len(cls.Eps_vars)} internal variables but {len(cls.Sig_vars)} thermodynamic forces")
        elif cls.Eps_vars and not cls.Sig_vars:
            issues.append(f"Mismatch: {len(cls.Eps_vars)} internal variables but 0 thermodynamic forces")
        
        return ValidationResult(is_valid=len(issues) == 0, issues=issues)

    def render_potentials(self) -> str:
        """
        Render thermodynamic potentials in LaTeX format.
        
        Returns:
            LaTeX string representation of the model potentials
        """
        latex_lines = []
        
        # Model header
        latex_lines.append(f"## {self.__class__.__name__} - Thermodynamic Potentials")
        latex_lines.append("")
        
        # Free energy (Helmholtz)
        latex_lines.append("### State Potential (Helmholtz Free Energy)")
        if self.F_expr != sp.S.Zero:
            latex_lines.append(f"$$F = {sp.latex(self.F_expr)}$$")
        else:
            latex_lines.append("$$F = 0$$ (not defined)")
        latex_lines.append("")
        
        # Dissipation potential
        latex_lines.append("### Dissipation Potential (Inequality Constraint)")
        if self.f_expr != sp.S.Zero:
            latex_lines.append(f"$$f = {sp.latex(self.f_expr)} \\leq 0$$")
        else:
            latex_lines.append("$$f = 0$$ (no dissipation)")
        latex_lines.append("")
        
        # External potential
        if self.phi_ext_expr != sp.S.Zero:
            latex_lines.append("### External Potential")
            latex_lines.append(f"$$\\phi_{{ext}} = {sp.latex(self.phi_ext_expr)}$$")
            latex_lines.append("")
        
        # Equality constraints
        if self.h_k:
            latex_lines.append("### Equality Constraints")
            for i, constraint in enumerate(self.h_k):
                latex_lines.append(f"$$h_{{{i}}} = {sp.latex(constraint)} = 0$$")
            latex_lines.append("")
        
        return "\n".join(latex_lines)
    
    def get_expression_summary(self) -> Dict[str, Any]:
        """
        Get summary of all symbolic expressions.
        
        Returns:
            Dictionary containing model summary information
        """
        return {
            "model_name": self.__class__.__name__,
            "external_variables": len(self.eps_vars),
            "external_stresses": len(self.sig_vars),
            "internal_variables": len(self.Eps_vars),
            "thermodynamic_forces": len(self.Sig_vars),
            "material_parameters": len(self.m_params),
            "has_free_energy": self.F_expr != sp.S.Zero,
            "has_dissipation": self.f_expr != sp.S.Zero,
            "has_external_potential": self.phi_ext_expr != sp.S.Zero,
            "equality_constraints": len(self.h_k),
            "free_energy_complexity": len(self.F_expr.free_symbols) if self.F_expr != sp.S.Zero else 0,
            "dissipation_complexity": len(self.f_expr.free_symbols) if self.f_expr != sp.S.Zero else 0
        }

    # Derived expressions as cached properties
    @cached_property
    def eps_a(self) -> sp.Matrix:
        """
        External strain variables as a vector (eps_a).
        
        Following the convention where _a index represents the expanded collection
        of all external control strain variables as a homogeneous vector structure.
        
        Returns:
            Vector (Matrix) of external strain variables
        """
        return sp.BlockMatrix([
            var if isinstance(var, sp.Matrix) and var.shape[1] == 1 else sp.Matrix([[var]]) 
            for var in self.eps_vars
        ]).as_explicit()

    @cached_property
    def sig_a(self) -> sp.Matrix:
        """
        External stress variables as a vector (sig_a).
        
        Following the convention where _a index represents the expanded collection
        of all external stress variables as a homogeneous vector structure.
        These are the conjugate variables to eps_a.
        
        Returns:
            Vector (Matrix) of external stress variables
        """
        return sp.BlockMatrix([
            var if isinstance(var, sp.Matrix) and var.shape[1] == 1 else sp.Matrix([[var]]) 
            for var in self.sig_vars
        ]).as_explicit()

    @cached_property
    def Eps_B(self) -> sp.Matrix:
        """
        Internal strain variables with global indexing for heterogeneous types (Eps_B).
        
        Following the gsm_engine.py convention where individual multi-dimensional internal 
        variables (scalar, vector, tensor) are organized in a BlockMatrix structure with 
        global indexing scheme. The _B suffix (capital) indicates the block-structured 
        representation with non-expanded heterogeneous variable types (e.g., plastic tensor, 
        isotropic hardening scalar, kinematic hardening tensor).
        
        Similar to gsm_engine.py's Eps property: BlockMatrix([Eps_i.T for Eps_i in self.Eps_vars]).T
        
        Returns:
            BlockMatrix of internal strain variables maintaining their original dimensions
        """
        if not self.Eps_vars:
            return sp.Matrix([])
        
        # Create list of internal variables transposed for BlockMatrix structure
        # Following gsm_engine.py pattern: [Eps_i.T for Eps_i in self.Eps_vars]
        Eps_list_transposed = [
            var.T if isinstance(var, sp.Matrix) 
            else sp.Matrix([[var]]) 
            for var in self.Eps_vars
        ]
        
        # Use BlockMatrix and transpose to match gsm_engine.py: BlockMatrix(...).T
        return sp.BlockMatrix(Eps_list_transposed).T

    @cached_property
    def Eps_b(self) -> sp.Matrix:
        """
        Internal strain variables as a flattened vector (Eps_b).
        
        Following the convention where _b index (lowercase) represents the expanded/flattened 
        collection of all internal strain variables as a homogeneous vector structure.
        The individual variables from Eps_B are converted from their semantically 
        distinguished form (Scalar, Vector, Tensor) to flat form with expanded indexing.
        
        This is derived from Eps_B using as_explicit() to flatten the block structure.
        
        Returns:
            Flattened vector (Matrix) of all internal strain variable components
        """
        return self.Eps_B.as_explicit()

    @cached_property
    def Sig_B(self) -> sp.Matrix:
        """
        Thermodynamic force variables with global indexing for heterogeneous types (Sig_B).
        
        Following the gsm_engine.py convention where individual multi-dimensional thermodynamic
        force variables (scalar, vector, tensor) are organized in a BlockMatrix structure with 
        global indexing scheme. The _B suffix (capital) indicates the block-structured 
        representation with non-expanded heterogeneous variable types.
        
        Returns:
            BlockMatrix of thermodynamic force variables maintaining their original dimensions
        """
        if not self.Sig_vars:
            return sp.Matrix([])
        
        # Create list of thermodynamic force variables transposed for BlockMatrix structure
        Sig_list_transposed = [
            var.T if isinstance(var, sp.Matrix) 
            else sp.Matrix([[var]]) 
            for var in self.Sig_vars
        ]
        
        # Use BlockMatrix and transpose to match gsm_engine.py pattern
        return sp.BlockMatrix(Sig_list_transposed).T

    @cached_property
    def Sig_b(self) -> sp.Matrix:
        """
        Thermodynamic force variables as a flattened vector (Sig_b).
        
        Following the convention where _b index (lowercase) represents the expanded/flattened 
        collection of all thermodynamic force variables as a homogeneous vector structure.
        The individual variables from Sig_B are converted from their semantically 
        distinguished form (Scalar, Vector, Tensor) to flat form with expanded indexing.
        
        This is derived from Sig_B using as_explicit() to flatten the block structure.
        
        Returns:
            Flattened vector (Matrix) of all thermodynamic force variable components
        """
        return self.Sig_B.as_explicit()

    @cached_property
    def dot_eps_a(self) -> sp.Matrix:
        """
        Rate of external strain variables (dot_eps_a).
        
        Creates time derivatives of all external strain variables following the
        gsm_engine.py pattern. Each variable gets a dot symbol prefix.
        
        Returns:
            Vector (Matrix) of external strain rate symbols
        """
        return sp.Matrix([sp.Symbol(f'\\dot{{{var.name}}}', real=True) for var in self.eps_a])

    @cached_property
    def dot_Eps_b(self) -> sp.Matrix:
        """
        Rate of internal strain variables (dot_Eps_b).
        
        Creates time derivatives of all internal strain variables in flattened form,
        following the gsm_engine.py pattern. Each variable gets a dot symbol prefix.
        
        Returns:
            Vector (Matrix) of internal strain rate symbols (flattened form)
        """
        return sp.Matrix([sp.Symbol(f'\\dot{{{var.name}}}', real=True) for var in self.Eps_b])

    @cached_property
    def sig_a_(self) -> sp.Matrix:
        """
        External stresses obtained as derivatives of state function w.r.t. external strains (expression).
        
        General formulation: σ = ∂F/∂ε (sign determined by flow variable configuration)
        Following the convention where _a index represents the expanded collection.
        The trailing underscore indicates this is a derived expression, not a symbol.
        
        Returns:
            Vector (Matrix) of stress expressions (σ) corresponding to eps_a
        """
        if self.F_expr == sp.S.Zero:
            return sp.Matrix([sp.S.Zero for _ in range(len(self.eps_a))])
        
        return self.F_expr.diff(self.eps_a)

    @cached_property
    def dF_dEps_b(self) -> sp.Matrix:
        """
        Derivatives of free energy with respect to internal strain variables.
        
        Direct differentiation w.r.t. flattened Eps_b to get vector of derivatives.
        
        Returns:
            Vector of derivatives ∂F/∂ℰ_b (expanded/flattened form)
        """
        if self.F_expr == sp.S.Zero or not self.Eps_vars:
            return sp.Matrix([])
        
        return sp.Matrix([sp.diff(self.F_expr, var) for var in self.Eps_b])

    @cached_property
    def Y_b(self) -> sp.Matrix:
        """
        Sign matrix for thermodynamic forces (expanded representation).
        
        Expands the block-wise signs in Sig_signs to match the flattened Eps_b structure.
        Creates a diagonal matrix with appropriate signs for each component.
        
        Returns:
            Diagonal matrix of signs for expanded internal variables
        """
        if not self.Eps_vars:
            return sp.Matrix([])
        
        # Expand signs to match flattened structure
        signs_expanded = []
        for i, var in enumerate(self.Eps_vars):
            sign = self.Sig_signs[i] # if i < len(cls.Sig_signs) else -1
            # Get the size of this variable when flattened
            if isinstance(var, sp.Matrix):
                var_size = var.shape[0] * var.shape[1]
            else:
                var_size = 1
            signs_expanded.extend([sign] * var_size)
        
        return sp.diag(*signs_expanded)

    @cached_property
    def Sig_b_(self) -> sp.Matrix:
        """
        Thermodynamic forces obtained from free energy derivatives with sign convention (expression).
        
        General formulation: 𝒮 = Y * ∂F/∂ℰ (sign convention configures state domain)
        Following the gsm_engine.py convention: Sig_ = Y_ * dF_dEps_.as_explicit()
        The trailing underscore indicates this is a derived expression, not a symbol.
        
        Returns:
            Vector (Matrix) of thermodynamic force expressions (𝒮) corresponding to Eps_b
        """
        if self.F_expr == sp.S.Zero or not self.Eps_vars:
            return sp.Matrix([])
        
        return self.Y_b * self.dF_dEps_b

    @cached_property
    def phi_(self) -> sp.Expr:
        """
        Combined dissipation and flow potential (phi_).
        
        Following the gsm_engine.py convention where phi_ combines the threshold 
        function (f_expr) and the flow potential extension (phi_ext_expr).
        This represents the complete dissipation criterion for the material model.
        
        In gsm_engine.py: phi_ = (f_expr + phi_ext_expr).subs(m_param_subs)
        Since m_param_subs is not used in practice, we omit the substitution.
        
        Returns:
            Combined dissipation potential expression
        """
        return self.f_expr + self.phi_ext_expr

    @cached_property
    def subs_sig_eps_a(self) -> Dict[sp.Symbol, sp.Expr]:
        """
        Constitutive relations for external variables: sig_a = sig_a_
        
        Creates substitution dictionary mapping external stress symbols to
        their expressions derived from free energy derivatives.
        
        Returns:
            Dictionary {sigma_i: ∂F/∂ε_i} for external constitutive relations
        """
        return dict(zip(self.sig_a, self.sig_a_))

    @cached_property  
    def subs_Sig_Eps_b(self) -> Dict[sp.Symbol, sp.Expr]:
        """
        Constitutive relations for internal variables: Sig_b = Sig_b_
        
        Creates substitution dictionary mapping thermodynamic force symbols to
        their expressions derived from free energy derivatives with sign convention.
        
        Returns:
            Dictionary {S_i: Y_b * ∂F/∂ℰ_i} for internal constitutive relations
        """
        return dict(zip(self.Sig_b, self.Sig_b_))

    @cached_property
    def gamma_mech_(self) -> sp.Expr:
        """
        Mechanical dissipation (gamma_mech_).
        
        Following the gsm_engine.py convention where mechanical dissipation is calculated as:
        gamma_mech = gamma_mech_sign * ((Y * Sig).T * dot_Eps)[0]
        
        This represents the power dissipated through internal variable evolution.
        The sign depends on the thermodynamic potential type:
        - Helmholtz (gamma_mech_sign = -1): negative dissipation
        - Gibbs (gamma_mech_sign = +1): positive dissipation
        
        Returns:
            Mechanical dissipation expression
        """
        if not self.Eps_vars:
            return sp.S.Zero
        
        # Calculate mechanical dissipation: gamma_mech_sign * ((Y_b * Sig_b).T * dot_Eps_b)[0]
        Y_Sig_product = self.Y_b * self.Sig_b
        dissipation_rate = (Y_Sig_product.T * self.dot_Eps_b)[0]
        
        return self.gamma_mech_sign * dissipation_rate

    @cached_property
    def lam_phi_f_(self) -> Tuple[List[sp.Symbol], sp.Expr, sp.Expr]:
        """
        Inequality constraint components: Lagrange multipliers, constraint term, and flow potential.
        
        Following gsm_engine.py pattern:
        lam_phi_f_ = (lam, lam_phi, f_)
        where lam is list of multipliers, lam_phi is constraint term, f_ is flow potential
        
        Returns:
            Tuple of (multiplier_list, constraint_term, flow_potential)
        """
        if self.phi_ == sp.S.Zero:
            return ([], sp.S.Zero, sp.S.Zero)
        
        lam_phi = sp.Symbol(r'\lambda_{\mathrm{\phi}}', real=True)
        return ([lam_phi], lam_phi * (-self.phi_), self.f_expr)

    @cached_property
    def H_Lam_(self) -> Tuple[List[sp.Symbol], sp.Matrix, sp.Expr]:
        """
        Equality constraint components: rate multipliers, increment multipliers, and constraint sum.
        
        Following gsm_engine.py pattern:
        H_Lam = (dot_Lam, delta_Lam, dot_lam_sum)
        where dot_Lam are rate multipliers, delta_Lam are incremental multipliers, 
        dot_lam_sum is the constraint term sum
        
        Returns:
            Tuple of (rate_multipliers, increment_multipliers, constraint_sum)
        """
        dot_Lam = [sp.Symbol(f'\\dot{{\\lambda}}_{{{k}}}', real=True) for k in range(len(self.h_k))]
        delta_Lam = sp.Matrix([sp.Symbol(f'\\Delta\\lambda_{{{k}}}', real=True) for k in range(len(self.h_k))])
        dot_lam_sum = sum(l * h for l, h in zip(dot_Lam, self.h_k)) if self.h_k else sp.S.Zero
        return (dot_Lam, delta_Lam, dot_lam_sum)

    @cached_property
    def L_(self) -> sp.Expr:
        """
        Lagrangian for the minimum principle of dissipation potential.
        
        Following gsm_engine.py convention:
        L_ = -gamma_mech - dot_Lam_H_ - lam_phi
        
        This represents the full Lagrangian combining:
        - Mechanical dissipation (negative for energy balance)
        - Equality constraint terms (from h_k constraints)
        - Inequality constraint terms (from flow potential phi_)
        
        Returns:
            Complete Lagrangian expression
        """
        # Get constraint components
        lam_phi_list, lam_phi_term, _ = self.lam_phi_f_
        dot_Lam_list, delta_Lam, dot_lam_sum = self.H_Lam_
        
        # Construct full Lagrangian: L_ = -gamma_mech - dot_Lam_H_ - lam_phi
        L_expr = -self.gamma_mech_ - dot_lam_sum - lam_phi_term
        
        return L_expr

    @cached_property
    def dL_dS_A_(self) -> sp.Matrix:
        """
        Continuous optimality criterion: derivative of Lagrangian w.r.t. generalized forces.
        
        Following gsm_engine.py convention where dL_dS_A_ represents the derivative of the
        Lagrangian with respect to the generalized forces after applying constitutive
        relations. This should be zero for the optimal state with maximum energy dissipation.
        
        The generalized forces vector S includes:
        - Internal thermodynamic forces (Sig_b)
        - Lagrange multipliers for equality constraints
        - Lagrange multipliers for inequality constraints
        
        Returns:
            Matrix of optimality conditions that should equal zero
        """
        if not self.Eps_vars:
            return sp.Matrix([])
        
        # Get constraint components
        lam_phi_list, lam_phi_term, f_constraint = self.lam_phi_f_
        dot_Lam_list, delta_Lam, dot_lam_sum = self.H_Lam_
        
        # Construct generalized forces vector S = [Sig_b, dot_Lam, lam_phi]
        S_components = [self.Sig_b]
        if dot_Lam_list:
            S_components.append(sp.Matrix(dot_Lam_list))
        if lam_phi_list:
            S_components.append(sp.Matrix(lam_phi_list))
        
        if len(S_components) == 1:
            S = S_components[0]
        else:
            S = sp.Matrix.vstack(*S_components)
        
        # Calculate derivative of Lagrangian with respect to generalized forces
        dL_dS = self.L_.diff(S)
        dL_dS_matrix = sp.Matrix(dL_dS) if not isinstance(dL_dS, sp.Matrix) else dL_dS
        
        # Apply constitutive relations (substitute Sig_b expressions)
        if self.subs_Sig_Eps_b and self.subs_sig_eps_a:
            # Combine both external and internal constitutive relations
            all_subs = {**self.subs_Sig_Eps_b, **self.subs_sig_eps_a}
            dL_dS_A = dL_dS_matrix.subs(all_subs)
        elif self.subs_Sig_Eps_b:
            dL_dS_A = dL_dS_matrix.subs(self.subs_Sig_Eps_b)
        else:
            dL_dS_A = dL_dS_matrix
        
        # Handle the special case for inequality constraint (flow potential)
        if lam_phi_list and self.f_expr != sp.S.Zero:
            # The last component should be the flow potential constraint f_
            dL_dS_A[-1] = self.f_expr
        
        return dL_dS_A

    @cached_property
    def subs_n1_(self) -> Dict[sp.Symbol, sp.Expr]:
        """
        Time discretization substitutions for state variables at time n+1.
        
        Following gsm_engine.py pattern from _get_Sig_f_R_dR_n1 method where
        the continuous time derivatives and state variables are replaced with
        discrete incremental forms for numerical time integration.
        
        Creates substitutions for:
        - Rate variables: dot_Eps, dot_eps, dot_Lam -> increments/delta_t
        - State variables: Eps, eps -> n-state + increments
        - Lagrange multipliers: lam -> delta_lam
        - Time differential: d_t -> delta_t
        
        Returns:
            Dictionary of substitutions for time discretization
        """
        from bmcs_matmod.gsm_lagrange.core.gsm_vars import Scalar
        
        # Time symbols
        delta_t = Scalar(r'\Delta t', codename='delta_t', real=True)
        
        # External variables - state at time n (previous time step)
        eps_n = sp.Matrix([
            Scalar(f'{var.name}_{{(n)}}', codename=f'{self._get_codename(var)}_n', real=True) 
            for var in self.eps_a
        ])
        
        # Internal variables - state at time n (previous time step)
        Eps_n = sp.Matrix([
            Scalar(f'{var.name}_{{(n)}}', codename=f'{self._get_codename(var)}_n', real=True) 
            for var in self.Eps_b
        ])
        delta_Eps = sp.Matrix([
            Scalar(f'\\Delta{{{var.name}}}', codename=f'delta_{self._get_codename(var)}', real=True) 
            for var in self.Eps_b
        ])
        
        # Updated state at time n+1
        eps_n1 = eps_n + self.delta_eps_a_
        Eps_n1 = Eps_n + delta_Eps
        
        # Rate of change (increment / time_step)
        dot_eps_n = self.delta_eps_a_ / delta_t
        dot_Eps_n = delta_Eps / delta_t
        
        # Build substitution dictionary
        subs_dict = {}
        
        # External strain substitutions - iterate over all components
        for i, (eps_var, dot_eps_var) in enumerate(zip(self.eps_a, self.dot_eps_a)):
            subs_dict[eps_var] = eps_n1[i]
            subs_dict[dot_eps_var] = dot_eps_n[i]
        
        # Internal strain substitutions
        for i, (eps_var, dot_eps_var) in enumerate(zip(self.Eps_b, self.dot_Eps_b)):
            subs_dict[eps_var] = Eps_n1[i]
            subs_dict[dot_eps_var] = dot_Eps_n[i]
        
        # Constraint multiplier substitutions
        lam_phi_list, lam_phi_term, _ = self.lam_phi_f_
        dot_Lam_list, delta_Lam, dot_lam_sum = self.H_Lam_
        
        # Equality constraint Lagrange multipliers
        for k, dot_lam in enumerate(dot_Lam_list):
            delta_lam_k = Scalar(f'\\Delta\\lambda_{{{k}}}', codename=f'delta_lam_{k}', real=True)
            subs_dict[dot_lam] = delta_lam_k
        
        # Inequality constraint Lagrange multipliers
        for lam in lam_phi_list:
            lam_codename = self._get_codename(lam)
            delta_lam_phi = Scalar(f'\\Delta{{{lam.name}}}', codename=f'delta_{lam_codename}', real=True)
            subs_dict[lam] = delta_lam_phi
        
        # Time differential substitution
        if hasattr(self, 'd_t'):
            subs_dict[self.d_t] = delta_t
        else:
            # Create d_t symbol if not present
            d_t = Scalar(r'\mathrm{d}t', codename='dt', real=True)
            subs_dict[d_t] = delta_t
        
        return subs_dict

    @cached_property
    def delta_A_c_(self) -> sp.Matrix:
        """
        Augmented vector of unknowns containing increments of internal variables and Lagrange multipliers.
        
        Following gsm_engine.py pattern from _get_Sig_f_R_dR_n1 method:
        delta_A = Matrix([delta_Eps, Matrix(delta_Lam), Matrix(delta_lam)])
        
        This vector contains all the unknown increments that need to be solved
        in the nonlinear system of equations at each time step.
        
        The _c suffix indicates the augmented dimension (internal + constraints).
        
        Returns:
            Matrix containing all unknown increments
        """
        from bmcs_matmod.gsm_lagrange.core.gsm_vars import Scalar
        
        # Internal variable increments
        delta_Eps = sp.Matrix([
            Scalar(f'\\Delta{{{var.name}}}', codename=f'delta_{self._get_codename(var)}', real=True) 
            for var in self.Eps_b
        ])
        
        # Constraint multiplier increments
        lam_phi_list, lam_phi_term, _ = self.lam_phi_f_
        dot_Lam_list, delta_Lam, dot_lam_sum = self.H_Lam_
        
        # Equality constraint multiplier increments
        delta_Lam_matrix = sp.Matrix([
            Scalar(f'\\Delta\\lambda_{{{k}}}', codename=f'delta_lam_{k}', real=True)
            for k in range(len(self.h_k))
        ])
        
        # Inequality constraint multiplier increments
        delta_lam_matrix = sp.Matrix([
            Scalar(f'\\Delta{{{lam.name}}}', codename=f'delta_{self._get_codename(lam)}', real=True)
            for lam in lam_phi_list
        ])
        
        # Assemble complete augmented vector
        delta_A_components = [delta_Eps]
        if len(delta_Lam_matrix) > 0:
            delta_A_components.append(delta_Lam_matrix)
        if len(delta_lam_matrix) > 0:
            delta_A_components.append(delta_lam_matrix)
        
        if len(delta_A_components) == 1:
            return delta_A_components[0]
        else:
            return sp.Matrix.vstack(*delta_A_components)

    @cached_property
    def S_c_(self) -> sp.Matrix:
        """
        Augmented vector of generalized forces conjugate to delta_A_c_.
        
        Following gsm_engine.py pattern from _get_Sig_f_R_dR_n1 method:
        S = Matrix([Sig, Matrix(dot_Lam), Matrix(lam)])
        
        This vector contains all the generalized forces that are conjugate to
        the unknown increments in delta_A_c_.
        
        The _c suffix indicates the augmented dimension (internal + constraints).
        
        Returns:
            Matrix containing all generalized forces
        """
        # Internal thermodynamic forces
        Sig = self.Sig_b
        
        # Constraint multiplier components
        lam_phi_list, lam_phi_term, _ = self.lam_phi_f_
        dot_Lam_list, delta_Lam, dot_lam_sum = self.H_Lam_
        
        # Assemble complete augmented vector
        S_components = [Sig]
        if len(dot_Lam_list) > 0:
            S_components.append(sp.Matrix(dot_Lam_list))
        if len(lam_phi_list) > 0:
            S_components.append(sp.Matrix(lam_phi_list))
        
        if len(S_components) == 1:
            return S_components[0]
        else:
            return sp.Matrix.vstack(*S_components)

    @cached_property
    def R_c_n1_(self) -> sp.Matrix:
        """
        Residuum vector evaluated at time n+1 with time discretization substitutions.
        
        Following gsm_engine.py pattern from _get_Sig_f_R_dR_n1 method where
        the residuum is constructed from the derivative of the Lagrangian with
        respect to generalized forces, then time discretization substitutions
        are applied.
        
        This represents the nonlinear system of equations that must be solved
        to find the state at time n+1.
        
        The _c suffix indicates the augmented dimension (internal + constraints).
        
        Returns:
            Matrix representing the residuum that should equal zero
        """
        # Start with the continuous optimality criterion
        dL_dS_A = self.dL_dS_A_
        
        # Apply time discretization substitutions
        R_n1 = dL_dS_A.subs(self.subs_n1_)
        
        # Special handling for inequality constraint (flow potential)
        lam_phi_list, lam_phi_term, f_constraint = self.lam_phi_f_
        if lam_phi_list and self.f_expr != sp.S.Zero:
            # The last component should be the flow potential constraint f_
            # evaluated at time n+1
            f_n1 = self.f_expr.subs(self.subs_n1_)
            R_n1[-1] = f_n1
        
        return R_n1

    @cached_property
    def dR_dA_cc_n1_(self) -> sp.Matrix:
        """
        Jacobian matrix of the residuum with respect to the augmented unknowns at time n+1.
        
        Following gsm_engine.py pattern from _get_Sig_f_R_dR_n1 method where
        the Jacobian is computed as R_n1.jacobian(delta_A) with special handling
        for derivatives and Dirac delta functions.
        
        This matrix is used in Newton-Raphson iterations to solve the nonlinear
        system of equations at each time step.
        
        The _cc suffix indicates the expanded Jacobian dimensions (augmented × augmented).
        
        Returns:
            Jacobian matrix dR/dA for Newton-Raphson method
        """
        # Calculate Jacobian of residuum with respect to augmented unknowns
        dR_dA_n1 = self.R_c_n1_.jacobian(self.delta_A_c_)
        
        # Replace derivatives and Dirac delta functions with zeros
        # Following gsm_engine.py pattern
        dR_dA_n1 = dR_dA_n1.replace(sp.Derivative, lambda *args: 0)
        dR_dA_n1 = dR_dA_n1.replace(sp.DiracDelta, lambda *args: 0)
        
        return dR_dA_n1

    @cached_property
    def f_n1_(self) -> sp.Expr:
        """
        Flow potential (threshold function) evaluated at time n+1.
        
        Following gsm_engine.py pattern from _get_Sig_f_R_dR_n1 method where
        the static threshold function is substituted with constitutive relations
        and then time discretization substitutions are applied.
        
        This is used to determine if the material point is in the elastic or
        inelastic state (f ≤ 0 for admissible states).
        
        Returns:
            Flow potential expression at time n+1
        """
        if self.f_expr == sp.S.Zero:
            return sp.S.Zero
        
        # Apply constitutive relations first
        f_Eps = self.f_expr
        if self.subs_Sig_Eps_b:
            f_Eps = f_Eps.subs(self.subs_Sig_Eps_b)
        
        # Then apply time discretization substitutions
        f_n1 = f_Eps.subs(self.subs_n1_)
        
        return f_n1

    @cached_property
    def Sig_b_n1_(self) -> sp.Matrix:
        """
        Internal thermodynamic forces evaluated at time n+1.
        
        Following gsm_engine.py pattern from _get_Sig_f_R_dR_n1 method where
        Sig_n1 = Sig_.subs(subs_n1) represents the internal thermodynamic forces
        (not external stresses) evaluated with time discretization substitutions.
        
        This provides the internal force response at the end of the time step.
        
        The _b suffix indicates the internal variable dimension.
        
        Returns:
            Internal thermodynamic forces vector at time n+1
        """
        # Apply time discretization substitutions to internal thermodynamic force expressions
        Sig_n1 = self.Sig_b_.subs(self.subs_n1_)
        
        return Sig_n1

    @cached_property
    def sig_a_n1_(self) -> sp.Matrix:
        """
        External stress vector evaluated at time n+1.
        
        Following gsm_engine.py pattern where external stresses are evaluated
        with time discretization substitutions applied to the stress expressions
        derived from free energy derivatives.
        
        This provides the external stress response at the end of the time step.
        
        The _a suffix indicates the external variable dimension.
        
        Returns:
            External stress vector at time n+1
        """
        # Apply time discretization substitutions to external stress expressions
        sig_n1 = self.sig_a_.subs(self.subs_n1_)
        
        return sig_n1

    @cached_property
    def delta_eps_a_(self) -> sp.Matrix:
        """
        External strain increments as a vector (delta_eps_a_).
        
        Following the convention where _a index represents the expanded collection
        of all external strain increment variables as a homogeneous vector structure.
        This is the external counterpart to delta_A_c_ for internal variables.
        
        The trailing underscore indicates this is a derived expression used for
        time discretization and differentiation.
        
        Returns:
            Vector (Matrix) of external strain increment symbols
        """
        from bmcs_matmod.gsm_lagrange.core.gsm_vars import Scalar
        
        return sp.Matrix([
            Scalar(f'\\Delta{{{var.name}}}', codename=f'delta_{self._get_codename(var)}', real=True) 
            for var in self.eps_a
        ])

    @cached_property
    def dsig_deps_aa_n1_(self) -> sp.Matrix:
        """
        Material matrix: Jacobian of external stresses w.r.t. external strain increments at time n+1.
        
        Following gsm_engine.py pattern for algorithmic stiffness matrix construction where
        the material matrix is computed as sig_n1.jacobian(delta_eps) with special handling
        for derivatives and Dirac delta functions.
        
        This matrix represents the tangent stiffness relating external stress increments
        to external strain increments and is used in the global stiffness matrix assembly.
        
        The _aa suffix indicates the external × external dimensions (strain increments → stress increments).
        
        Returns:
            Material matrix dsig/deps for algorithmic stiffness
        """
        # Calculate Jacobian of external stresses with respect to external strain increments
        dsig_deps_n1 = self.sig_a_n1_.jacobian(self.delta_eps_a_)
        
        # Replace derivatives and Dirac delta functions with zeros
        # Following gsm_engine.py pattern
        dsig_deps_n1 = dsig_deps_n1.replace(sp.Derivative, lambda *args: 0)
        dsig_deps_n1 = dsig_deps_n1.replace(sp.DiracDelta, lambda *args: 0)
        
        return dsig_deps_n1

    def _get_codename(self, symbol: sp.Symbol) -> str:
        """
        Get codename for a symbol, with fallback to symbol name.
        
        Args:
            symbol: SymPy symbol to get codename for
        
        Returns:
            Codename string suitable for code generation
        """
        # First check if it's in any of our codename dictionaries
        for codename_dict in [
            self.eps_codenames, self.sig_codenames, 
            self.Eps_codenames, self.Sig_codenames, 
            self.param_codenames
        ]:
            if symbol in codename_dict:
                return codename_dict[symbol]
        
        # Check if symbol has codename attribute
        if hasattr(symbol, 'codename') and symbol.codename:
            return symbol.codename
            
        # Fall back to symbol name if it's a valid identifier
        if is_valid_variable_name(symbol.name):
            return symbol.name
            
        # Last resort: create a safe codename from symbol name
        safe_name = ''.join(c if c.isalnum() else '_' for c in symbol.name)
        return safe_name

    def latex_potentials(self) -> str:
        """
        Returns a LaTeX-friendly string with thermodynamic potentials and constitutive relations.
        
        Following the pattern from gsm_def.py but adapted for pure symbolic GSMSymbDef.
        Displays free energy, constitutive relations, and placeholders for future extensions.
        
        Returns:
            LaTeX string representation of the model potentials and relations
        """
        latex_lines = []
        
        # Model header
        latex_lines.append(f"## {self.__class__.__name__}")
        
        # Free energy
        latex_lines.append("### Free energy")
        if self.F_expr != sp.S.Zero:
            latex_lines.append(f"$$F = {sp.latex(self.F_expr)}$$")
        else:
            latex_lines.append("$$F = 0$$ (not defined)")
        
        # External constitutive relations
        if self.subs_sig_eps_a:
            latex_lines.append("### External constitutive relations")
            latex_lines.append("$$" + sp.latex(self.subs_sig_eps_a) + "$$")
        
        # Internal constitutive relations  
        if self.subs_Sig_Eps_b:
            latex_lines.append("### Internal constitutive relations")
            latex_lines.append("$$" + sp.latex(self.subs_Sig_Eps_b) + "$$")
        
        # Dissipation potential
        if self.phi_ != sp.S.Zero:
            latex_lines.append("### Dissipation potential")
            latex_lines.append(f"$$\\phi = {sp.latex(self.phi_)}$$")
        
        # Mechanical dissipation
        if self.gamma_mech_ != sp.S.Zero:
            latex_lines.append("#### Mechanical dissipation")
            latex_lines.append(f"$$\\gamma_{{\\mathrm{{mech}}}} = {sp.latex(self.gamma_mech_)}$$")
        else:
            latex_lines.append("#### Mechanical dissipation")
            latex_lines.append("$$\\gamma_{\\mathrm{mech}} = 0$$ (no internal variables)")
        
        # Lagrangian
        if self.L_ != sp.S.Zero:
            latex_lines.append("#### Lagrangian")  
            latex_lines.append(f"$$\\mathcal{{L}} = {sp.latex(self.L_)}$$")
        else:
            latex_lines.append("#### Lagrangian")  
            latex_lines.append("$$\\mathcal{L} = 0$$ (no constraints or dissipation)")
        
        # Optimality criterion
        if len(self.dL_dS_A_) > 0:
            latex_lines.append("#### Optimality criterion")
            latex_lines.append(f"$$\\frac{{\\partial \\mathcal{{L}}}}{{\\partial \\mathcal{{S}}}} = {sp.latex(self.dL_dS_A_)} = 0$$")
        else:
            latex_lines.append("#### Optimality criterion")
            latex_lines.append("$$\\frac{\\partial \\mathcal{L}}{\\partial \\mathcal{S}} = 0$$ (no generalized forces)")
        
        # Time discretization substitutions
        if self.subs_n1_:
            latex_lines.append("#### Time discretization substitutions")
            # Display only a subset to avoid overwhelming output
            subs_items = list(self.subs_n1_.items())[:6]  # Show first 6 substitutions
            subs_display = dict(subs_items)
            if len(subs_items) < len(self.subs_n1_):
                latex_lines.append(f"$$\\text{{(showing first 6 of {len(self.subs_n1_)} substitutions)}}$$")
            latex_lines.append("$$" + sp.latex(subs_display) + "$$")
        
        # Bounds of inelastic process (if applicable)
        if hasattr(self, 'dot_Eps_bounds_expr') and self.dot_Eps_bounds_expr != sp.S.Zero:
            latex_lines.append("#### Bounds of inelastic process")
            latex_lines.append("$$" + sp.latex(self.dot_Eps_bounds_expr) + " \\leq 0$$")
        
        return "\n".join(latex_lines)

    def print_potentials(self) -> None:
        """
        Print thermodynamic potentials in a formatted display.
        
        Convenience method for interactive use, displaying LaTeX expressions
        using IPython display capabilities when available.
        """
        print('=' * 50)
        print(f'class {self.__class__.__name__}')
        print('=' * 50)
        
        try:
            from IPython.display import display, Math
            print('Free energy:')
            display(Math(f'F = {sp.latex(self.F_expr)}'))
            
            if self.subs_sig_eps_a:
                print('External constitutive relations:')
                display(Math(sp.latex(self.subs_sig_eps_a)))
                
            if self.subs_Sig_Eps_b:
                print('Internal constitutive relations:')
                display(Math(sp.latex(self.subs_Sig_Eps_b)))
                
            if self.phi_ != sp.S.Zero:
                print('Dissipation potential:')
                display(Math(f'\\phi = {sp.latex(self.phi_)}'))
                
            if self.gamma_mech_ != sp.S.Zero:
                print('Mechanical dissipation:')
                display(Math(f'\\gamma_{{\\mathrm{{mech}}}} = {sp.latex(self.gamma_mech_)}'))
                
            if self.L_ != sp.S.Zero:
                print('Lagrangian:')
                display(Math(f'\\mathcal{{L}} = {sp.latex(self.L_)}'))
                
            if len(self.dL_dS_A_) > 0:
                print('Optimality criterion:')
                display(Math(f'\\frac{{\\partial \\mathcal{{L}}}}{{\\partial \\mathcal{{S}}}} = {sp.latex(self.dL_dS_A_)} = 0'))
                
            if self.subs_n1_:
                print('Time discretization substitutions:')
                # Display only first few to avoid overwhelming output
                subs_items = list(self.subs_n1_.items())[:4]
                subs_display = dict(subs_items)
                display(Math(sp.latex(subs_display)))
                if len(self.subs_n1_) > 4:
                    print(f'... (showing 4 of {len(self.subs_n1_)} total substitutions)')
                
        except ImportError:
            # Fallback for non-IPython environments
            print(f'Free energy: F = {self.F_expr}')
            if self.subs_sig_eps_a:
                print(f'External relations: {self.subs_sig_eps_a}')
            if self.subs_Sig_Eps_b:
                print(f'Internal relations: {self.subs_Sig_Eps_b}')
            if self.phi_ != sp.S.Zero:
                print(f'Dissipation: phi = {self.phi_}')
            if self.gamma_mech_ != sp.S.Zero:
                print(f'Mechanical dissipation: gamma_mech = {self.gamma_mech_}')
            if self.L_ != sp.S.Zero:
                print(f'Lagrangian: L = {self.L_}')
            if len(self.dL_dS_A_) > 0:
                print(f'Optimality criterion: dL_dS_A = {self.dL_dS_A_} = 0')
            if self.subs_n1_:
                print(f'Time discretization: {len(self.subs_n1_)} substitutions')
