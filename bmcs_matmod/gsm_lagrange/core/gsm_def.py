import traits.api as tr
import sympy as sp
import keyword
from typing import Dict, Tuple, Any, Optional, Type, Union, List, ClassVar

# Type aliases for SymPy objects
SymExpr = Union[sp.Expr, sp.Basic, sp.Symbol]
SymMatrix = sp.Matrix
SymDict = Dict[SymExpr, SymExpr]

# Skip init_printing for now to avoid potential hanging
# sp.init_printing()
from traits.api import \
    HasTraits, Property, cached_property, \
    Instance, Dict as TraitsDict, Str

from IPython.display import display, Math, Markdown

from .gsm_engine import GSMEngine

"""
Framework for GSM-based material models.

This file defines the GSMDef class, which outlines the symbolic attributes
(eps_vars, T_var, Eps_vars, Sig_vars, Sig_signs, F_expr, f_expr, phi_ext_expr,
g_expr_list) for thermodynamic potentials and constraints using Sympy. It also
provides a property for transforming the Helmholtz free energy (F_expr)
into the Gibbs free energy (G_expr) via a Legendre transform.
"""

def is_valid_variable_name(name):
    """Check if the given name is a valid Python variable name."""
    if not name.isidentifier():
        return False
    if keyword.iskeyword(name):
        return False
    return True

class GSMDef:
    """
    Base class for setting up thermodynamic models within the GSM framework.
    """
    
    # Add proper type annotations including SymPy types
    param_codenames: ClassVar[Dict[sp.Symbol, str]]
    eps_codenames: ClassVar[Dict[sp.Symbol, str]] 
    sig_codenames: ClassVar[Dict[sp.Symbol, str]]
    Eps_codenames: ClassVar[Dict[sp.Symbol, str]]
    Sig_codenames: ClassVar[Dict[sp.Symbol, str]]
    F_engine: ClassVar[GSMEngine]
    G_engine: ClassVar[GSMEngine]
    subs_eps_sig: ClassVar[SymDict]
    subs_dot_eps_sig: ClassVar[SymDict]
    eps_a_: ClassVar[SymMatrix]
    dot_eps_a_: ClassVar[SymMatrix]
    sig_x_eps_: ClassVar[SymExpr]
    _missing_symbol_reported: ClassVar[bool] = False

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        This method is called when a subclass of GSMDef is created.
        It initializes all class-level symbolic computations.
        """
        super().__init_subclass__(**kwargs)
        
        # Only proceed if this is a concrete subclass with a proper F_engine
        if hasattr(cls, 'F_engine') and cls.F_engine is not None:
            # Build codename mappings and collect missing codenames
            cls.param_codenames, param_missing = cls._build_symbol_codenames('m_params')
            cls.eps_codenames, eps_missing = cls._build_symbol_codenames('eps_vars')
            cls.sig_codenames, sig_missing = cls._build_symbol_codenames('sig_vars')
            cls.Eps_codenames, Eps_missing = cls._build_symbol_codenames('Eps_vars')
            cls.Sig_codenames, Sig_missing = cls._build_symbol_codenames('Sig_vars')

            missing = param_missing + eps_missing + sig_missing + Eps_missing + Sig_missing
            if hasattr(cls, '_missing_symbol_reported'):
                pass
            elif missing:
                msg = (
                    f"\n[WARNING] The following symbols in '{cls.__name__}' are not valid Python variable names "
                    f"and are missing a 'codename' attribute:\n"
                )
                for group, sym, name in missing:
                    msg += f"  - {group}: {repr(sym)} (name: '{name}')\n"
                msg += (
                    "Please add a 'codename' attribute to these symbols in your variable definitions, e.g.:\n\n"
                    "    sym.codename = 'your_python_name'\n"
                )
                print(msg)
                cls._missing_symbol_reported = True

            cls._calculate_symbolic_expressions()
            cls._initialize_gibbs_engine()

    @classmethod
    def _build_symbol_codenames(cls, attr_name: str) -> Tuple[Dict[sp.Symbol, str], List[Tuple[str, sp.Symbol, str]]]:
        """
        Build mapping from symbolic variable to codename.
        If the symbol's name is a valid Python identifier, use it.
        If not, use the .codename attribute if present.
        If neither, collect for reporting.
        Returns (symbol_codenames, missing_symbols)
        """
        symbol_codenames = {}
        missing_symbols = []
        F_engine = cls.F_engine
        symbols = getattr(F_engine, attr_name, ())
        for sym in symbols:
            if hasattr(sym, 'shape') and sym.shape == (1, 1):
                s = sym[0, 0]
            else:
                s = sym
            sym_name = s.name
            if is_valid_variable_name(sym_name):
                codename = sym_name
            elif hasattr(s, 'codename'):
                codename = s.codename
            else:
                codename = None
                missing_symbols.append((attr_name, s, sym_name))
            symbol_codenames[s] = codename
        return symbol_codenames, missing_symbols

    @classmethod
    def _calculate_symbolic_expressions(cls) -> None:
        """
        Calculate all symbolic expressions needed for the model at class level.
        """
        F_engine = cls.F_engine
        eps_a = F_engine.eps_a
        sig_a = F_engine.sig_a
        dF_deps = F_engine.F_expr.diff(eps_a)
        
        # Calculate eps_a_
        cls.eps_a_ = sp.Matrix([
            sp.solve(sp.Eq(sig_i, dF_deps_i), eps_i)[0]
            for sig_i, eps_i, dF_deps_i in zip(sig_a, eps_a, dF_deps)
        ])
        
        # Calculate subs_eps_sig
        cls.subs_eps_sig = dict(zip(F_engine.eps_a, cls.eps_a_))  # type: ignore[arg-type]
        
        # Calculate dot_eps_a_
        sigEps = sp.Matrix.vstack(F_engine.sig_a, F_engine.Eps.as_explicit())
        dot_sigEps = sp.Matrix.vstack(F_engine.dot_sig_a, F_engine.dot_Eps.as_explicit())
        cls.dot_eps_a_ = sp.simplify(cls.eps_a_.jacobian(sigEps) * dot_sigEps)
        
        # Calculate subs_dot_eps_sig
        cls.subs_dot_eps_sig = dict(zip(F_engine.dot_eps_a, cls.dot_eps_a_))  # type: ignore[arg-type]
        
        # Calculate sig_x_eps_
        cls.sig_x_eps_ = (F_engine.sig_a.T * F_engine.eps_a)[0]
    
    @classmethod
    def _initialize_gibbs_engine(cls) -> None:
        """
        Initialize the Gibbs free energy engine at class level.
        """
        F_engine = cls.F_engine
        G_expr = cls.sig_x_eps_ - F_engine.F_expr

        subs_eps_sig_ = {**cls.subs_eps_sig, **cls.subs_dot_eps_sig}
        G_ = sp.simplify(G_expr.subs(subs_eps_sig_))
        f_ = sp.simplify(F_engine.f_expr.subs(subs_eps_sig_))
        phi_ = sp.simplify(F_engine.phi_ext_expr.subs(subs_eps_sig_))
        h_k_ = [sp.simplify(h_.subs(subs_eps_sig_)) for h_ in F_engine.h_k]

        # Create a class-level G_engine (will be shared by all instances)
        cls.G_engine = GSMEngine(
            name=f'G_{F_engine.name}',
            eps_vars=F_engine.sig_vars,
            sig_vars=cls.eps_a_,
            T_var=F_engine.T_var,
            m_params=F_engine.m_params,
            Eps_vars=F_engine.Eps_vars,
            Sig_vars=F_engine.Sig_vars,
            gamma_mech_sign=(1),
            Sig_signs=F_engine.Sig_signs * (-1),
            F_expr=G_,
            f_expr=f_,
            phi_ext_expr=phi_,
            h_k=h_k_
        )

    name = Property(Str)
    @classmethod
    def _get_name(cls) -> str:
        return cls.__name__

    @classmethod
    def print_potentials(cls):
        print('=============================================')
        print(f'class {cls._get_name()}')
        print('=============================================')
        print(f'Helmholtz')
        display(Math(r'F =' + sp.latex(sp.simplify(cls.F_engine.F_expr))))
        display(cls.F_engine.subs_Sig_Eps)
        print(f'Gibbs')
        display(Math(r'G =' + sp.latex(sp.simplify(cls.G_engine.F_expr))))
        display(cls.G_engine.subs_Sig_Eps)
        (gamma_mech, L_, dL_dS_, dL_dS_A_, dR_dA_n1), (eps_n, delta_eps, Eps_n, delta_A, delta_t, Ox, Ix), Sig_n1, f_n1, R_n1, dR_dA_n1_OI = cls.G_engine.Sig_f_R_dR_n1
        print(f'Mechanical dissipation')
        display(Math(r'\gamma_{\mathrm{mech}} = ' + sp.latex(sp.simplify(gamma_mech))))
        print(f'Lagrangian')
        display(Math(r'L =' + sp.latex(L_)))
        print(f'Residuum')
        display(Math(r'\frac{\partial L}{\partial S} =' + sp.latex(dL_dS_) + ' = 0'))


    @classmethod
    def latex_potentials(cls):
        """
        Returns a KaTeX-friendly string with minimal LaTeX commands.
        """
        (gamma_mech, L_, dL_dS_, _, _), _, _, _, _, _ = cls.F_engine.Sig_f_R_dR_n1

        latex_lines = []
        latex_lines.append("## class " + cls._get_name())
        latex_lines.append("### Helmholtz free energy")
        latex_lines.append("$$F = " + sp.latex(sp.simplify(cls.F_engine.F_expr)) + "$$")
        latex_lines.append("$$" + sp.latex(cls.F_engine.subs_Sig_Eps) + "$$")
        latex_lines.append("#### Mechanical dissipation")
        latex_lines.append("$$\\gamma_{\\mathrm{mech}} = " + sp.latex(sp.simplify(gamma_mech)) + "$$")
        latex_lines.append("#### Lagrangian")
        latex_lines.append("$$\mathcal{L} = " + sp.latex(L_) + "$$")
        latex_lines.append("#### Residuum")
        latex_lines.append("$$\\frac{\\partial \mathcal{L}}{\\partial \mathcal{S}} = " + sp.latex(dL_dS_) + " = 0$$")
        if cls.F_engine.dot_Eps_bounds_expr is not sp.S.Zero:
            latex_lines.append("#### Bounds of inelastic process")
            latex_lines.append("$$" + sp.latex(cls.F_engine.dot_Eps_bounds_expr) + " \leq 0$$")

        (gamma_mech, L_, dL_dS_, _, _), _, _, _, _, _ = cls.G_engine.Sig_f_R_dR_n1
        latex_lines.append("### Legendre transform")

        latex_lines.append("#### Strain substitutions in dissipative terms")
        latex_lines.append("$$" + sp.latex(cls.subs_eps_sig) + "$$")
        latex_lines.append("$$" + sp.latex(cls.subs_dot_eps_sig) + "$$")

        latex_lines.append("### Gibbs free energy")
        latex_lines.append("$$G = " + sp.latex(cls.sig_x_eps_) + r"- \left[" + sp.latex(sp.simplify(cls.F_engine.F_expr) )+ r"\right] $$")
        latex_lines.append("#### Gibbs free energy after strain substitutions")
        latex_lines.append("$$G = " + sp.latex(sp.simplify(cls.G_engine.F_expr)) + "$$")
        latex_lines.append("$$" + sp.latex(cls.G_engine.subs_Sig_Eps) + "$$")

        latex_lines.append("#### Mechanical dissipation")
        latex_lines.append("$$\\gamma_{\\mathrm{mech}} = " + sp.latex(sp.simplify(gamma_mech)) + "$$")
        latex_lines.append("#### Lagrangian")
        latex_lines.append("$$\mathcal{L} = " + sp.latex(L_) + "$$")
        latex_lines.append("#### Residuum")
        latex_lines.append("$$\\frac{\\partial \mathcal{L}}{\\partial \mathcal{S}} = " + sp.latex(dL_dS_) + " = 0$$")
        if cls.F_engine.dot_Eps_bounds_expr is not sp.S.Zero:
            latex_lines.append("#### Bounds of inelastic process")
            latex_lines.append("$$" + sp.latex(cls.F_engine.dot_Eps_bounds_expr) + " \leq 0$$")

        return "\n".join(latex_lines)

    @classmethod
    def markdown(cls):
        """
        Returns a markdown string with minimal LaTeX commands.
        """
        return Markdown(cls.latex_potentials())

    @classmethod
    def get_args(cls, **kwargs: float) -> List[float]:
        """Convert keyword parameters to args."""
        # Ensure that all required parameters are provided
        missing_params = [codename for codename in cls.param_codenames.values() if codename not in kwargs]
        if missing_params:
            raise ValueError(f"Missing parameter values for: {missing_params}")
        
        # Extract arguments in the correct order
        args = [kwargs[cls.param_codenames[var]] for var in cls.F_engine.m_params]
        return args

    # Helmholtz free energy based solver - with consistent naming
    @classmethod
    def get_F_sig(cls, eps, *args):
        """Calculate the stress for the given strain."""
        return cls.F_engine.get_sig(eps, *args)

    @classmethod
    def get_F_response(cls, eps_ta, t_t, *args):
        """Calculate the stress and internal variables over time for the given strain history.
        
        Args:
            eps_ta: Array of strains over time
            t_t: Array of time points
            *args: Material parameters
        """
        return cls.F_engine.get_response(eps_ta, t_t, *args)

    @classmethod
    def get_F_Sig(cls, eps, *args):
        """Calculate the thermodynamic forces for the given strain."""
        return cls.F_engine.get_Sig(eps, *args)
    
    # For backward compatibility - these will be deprecated in future versions
    @classmethod
    def get_sig(cls, eps, Eps=None, *args):
        """Backward compatibility function, will be deprecated. Use get_F_sig instead."""
        return cls.get_F_sig(eps, *args)
    
    @classmethod
    def get_response(cls, eps_ta, t_t, *args):
        """Backward compatibility function, will be deprecated. Use get_F_response instead."""
        return cls.get_F_response(eps_ta, t_t, *args)
    
    @classmethod
    def get_Sig(cls, eps, Eps=None, *args):
        """Backward compatibility function, will be deprecated. Use get_F_Sig instead."""
        return cls.get_F_Sig(eps, *args)

    ### Gibbs free energy based solver
    
    @classmethod
    def get_G_eps(cls, sig, *args):
        """Calculate the strain for the given stress."""
        return cls.G_engine.get_sig(sig, *args)

    @classmethod
    def get_G_response(cls, sig_ta, t_t, *args):
        """Calculate the strain and internal variables over time for the given stress history.
        Args:
            sig_ta: Array of stresses over time
            t_t: Array of time points
            *args: Material parameters
        """
        resp = cls.G_engine.get_response(sig_ta, t_t, *args)
        # Rearrange the response to match the expected format
        return (resp[0], resp[2], resp[1], resp[3], resp[4], resp[5], resp[6], resp[7])

    @classmethod
    def get_G_Sig(cls, sig, *args):
        """Calculate the thermodynamic forces for the given stress."""
        return cls.G_engine.get_Sig(sig, *args)