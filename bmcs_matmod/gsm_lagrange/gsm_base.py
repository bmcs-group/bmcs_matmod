import traits.api as tr
import sympy as sp
sp.init_printing()
from traits.api import \
    HasTraits, Property, cached_property, \
    Instance, Dict, Str

from IPython.display import display, Math, Markdown

from .gsm_mpdp import GSMMPDP

"""
Framework for GSM-based material models.

This file defines the GSMBase class, which outlines the symbolic attributes
(eps_vars, T_var, Eps_vars, Sig_vars, Sig_signs, F_expr, f_expr, phi_ext_expr,
g_expr_list) for thermodynamic potentials and constraints using Sympy. It also
provides a property for transforming the Helmholtz free energy (F_expr)
into the Gibbs free energy (G_expr) via a Legendre transform.
"""

import sympy as sp
import keyword

def is_valid_variable_name(name):
    """Check if the given name is a valid Python variable name."""
    if not name.isidentifier():
        return False
    if keyword.iskeyword(name):
        return False
    return True

class GSMBase(HasTraits):
    """
    Base class for setting up thermodynamic models within the GSM framework.
    """

    # Class-level initialization that happens during class definition
    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        This method is called when a subclass of GSMBase is created.
        It initializes all class-level symbolic computations.
        """
        super().__init_subclass__(**kwargs)
        
        # Only proceed if this is a concrete subclass with a proper F_engine
        if hasattr(cls, 'F_engine') and cls.F_engine is not None:
            # Build parameter codenames
            cls.param_codenames = cls._build_param_codenames()
            
            # Calculate symbolic expressions
            cls._calculate_symbolic_expressions()
            
            # Initialize the Gibbs engine
            cls._initialize_gibbs_engine()
    
    @classmethod
    def _build_param_codenames(cls):
        """
        Build mapping from symbolic parameter names to valid Python variable names.
        """
        param_codenames = {}
        # Get the F_engine class variable directly
        F_engine = cls.F_engine
        
        for sym in F_engine.m_params:
            sym_name = sym.name
            if is_valid_variable_name(sym_name):
                codename = sym_name
            else:
                # Check if a codename is provided in m_param_codenames
                m_param_codenames = getattr(cls, 'm_param_codenames', {})
                if sym in m_param_codenames:
                    codename = m_param_codenames[sym]
                else:
                    raise ValueError(
                        f"Symbol '{sym}' has an invalid name '{sym_name}' "
                        f"and no codename was provided in 'm_param_codenames'."
                    )
            param_codenames[sym] = codename
        return param_codenames
    
    @classmethod
    def _calculate_symbolic_expressions(cls):
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
        cls.subs_eps_sig = dict(zip(F_engine.eps_a, cls.eps_a_))
        
        # Calculate dot_eps_a_
        sigEps = sp.Matrix.vstack(F_engine.sig_a, F_engine.Eps.as_explicit())
        dot_sigEps = sp.Matrix.vstack(F_engine.dot_sig_a, F_engine.dot_Eps.as_explicit())
        cls.dot_eps_a_ = sp.simplify(cls.eps_a_.jacobian(sigEps) * dot_sigEps)
        
        # Calculate subs_dot_eps_sig
        cls.subs_dot_eps_sig = dict(zip(F_engine.dot_eps_a, cls.dot_eps_a_))
        
        # Calculate sig_x_eps_
        cls.sig_x_eps_ = (F_engine.sig_a.T * F_engine.eps_a)[0]
    
    @classmethod
    def _initialize_gibbs_engine(cls):
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
        cls.G_engine = GSMMPDP(
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
    def _get_name(self):
        return self.__class__.__name__

    def print_potentials(self):
        print('=============================================')
        print(f'class {self.name}')
        print('=============================================')
        print(f'Helmholtz')
        display(Math(r'F =' + sp.latex(sp.simplify(self.F_engine.F_expr))))
        display(self.F_engine.subs_Sig_Eps)
        print(f'Gibbs')
        display(Math(r'G =' + sp.latex(sp.simplify(self.G_engine.F_expr))))
        display(self.G_engine.subs_Sig_Eps)
        (gamma_mech, L_, dL_dS_, dL_dS_A_, dR_dA_n1), (eps_n, delta_eps, Eps_n, delta_A, delta_t, Ox, Ix), Sig_n1, f_n1, R_n1, dR_dA_n1_OI = self.G_engine.Sig_f_R_dR_n1
        print(f'Mechanical dissipation')
        display(Math(r'\gamma_{\mathrm{mech}} = ' + sp.latex(sp.simplify(gamma_mech))))
        print(f'Lagrangian')
        display(Math(r'L =' + sp.latex(L_)))
        print(f'Residuum')
        display(Math(r'\frac{\partial L}{\partial S} =' + sp.latex(dL_dS_) + ' = 0'))


    def latex_potentials(self):
        """
        Returns a KaTeX-friendly string with minimal LaTeX commands.
        """
        (gamma_mech, L_, dL_dS_, _, _), _, _, _, _, _ = self.F_engine.Sig_f_R_dR_n1

        latex_lines = []
        latex_lines.append("## class " + self.name)
        latex_lines.append("### Helmholtz free energy")
        latex_lines.append("$$F = " + sp.latex(sp.simplify(self.F_engine.F_expr)) + "$$")
        latex_lines.append("$$" + sp.latex(self.F_engine.subs_Sig_Eps) + "$$")
        latex_lines.append("#### Mechanical dissipation")
        latex_lines.append("$$\\gamma_{\\mathrm{mech}} = " + sp.latex(sp.simplify(gamma_mech)) + "$$")
        latex_lines.append("#### Lagrangian")
        latex_lines.append("$$\mathcal{L} = " + sp.latex(L_) + "$$")
        latex_lines.append("#### Residuum")
        latex_lines.append("$$\\frac{\\partial \mathcal{L}}{\\partial \mathcal{S}} = " + sp.latex(dL_dS_) + " = 0$$")
        if self.F_engine.dot_Eps_bounds_expr is not sp.S.Zero:
            latex_lines.append("#### Bounds of inelastic process")
            latex_lines.append("$$" + sp.latex(self.F_engine.dot_Eps_bounds_expr) + " \leq 0$$")

        (gamma_mech, L_, dL_dS_, _, _), _, _, _, _, _ = self.G_engine.Sig_f_R_dR_n1
        latex_lines.append("### Legendre transform")

        latex_lines.append("#### Strain substitutions in dissipative terms")
        latex_lines.append("$$" + sp.latex(self.subs_eps_sig) + "$$")
        latex_lines.append("$$" + sp.latex(self.subs_dot_eps_sig) + "$$")

        latex_lines.append("### Gibbs free energy")
        latex_lines.append("$$G = " + sp.latex(self.sig_x_eps_) + r"- \left[" + sp.latex(sp.simplify(self.F_engine.F_expr) )+ r"\right] $$")
        latex_lines.append("#### Gibbs free energy after strain substitutions")
        latex_lines.append("$$G = " + sp.latex(sp.simplify(self.G_engine.F_expr)) + "$$")
        latex_lines.append("$$" + sp.latex(self.G_engine.subs_Sig_Eps) + "$$")

        latex_lines.append("#### Mechanical dissipation")
        latex_lines.append("$$\\gamma_{\\mathrm{mech}} = " + sp.latex(sp.simplify(gamma_mech)) + "$$")
        latex_lines.append("#### Lagrangian")
        latex_lines.append("$$\mathcal{L} = " + sp.latex(L_) + "$$")
        latex_lines.append("#### Residuum")
        latex_lines.append("$$\\frac{\\partial \mathcal{L}}{\\partial \mathcal{S}} = " + sp.latex(dL_dS_) + " = 0$$")
        if self.F_engine.dot_Eps_bounds_expr is not sp.S.Zero:
            latex_lines.append("#### Bounds of inelastic process")
            latex_lines.append("$$" + sp.latex(self.F_engine.dot_Eps_bounds_expr) + " \leq 0$$")

        return "\n".join(latex_lines)

    def markdown(self):
        """
        Returns a markdown string with minimal LaTeX commands.
        """
        return Markdown(self.latex_potentials())

    def get_args(self, **kwargs):
        """Convert keyword parameters to args."""
        # Ensure that all required parameters are provided
        missing_params = [codename for codename in self.param_codenames.values() if codename not in kwargs]
        if missing_params:
            raise ValueError(f"Missing parameter values for: {missing_params}")
        
        # Extract arguments in the correct order
        args = [kwargs[self.param_codenames[var]] for var in self.F_engine.m_params]
        return args

    # Helmholtz free energy based solver - with consistent naming
    def get_F_sig(self, eps, *args):
        """Calculate the stress for the given strain."""
        return self.F_engine.get_sig(eps, *args)

    def get_F_response(self, eps_ta, t_t, *args):
        """Calculate the stress and internal variables over time for the given strain history.
        
        Args:
            eps_ta: Array of strains over time
            t_t: Array of time points
            *args: Material parameters
        """
        return self.F_engine.get_response(eps_ta, t_t, *args)

    def get_F_Sig(self, eps, *args):
        """Calculate the thermodynamic forces for the given strain."""
        return self.F_engine.get_Sig(eps, *args)
    
    # For backward compatibility - these will be deprecated in future versions
    def get_sig(self, eps, Eps=None, *args):
        """Backward compatibility function, will be deprecated. Use get_F_sig instead."""
        return self.get_F_sig(eps, *args)
    
    def get_response(self, eps_ta, t_t, *args):
        """Backward compatibility function, will be deprecated. Use get_F_response instead."""
        return self.get_F_response(eps_ta, t_t, *args)
    
    def get_Sig(self, eps, Eps=None, *args):
        """Backward compatibility function, will be deprecated. Use get_F_Sig instead."""
        return self.get_F_Sig(eps, *args)

    ### Gibbs free energy based solver
    
    def get_G_eps(self, sig, *args):
        """Calculate the strain for the given stress."""
        return self.G_engine.get_sig(sig, *args)

    def get_G_response(self, sig_ta, t_t, *args):
        """Calculate the strain and internal variables over time for the given stress history.
        Args:
            sig_ta: Array of stresses over time
            t_t: Array of time points
            *args: Material parameters
        """
        resp = self.G_engine.get_response(sig_ta, t_t, *args)
        # Rearrange the response to match the expected format
        return (resp[0], resp[1], resp[2], resp[3], resp[4], resp[5], resp[6], resp[7])

    def get_G_Sig(self, sig, *args):
        """Calculate the thermodynamic forces for the given stress."""
        return self.G_engine.get_Sig(sig, *args)
