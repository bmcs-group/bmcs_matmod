import traits.api as tr
import sympy as sp
sp.init_printing()
from traits.api import \
    HasTraits, Property, cached_property, \
    Instance

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
    It defines the symbolic expressions and their standard names, as well as
    the property to transform Helmholtz free energy to Gibbs free energy.
    """

    param_codenames = tr.Property
    @cached_property
    def _get_param_codenames(self):
        """Construct a mapping from symbols to codenames for parameter substitution."""
        param_codenames = {}
        for sym in self.F_engine.m_params:
            sym_name = sym.name
            if is_valid_variable_name(sym_name):
                codename = sym_name
            else:
                # Check if a codename is provided in m_param_codenames
                m_param_codenames = getattr(self, 'm_param_codenames', {})
                if sym in m_param_codenames:
                    codename = m_param_codenames[sym]
                else:
                    raise ValueError(
                        f"Symbol '{sym}' has an invalid name '{sym_name}' "
                        f"and no codename was provided in 'm_param_codenames'."
                    )
            param_codenames[sym] = codename
        return param_codenames

    # Symbolic attributes (to be defined in subclasses)

    F_engine = Instance(sp.Expr)

    G_engine = Property(Instance(sp.Expr), depends_on='F_engine')
    """
    Transform Helmholtz free energy (F_expr) into Gibbs free energy (G_expr)
    using the Legendre transform. Override in subclass if needed.
    """
    @cached_property
    def _get_G_engine(self):
        F_gsm = self.F_engine
        eps_a = self.eps_a
        sig_a = self.sig_a
        dF_du = F_gsm.F_expr.diff(eps_a)
        u_sig_ = sp.Matrix(
            [
            sp.solve(sp.Eq(sig_i, dF_du_i), u_i)[0]
            for sig_i, u_i, dF_du_i in zip(sig_a, eps_a, dF_du)
            ]
        )
        subs_u_sig_ = dict(zip(eps_a, u_sig_))

        sig_x_u_ = (sig_a.T * eps_a)[0]
        G_expr = sig_x_u_ - F_gsm.F_expr
        G_ = sp.simplify(G_expr.subs(subs_u_sig_))

        G_gsm = GSMMPDP(
            name=f'G_{F_gsm.name}',
            u_vars=F_gsm.sig_vars,
            sig_vars=eps_a,
            T_var=F_gsm.T_var,
            m_params=F_gsm.m_params,
            Eps_vars=F_gsm.Eps_vars,
            Sig_vars=F_gsm.Sig_vars,
            Sig_signs=F_gsm.Sig_signs,
            F_expr=G_,
            dF_sign=-1,
            f_expr=F_gsm.f_expr,
            phi_ext_expr=F_gsm.phi_ext_expr,
            h_k=F_gsm.h_k
        )
        return G_gsm

    name = Property(tr.Str)
    
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
        (gamma_mech, L_, dL_dS_, dL_dS_A_, dR_dA_n1), (eps_n, delta_eps, Eps_n, delta_A, delta_t, Ox, Ix), Sig_n1, f_n1, R_n1, dR_dA_n1_OI = self.F_engine.Sig_f_R_dR_n1
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
        latex_lines.append("### Gibbs free energy")
        latex_lines.append("$$G = " + sp.latex(sp.simplify(self.G_engine.F_expr)) + "$$")
        latex_lines.append("$$" + sp.latex(self.G_engine.subs_Sig_Eps) + "$$")
        latex_lines.append("### Mechanical dissipation")
        latex_lines.append("$$\\gamma_{\\mathrm{mech}} = " + sp.latex(sp.simplify(gamma_mech)) + "$$")
        latex_lines.append("### Lagrangian")
        latex_lines.append("$$\mathcal{L} = " + sp.latex(L_) + "$$")
        latex_lines.append("### Residuum")
        latex_lines.append("$$\\frac{\\partial \mathcal{L}}{\\partial \mathcal{S}} = " + sp.latex(dL_dS_) + " = 0$$")
        if self.F_engine.dot_Eps_bounds_expr is not sp.S.Zero:
            latex_lines.append("### Bounds of inelastic process")
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

    def get_sig(self, eps, Eps, **kwargs):
        """Calculate the stress for the given strain and internal variables."""
        args = self.get_args(**kwargs)
        return self.F_engine.get_sig(eps, Eps, *args)

    def get_response(self, eps, Eps, **kwargs):
        """Calculate the stress and internal variables for the given strain."""
        args = self.get_args(**kwargs)
        return self.F_engine.get_response(eps, Eps, *args)

    def get_Sig(self, eps, Eps, **kwargs):
        """Calculate the thermodynamic forces for the given strain and internal variables."""
        args = self.get_args(**kwargs)
        return self.F_engine.get_Sig(eps, Eps, *args)

    ### Gibbs free energy based solver
    
    def get_G_eps(self, sig, Eps, **kwargs):
        """Calculate the strain for the given stress and internal variables."""
        args = self.get_args(**kwargs)
        return self.G_engine.get_sig(sig, Eps, *args)

    def get_G_response(self, sig, Eps, **kwargs):
        """Calculate the strain and internal variables for the given stress."""
        args = self.get_args(**kwargs)
        return self.G_engine.get_response(sig, Eps, *args)

    def get_G_Sig(self, sig, Eps, **kwargs):
        """Calculate the thermodynamic forces for the given stress and internal variables."""
        args = self.get_args(**kwargs)
        return self.G_engine.get_Sig(sig, Eps, *args)
