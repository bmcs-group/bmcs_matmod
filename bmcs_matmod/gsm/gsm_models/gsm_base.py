import sympy as sp
sp.init_printing()
from traits.api import \
    HasTraits, Property, cached_property, \
    Instance

from bmcs_matmod.api import GSMMPDP

"""
Framework for GSM-based material models.

This file defines the GSMBase class, which outlines the symbolic attributes
(eps_vars, T_var, Eps_vars, Sig_vars, Sig_signs, F_expr, f_expr, phi_ext_expr,
g_expr_list) for thermodynamic potentials and constraints using Sympy. It also
provides a property for transforming the Helmholtz free energy (F_expr)
into the Gibbs free energy (G_expr) via a Legendre transform.
"""

class GSMBase(HasTraits):
    """
    Base class for setting up thermodynamic models within the GSM framework.
    It defines the symbolic expressions and their standard names, as well as
    the property to transform Helmholtz free energy to Gibbs free energy.
    """

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
            phi_ext_expr=F_gsm.phi_ext_expr
        )
        return G_gsm
