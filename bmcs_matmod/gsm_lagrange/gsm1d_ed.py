import sympy as sp
from .gsm_base import GSMBase
from .gsm_mpdp import GSMMPDP

class GSM1D_ED(GSMBase):
    """Single variable one-dimensional potential that can be used to demonstrate the
    interaction between the individual dissipative mechanisms.
    """
    E = sp.Symbol(r'E', real=True, nonnegative=True)
    S = sp.Symbol(r'S', real=True, nonnegative=True)
    c = sp.Symbol(r'c', real=True, nonnegative=True)
    r = sp.Symbol(r'r', real=True, nonnegative=True)
    eps_0 = sp.Symbol(r'\varepsilon_0', real=True, nonnegative=True)

    # ## External state variables

    eps = sp.Symbol(r'\varepsilon', real=True)
    eps_a = sp.Matrix([eps])
    sig = sp.Symbol(r'\sigma', real=True)
    sig_a = sp.Matrix([sig])

    # ## Internal state variables

    z = sp.Symbol(r'z', real=True, nonnegative=True)
    z_a = sp.Matrix([z])
    Z = sp.Symbol(r'Z', real=True, nonnegative=True)
    Z_a = sp.Matrix([Z])

    omega = sp.Symbol(r'\omega', real=True)
    omega_a = sp.Matrix([omega])
    Y = sp.Symbol(r'Y', real=True)
    Y_a = sp.Matrix([Y])

    # ## Free energy potential
    eps_el = eps
    U_e_ = sp.Rational(1,2) * (1 - omega) * E * eps_el**2 + sp.Rational(1,2) * z**2
    F_ = U_e_

    m_param_codenames = {eps_0: 'eps_0'}

    F_engine = GSMMPDP(
        name = 'gsm_F_1d_mpdp_ed',
        eps_vars = eps_a,
        T_var = sp.Symbol('T', real=True),
        sig_vars = sig_a,
        m_params = (E, S, c, r, eps_0),
        Eps_vars = (omega_a, z_a),
        Sig_vars = (Y_a, Z_a),
        Sig_signs = (-1, 1),
        F_expr = F_,
        f_expr = sp.Abs(eps) - (eps_0 + Z),
        phi_ext_expr =  (1 - omega)**c * (S/(r+1)) * (Y / S)**(r+1)
    )


