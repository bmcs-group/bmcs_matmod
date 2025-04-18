import sympy as sp
from .gsm_base import GSMBase
from .gsm_mpdp import GSMMPDP

class GSM1D_EPD(GSMBase):
    """Single variable one-dimensional potential that can be used to demonstrate the
    interaction between the individual dissipative mechanisms.
    """

    E = sp.Symbol(r'E', real=True, nonnegative=True)
    K = sp.Symbol(r'K', real=True)
    S = sp.Symbol(r'S', real=True, nonnegative=True)
    c = sp.Symbol(r'c', real=True, nonnegative=True)
    r = sp.Symbol(r'r', real=True, nonnegative=True)
    f_c = sp.Symbol(r'f_\mathrm{c}')

    mparams = (E, K, f_c, S, c, r)
    m_param_codenames = {f_c: 'f_c'}

    # ## External state variables

    eps = sp.Symbol(r'\varepsilon', codename='eps_', real=True)
    eps_a = sp.Matrix([eps])
    sig = sp.Symbol(r'\sigma', codename='sig_', real=True)
    sig_a = sp.Matrix([sig])

    # ## Internal state variables

    eps_p = sp.Symbol(r'\varepsilon^\mathrm{p}', codename='eps_p_', real=True)
    eps_p_a = sp.Matrix([eps_p])
    sig_p = sp.Symbol(r'\sigma^\mathrm{p}', codename='sig_p_', real=True)
    sig_p_a = sp.Matrix([sig_p])

    omega = sp.Symbol(r'\omega', real=True)
    omega_a = sp.Matrix([omega])
    Y = sp.Symbol(r'Y', real=True)
    Y_a = sp.Matrix([Y])

    z = sp.Symbol(r'z', codename='z_', real=True, nonnegative=True)
    z_a = sp.Matrix([z])
    Z = sp.Symbol(r'Z', codename='Z_', real=True, nonnegative=True)
    Z_a = sp.Matrix([Z])

    # ## Free energy potential
    eps_el = eps - eps_p
    U_e_ = sp.Rational(1,2) * (1 - omega) * E * eps_el**2
    U_p_ =  sp.Rational(1,2) * K * z**2
    F_ = U_e_ + U_p_

    # ## Flow potential
    f_ = sp.sqrt((sig_p/(1-omega))**2) - (f_c + Z)

    Eps_vars = (eps_p_a, omega_a, z_a)
    Sig_vars = (sig_p_a, Y_a, Z_a)
    Sig_signs =  (-1, -1, 1)

    F_engine = GSMMPDP(
        name = 'gsm_F_1d_mpdp_epd_lih',
        u_vars = eps_a,
        sig_vars = sig_a,
        m_params = mparams,
        Eps_vars = Eps_vars,
        Sig_vars = Sig_vars,
        Sig_signs = Sig_signs,
        F_expr = F_,
        f_expr = f_,
        phi_ext_expr = (1 - omega)**c * (S/(r+1)) * (Y/ S)**(r+1)
    )
