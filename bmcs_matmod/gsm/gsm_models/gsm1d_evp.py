import sympy as sp
from .gsm_base import GSMBase
from bmcs_matmod.api import GSMMPDP

class GSM1D_EVP(GSMBase):
    """Single variable one-dimensional potential that can be used to demonstrate the
    interaction between the individual dissipative mechanisms.
    """

    E = sp.Symbol(r'E', real=True, nonnegative=True)
    K = sp.Symbol(r'K', real=True)
    f_c = sp.Symbol(r'f_\mathrm{c}')
    eta = sp.Symbol(r'\eta_\mathrm{vp}', real=True, nonnegative=True)

    mparams = (E, K, f_c, eta)

    # ## External state variables

    eps = sp.Symbol(r'\varepsilon', real=True)
    eps_a = sp.Matrix([eps])
    sig = sp.Symbol(r'\sigma', real=True)
    sig_a = sp.Matrix([sig])

    # ## Internal state variables

    eps_p = sp.Symbol(r'\varepsilon^\mathrm{p}', real=True)
    eps_p_a = sp.Matrix([eps_p])
    sig_p = sp.Symbol(r'\sigma^\mathrm{p}', real=True)
    sig_p_a = sp.Matrix([sig_p])

    z = sp.Symbol(r'z', real=True, nonnegative=True)
    z_a = sp.Matrix([z])
    Z = sp.Symbol(r'Z', real=True, nonnegative=True)
    Z_a = sp.Matrix([Z])

    # ## Free energy potential
    eps_el = eps - eps_p
    U_e_ = sp.Rational(1,2) * E * eps_el**2
    U_p_ =  sp.Rational(1,2) * K * z**2
    F_ = U_e_ + U_p_

    # ## Flow potential
    f_ = sp.sqrt(sig_p**2) - (f_c + Z)
    phi_ext_ = 0

    Eps_vars = (eps_p_a, z_a)
    Sig_vars = (sig_p_a, Z_a)
    Sig_signs =  (-1, 1)


    F_engine = GSMMPDP(
        name = 'gsm_F_1d_mpdp_vep_lih',
        diff_along_rates = False,
        u_vars = eps_a,
        sig_vars = sig_a,
        m_params = mparams,
        Eps_vars = Eps_vars,
        Sig_vars = Sig_vars,
        Sig_signs = Sig_signs,
        F_expr = F_,
        f_expr = f_,
    )

    dot_eps_p = F_engine.dot_Eps[0,0]
    f_d_ = F_engine.f_expr - eta * dot_eps_p
    F_engine.f_expr = f_d_

