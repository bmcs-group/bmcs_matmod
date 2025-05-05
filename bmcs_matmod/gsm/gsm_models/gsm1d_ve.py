import sympy as sp
from .gsm_base import GSMBase
from .gsm_mpdp import GSMMPDP

class GSM1D_VE(GSMBase):
    """Single variable one-dimensional potential that can be used to demonstrate the
    interaction between the individual dissipative mechanisms.
    """

    E = sp.Symbol(r'E', real=True, nonnegative=True)
    eta_ve = sp.Symbol(r'\eta_\mathrm{ve}', real=True, nonnegative=True)

    mparams = (E, eta_ve)

    # ## External state variables

    eps = sp.Symbol(r'\varepsilon', real=True)
    eps_a = sp.Matrix([eps])
    sig = sp.Symbol(r'\sigma', real=True)
    sig_a = sp.Matrix([sig])

    # ## Internal state variables

    eps_v = sp.Symbol(r'\varepsilon^\mathrm{v}', real=True)
    eps_v_a = sp.Matrix([eps_v])
    sig_v = sp.Symbol(r'\sigma^\mathrm{v}', real=True)
    sig_v_a = sp.Matrix([sig_v])

    # ## Free energy potential
    eps_el = eps - eps_v
    U_e_ = sp.Rational(1,2) * E * eps_el**2
    F_ = U_e_

    Eps_vars = (eps_v_a,)
    Sig_vars = (sig_v_a,)
    Sig_signs =  (-1,)

    m_param_codenames = {eta_ve: 'eta_ve'}

    F_engine = GSMMPDP(
        name = 'gsm_F_1d_mpdp_ve',
        diff_along_rates = False,
        eps_vars = eps_a,
        sig_vars = sig_a,
        m_params = mparams,
        Eps_vars = Eps_vars,
        Sig_vars = Sig_vars,
        Sig_signs = Sig_signs,
        F_expr = F_,
    )
    dot_eps_ve = F_engine.dot_Eps[0, 0]
    sig_ve = F_engine.Sig[0, 0]
    #sig_ = F_engine.sig_
    F_engine.h_k = [dot_eps_ve * eta_ve - sig_ve]
