import sympy as sp
from ..core.gsm_def import GSMDef
from ..core.gsm_engine import GSMEngine
from ..core.gsm_vars import Scalar, Vector

class GSM1D_VE(GSMDef):
    """Single variable one-dimensional potential that can be used to demonstrate the
    interaction between the individual dissipative mechanisms.
    """

    E = Scalar(r'E', real=True, nonnegative=True)
    eta_ve = Scalar(r'\eta_\mathrm{ve}', real=True, nonnegative=True, codename='eta_ve')

    mparams = (E, eta_ve)

    # ## External state variables

    eps = Scalar(r'\varepsilon', real=True, codename='eps')
    eps_a = Vector(r'\varepsilon_{a}', [eps], codename='eps_a')
    sig = Scalar(r'\sigma', real=True, codename='sig')
    sig_a = Vector(r'\sigma_{a}', [sig], codename='sig_a')

    # ## Internal state variables

    eps_v = Scalar(r'\varepsilon^\mathrm{v}', real=True, codename='eps_v')
    eps_v_a = Vector(r'\varepsilon^\mathrm{v}_{a}', [eps_v], codename='eps_v_a')
    sig_v = Scalar(r'\sigma', real=True, codename='sig_v')
    sig_v_a = Vector(r'\sigma_{a}', [sig_v], codename='sig_v_a')

    eps_el_a = eps_a - eps_v_a
    U_e_a = sp.Rational(1,2) * E * eps_el_a.T * eps_el_a

    # ## Free energy potential
    eps_el = eps - eps_v
    U_e_ = sp.Rational(1,2) * E * eps_el**2
    F_ = U_e_

    Eps_vars = (eps_v_a,)
    Sig_vars = (sig_v_a,)
    Sig_signs =  (-1,)

    F_engine = GSMEngine(
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
    dot_eps_ve_a = F_engine.dot_Eps[0, 0]
    sig_ve_a = F_engine.Sig[0, 0]
    #sig_ = F_engine.sig_
    F_engine.h_k = [dot_eps_ve_a * eta_ve - sig_ve_a]
