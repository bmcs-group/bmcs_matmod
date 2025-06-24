import sympy as sp
from ..core.gsm_def import GSMDef
from ..core.gsm_engine import GSMEngine
from ..core.gsm_vars import Scalar, Vector

class GSM1D_EVP(GSMDef):
    """Single variable one-dimensional potential that can be used to demonstrate the
    interaction between the individual dissipative mechanisms.
    """

    E = Scalar(r'E', real=True, positive=True)
    K = Scalar(r'K', real=True, positive=True)
    f_c = Scalar(r'f_\mathrm{c}', positive=True, real=True, codename='f_c')
    eta_vp = Scalar(r'\eta_\mathrm{vp}', real=True, nonnegative=True, codename='eta_vp')
    

    mparams = (E, K, f_c, eta_vp)
    
    # ## External state variables

    eps = Scalar(r'\varepsilon', real=True, codename='eps')
    eps_a = Vector(r'\varepsilon_{a}', [eps], codename='eps')
    sig = Scalar(r'\sigma', real=True, codename='sig')
    sig_a = Vector(r'\sigma_{a}', [sig], codename='sig')

    # ## Internal state variables

    eps_p = Scalar(r'\varepsilon^\mathrm{p}', real=True, codename='eps_p')
    eps_p_a = Vector(r'\varepsilon^\mathrm{p}_{a}', [eps_p], codename='eps_p')
    sig_p = Scalar(r'\sigma^\mathrm{p}', real=True, codename='sig_p')
    sig_p_a = Vector(r'\sigma^\mathrm{p}_{a}', [sig_p], codename='sig_p')

    z = Scalar(r'z', real=True, nonnegative=True, codename='z')
    z_a = Vector(r'z_{a}', [z], codename='z')
    Z = Scalar(r'Z', real=True, nonnegative=True, codename='Z')
    Z_a = Vector(r'Z_{a}', [Z], codename='Z')

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


    F_engine = GSMEngine(
        name = 'gsm_F_1d_mpdp_vep_lih',
        diff_along_rates = False,
        eps_vars = eps_a,
        sig_vars = sig_a,
        m_params = mparams,
        Eps_vars = Eps_vars,
        Sig_vars = Sig_vars,
        Sig_signs = Sig_signs,
        F_expr = F_,
        f_expr = f_,
    )

    dot_eps_p = F_engine.dot_Eps[0,0]
    f_d_ = F_engine.f_expr - eta_vp * sp.Abs(dot_eps_p)
    F_engine.f_expr = f_d_

