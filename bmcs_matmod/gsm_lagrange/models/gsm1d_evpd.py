import sympy as sp
from ..core.gsm_def import GSMDef
from ..core.gsm_engine import GSMEngine
from ..core.gsm_vars import Scalar, Vector

class GSM1D_EVPD(GSMDef):
    """Single variable one-dimensional potential that can be used to demonstrate the
    interaction between the individual dissipative mechanisms.
    """

    # ## Material parameters
    E = Scalar(r'E', real=True, positive=True, codename='E')
    K = Scalar(r'K', real=True, positive=True, codename='K')
    S = Scalar(r'S', real=True, positive=True, codename='S')
    c = Scalar(r'c', real=True, positive=True, codename='c')
    r = Scalar(r'r', real=True, positive=True, codename='r')
    f_c = Scalar(r'f_\mathrm{c}', positive=True, real=True, codename='f_c')
    eta_vp = Scalar(r'\eta_\mathrm{vp}', real=True, nonnegative=True, codename='eta_vp')

    mparams = (E, K, f_c, S, c, r, eta_vp)

    # ## External state variables
    eps = Scalar(r'\varepsilon', real=True, codename='eps')
    eps_a = Vector(r'\varepsilon_{a}', [eps], codename='eps_a')
    sig = Scalar(r'\sigma', real=True, codename='sig')
    sig_a = Vector(r'\sigma_{a}', [sig], codename='sig_a')

    # ## Internal state variables
    eps_p = Scalar(r'\varepsilon^\mathrm{p}', real=True, codename='eps_p')
    eps_p_a = Vector(r'\varepsilon^\mathrm{p}_{a}', [eps_p], codename='eps_p_a')
    sig_p = Scalar(r'\sigma^\mathrm{p}', real=True, codename='sig_p')
    sig_p_a = Vector(r'\sigma^\mathrm{p}_{a}', [sig_p], codename='sig_p_a')

    omega = Scalar(r'\omega', nonnegative=True, real=True, codename='omega')
    omega_a = Vector(r'\omega_{a}', [omega], codename='omega_a')
    Y = Scalar(r'Y', real=True, codename='Y')
    Y_a = Vector(r'Y_{a}', [Y], codename='Y_a')

    z = Scalar(r'z', real=True, nonnegative=True, codename='z')
    z_a = Vector(r'z_{a}', [z], codename='z_a')
    Z = Scalar(r'Z', real=True, nonnegative=True, codename='Z')
    Z_a = Vector(r'Z_{a}', [Z], codename='Z_a')

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

    F_engine = GSMEngine(
        name = 'gsm_F_1d_mpdp_evpd_lih',
        eps_vars = eps_a,
        sig_vars = sig_a,
        m_params = mparams,
        Eps_vars = Eps_vars,
        Sig_vars = Sig_vars,
        Sig_signs = Sig_signs,
        F_expr = F_,
        f_expr = f_,
    )

    dot_eps_p = F_engine.dot_Eps[0, 0]
    dot_eps = F_engine.dot_eps_a[0, 0]

    f_d_ = F_engine.f_expr - eta_vp * sp.sqrt(dot_eps_p**2) 
    F_engine.f_expr = f_d_

    epsEps = sp.Matrix.vstack(F_engine.eps_vars, F_engine.Eps.as_explicit())
    dot_epsEps = sp.Matrix.vstack(sp.Matrix([[dot_eps]]), F_engine.dot_Eps.as_explicit())
    dot_f_ = (F_engine.f_expr.subs(F_engine.subs_Sig_Eps).diff(epsEps).T * dot_epsEps)[0,0]

    sig = F_engine.sig_a[0, 0]
    F_engine.phi_ext_expr = sp.Heaviside(sig * dot_eps) * sp.Abs(dot_eps-dot_eps_p) * (1 - omega)**c * (S/(r+1)) * (Y/ S)**(r+1)
