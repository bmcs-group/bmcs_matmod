import sympy as sp
from .gsm_base import GSMBase
from .gsm_mpdp import GSMEngine

class GSM1D_EVPD(GSMBase):
    """Single variable one-dimensional potential that can be used to demonstrate the
    interaction between the individual dissipative mechanisms.
    """


    E = sp.Symbol(r'E', real=True, positive=True)
    K = sp.Symbol(r'K', real=True, positive=True)
    S = sp.Symbol(r'S', real=True, positive=True)
    c = sp.Symbol(r'c', real=True, positive=True)
    r = sp.Symbol(r'r', real=True, positive=True)
    f_c = sp.Symbol(r'f_\mathrm{c}', positive=True, real=True)
    eta_vp = sp.Symbol(r'\eta_\mathrm{vp}', real=True, nonnegative=True)

    mparams = (E, K, f_c, S, c, r, eta_vp)
    m_param_codenames = {f_c: 'f_c', eta_vp: 'eta_vp'}

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

    omega = sp.Symbol(r'\omega', nonnegative=True, real=True)
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

    sig = F_engine.sig_
    F_engine.phi_ext_expr = sp.Heaviside(sig * dot_eps) * sp.Abs(dot_eps-dot_eps_p) * (1 - omega)**c * (S/(r+1)) * (Y/ S)**(r+1)
    