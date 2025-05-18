import sympy as sp
from .gsm_base import GSMBase
from .gsm_mpdp import GSMMPDP

class GSM1D_VED(GSMBase):
    """Single variable one-dimensional potential that can be used to demonstrate the
    interaction between the individual dissipative mechanisms.
    """
    E = sp.Symbol(r'E', real=True, nonnegative=True)
    S = sp.Symbol(r'S', real=True, nonnegative=True)
    c = sp.Symbol(r'c', real=True, nonnegative=True)
    r = sp.Symbol(r'r', real=True, nonnegative=True)
    eta_ve = sp.Symbol(r'\eta_\mathrm{ve}', real=True, nonnegative=True)
    eps_0 = sp.Symbol(r'\varepsilon_0', real=True, positive=True)

    m_param_codenames = {eta_ve: 'eta_ve', eps_0: 'eps_0'}

    # ## External state variables

    eps = sp.Symbol(r'\varepsilon', real=True)
    eps_a = sp.Matrix([eps])
    sig = sp.Symbol(r'\sigma', real=True)
    sig_a = sp.Matrix([sig])

    # ## Internal state variables

    omega = sp.Symbol(r'\omega', real=True)
    omega_a = sp.Matrix([omega])
    Y = sp.Symbol(r'Y', real=True)
    Y_a = sp.Matrix([Y])

    z = sp.Symbol(r'z', codename='z_', real=True, nonnegative=True)
    z_a = sp.Matrix([z])
    Z = sp.Symbol(r'Z', codename='Z_', real=True, nonnegative=True)
    Z_a = sp.Matrix([Z])

    eps_v = sp.Symbol(r'\varepsilon^\mathrm{ve}', real=True)
    eps_v_a = sp.Matrix([eps_v])
    sig_v = sp.Symbol(r'\sigma^\mathrm{ve}', real=True)
    sig_v_a = sp.Matrix([sig_v])

    # ## Free energy potential
    eps_el = eps - eps_v
    U_e_ = sp.Rational(1,2) * (1 - omega) * E * eps_el**2 + sp.Rational(1,2) * z**2 / E
    F_ = U_e_

    F_engine = GSMMPDP(
        name = 'gsm_F_1d_mpdp_ved',
        eps_vars = eps_a,
        sig_vars = sig_a,
        T_var = sp.Symbol('T', real=True),
        m_params = (E, eta_ve, S, c, r, eps_0),
        Eps_vars = (eps_v_a, omega_a, z_a),
        Sig_vars = (sig_v_a, Y_a, Z_a),
        Sig_signs = (-1, -1, 1),
        F_expr = F_,
        f_expr = sp.Abs(eps) - (eps_0 + Z),
    )
    
    dot_eps_ve_ = F_engine.dot_Eps[0, 0]
    dot_eps = F_engine.dot_eps_a[0, 0]
    sig_ve_ = F_engine.Sig[0, 0]
    F_engine.h_k = [dot_eps_ve_ - sig_ve_ / (1 - omega) / eta_ve]
    F_engine.phi_ext_expr = sp.Abs(dot_eps-dot_eps_ve_) * (1 - omega)**c * (S/(r+1)) * (Y / S)**(r+1)

