import sympy as sp
from ..core.gsm_def import GSMDef
from ..core.gsm_engine import GSMEngine
from ..core.gsm_vars import Scalar, Vector

class GSM1D_VED(GSMDef):
    """Single variable one-dimensional potential that can be used to demonstrate the
    interaction between the individual dissipative mechanisms.
    """
    E = Scalar(r'E', real=True, nonnegative=True)
    S = Scalar(r'S', real=True, nonnegative=True)
    c = Scalar(r'c', real=True, nonnegative=True)
    r = Scalar(r'r', real=True, nonnegative=True)
    eta_ve = Scalar(r'\eta_\mathrm{ve}', codename='eta_ve', real=True, nonnegative=True)
    eps_0 = Scalar(r'\varepsilon_0', codename='eps_0', real=True, positive=True)

    # ## External state variables

    eps = Scalar(r'\varepsilon', codename='eps', real=True)
    eps_a = Vector(r'\varepsilon_{a}', [eps], codename='eps_a')
    sig = Scalar(r'\sigma', codename='sig', real=True)
    sig_a = Vector(r'\sigma_{a}', [sig], codename='sig_a')

    # ## Internal state variables

    omega = Scalar(r'\omega', codename='omega', real=True)
    omega_a = Vector(r'\omega_{a}', [omega], codename='omega_a')
    Y = Scalar(r'Y', codename='Y', real=True)
    Y_a = Vector(r'Y_{a}', [Y], codename='Y_a')

    z = Scalar(r'z', codename='z', real=True, nonnegative=True)
    z_a = Vector(r'z_{a}', [z], codename='z_a')
    Z = Scalar(r'Z', codename='Z', real=True, nonnegative=True)
    Z_a = Vector(r'Z_{a}', [Z], codename='Z_a')

    eps_v = Scalar(r'\varepsilon^\mathrm{ve}', codename='eps_v', real=True)
    eps_v_a = Vector(r'\varepsilon^\mathrm{ve}_{a}', [eps_v], codename='eps_v_a')
    sig_v = Scalar(r'\sigma^\mathrm{ve}', codename='sig_v', real=True)
    sig_v_a = Vector(r'\sigma^\mathrm{ve}_{a}', [sig_v], codename='sig_v_a')

    # ## Free energy potential
    eps_el = eps - eps_v
    U_e_ = sp.Rational(1,2) * (1 - omega) * E * eps_el**2 + sp.Rational(1,2) * z**2 / E
    F_ = U_e_

    F_engine = GSMEngine(
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

