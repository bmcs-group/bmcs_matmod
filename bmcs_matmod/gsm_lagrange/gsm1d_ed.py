import sympy as sp
from .gsm_def import GSMDef
from .gsm_engine import GSMEngine
from .gsm_vars import Scalar, Vector

class GSM1D_ED(GSMDef):
    """Single variable one-dimensional potential that can be used to demonstrate the
    interaction between the individual dissipative mechanisms.
    """
    # Material parameters
    E = Scalar(r'E', codename='E', real=True, nonnegative=True)
    S = Scalar(r'S', codename='S', real=True, nonnegative=True)
    c = Scalar(r'c', codename='c', real=True, nonnegative=True)
    r = Scalar(r'r', codename='r', real=True, nonnegative=True)
    eps_0 = Scalar(r'\varepsilon_0', codename='eps_0', real=True, nonnegative=True)

    # ## External state variables
    eps = Scalar(r'\varepsilon', codename='eps', real=True)
    eps_a = Vector(r'{\varepsilon}_{a}', [eps], codename='eps_a')
    sig = Scalar(r'\sigma', codename='sig', real=True)
    sig_a = Vector(r'{\sigma}_{a}', [sig], codename='sig_a')

    # ## Internal state variables
    z = Scalar(r'z', codename='z', real=True, nonnegative=True)
    z_a = Vector(r'{z}_{a}', [z], codename='z_a')
    Z = Scalar(r'Z', codename='Z', real=True, nonnegative=True)
    Z_a = Vector(r'{Z}_{a}', [Z], codename='Z_a')
    omega = Scalar(r'\omega', codename='omega', real=True)
    omega_a = Vector(r'{\omega}_{a}', [omega], codename='omega_a')
    Y = Scalar(r'Y', codename='Y', real=True)
    Y_a = Vector(r'{Y}_{a}', [Y], codename='Y_a')

    # ## Free energy potential
    eps_el = eps
    U_e_ = sp.Rational(1,2) * (1 - omega) * E * eps_el**2 + sp.Rational(1,2) * z**2
    F_ = U_e_

    F_engine = GSMEngine(
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


