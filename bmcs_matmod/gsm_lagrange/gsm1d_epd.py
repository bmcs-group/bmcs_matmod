import sympy as sp
from .gsm_def import GSMDef
from .gsm_engine import GSMEngine
from .gsm_vars import Scalar, Vector  # <-- import Scalar and Vector

class GSM1D_EPD(GSMDef):
    """1D Elastic-Plastic-Damage model."""

    # ## Material parameters
    E = Scalar(r'E', real=True, nonnegative=True, codename='E')
    K = Scalar(r'K', real=True, positive=True, codename='K')
    gamma = Scalar(r'\gamma', real=True, positive=True, codename='gamma')
    S = Scalar(r'S', real=True, nonnegative=True, codename='S')
    c = Scalar(r'c', real=True, nonnegative=True, codename='c')
    r = Scalar(r'r', real=True, nonnegative=True, codename='r')
    f_c = Scalar(r'f_\mathrm{c}', positive=True, real=True, codename='f_c')

    mparams = (E, K, gamma, S, c, r, f_c)

    # ## External state variables
    eps = Scalar(r'\varepsilon', real=True, codename='eps')
    eps_a = Vector(r'\varepsilon_{a}', [eps], codename='eps_a')
    sig = Scalar(r'\sigma', real=True, codename='sig')
    sig_a = Vector(r'\sigma_{a}', [sig], codename='sig_a')

    # ## Internal state variables
    omega = Scalar(r'\omega', real=True, codename='omega')
    omega_a = Vector(r'\omega_{a}', [omega], codename='omega_a')
    Y = Scalar(r'Y', real=True, codename='Y')
    Y_a = Vector(r'Y_{a}', [Y], codename='Y_a')

    eps_p = Scalar(r'\varepsilon^\mathrm{p}', real=True, codename='eps_p')
    eps_p_a = Vector(r'\varepsilon^\mathrm{p}_{a}', [eps_p], codename='eps_p_a')
    sig_p = Scalar(r'\sigma^\mathrm{p}', real=True, codename='sig_p')
    sig_p_a = Vector(r'\sigma^\mathrm{p}_{a}', [sig_p], codename='sig_p_a')

    z = Scalar(r'z', real=True, nonnegative=True, codename='z')
    z_a = Vector(r'z_{a}', [z], codename='z_a')
    Z = Scalar(r'Z', real=True, nonnegative=True, codename='Z')
    Z_a = Vector(r'Z_{a}', [Z], codename='Z_a')

    alpha = Scalar(r'\alpha', real=True, nonnegative=True, codename='alpha')
    alpha_a = Vector(r'\alpha_{a}', [alpha], codename='alpha_a')
    X = Scalar(r'X', real=True, nonnegative=True, codename='X')
    X_a = Vector(r'X_{a}', [X], codename='X_a')

    # ## Free energy potential
    eps_el = eps - eps_p
    U_e_ = sp.Rational(1,2) * (1 - omega) * E * eps_el**2
    U_p_ = sp.Rational(1,2) * K * z**2 + sp.Rational(1,2) * gamma * alpha**2
    F_ = U_e_ + U_p_

    # ## Flow potential
    f_ = sp.sqrt((sig_p/(1-omega) - X)**2) - (f_c + Z)
    phi_ext_ = (1 - omega)**c * (S/(r+1)) * (Y / S)**(r+1)

    Eps_vars = (eps_p_a, omega_a, z_a, alpha_a)
    Sig_vars = (sig_p_a, Y_a, Z_a, X_a)
    Sig_signs =  (-1, -1, 1, 1)

    F_engine = GSMEngine(
        name = 'gsm_F_1d_mpdp_epd',
        eps_vars = eps_a,
        T_var = None,  # T_var is not used here
        sig_vars = sig_a,
        m_params = mparams,
        Eps_vars = Eps_vars,
        Sig_vars = Sig_vars,
        Sig_signs = Sig_signs,
        F_expr = F_,
        f_expr = f_,
        phi_ext_expr = phi_ext_
    )
