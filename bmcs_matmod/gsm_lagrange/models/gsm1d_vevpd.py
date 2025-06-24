import sympy as sp
from ..core.gsm_def import GSMDef
from ..core.gsm_engine import GSMEngine
from ..core.gsm_vars import Scalar, Vector

class GSM1D_VEVPD(GSMDef):
    """Single variable one-dimensional potential that can be used to demonstrate the
    interaction between the individual dissipative mechanisms.
    """

    E = Scalar(r'E', real=True, nonnegative=True)
    K = Scalar(r'K', real=True)
    f_c = Scalar(r'f_\mathrm{c}', codename='f_c')
    S = Scalar(r'S', real=True, nonnegative=True)
    c = Scalar(r'c', real=True, nonnegative=True)
    r = Scalar(r'r', real=True, nonnegative=True)
    eta_vp = Scalar(r'\eta_\mathrm{vp}', real=True, nonnegative=True, codename='eta_vp')
    eta_ve = Scalar(r'\eta_\mathrm{ve}', real=True, nonnegative=True, codename='eta_ve')

    eps = Scalar(r'\varepsilon', real=True, codename='eps')
    eps_a = Vector(r'\varepsilon_{a}', [eps], codename='eps_a')
    sig = Scalar(r'\sigma', real=True, codename='sig')
    sig_a = Vector(r'\sigma_{a}', [sig], codename='sig_a')

    omega = Scalar(r'\omega', real=True, codename='omega')
    omega_a = Vector(r'\omega_{a}', [omega], codename='omega_a')
    Y = Scalar(r'Y', real=True)
    Y_a = Vector(r'Y_{a}', [Y], codename='Y_a')

    eps_p = Scalar(r'\varepsilon^\mathrm{vp}', real=True, codename='eps_p')
    eps_p_a = Vector(r'\varepsilon^\mathrm{vp}_{a}', [eps_p], codename='eps_p_a')
    sig_p = Scalar(r'\sigma^\mathrm{vp}', real=True, codename='sig_p')
    sig_p_a = Vector(r'\sigma^\mathrm{vp}_{a}', [sig_p], codename='sig_p_a')

    eps_v = Scalar(r'\varepsilon^\mathrm{ve}', real=True, codename='eps_v')
    eps_v_a = Vector(r'\varepsilon^\mathrm{ve}_{a}', [eps_v], codename='eps_v_a')
    sig_v = Scalar(r'\sigma^\mathrm{ve}', real=True, codename='sig_v')
    sig_v_a = Vector(r'\sigma^\mathrm{ve}_{a}', [sig_v], codename='sig_v_a')

    z = Scalar(r'z', real=True, nonnegative=True)
    z_a = Vector(r'z_{a}', [z], codename='z_a')
    Z = Scalar(r'Z', real=True, nonnegative=True)
    Z_a = Vector(r'Z_{a}', [Z], codename='Z_a')

    # # ## Free energy potential
    # eps_el = eps - eps_v - eps_p
    # U_e_ = sp.Rational(1,2) * (1 - omega) * E * eps_el**2
    # U_p_ =  sp.Rational(1,2) * K * z**2
    # F_ = U_e_ + U_p_

    # ## Flow potential
    f_ = sp.sqrt(sig_p**2) - (f_c + Z)

    Eps_vars = (eps_v_a, eps_p_a, omega_a, z_a)
    Sig_vars = (sig_v_a, sig_p_a, Y_a, Z_a)
    Sig_signs =  (-1, -1, -1, 1)

    F_engine = GSMEngine(
        name = 'gsm_F_1d_mpdp_vevp_lih',
        diff_along_rates = False,
        eps_vars = eps_a,
        sig_vars = sig_a,
        m_params = (E, K, f_c, S, c, r, eta_vp, eta_ve),
        Eps_vars = Eps_vars,
        Sig_vars = Sig_vars,
        Sig_signs = Sig_signs,
        F_expr = sp.Rational(1,2) * (1 - omega) * E * (eps - eps_v - eps_p)**2 + sp.Rational(1,2) * K * z**2,
        f_expr = sp.sqrt(sig_p**2) - (f_c + Z),
    )
    dot_eps = F_engine.dot_eps_a[0, 0]
    dot_eps_vp = F_engine.dot_Eps[1, 0]
    dot_eps_ve = F_engine.dot_Eps[0, 0]

    # evaluate the rate of the static threshold function f_s
    epsEps = sp.Matrix.vstack(F_engine.eps_vars, F_engine.Eps.as_explicit())
    dot_epsEps = sp.Matrix.vstack(sp.Matrix([[dot_eps]]), F_engine.dot_Eps.as_explicit())
    dot_f_ = (F_engine.f_expr.subs(F_engine.subs_Sig_Eps).diff(epsEps).T * dot_epsEps)[0,0]

    # dynamic threshold function f_d depending on the rate of the plastic strain norm
    f_d_ = F_engine.f_expr -  eta_vp * (sp.sqrt(dot_eps_vp**2))
    F_engine.f_expr = f_d_

    # viscoelastic strain 
    sig_ve = F_engine.Sig[0, 0]
    F_engine.h_k = [eta_ve * dot_eps_ve - sig_ve]
    
    # non-associative flow potential extension is normed by the norm of elastic strain rate
    # it is only active if the rate os stress power is positive, i.e. Heaviside(sig_ * dot_eps) > 0
    sig_ = F_engine.sig_a[0, 0]
    F_engine.phi_ext_expr = sp.Heaviside(sig_ * dot_eps) * sp.Abs(dot_eps-dot_eps_vp-dot_eps_ve) * (1 - omega)**c * (S/(r+1)) * (Y/ S)**(r+1)

    F_engine.dot_Eps_bounds_expr = -(sp.Abs(dot_eps) - sp.Abs(dot_eps_ve))

