# # Single variable thermo-elasto-plastic damage

import sympy as sp
from bmcs_utils.api import Cymbol, SymbExpr

class Potential1D_T_E_VP_NIKH_SymbExpr(SymbExpr):
      """Single variable one-dimensional potential that can be used to demonstrate the 
      interaction between the thermal and mechanical loading. 
      """
      E = Cymbol(r'E', codename='E_', real=True, nonnegative=True)
      gamma = Cymbol(r'\gamma', codename='gamma_', real=True)
      X_0 = Cymbol(r'X_0', codename='X_0_', real=True)
      K_lin = Cymbol(r'K^\mathrm{lin}', codename='K_lin_', real=True)
      k_exp = Cymbol(r"k_\mathrm{exp}", codename='k_exp_', real=True, positive=True)
      z_0 = Cymbol(r'z_0', codename='z_0_', real=True, nonnegative=True)
      eta = Cymbol(r'\eta', codename='eta_', real=True, nonnegative=True)
      # temperature 
      C_v = Cymbol(r'C_{\mathrm{v}}', codename='C_v_', real=True, nonnegative=True)
      T_0 = Cymbol(r'\vartheta_0', codename='T_0_', real=True, nonnegative=True)
      # threshold
      f_c = Cymbol(r'f_\mathrm{c}', codename='f_c_')

      mparams = (E, gamma, X_0, K_lin, k_exp, z_0, f_c, eta, C_v, T_0)

      # ## External state variables
      eps = Cymbol(r'\varepsilon', codename='eps_', real=True)
      eps_a = sp.Matrix([eps])
      sig = Cymbol(r'\sigma', codename='sig_', real=True)
      sig_a = sp.Matrix([sig])

      T = Cymbol(r'\vartheta', codename='T_', real=True)

      # ## Internal state variables
      eps_p = Cymbol(r'\varepsilon^\mathrm{p}', codename='eps_p_', real=True)
      eps_p_a = sp.Matrix([eps_p])
      sig_p = Cymbol(r'\sigma^\mathrm{p}', codename='sig_p_', real=True)
      sig_p_a = sp.Matrix([sig_p])

      z = Cymbol(r'z', codename='z_', real=True, nonnegative=True)
      z_a = sp.Matrix([z])
      K_ab = sp.Matrix([[K_lin]])
      Z = Cymbol(r'Z', codename='Z_', real=True, nonnegative=True)
      Z_a = sp.Matrix([Z])

      alpha = Cymbol(r'\alpha', codename='alpha_', real=True, nonnegative=True)
      gamma_ab = sp.Matrix([[gamma]])
      alpha_a = sp.Matrix([alpha])
      X = Cymbol(r'X', codename='X_', real=True, nonnegative=True)
      X_a = sp.Matrix([X])

      E_ab = sp.Matrix([[E]])
      eps_el_a = eps_a - eps_p_a
      E_eff_ab = E_ab

      Z_z = K_lin * z * sp.exp(-(z/z_0)**k_exp)
      int_Z_z = sp.integrate(Z_z, z)

      U_e_ = sp.Rational(1,2) * (eps_el_a.T * E_eff_ab * eps_el_a)[0]
      U_p_ =  int_Z_z + sp.Rational(1,2) * (alpha_a.T * gamma_ab * alpha_a)[0]
      TS_ = C_v * (T - T_0) **2 / (2 * T_0)
      F_ = U_e_ + U_p_ - TS_

      # ## Dissipation potential
      sig_eff = sp.Symbol(r'\sigma^{\mathrm{eff}}')
      q = sp.Symbol(r'q')
      norm_q = sp.sqrt(q*q)
      subs_q = {q: ((sig_eff  - X_0) - X)}
      subs_sig_eff = {sig_eff: sig_p}
      y = Cymbol(r'y')
      f_solved_ = sp.sqrt(y**2) - f_c
      f_ = (f_solved_
            .subs({y: norm_q})
            .subs(subs_q)
            .subs(subs_sig_eff)
            .subs(f_c,((f_c + Z) ))
            )

      phi_ext_ = 0

      # %%
      t_relax_ = eta / (E + K_lin + gamma)
      t_relax_ = sp.Matrix([
                        t_relax_,
                        t_relax_,
                        t_relax_,
                        ] 
                  )

      Eps_vars = (eps_p_a, z_a, alpha_a)
      Sig_vars = (sig_p_a, Z_a, X_a)
      Sig_signs =  (-1, 1, 1)
