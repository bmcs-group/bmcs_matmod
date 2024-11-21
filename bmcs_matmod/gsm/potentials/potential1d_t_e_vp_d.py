# # Single variable thermo-elasto-plastic damage

import sympy as sp
from bmcs_utils.api import Cymbol, SymbExpr
import traits.api as tr

import os

class Potential1D_T_E_VP_D_SymbExpr(SymbExpr):
      """Single variable one-dimensional potential that can be used to demonstrate the 
      interaction between the thermal and mechanical loading. 
      """
      # %%
      E = Cymbol(r'E', codename='E_', real=True, nonnegative=True)
      gamma_lin = Cymbol(r'\gamma_\mathrm{lin}', codename='gamma_lin_', real=True)
      gamma_exp = Cymbol(r'\gamma_\mathrm{exp}', codename='gamma_exp_', real=True, positive=True)
      alpha_0 = Cymbol(r'\alpha_0', codename='alpha_0_', real=True, nonnegative=True)
      X_0 = Cymbol(r'X_0', codename='X_0_', real=True)
      K_lin = Cymbol(r'K^\mathrm{lin}', codename='K_lin_', real=True)
      k_exp = Cymbol(r"k_\mathrm{exp}", codename='k_exp_', real=True, positive=True)
      z_0 = Cymbol(r'z_0', codename='z_0_', real=True, nonnegative=True)
      S = Cymbol(r'S', codename='S_', real=True, nonnegative=True)
      r = Cymbol(r'r', codename='r_', real=True, nonnegative=True)
      c = Cymbol(r'c', codename='c_', real=True, nonnegative=True)
      eta = Cymbol(r'\eta', codename='eta_', real=True, nonnegative=True)
      # temperature 
      C_v = Cymbol(r'C_{\mathrm{v}}', codename='C_v_', real=True, nonnegative=True)
      T_0 = Cymbol(r'\vartheta_0', codename='T_0_', real=True, nonnegative=True)
      alpha_therm = Cymbol(r'\alpha_\vartheta', codename='alpha_therm_', real=True, nonnegative=True)
      beta = Cymbol(r'\beta', codename='beta_', real=True, nonnegative=True)

      # %%
      f_s = Cymbol(r'f_\mathrm{c}', codename='f_c_')

      # %%
      mparams = (E, gamma_lin, gamma_exp, alpha_0, X_0, K_lin, k_exp, z_0, S, f_s, c, r, eta, C_v, T_0, alpha_therm, beta)
      mparams

      # %% [markdown]
      # ## External state variables

      # %%
      eps = Cymbol(r'\varepsilon', codename='eps_', real=True)
      eps_a = sp.Matrix([eps])
      sig = Cymbol(r'\sigma', codename='sig_', real=True)
      sig_a = sp.Matrix([sig])
      sig_a

      # %%
      T = Cymbol(r'\vartheta', codename='T_', real=True)

      # %% [markdown]
      # ## Internal state variables

      # %%
      eps_p = Cymbol(r'\varepsilon^\mathrm{p}', codename='eps_p_', real=True)
      eps_p_a = sp.Matrix([eps_p])
      sig_p = Cymbol(r'\sigma^\mathrm{p}', codename='sig_p_', real=True)
      sig_p_a = sp.Matrix([sig_p])

      # %%
      omega = Cymbol(r'\omega', codename='omega_', real=True)
      omega_ab = sp.Matrix([[omega]])
      omega_a = sp.Matrix([omega])
      Y = Cymbol(r'Y', codename='Y_', real=True)
      Y_a = sp.Matrix([Y])

      # %%
      z = Cymbol(r'z', codename='z_', real=True, nonnegative=True)
      z_a = sp.Matrix([z])
      K_ab = sp.Matrix([[K_lin]])
      Z = Cymbol(r'Z', codename='Z_', real=True, nonnegative=True)
      Z_a = sp.Matrix([Z])

      # %%
      alpha = Cymbol(r'\alpha', codename='alpha_', real=True, nonnegative=True)
      gamma_ab = sp.Matrix([[gamma_lin]])
      alpha_a = sp.Matrix([alpha])
      X = Cymbol(r'X', codename='X_', real=True, nonnegative=True)
      X_a = sp.Matrix([X])

      # %% [markdown]
      # ## Free energy potential

      # %%
      E_ab = sp.Matrix([[E]])
      eps_el_a = eps_a - eps_p_a
      E_eff_ab = (sp.eye(1) - omega_ab) * E_ab
      E_eff_ab

      Gamma = sp.exp(-beta * (T - T_0))
      Gamma

      F_ = tr.Property()
      @tr.cached_property
      def _get_F_(self):
            X_alpha = self.gamma_lin * self.alpha * sp.exp(-(self.alpha/self.alpha_0)**self.gamma_exp)
            int_X_alpha = sp.integrate(X_alpha, self.alpha)
            #int_X_alpha = cached_integrate('int_X_alpha', X_alpha, alpha)

            Z_z = self.K_lin * self.z * sp.exp(-(self.z/self.z_0)**self.k_exp)
            int_Z_z = sp.integrate(Z_z, self.z)
            #int_Z_z = cached_integrate('int_Z_z', Z_z, z)

            #k = 1
            # %%
            U_e_ = sp.Rational(1,2) * (self.eps_el_a.T * self.E_eff_ab * self.eps_el_a)[0]
            U_p_ =  int_Z_z + int_X_alpha
            TS_ = self.C_v * (self.T - self.T_0) **2 / (2 * self.T_0)
            return U_e_ + U_p_ - TS_

      # %% [markdown]
      # ## Dissipation potential

      # %%
      sig_eff = sp.Function(r'\sigma^{\mathrm{eff}}')(sig_p, omega)
      q = sp.Function(r'q')(sig_eff,X)
      norm_q = sp.sqrt(q*q)
      subs_q = {q: ((sig_eff  - X_0) - X)}
      subs_sig_eff = {sig_eff: sig_p / (1-omega) }
      y = Cymbol(r'y')
      f_solved_ = sp.sqrt(y**2) - f_s
      f_ = (f_solved_
            .subs({y: norm_q})
            .subs(subs_q)
            .subs(subs_sig_eff)
            .subs(f_s,((f_s + Z) * Gamma ))
            )

      # %%
      f_

      dot_omega = (1 - omega)**c * (Y / S)**(r) 
      int_dot_omega = sp.integrate(dot_omega, Y) 

      # %%
      phi_ext_ = int_dot_omega

      # %%
      t_relax_ = eta / (E + K_lin + gamma_lin)
      t_relax_ = sp.Matrix([
                        t_relax_,
                        t_relax_,
                        t_relax_,
                        ] 
                  )

      # %%
      Eps_vars = (eps_p_a, z_a, alpha_a, omega_a)

      # %%
      Sig_vars = (sig_p_a, Z_a, X_a, Y_a)

      Sig_signs =  (-1, 1, 1, -1)
