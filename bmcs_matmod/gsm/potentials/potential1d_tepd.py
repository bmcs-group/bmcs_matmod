# # Single variable thermo-elasto-plastic damage

import sympy as sp
from bmcs_utils.api import Cymbol, SymbExpr

class Potential1DTEPDSymbExpr(SymbExpr):
      """Single variable one-dimensional potential that can be used to demonstrate the 
      interaction between the thermal and mechanical loading. 
      """
      # %%
      E_T = Cymbol(r'E_{\mathrm{T}}', codename='E_T_', real=True, nonnegative=True)
      gamma_T = Cymbol(r'\gamma_{\mathrm{T}}', codename='gamma_T_', real=True)
      X_0 = Cymbol(r'X_0', codename='X_0_', real=True)
      K_T = Cymbol(r'K_{\mathrm{T}}', codename='K_T_', real=True)
      S_T = Cymbol(r'S_{\mathrm{T}}', codename='S_T_', real=True, nonnegative=True)
      r_T = Cymbol(r'r_{\mathrm{T}}', codename='r_T_', real=True, nonnegative=True)
      c_T = Cymbol(r'c_{\mathrm{T}}', codename='c_T_', real=True, nonnegative=True)
      eta_T = Cymbol(r'\eta_{\mathrm{T}}', codename='eta_T_', real=True, nonnegative=True)
      # temperature 
      C_v = Cymbol(r'C_{\mathrm{v}}', codename='C_v_', real=True, nonnegative=True)
      T_0 = Cymbol(r'\vartheta_0', codename='T_0_', real=True, nonnegative=True)
      alpha_therm = Cymbol(r'\alpha_\vartheta', codename='alpha_therm_', real=True, nonnegative=True)
      beta = Cymbol(r'\beta', codename='beta_', real=True, nonnegative=True)

      # %%
      f_s = Cymbol(r'f_\mathrm{T}', codename='f_s_')

      # %%
      mparams = (E_T, gamma_T, X_0, K_T, S_T, f_s, c_T, r_T, eta_T, C_v, T_0, alpha_therm, beta)
      mparams

      # %% [markdown]
      # ## External state variables

      # %%
      u_T = Cymbol(r'\varepsilon', codename='eps_', real=True)
      u_a = sp.Matrix([u_T])
      sig_T = Cymbol(r'\sigma', codename='sig_', real=True)
      sig_a = sp.Matrix([sig_T])
      sig_a

      # %%
      T = Cymbol(r'\vartheta', codename='T_', real=True)
      Gamma = sp.exp(-beta * (T - T_0))
      Gamma

      # %% [markdown]
      # ## Internal state variables

      # %%
      u_p_T = Cymbol(r'\varepsilon^\mathrm{p}', codename='eps_p_', real=True)
      u_p_a = sp.Matrix([u_p_T])
      sig_p_T = Cymbol(r'\sigma^\mathrm{p}', codename='sig_p_', real=True)
      sig_p_a = sp.Matrix([sig_p_T])

      # %%
      omega_T = Cymbol(r'\omega_\mathrm{T}', codename='omega_T_', real=True)
      omega_ab = sp.Matrix([[omega_T]])
      omega_a = sp.Matrix([omega_T])
      Y_T = Cymbol(r'Y_\mathrm{T}', codename='Y_T_', real=True)
      Y_a = sp.Matrix([Y_T])

      # %%
      z_T = Cymbol(r'z_\mathrm{T}', codename='z_T_', real=True, nonnegative=True)
      z_a = sp.Matrix([z_T])
      K_ab = sp.Matrix([[K_T]])
      Z_T = Cymbol(r'Z_\mathrm{T}', codename='Z_T_', real=True, nonnegative=True)
      Z_a = sp.Matrix([Z_T])

      # %%
      alpha_T = Cymbol(r'\alpha_\mathrm{T}', codename='alpha_T_', real=True, nonnegative=True)
      gamma_ab = sp.Matrix([[gamma_T]])
      alpha_a = sp.Matrix([alpha_T])
      X_T = Cymbol(r'X_\mathrm{T}', codename='X_T_', real=True, nonnegative=True)
      X_a = sp.Matrix([X_T])

      # %% [markdown]
      # ## Free energy potential

      # %%
      E_ab = sp.Matrix([[E_T]])
      u_el_a = u_a - u_p_a
      E_eff_ab = (sp.eye(1) - omega_ab) * E_ab
      E_eff_ab

      # %%
      U_T_ = ( (1 - omega_T) * E_T * alpha_therm * (T - T_0) * (u_T - u_p_T) )
      U_e_ = sp.Rational(1,2) * (u_el_a.T * E_eff_ab * u_el_a)[0]
      U_p_ = sp.Rational(1,2) * (z_a.T * K_ab * z_a + alpha_a.T * gamma_ab * alpha_a)[0]
      TS_ = C_v * (T - T_0) **2 / (2 * T_0)
      F_ = U_e_ + U_p_ + - TS_
      F_ = U_e_ + U_p_ - U_T_ - TS_
      F_

      # %% [markdown]
      # ## Dissipation potential

      # %%
      sig_eff_T = sp.Function(r'\sigma^{\mathrm{eff}}_{\mathrm{T}}')(sig_p_T, omega_T)
      q_T = sp.Function(r'q_Tx')(sig_eff_T,X_T)
      norm_q_T = sp.sqrt(q_T*q_T)
      subs_q_T = {q_T: ((sig_eff_T  - X_0) - X_T)}
      subs_sig_eff = {sig_eff_T: sig_p_T / (1-omega_T) }
      y = Cymbol(r'y')
      f_s = Cymbol(r'f_s_')
      f_solved_ = sp.sqrt(y**2) - f_s
      f_ = (f_solved_
            .subs({y: norm_q_T})
            .subs(subs_q_T)
            .subs(subs_sig_eff)
            .subs(f_s,((f_s + Z_T) * Gamma ))
            )

      # %%
      f_

      # %%
      phi_T = (1 - omega_T)**c_T * S_T / (r_T+1) * (Y_T / S_T)**(r_T+1)
      phi_ext_ = phi_T
      phi_ext_

      # %%
      t_relax_T_ = eta_T / (E_T + K_T + gamma_T)
      t_relax_ = sp.Matrix([
                        t_relax_T_,
                        t_relax_T_,
                        t_relax_T_,
                        ] 
                  )

      # %%
      Eps_vars = (u_p_a, z_a, alpha_a, omega_a)

      # %%
      Sig_vars = (sig_p_a, Z_a, X_a, Y_a)
