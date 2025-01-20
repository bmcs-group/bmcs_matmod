# # Single variable thermo-elasto-plastic damage

import sympy as sp
from bmcs_utils.api import Cymbol, SymbExpr
import traits.api as tr

import os

class Potential1D_T_E_VP_LIH_SymbExpr(SymbExpr):
      """Single variable one-dimensional potential that can be used to demonstrate the 
      interaction between the thermal and mechanical loading. 
      """
      # %%
      E = Cymbol(r'E', codename='E_', real=True, nonnegative=True)
      K_lin = Cymbol(r'K^\mathrm{lin}', codename='K_lin_', nonnegative=True)
      eta = Cymbol(r'\eta', codename='eta_', real=True, nonnegative=True)
      # temperature 
      C_v = Cymbol(r'C_{\mathrm{v}}', codename='C_v_', real=True, nonnegative=True)
      T_0 = Cymbol(r'\vartheta_0', codename='T_0_', real=True, nonnegative=True)

      # %%
      f_c = Cymbol(r'f_\mathrm{c}', codename='f_c_')

      # %%
      mparams = (E, K_lin, f_c, eta, C_v, T_0)
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
      z = Cymbol(r'z', codename='z_', real=True, nonnegative=True)
      z_a = sp.Matrix([z])
      K_ab = sp.Matrix([[K_lin]])
      Z = Cymbol(r'Z', codename='Z_', real=True, nonnegative=True)
      Z_a = sp.Matrix([Z])

      # %% [markdown]
      # ## Free energy potential

      # %%
      E_ab = sp.Matrix([[E]])
      eps_el_a = eps_a - eps_p_a
      E_eff_ab = E_ab
      E_eff_ab

      F_ = tr.Property()
      @tr.cached_property
      def _get_F_(self):

            Z_z = self.K_lin * self.z
            int_Z_z = sp.integrate(Z_z, self.z)

            #k = 1
            # %%
            U_e_ = sp.Rational(1,2) * (self.eps_el_a.T * self.E_eff_ab * self.eps_el_a)[0]
            U_p_ =  int_Z_z
            TS_ = self.C_v * (self.T - self.T_0) **2 / (2 * self.T_0)
            return U_e_ + U_p_ - TS_

      # %% [markdown]
      # ## Dissipation potential

      # %%
      sig_eff = sp.Function(r'\sigma^{\mathrm{eff}}')(sig_p)
      q = sp.Function(r'q')(sig_eff)
      norm_q = sp.sqrt(q*q)
      subs_q = {q: sig_eff}
      subs_sig_eff = {sig_eff: sig_p }
      y = Cymbol(r'y')
      f_solved_ = sp.sqrt(y**2) - f_c
      f_ = (f_solved_
            .subs({y: norm_q})
            .subs(subs_q)
            .subs(subs_sig_eff)
            .subs(f_c,((f_c + Z)))
            )

      # %%
      f_

      # %%
      phi_ext_ = 0

      # %%
      t_relax_ = eta / (E + K_lin)
      t_relax_ = sp.Matrix([
                        t_relax_,
                        t_relax_,
                        t_relax_,
                        ] 
                  )

      # %%
      Eps_vars = (eps_p_a, z_a)

      # %%
      Sig_vars = (sig_p_a, Z_a)

      Sig_signs =  (-1, 1)
