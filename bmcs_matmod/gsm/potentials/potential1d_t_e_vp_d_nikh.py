# # Single variable thermo-elasto-plastic damage

import sympy as sp
from bmcs_utils.api import Cymbol, SymbExpr
import traits.api as tr

import os

class Potential1D_T_E_VP_D_NIKH_SymbExpr(SymbExpr):
      """
      Single variable one-dimensional potential that can be used to demonstrate the 

      Attributes
      ----------
      E : Cymbol
            Young's modulus, nonnegative.
      gamma_lin : Cymbol
            Linear hardening parameter.
      gamma_exp : Cymbol
            Exponential hardening parameter, positive.
      alpha_0 : Cymbol
            Initial back stress, nonnegative.
      X_0 : Cymbol
            Initial kinematic hardening variable.
      K_lin : Cymbol
            Linear isotropic hardening parameter.
      k_exp : Cymbol
            Exponential isotropic hardening parameter, positive.
      z_0 : Cymbol
            Initial isotropic hardening variable, nonnegative.
      S : Cymbol
            Saturation value for damage, nonnegative.
      r : Cymbol
            Damage exponent, nonnegative.
      c : Cymbol
            Damage parameter, nonnegative.
      eta : Cymbol
            Viscosity parameter, nonnegative.
      C_v : Cymbol
            Heat capacity at constant volume, nonnegative.
      T_0 : Cymbol
            Reference temperature, nonnegative.
      alpha_therm : Cymbol
            Thermal expansion coefficient, nonnegative.
      beta : Cymbol
            Thermal softening parameter, nonnegative.
      f_s : Cymbol
            Yield function.
      mparams : tuple
            Material parameters tuple.
      eps : Cymbol
            Strain.
      eps_a : sp.Matrix
            Strain matrix.
      sig : Cymbol
            Stress.
      sig_a : sp.Matrix
            Stress matrix.
      T : Cymbol
            Temperature.
      eps_p : Cymbol
            Plastic strain.
      eps_p_a : sp.Matrix
            Plastic strain matrix.
      sig_p : Cymbol
            Plastic stress.
      sig_p_a : sp.Matrix
            Plastic stress matrix.
      omega : Cymbol
            Damage variable.
      omega_ab : sp.Matrix
            Damage matrix.
      omega_a : sp.Matrix
            Damage vector.
      Y : Cymbol
            Energy release rate.
      Y_a : sp.Matrix
            Energy release rate vector.
      z : Cymbol
            Isotropic hardening variable, nonnegative.
      z_a : sp.Matrix
            Isotropic hardening vector.
      K_ab : sp.Matrix
            Isotropic hardening matrix.
      Z : Cymbol
            Isotropic hardening variable, nonnegative.
      Z_a : sp.Matrix
            Isotropic hardening vector.
      alpha : Cymbol
            Kinematic hardening variable, nonnegative.
      gamma_ab : sp.Matrix
            Kinematic hardening matrix.
      alpha_a : sp.Matrix
            Kinematic hardening vector.
      X : Cymbol
            Kinematic hardening variable, nonnegative.
      X_a : sp.Matrix
            Kinematic hardening vector.
      E_ab : sp.Matrix
            Elastic modulus matrix.
      eps_el_a : sp.Matrix
            Elastic strain vector.
      E_eff_ab : sp.Matrix
            Effective elastic modulus matrix.
      Gamma : sp.Expr
            Thermal softening factor.
      F_ : tr.Property
            Free energy potential.
      sig_eff : sp.Function
            Effective stress function.
      q : sp.Function
            Equivalent stress function.
      norm_q : sp.Expr
            Norm of the equivalent stress.
      subs_q : dict
            Substitution dictionary for equivalent stress.
      subs_sig_eff : dict
            Substitution dictionary for effective stress.
      y : Cymbol
            Auxiliary variable.
      f_solved_ : sp.Expr
            Solved yield function.
      f_ : sp.Expr
            Yield function with substitutions.
      dot_omega : sp.Expr
            Damage evolution rate.
      int_dot_omega : sp.Expr
            Integrated damage evolution rate.
      phi_ext_ : sp.Expr
            External dissipation potential.
      t_relax_ : sp.Matrix
            Relaxation time matrix.
      Eps_vars : tuple
            Tuple of internal state variables (plastic strain, isotropic hardening, kinematic hardening, damage).
      Sig_vars : tuple
            Tuple of conjugate forces (plastic stress, isotropic hardening force, kinematic hardening force, energy release rate).
      Sig_signs : tuple
            Tuple of signs for the conjugate forces.
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

      T = Cymbol(r'\vartheta', codename='T_', real=True)


      # %%
      eps_p = Cymbol(r'\varepsilon^\mathrm{p}', codename='eps_p_', real=True)
      eps_p_a = sp.Matrix([eps_p])
      sig_p = Cymbol(r'\sigma^\mathrm{p}', codename='sig_p_', real=True)
      sig_p_a = sp.Matrix([sig_p])

      omega = Cymbol(r'\omega', codename='omega_', real=True)
      omega_ab = sp.Matrix([[omega]])
      omega_a = sp.Matrix([omega])
      Y = Cymbol(r'Y', codename='Y_', real=True)
      Y_a = sp.Matrix([Y])

      z = Cymbol(r'z', codename='z_', real=True, nonnegative=True)
      z_a = sp.Matrix([z])
      K_ab = sp.Matrix([[K_lin]])
      Z = Cymbol(r'Z', codename='Z_', real=True, nonnegative=True)
      Z_a = sp.Matrix([Z])

      alpha = Cymbol(r'\alpha', codename='alpha_', real=True, nonnegative=True)
      gamma_ab = sp.Matrix([[gamma_lin]])
      alpha_a = sp.Matrix([alpha])
      X = Cymbol(r'X', codename='X_', real=True, nonnegative=True)
      X_a = sp.Matrix([X])

      # ## Free energy potential

      E_ab = sp.Matrix([[E]])
      eps_el_a = eps_a - eps_p_a
      E_eff_ab = (sp.eye(1) - omega_ab) * E_ab

      Gamma = sp.exp(-beta * (T - T_0))

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
            U_e_ = sp.Rational(1,2) * (self.eps_el_a.T * self.E_eff_ab * self.eps_el_a)[0]
            U_p_ =  int_Z_z + int_X_alpha
            # TS_ = self.C_v * (self.T - self.T_0) **2 / (2 * self.T_0)
            TS_ = self.C_v * (self.T - self.T * sp.log(self.T / self.T_0))
            return U_e_ + U_p_ - TS_

      # ## Dissipation potential

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


      dot_omega = (1 - omega)**c * (Y / S)**(r) 
      int_dot_omega = sp.integrate(dot_omega, Y) 

      phi_ext_ = int_dot_omega

      t_relax_ = eta / (E + K_lin + gamma_lin)
      t_relax_ = sp.Matrix([
                        t_relax_,
                        t_relax_,
                        t_relax_,
                        ] 
                  )

      Eps_vars = (eps_p_a, z_a, alpha_a, omega_a)

      Sig_vars = (sig_p_a, Z_a, X_a, Y_a)

      Sig_signs =  (-1, 1, 1, -1)
