# # Single variable thermo-elasto-plastic damage

import sympy as sp
from bmcs_utils.api import Cymbol, SymbExpr
import traits.api as tr

import os

class Potential1D_T_E_VP_D_LIKH_SymbExpr(SymbExpr):
      """
      Single variable one-dimensional potential that can be used to demonstrate the 

      Attributes
      ----------
      E : Cymbol
            Young's modulus, nonnegative.
      gamma : Cymbol
            Linear hardening parameter.
      X_0 : Cymbol
            Initial kinematic hardening variable.
      K : Cymbol
            Isotropic hardening parameter.
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
      E = sp.Symbol(r'E', real=True, nonnegative=True)
      gamma = sp.Symbol(r'\gamma', real=True)
      X_0 = sp.Symbol(r'X_0', real=True)
      K = sp.Symbol(r'K', real=True)
      z_0 = sp.Symbol(r'z_0', real=True, nonnegative=True)
      S = sp.Symbol(r'S', real=True, nonnegative=True)
      r = sp.Symbol(r'r', real=True, nonnegative=True)
      c = sp.Symbol(r'c', real=True, nonnegative=True)
      eta = sp.Symbol(r'\eta', real=True, nonnegative=True)
      # temperature 
      C_v = sp.Symbol(r'C_{\mathrm{v}}', real=True, nonnegative=True)
      T_0 = sp.Symbol(r'\vartheta_0', real=True, nonnegative=True)
      alpha_therm = sp.Symbol(r'\alpha_\vartheta', real=True, nonnegative=True)
      beta = sp.Symbol(r'\beta', real=True, nonnegative=True)
      T = Cymbol(r'\vartheta', codename='T_', real=True)

      f_s = sp.Symbol(r'f_\mathrm{c}')
      mparams = (E, gamma, X_0, K, z_0, S, f_s, c, r, eta, C_v, T_0, alpha_therm, beta)

      m_param_codenames = {gamma: 'gamma',
                           beta: 'beta',
                           X_0: 'X_0',
                           K: 'K',
                           z_0: 'z_0',
                           S: 'S',
                           c: 'c',
                           r: 'r',
                           eta: 'eta',
                           C_v: 'C_v',
                           alpha_therm: 'alpha_therm', 
                           f_s: 'f_c',
                           T_0: 'T_0'}

      eps = Cymbol(r'\varepsilon', codename='eps_', real=True)
      eps_a = sp.Matrix([eps])
      sig = Cymbol(r'\sigma', codename='sig_', real=True)
      sig_a = sp.Matrix([sig])

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
      K_ab = sp.Matrix([[K]])
      Z = Cymbol(r'Z', codename='Z_', real=True, nonnegative=True)
      Z_a = sp.Matrix([Z])

      alpha = Cymbol(r'\alpha', codename='alpha_', real=True, nonnegative=True)
      gamma_ab = sp.Matrix([[gamma]])
      alpha_a = sp.Matrix([alpha])
      X = Cymbol(r'X', codename='X_', real=True, nonnegative=True)
      X_a = sp.Matrix([X])

      E_ab = sp.Matrix([[E]])
      eps_el_a = eps_a - eps_p_a
      E_eff_ab = (sp.eye(1) - omega_ab) * E_ab

      Gamma = sp.exp(-beta * (T - T_0))

      F_ = tr.Property()

      @tr.cached_property
      def _get_F_(self):
            X_alpha = self.gamma * self.alpha
            int_X_alpha = sp.integrate(X_alpha, self.alpha)

            Z_z = self.K * self.z
            int_Z_z = sp.integrate(Z_z, self.z)

            U_e_ = sp.Rational(1, 2) * (self.eps_el_a.T * self.E_eff_ab * self.eps_el_a)[0]
            U_p_ = int_Z_z + int_X_alpha
            TS_ = self.C_v * (self.T - self.T_0) ** 2 / (2 * self.T_0)
            return U_e_ + U_p_ - TS_

      sig_eff = sp.Function(r'\sigma^{\mathrm{eff}}')(sig_p, omega)
      q = sp.Function(r'q')(sig_eff, X)
      norm_q = sp.sqrt(q * q)
      subs_q = {q: ((sig_eff - X_0) - X)}
      subs_sig_eff = {sig_eff: sig_p / (1 - omega)}
      y = Cymbol(r'y')
      f_solved_ = sp.sqrt(y ** 2) - f_s
      f_ = (f_solved_
              .subs({y: norm_q})
              .subs(subs_q)
              .subs(subs_sig_eff)
              .subs(f_s, ((f_s + Z) * Gamma)))

      dot_omega = (1 - omega) ** c * (Y / S) ** r
      int_dot_omega = sp.integrate(dot_omega, Y)

      phi_ext_ = int_dot_omega

      t_relax_ = eta / (E + K + gamma)
      t_relax_ = sp.Matrix([
            t_relax_,
            t_relax_,
            t_relax_,
      ])

      Eps_vars = (eps_p_a, z_a, alpha_a, omega_a)

      Sig_vars = (sig_p_a, Z_a, X_a, Y_a)

      Sig_signs = (-1, 1, 1, -1)
