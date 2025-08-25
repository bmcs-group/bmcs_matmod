"""
Example GSMSymbDef Models - Phase 1

This module contains concrete implementations of the GSMSymbDef base class,
providing specific thermodynamic model definitions for common material behaviors.

Example Models:
- GSM1D_ED: 1D elastic-damage model
- GSM1D_VE: 1D viscoelastic model
"""

import sympy as sp
from bmcs_matmod.gsm_lagrange.core2.gsm_symb_def import GSMSymbDef
from bmcs_matmod.gsm_lagrange.core.gsm_vars import Scalar, Vector


class GSM1D_ED(GSMSymbDef):
    """
    1D Elastic-Damage Model - Pure Symbolic Definition
    
    This model defines a one-dimensional elastic material with damage evolution.
    Following VARIABLE_NAMING.md conventions for damage variables.
    
    Variables:
    - ε: external strain
    - ω: damage variable (0=undamaged, 1=fully damaged)
    
    Thermodynamic Forces:
    - σ: external stress = ∂F/∂ε  
    - Y: damage driving force = -∂F/∂ω (energy release rate)
    
    Parameters:
    - E: elastic modulus
    - κ: damage evolution parameter
    """
    
    # Define symbolic variables with proper LaTeX and codenames
    eps = Scalar(r'\varepsilon', codename='eps', real=True)                    # External strain
    sig = Scalar(r'\sigma', codename='sig', real=True)                        # External stress
    omega = Scalar(r'\omega', codename='omega', real=True, nonnegative=True)   # Damage variable
    
    # Thermodynamic forces (conjugate variables)
    Y = Scalar(r'Y', codename='Y', real=True)                                 # Damage driving force
    
    # Material parameters
    E = Scalar(r'E', codename='E', positive=True, real=True)                  # Elastic modulus
    kappa = Scalar(r'\kappa', codename='kappa', positive=True, real=True)     # Damage parameter
    
    # GSMSymbDef variable assignments
    eps_vars = (eps,)           # External strain variables
    sig_vars = (sig,)           # External stress variables
    Eps_vars = (omega,)         # Internal damage variable
    Sig_vars = (Y,)             # Thermodynamic forces
    m_params = (E, kappa)       # Material parameters
    
    # Thermodynamic potentials
    F_expr = sp.Rational(1,2) * E * (1 - omega) * eps**2  # Free energy with damage
    f_expr = sp.Rational(1,2) * E * eps**2 - kappa        # Damage criterion (f ≤ 0)
    
    # Sign convention: damage force drives damage growth (negative sign)
    Sig_signs = (-1,)
    
    # Variable codenames for display (inherited from Scalar definitions)
    eps_codenames = {eps: eps.codename}
    sig_codenames = {sig: sig.codename}
    Eps_codenames = {omega: omega.codename}
    Sig_codenames = {Y: Y.codename}
    param_codenames = {E: E.codename, kappa: kappa.codename}


class GSM1D_VE(GSMSymbDef):
    """
    1D Viscoelastic Model - Pure Symbolic Definition
    
    This model defines a one-dimensional viscoelastic material with internal
    strain variables representing viscous deformation.
    Following VARIABLE_NAMING.md conventions for viscoelastic variables.
    
    Variables:
    - ε: total strain
    - ε_v: viscous strain
    
    Thermodynamic Forces:
    - σ: external stress = ∂F/∂ε
    - σ_v: viscous stress = -∂F/∂ε_v (opposes viscous flow)
    
    Parameters:
    - E: elastic modulus
    - η: viscosity parameter
    """
    
    # Define symbolic variables with proper LaTeX and codenames
    eps = Scalar(r'\varepsilon', codename='eps', real=True)                    # Total strain
    sig = Scalar(r'\sigma', codename='sig', real=True)                        # External stress
    eps_v = Scalar(r'\varepsilon^\mathrm{v}', codename='eps_v', real=True)     # Viscous strain
    
    # Thermodynamic forces (conjugate variables)
    sig_v = Scalar(r'\sigma^\mathrm{v}', codename='sig_v', real=True)          # Viscous stress
    
    # Material parameters
    E = Scalar(r'E', codename='E', positive=True, real=True)                  # Elastic modulus
    eta = Scalar(r'\eta', codename='eta', positive=True, real=True)           # Viscosity
    
    # GSMSymbDef variable assignments
    eps_vars = (eps,)           # External strain variables
    sig_vars = (sig,)           # External stress variables
    Eps_vars = (eps_v,)         # Internal viscous strain variables
    Sig_vars = (sig_v,)         # Thermodynamic forces
    m_params = (E, eta)         # Material parameters
    
    # Thermodynamic potentials
    F_expr = sp.Rational(1,2) * E * (eps - eps_v)**2          # Elastic energy in spring
    f_expr = sp.Rational(1,2) * eta * eps_v**2                # Viscous dissipation (always ≤ 0)
    
    # Sign convention: viscous stress opposes viscous flow (negative sign)
    Sig_signs = (-1,)
    
    # Variable codenames for display (inherited from Scalar definitions)
    eps_codenames = {eps: eps.codename}
    sig_codenames = {sig: sig.codename}
    Eps_codenames = {eps_v: eps_v.codename}
    Sig_codenames = {sig_v: sig_v.codename}
    param_codenames = {E: E.codename, eta: eta.codename}


class GSM1D_EPD(GSMSymbDef):
    """
    1D Elastic-Plastic-Damage Model - Pure Symbolic Definition
    
    This model combines elastic deformation with plasticity and damage evolution.
    Following VARIABLE_NAMING.md conventions for EPD variables.
    
    Variables:
    - ε: total strain
    - ε_p: plastic strain  
    - ω: damage variable (0=undamaged, 1=fully damaged)
    - z: isotropic hardening variable
    - α: kinematic hardening variable
    
    Thermodynamic Forces:
    - σ: external stress = ∂F/∂ε
    - σ_p: plastic stress = -∂F/∂ε_p (opposes plastic flow)
    - Y: damage driving force = -∂F/∂ω (energy release rate)
    - Z: isotropic hardening force = ∂F/∂z (strengthens material)
    - X: kinematic hardening force = ∂F/∂α (translates yield surface)
    
    Parameters:
    - E: elastic modulus
    - K: isotropic hardening modulus
    - γ: kinematic hardening modulus
    - S: damage evolution parameter
    - c: damage evolution exponent
    - r: damage evolution exponent
    - f_c: yield stress
    """
    
    # Define symbolic variables with proper LaTeX and codenames
    eps = Scalar(r'\varepsilon', codename='eps', real=True)                        # Total strain
    sig = Scalar(r'\sigma', codename='sig', real=True)                            # External stress
    eps_p = Scalar(r'\varepsilon^\mathrm{p}', codename='eps_p', real=True)         # Plastic strain
    omega = Scalar(r'\omega', codename='omega', real=True, nonnegative=True)       # Damage variable
    z = Scalar(r'z', codename='z', real=True, nonnegative=True)                   # Isotropic hardening variable
    alpha = Scalar(r'\alpha', codename='alpha', real=True, nonnegative=True)      # Kinematic hardening variable
    
    # Thermodynamic forces (conjugate variables)
    sig_p = Scalar(r'\sigma^\mathrm{p}', codename='sig_p', real=True)              # Plastic stress
    Y = Scalar(r'Y', codename='Y', real=True)                                     # Damage driving force
    Z = Scalar(r'Z', codename='Z', real=True, nonnegative=True)                   # Isotropic hardening force
    X = Scalar(r'X', codename='X', real=True, nonnegative=True)                   # Kinematic hardening force
    
    # Material parameters
    E = Scalar(r'E', codename='E', positive=True, real=True)                      # Elastic modulus
    K = Scalar(r'K', codename='K', positive=True, real=True)                      # Isotropic hardening modulus
    gamma = Scalar(r'\gamma', codename='gamma', positive=True, real=True)         # Kinematic hardening modulus
    S = Scalar(r'S', codename='S', positive=True, real=True)                      # Damage evolution parameter
    c = Scalar(r'c', codename='c', positive=True, real=True)                      # Damage evolution exponent
    r = Scalar(r'r', codename='r', positive=True, real=True)                      # Damage evolution exponent
    f_c = Scalar(r'f_\mathrm{c}', codename='f_c', positive=True, real=True)       # Yield stress
    
    # GSMSymbDef variable assignments
    eps_vars = (eps,)                           # External strain variables
    sig_vars = (sig,)                           # External stress variables
    Eps_vars = (eps_p, omega, z, alpha)        # Internal variables (plastic strain, damage, hardening)
    Sig_vars = (sig_p, Y, Z, X)                # Thermodynamic forces
    m_params = (E, K, gamma, S, c, r, f_c)     # Material parameters
    
    # Elastic strain
    eps_e = eps - eps_p
    
    # Thermodynamic potentials
    # Free energy: elastic energy with damage + hardening energies
    U_e = sp.Rational(1,2) * (1 - omega) * E * eps_e**2    # Elastic energy with damage
    U_p = sp.Rational(1,2) * K * z**2 + sp.Rational(1,2) * gamma * alpha**2  # Hardening energies
    F_expr = U_e + U_p
    
    # Dissipation potential: plastic yield function
    # Following reference: sqrt((sig_p/(1-omega) - X)^2) - (f_c + Z)
    f_expr = sp.sqrt((sig_p/(1-omega) - X)**2) - (f_c + Z)
    
    # External potential: damage evolution function
    # Following reference: (1-omega)^c * (S/(r+1)) * (Y/S)^(r+1)
    phi_ext_expr = (1 - omega)**c * (S/(r+1)) * (Y / S)**(r+1)
    
    # Sign conventions (following VARIABLE_NAMING.md):
    # eps_p → sig_p: plastic stress opposes plastic flow (-1)
    # omega → Y: damage force drives damage growth (-1)
    # z → Z: isotropic hardening strengthens material (+1)
    # alpha → X: kinematic hardening contributes to yield surface (+1)
    Sig_signs = (-1, -1, 1, 1)
    
    # Variable codenames for display (inherited from Scalar definitions)
    eps_codenames = {eps: eps.codename}
    sig_codenames = {sig: sig.codename}
    Eps_codenames = {eps_p: eps_p.codename, omega: omega.codename, z: z.codename, alpha: alpha.codename}
    Sig_codenames = {sig_p: sig_p.codename, Y: Y.codename, Z: Z.codename, X: X.codename}
    param_codenames = {
        E: E.codename, 
        K: K.codename,
        gamma: gamma.codename,
        S: S.codename,
        c: c.codename,
        r: r.codename,
        f_c: f_c.codename
    }
