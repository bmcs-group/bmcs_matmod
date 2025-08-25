"""
3D Interface GSM Models

This module contains GSM models for interface mechanics with multiple strain components.
Suitable for modeling cohesive zones, interfaces, and contact problems.

Example Models:
- GSM3D_IFC_ED: 3D interface elastic-damage model with isotropic damage
"""

import sympy as sp
from bmcs_matmod.gsm_lagrange.core2.gsm_symb_def import GSMSymbDef
from bmcs_matmod.gsm_lagrange.core.gsm_vars import Scalar, Vector


class GSM3D_IFC_ED(GSMSymbDef):
    """
    3D Interface Elastic-Damage Model - Pure Symbolic Definition
    
    This model defines a three-dimensional interface material with isotropic damage evolution.
    Suitable for cohesive zone modeling and interface degradation analysis.
    Following VARIABLE_NAMING.md conventions for damage variables.
    
    Variables:
    - ε_N: normal strain component
    - ε_Tx: tangential strain component (x-direction)  
    - ε_Ty: tangential strain component (y-direction)
    - ω: isotropic damage variable (0=undamaged, 1=fully damaged)
    
    Thermodynamic Forces:
    - σ_N: normal stress = ∂F/∂ε_N
    - σ_Tx: tangential stress (x) = ∂F/∂ε_Tx
    - σ_Ty: tangential stress (y) = ∂F/∂ε_Ty
    - Y: damage driving force = -∂F/∂ω (energy release rate)
    
    Parameters:
    - E_N: normal stiffness
    - E_T: tangential stiffness
    - κ: damage evolution parameter
    """
    
    # Define symbolic strain variables with proper LaTeX and codenames
    eps_N = Scalar(r'\varepsilon_N', codename='eps_N', real=True)               # Normal strain
    eps_Tx = Scalar(r'\varepsilon_{Tx}', codename='eps_Tx', real=True)          # Tangential strain (x)
    eps_Ty = Scalar(r'\varepsilon_{Ty}', codename='eps_Ty', real=True)          # Tangential strain (y)
    
    # Define symbolic stress variables with proper LaTeX and codenames
    sig_N = Scalar(r'\sigma_N', codename='sig_N', real=True)                    # Normal stress
    sig_Tx = Scalar(r'\sigma_{Tx}', codename='sig_Tx', real=True)               # Tangential stress (x)
    sig_Ty = Scalar(r'\sigma_{Ty}', codename='sig_Ty', real=True)               # Tangential stress (y)
    
    # Internal damage variable
    omega = Scalar(r'\omega', codename='omega', real=True, nonnegative=True)    # Isotropic damage
    
    # Thermodynamic forces (conjugate variables)
    Y = Scalar(r'Y', codename='Y', real=True)                                  # Damage driving force
    
    # Material parameters
    E_N = Scalar(r'E_N', codename='E_N', positive=True, real=True)             # Normal stiffness
    E_T = Scalar(r'E_T', codename='E_T', positive=True, real=True)             # Tangential stiffness
    kappa = Scalar(r'\kappa', codename='kappa', positive=True, real=True)      # Damage parameter
    
    # GSMSymbDef variable assignments
    eps_vars = (eps_N, eps_Tx, eps_Ty)     # External strain variables (3 components)
    sig_vars = (sig_N, sig_Tx, sig_Ty)     # External stress variables (3 components)
    Eps_vars = (omega,)                     # Internal damage variable
    Sig_vars = (Y,)                         # Thermodynamic forces
    m_params = (E_N, E_T, kappa)           # Material parameters
    
    # Thermodynamic potentials
    # Free energy: elastic energy with isotropic damage affecting all components
    F_expr = (sp.Rational(1,2) * (1 - omega) * 
              (E_N * eps_N**2 + E_T * (eps_Tx**2 + eps_Ty**2)))
    
    # Damage threshold: equivalent strain-based criterion
    # Using von Mises-like equivalent strain for interface
    eps_eq = sp.sqrt(eps_N**2 + eps_Tx**2 + eps_Ty**2)
    f_expr = eps_eq - kappa                # Damage criterion (f ≤ 0)
    
    # Sign convention: damage force drives damage growth (negative sign)
    Sig_signs = (-1,)
    
    # Variable codenames for display (inherited from Scalar definitions)
    eps_codenames = {
        eps_N: eps_N.codename,
        eps_Tx: eps_Tx.codename, 
        eps_Ty: eps_Ty.codename
    }
    sig_codenames = {
        sig_N: sig_N.codename,
        sig_Tx: sig_Tx.codename,
        sig_Ty: sig_Ty.codename
    }
    Eps_codenames = {omega: omega.codename}
    Sig_codenames = {Y: Y.codename}
    param_codenames = {
        E_N: E_N.codename, 
        E_T: E_T.codename, 
        kappa: kappa.codename
    }
