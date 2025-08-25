"""
GSM Engine - Numerical Execution Layer

This module provides numerical execution capabilities for thermodynamic models
defined using GSMSymbDef. It separates symbolic definitions from numerical 
execution, providing efficient lambdified functions for time-stepping algorithms.

Key Features:
- Takes GSMSymbDef instances as input for symbolic expressions
- Provides lambdified numerical methods for time integration
- Implements the maximum dissipation time-stepping algorithm
- Uses disk caching for expensive lambdification operations
- Handles symbolic zeros and ones for proper numpy broadcasting
"""

import numpy as np
import numpy.typing as npt
import sympy as sp
from typing import Tuple, Any, Callable, Union
from functools import cached_property

from bmcs_matmod.gsm_lagrange.core2.gsm_symb_def import GSMSymbDef
from bmcs_matmod.gsm_lagrange.core2.disk_cached_property import disk_cached_property


def get_dirac_delta(x: Any, x_0: float = 0) -> int:
    """Dirac delta function replacement for numpy."""
    return 0


# Numpy modules for lambdification
numpy_dirac = [{'DiracDelta': get_dirac_delta}, 'numpy']


class GSMEngine:
    """
    Numerical execution engine for GSM models.
    
    This class takes a GSMSymbDef instance and provides efficient numerical
    methods for time integration and state evolution using lambdified SymPy
    expressions.
    
    Args:
        symb_def: GSMSymbDef instance containing symbolic expressions
        k_max: Maximum iterations for Newton-Raphson solver (default: 10)
        tol: Convergence tolerance (default: 1e-8)
        cache_dir: Directory for disk caching (default: ".gsm_cache")
    """
    
    def __init__(self, 
                 symb_def: GSMSymbDef, 
                 k_max: int = 10, 
                 tol: float = 1e-8,
                 cache_dir: str = ".gsm_cache"):
        self.symb_def = symb_def
        self.k_max = k_max
        self.tol = tol
        self.cache_dir = cache_dir
        
        # Symbolic variables for zero/one replacement
        self.Ox = sp.Symbol('O', real=True)
        self.Ix = sp.Symbol('I', real=True)
        
        # Validate the symbolic definition
        validation = symb_def.validate_symbolic_expressions()
        if not validation.is_valid:
            print(f"⚠️ Warning: Issues in symbolic definition:")
            for issue in validation.issues:
                print(f"  - {issue}")
    
    @property
    def name(self) -> str:
        """Get model name for caching purposes."""
        return self.symb_def.__class__.__name__
    
    @property
    def n_eps_a(self) -> int:
        """Number of external strain variables."""
        return len(self.symb_def.eps_a)
    
    @property
    def n_Eps_b(self) -> int:
        """Number of internal strain variables (flattened)."""
        return len(self.symb_def.Eps_b)
    
    @property
    def n_lam_phi(self) -> int:
        """Number of inequality constraint multipliers."""
        lam_phi_list, _, _ = self.symb_def.lam_phi_f_
        return len(lam_phi_list)
    
    @property
    def n_Lam(self) -> int:
        """Number of equality constraint multipliers."""
        return len(self.symb_def.h_k)
    
    @property
    def n_delta_A_c(self) -> int:
        """Total number of augmented unknowns."""
        return len(self.symb_def.delta_A_c_)
    
    def replace_zeros_and_ones_with_symbolic(self, 
                                           dR_dA_: sp.Matrix, 
                                           delta_A: sp.Matrix) -> sp.Matrix:
        """
        Replace zero elements and constant terms with symbolic variables for proper broadcasting.
        
        This method processes the Jacobian matrix to handle numpy broadcasting issues:
        - Zero elements are replaced with symbolic Ox for proper array broadcasting
        - Constant terms (independent of delta_A) are multiplied by Ix
        - Variable terms are left unchanged
        
        Args:
            dR_dA_: Jacobian matrix to process
            delta_A: Vector of unknowns for dependency checking
            
        Returns:
            Processed matrix with symbolic placeholders for zeros and ones
        """
        new_matrix = sp.Matrix(dR_dA_)
        rows, cols = new_matrix.shape
        
        for i in range(rows):
            for j in range(cols):
                if dR_dA_[i, j] == 0:
                    # Replace zeros with symbolic variable
                    new_matrix[i, j] = self.Ox
                else:
                    # Check if element depends on delta_A
                    grad_entry = dR_dA_[i, j].diff(delta_A)[:, 0]
                    ones_vector = sp.Matrix([1 for k in range(len(grad_entry))])
                    depends_on_Eps = (ones_vector.T * grad_entry)[0]
                    
                    if depends_on_Eps == 0:
                        # Constant term - multiply by symbolic one
                        new_matrix[i, j] = dR_dA_[i, j] * self.Ix
                    else:
                        # Variable term - leave unchanged
                        new_matrix[i, j] = dR_dA_[i, j]
        
        return new_matrix
    
    @disk_cached_property(cache_dir=".gsm_cache")
    def _get_sig_Sig_f_R_dR_dsig_n1_lambdified(self) -> Callable[..., Tuple[Any, Any, Any, Any, Any, Any]]:
        """
        Create lambdified function for numerical time-stepping operations including material stiffness.
        
        This property processes the symbolic expressions from GSMSymbDef and creates
        efficient numerical functions for:
        - External stress evaluation (sig_a_n1)
        - Internal thermodynamic forces evaluation (Sig_b_n1)
        - Flow potential evaluation (f_n1) 
        - Residuum vector (R_c_n1)
        - Jacobian matrix with symbolic zeros/ones (dR_dA_cc_n1_OI)
        - Material stiffness matrix (dsig_deps_aa_n1)
        
        Returns:
            Lambdified function for numerical evaluation
        """
        # Get symbolic variables from GSMSymbDef
        delta_eps_a = self.symb_def.delta_eps_a_
        delta_A_c = self.symb_def.delta_A_c_
        
        # Get evaluation expressions
        sig_a_n1 = self.symb_def.sig_a_n1_
        Sig_b_n1 = self.symb_def.Sig_b_n1_
        f_n1 = self.symb_def.f_n1_
        R_c_n1 = self.symb_def.R_c_n1_
        dR_dA_cc_n1 = self.symb_def.dR_dA_cc_n1_
        dsig_deps_aa_n1 = self.symb_def.dsig_deps_aa_n1_
        
        # Apply symbolic replacement for zeros and ones
        dR_dA_n1_OI = self.replace_zeros_and_ones_with_symbolic(dR_dA_cc_n1, delta_A_c)
        
        # Get all free symbols for lambdification arguments
        all_symbols = set()
        for expr in [sig_a_n1, Sig_b_n1, f_n1, R_c_n1, dR_dA_n1_OI, dsig_deps_aa_n1]:
            if hasattr(expr, 'free_symbols'):
                all_symbols.update(expr.free_symbols)
            elif hasattr(expr, 'atoms'):
                all_symbols.update(expr.atoms(sp.Symbol))
        
        # Remove Ox and Ix from symbols (they'll be provided as arguments)
        all_symbols.discard(self.Ox)
        all_symbols.discard(self.Ix)
        
        # Build argument list: state variables, increments, parameters, Ox, Ix
        eps_a = self.symb_def.eps_a
        Eps_b = self.symb_def.Eps_b
        m_params = self.symb_def.m_params
        
        # Get time discretization variables from subs_n1_
        subs_dict = self.symb_def.subs_n1_
        
        # Extract the fundamental variables that appear in substitutions
        eps_n_symbols = []
        delta_eps_symbols = []
        Eps_n_symbols = []
        delta_Eps_symbols = []
        delta_t_symbol = None
        
        # Parse substitution dictionary to get the base symbols
        for original, substituted in subs_dict.items():
            if hasattr(substituted, 'free_symbols'):
                for sym in substituted.free_symbols:
                    if '_n' in str(sym) and sym not in eps_n_symbols + Eps_n_symbols:
                        if any(str(eps_var) in str(sym) for eps_var in eps_a):
                            eps_n_symbols.append(sym)
                        elif any(str(Eps_var) in str(sym) for Eps_var in Eps_b):
                            Eps_n_symbols.append(sym)
                    elif 'Delta' in str(sym) and 'lambda' not in str(sym):
                        if any(str(eps_var) in str(sym) for eps_var in eps_a):
                            if sym not in delta_eps_symbols:
                                delta_eps_symbols.append(sym)
                        elif any(str(Eps_var) in str(sym) for Eps_var in Eps_b):
                            if sym not in delta_Eps_symbols:
                                delta_Eps_symbols.append(sym)
                    elif 'Delta' in str(sym) and 't' in str(sym):
                        delta_t_symbol = sym
        
        # Construct argument tuple
        args_tuple = tuple(eps_n_symbols + delta_eps_symbols + Eps_n_symbols + 
                          delta_Eps_symbols) + (delta_t_symbol,) + (self.Ox, self.Ix) + m_params
        
        # Create lambdified function
        try:
            lambdified_func = sp.lambdify(
                args_tuple,
                (sig_a_n1, Sig_b_n1, f_n1, R_c_n1, dR_dA_n1_OI, dsig_deps_aa_n1),
                modules=numpy_dirac,
                cse=True
            )
            return lambdified_func
        except Exception as e:
            print(f"Error in lambdification: {e}")
            print(f"Arguments: {[str(arg) for arg in args_tuple]}")
            raise
    
    def get_sig_Sig_f_R_dR_dsig_n1(self, 
                                   eps_n: npt.NDArray[np.float64],
                                   d_eps: npt.NDArray[np.float64], 
                                   Eps_b_n: npt.NDArray[np.float64],
                                   d_A_c: npt.NDArray[np.float64],
                                   d_t: float,
                                   *args: float) -> Tuple[npt.NDArray[np.float64],
                                                        npt.NDArray[np.float64], 
                                                        npt.NDArray[np.float64],
                                                        npt.NDArray[np.float64], 
                                                        npt.NDArray[np.float64],
                                                        npt.NDArray[np.float64]]:
        """
        Evaluate external stress, internal forces, flow potential, residuum, Jacobian and material stiffness at time n+1.
        
        Args:
            eps_n: External strain at time n
            d_eps: External strain increment
            Eps_b_n: Internal strains at time n (flattened)
            d_A_c: Augmented increments (internal strains + multipliers)
            d_t: Time increment
            *args: Material parameters
            
        Returns:
            Tuple of (sig_a_n1, Sig_b_n1, f_n1, R_c_n1, dR_dA_cc_n1, dsig_deps_aa_n1)
        """
        # Prepare arrays for evaluation
        eps_n_sp_ = np.moveaxis(np.atleast_1d(eps_n), -1, 0)
        d_eps_sp_ = np.moveaxis(np.atleast_1d(d_eps), -1, 0)
        Eps_b_n_sp_ = np.moveaxis(Eps_b_n, -1, 0)
        d_A_c_sp_ = np.moveaxis(d_A_c, -1, 0)
        
        # Create zero and one arrays for symbolic replacement
        O_ = np.zeros_like(eps_n_sp_)
        I_ = np.ones_like(eps_n_sp_)
        
        # Call lambdified function
        sig_sp_, Sig_sp_, f_sp_, R_sp_, dR_sp_, dsig_deps_sp_ = self._get_sig_Sig_f_R_dR_dsig_n1_lambdified(
            eps_n_sp_, d_eps_sp_, Eps_b_n_sp_, d_A_c_sp_, d_t, O_, I_, *args
        )
        
        # Handle case where flow potential is zero (no inequality constraints)
        if self.symb_def.phi_ == sp.S.Zero:
            f_sp_ = -np.ones_like(eps_n_sp_)
        
        # Reshape results
        sig_sp_ = sig_sp_.reshape(eps_n_sp_.shape)
        Sig_sp_ = Sig_sp_.reshape(Eps_b_n_sp_.shape)
        
        return (np.moveaxis(sig_sp_, 0, -1),
                np.moveaxis(Sig_sp_, 0, -1), 
                np.moveaxis(f_sp_, 0, -1),
                np.moveaxis(R_sp_[:, 0], 0, -1), 
                np.moveaxis(dR_sp_, (0, 1), (-2, -1)),
                np.moveaxis(dsig_deps_sp_, (0, 1), (-2, -1)))
    
    def get_state_n1(self, 
                     eps_n: npt.NDArray[np.float64],
                     d_eps: npt.NDArray[np.float64],
                     d_t: float,
                     Eps_b_n: npt.NDArray[np.float64],
                     *args: float) -> Tuple[npt.NDArray[np.float64],
                                          npt.NDArray[np.float64], 
                                          npt.NDArray[np.float64],
                                          npt.NDArray[np.float64],
                                          npt.NDArray[np.float64], 
                                          npt.NDArray[np.int32]]:
        """
        Calculate the state at time n+1 using maximum dissipation algorithm.
        
        This method implements the iterative Newton-Raphson solution for the
        constrained optimization problem arising from the maximum dissipation
        principle in thermodynamics.
        
        Args:
            eps_n: External strain at time n
            d_eps: External strain increment
            d_t: Time increment
            Eps_b_n: Internal strains at time n (flattened)
            *args: Material parameters
            
        Returns:
            Tuple of (sig_a_n1, dsig_deps_aa_n1, Eps_b_n1, Sig_b_n1, lam_k, k_I)
            - sig_a_n1: External stresses at time n+1
            - dsig_deps_aa_n1: Material stiffness matrix at time n+1
            - Eps_b_n1: Internal strains at time n+1 (flattened)
            - Sig_b_n1: Internal thermodynamic forces at time n+1
            - lam_k: Lagrange multipliers 
            - k_I: Iteration counts for each integration point
        """
        n_I = np.atleast_1d(eps_n).shape[0]
        k_I = np.zeros((n_I,), dtype=np.int32)
        d_A_c = np.zeros((n_I, self.n_delta_A_c), dtype=np.float64)
        
        # Initial evaluation
        sig_a_n1, Sig_b_n1, f_n1, R_c_n1, dR_dA_cc_n1, dsig_deps_aa_n1 = self.get_sig_Sig_f_R_dR_dsig_n1(
            eps_n, d_eps, Eps_b_n, d_A_c, d_t, *args
        )
        
        # Determine elastic/inelastic states
        I = f_n1 >= 0  # Inelastic points
        I_inel = np.copy(I)
        I_el = ~I_inel  # Elastic points
        
        # Inelastic state update (inequality constraint active)
        if self.n_lam_phi > 0:
            for k in range(self.k_max):
                if np.all(I == False):
                    break
                    
                try:
                    # Newton-Raphson update
                    d_A_c[I] -= np.linalg.solve(
                        dR_dA_cc_n1[I], 
                        R_c_n1[I][..., np.newaxis]
                    )[..., 0]
                except np.linalg.LinAlgError as e:
                    print(f"Singular matrix in inelastic update: {e}")
                    print(f"eps_n = {eps_n}, d_eps = {d_eps}, Eps_b_n = {Eps_b_n}")
                    raise
                
                # Re-evaluate at updated state
                sig_a_n1[I], Sig_b_n1[I], f_n1[I], R_c_n1[I], dR_dA_cc_n1[I], dsig_deps_aa_n1[I] = self.get_sig_Sig_f_R_dR_dsig_n1(
                    eps_n[I], d_eps[I], Eps_b_n[I], d_A_c[I], d_t, *args
                )
                
                # Check convergence
                norm_R_n1 = np.linalg.norm(R_c_n1, axis=-1)
                I[norm_R_n1 <= self.tol] = False
                k_I[I] += 1
            
            # Check bounds for internal variables (if defined)
            if hasattr(self.symb_def, 'dot_Eps_bounds_expr') and self.symb_def.dot_Eps_bounds_expr != sp.S.Zero:
                # TODO: Implement bounds checking using dot_Eps_bounds evaluation
                pass
        
        # Elastic state update (equality constraints only)
        if self.n_Lam > 0:
            for k in range(self.k_max):
                if np.all(I_el == False):
                    break
                    
                try:
                    # Determine slice for elastic update
                    if self.n_lam_phi == 0:
                        i1 = None
                    else:
                        i1 = -1
                        d_A_c[I_el, i1] = 0  # Set inequality multiplier to zero
                    
                    # Newton-Raphson update for elastic points
                    d_A_c[I_el, :i1] -= np.linalg.solve(
                        dR_dA_cc_n1[I_el, :i1, :i1], 
                        R_c_n1[I_el, :i1, np.newaxis]
                    )[..., 0]
                    
                except np.linalg.LinAlgError as e:
                    print(f"Singular matrix in elastic update: {e}")
                    print(f"eps_n = {eps_n}, d_eps = {d_eps}, Eps_b_n = {Eps_b_n}")
                    raise
                
                # Re-evaluate at updated state
                sig_a_n1[I_el], Sig_b_n1[I_el], f_n1[I_el], R_c_n1[I_el], dR_dA_cc_n1[I_el], dsig_deps_aa_n1[I_el] = self.get_sig_Sig_f_R_dR_dsig_n1(
                    eps_n[I_el], d_eps[I_el], Eps_b_n[I_el], d_A_c[I_el], d_t, *args
                )
                
                # Check convergence for elastic equations only
                norm_R_n1 = np.linalg.norm(R_c_n1[..., :i1], axis=-1)
                k_I[I_el] += 1
                I_el[norm_R_n1 <= self.tol] = False
        
        # Extract results
        lam_k = d_A_c[..., self.n_Eps_b:]  # Lagrange multipliers
        Eps_b_n1 = Eps_b_n + d_A_c[..., :self.n_Eps_b]  # Updated internal strains
        
        return sig_a_n1, dsig_deps_aa_n1, Eps_b_n1, Sig_b_n1, lam_k, k_I
