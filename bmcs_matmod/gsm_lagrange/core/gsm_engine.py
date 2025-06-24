"""Generalized standard material

Formulation of a generic framework for the derivation of constitutive models capturing 
dissipative processes in response of inelastic materials to mechanical and environmental 
loading. This framework aspires to combine the high-level mathematical formulation of the
thermodynamic potentials with an efficient execution of the automatically derived evolution
equations in the computational environment.
"""

from typing import TYPE_CHECKING, Any, Tuple, List, Union, cast, Dict, Optional, Callable
import traits.api as tr
import sympy as sp
import numpy as np
import numpy.typing as npt
import dill
import os
import functools
from functools import cached_property  # Python 3.8+

# Use TYPE_CHECKING to avoid runtime overhead and typing conflicts with SymPy
if TYPE_CHECKING:
    # Tell type checkers what the properties should return
    _sig_lambdified: Callable[..., npt.NDArray[np.float64]]
    phi_: Any  # SymPy expression
    Y_: Any    # SymPy matrix
    Phi_: Any  # SymPy matrix

    from typing import Any as SymExpr  # Simplified for type checking
    SymMatrix = Any
    SymDict = Dict[Any, Any]
else:
    # At runtime, use actual SymPy types without constraints
    SymExpr = Union[sp.Expr, sp.Basic, sp.Symbol]
    SymMatrix = sp.Matrix
    SymDict = Dict[SymExpr, SymExpr]

def get_dirac_delta(x: Any, x_0: float = 0) -> int:
    return 0

numpy_dirac = [{'DiracDelta': get_dirac_delta}, 'numpy']

# Directory to store the serialized symbolic instances
CACHE_DIR = '_lambdified_cache'

class GSMEngine(tr.HasTraits):
    """Generalized Standard Material

    The class definition consists of symbolic expressions and numerical execution methods
    for thermodynamically consistent material modeling.
    """

    name = tr.Str('unnamed')
    eps_vars = tr.Any()
    sig_vars = tr.Any()
    T_var = tr.Any()
    Eps_vars = tr.Tuple()
    Sig_vars = tr.Tuple()
    m_params = tr.Tuple()
    m_param_subs = tr.Dict({})
    F_expr = tr.Any()
    pi_expr = tr.Any(default_value=0)
    Sig_signs = tr.Array()
    sig_sign = tr.Float(1)
    gamma_mech_sign = tr.Int(-1)
    dot_Eps_bounds_expr = tr.Any(sp.S.Zero)
    phi_ext_expr = tr.Any(sp.S.Zero)
    f_expr = tr.Any(sp.S.Zero)
    d_t = sp.Symbol(r'\mathrm{d}t', real=True)
    h_k = tr.List(tr.Any, value=[])
    vp_on = tr.Bool(True)
    update_at_k = tr.Bool(True)
    k_max = tr.Int(10)

    # Properties - let SymPy handle its own typing internally
    eps_a = tr.Property()
    @tr.cached_property
    def _get_eps_a(self):
        return sp.BlockMatrix([
            var if isinstance(var, sp.Matrix) and var.shape[1] == 1 else sp.Matrix([[var]]) 
            for var in self.eps_vars
        ]).as_explicit()

    dot_eps_a = tr.Property()
    @tr.cached_property
    def _get_dot_eps_a(self):
        return sp.Matrix([sp.Symbol(f'\\dot{{{var.name}}}') for var in self.eps_a])

    sig_a = tr.Property()
    @tr.cached_property
    def _get_sig_a(self):
        return sp.BlockMatrix([
            var if isinstance(var, sp.Matrix) and var.shape[1] == 1 else sp.Matrix([[var]]) 
            for var in self.sig_vars
        ]).as_explicit()

    dot_sig_a = tr.Property()
    @tr.cached_property
    def _get_dot_sig_a(self):
        return sp.Matrix([sp.Symbol(f'\\dot{{{var.name}}}') for var in self.sig_vars])

    # For complex SymPy operations, don't over-constrain with typing
    lam_phi_f_ = tr.Property()
    @tr.cached_property
    def _get_lam_phi_f_(self):
        if self.phi_ == sp.S.Zero:
            return ([], sp.S.Zero, sp.S.Zero)
        lam_phi = sp.Symbol(r'\lambda_{\mathrm{\phi}}', real=True)
        return ([lam_phi], lam_phi * (-self.phi_), self.f_expr)

    n_lam = tr.Property()
    @tr.cached_property
    def _get_n_lam(self):
        lam_phi_k, _, _ = self.lam_phi_f_
        return len(lam_phi_k)

    H_Lam = tr.Property()
    @tr.cached_property
    def _get_H_Lam(self):
        dot_Lam = [sp.Symbol(f'\\dot{{\\lambda}}_{{{k}}}', real=True) for k in range(len(self.h_k))]
        delta_Lam = sp.Matrix([sp.Symbol(f'\\Delta\\lambda_{{{k}}}', real=True) for k in range(len(self.h_k))])
        dot_lam_sum = sum(l * h for l, h in zip(dot_Lam, self.h_k)) if self.h_k else sp.S.Zero
        return (dot_Lam, delta_Lam, dot_lam_sum)

    n_Lam = tr.Property()
    @tr.cached_property
    def _get_n_Lam(self):
        return len(self.h_k)

    # Internal variable representations - let SymPy handle its types
    Eps_list = tr.Property()
    @tr.cached_property
    def _get_Eps_list(self):
        return [Eps_i.T for Eps_i in self.Eps_vars]

    Eps = tr.Property()
    @tr.cached_property
    def _get_Eps(self):
        return sp.BlockMatrix(self.Eps_list).T
    
    n_Eps = tr.Property()
    @tr.cached_property
    def _get_n_Eps(self):
        return len(self.Eps.as_explicit())

    _Eps_as_array_lambdified = tr.Property()
    @tr.cached_property
    def _get__Eps_as_array_lambdified(self):
        return sp.lambdify(self.Eps.blocks, self.Eps.as_explicit())

    def Eps_as_array(self, arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self._Eps_as_array_lambdified(arr)[:, 0]

    _Eps_as_blocks_lambdified = tr.Property()
    @tr.cached_property
    def _get__Eps_as_blocks_lambdified(self):
        return sp.lambdify(self.Eps.as_explicit(), self.Eps_list)

    def Eps_as_blocks(self, arr: npt.NDArray[np.float64]) -> List[Any]:
        return [Eps_i[0] for Eps_i in self._Eps_as_blocks_lambdified(arr)]
       
    dot_Eps_list = tr.Property()
    @tr.cached_property
    def _get_dot_Eps_list(self):
        return [
            sp.Matrix(eps_var.shape[0], eps_var.shape[1], 
                    lambda i, j: sp.Symbol(name=f'\\dot{{{eps_var[i, j].name}}}'))
            for eps_var in self.Eps_list
        ]
    
    dot_Eps = tr.Property()
    @tr.cached_property
    def _get_dot_Eps(self):
        return sp.BlockMatrix(self.dot_Eps_list).T

    # Conjugate state variable representations
    Sig_list = tr.Property()
    @tr.cached_property
    def _get_Sig_list(self):
        return [Sig_i.T for Sig_i in self.Sig_vars]

    Sig = tr.Property()
    @tr.cached_property
    def _get_Sig(self):
        return sp.BlockMatrix([Sig_i.T for Sig_i in self.Sig_vars]).T

    dot_Sig_list = tr.Property()
    @tr.cached_property
    def _get_dot_Sig_list(self):
        return [
            sp.Matrix(sig_var.shape[0], sig_var.shape[1], 
                    lambda i, j: sp.Symbol(name=f'\\dot{{{sig_var[i, j].name}}}'))
            for sig_var in self.Sig_list
        ]
    
    dot_Sig = tr.Property()
    @tr.cached_property
    def _get_dot_Sig(self):
        return sp.BlockMatrix(self.dot_Sig_list).T

    dF_dEps_ = tr.Property()
    @tr.cached_property
    def _get_dF_dEps_(self):
        return sp.BlockMatrix([sp.simplify(sp.diff(self.F_expr, var)).T for var in self.Eps.blocks]).T.subs(self.m_param_subs)

    # Numerical methods - type the interfaces clearly
    def get_sig_a(self, eps: Union[float, npt.NDArray[np.float64]], Eps: Optional[npt.NDArray[np.float64]] = None, *args: float) -> Union[float, npt.NDArray[np.float64]]:
        """Calculate the displacement for a given stress level"""
        eps_sp_ = np.moveaxis(np.atleast_1d(eps), -1, 0)
        if Eps is not None:
            Eps_sp_ = np.moveaxis(Eps, -1, 0)
        else:
            Eps_sp_ = np.zeros((1, self.n_Eps))
        sig_sp_ = self._sig_a_lambdified(eps_sp_, Eps_sp_, *args)  # type: ignore[misc]
        sig_sp_ = sig_sp_.reshape(eps_sp_.shape)
        return np.moveaxis(sig_sp_, 0, -1)

    # For new properties, use Python's cached_property instead of traits
    @cached_property  
    def _sig_a_lambdified(self) -> Callable[..., npt.NDArray[np.float64]]:
        return sp.lambdify((self.eps_vars[0], self.Eps.as_explicit()) + self.m_params + ('*args',), 
                           self.sig_a_, numpy_dirac, cse=True)

    # Keep existing traits properties as-is, just ignore typing for them
    phi_ = tr.Property()  # type: ignore[misc]
    @tr.cached_property
    def _get_phi_(self):
        return (self.f_expr + self.phi_ext_expr).subs(self.m_param_subs)

    sig_a_ = tr.Property()
    @tr.cached_property
    def _get_sig_a_(self):
        return self.F_expr.diff(self.eps_a)

    subs_sig_eps = tr.Property()
    @tr.cached_property
    def _get_subs_sig_eps(self):
        return dict(zip(self.sig_a, self.sig_a_))  # type: ignore[arg-type]

    # More numerical interface methods...
    def get_dDiss_dEps(self, eps: Any, T: Any, Eps: Any, Sig: Any, *args: Any) -> npt.NDArray[np.float64]:
        """Calculate the derivative of the dissipation rate with respect to internal variables"""
        return self._dDiss_dEps_lambdified(eps, T, Eps, Sig, *args)[:, 0]

    _dDiss_dEps_lambdified = tr.Property()
    @tr.cached_property
    def _get__dDiss_dEps_lambdified(self):
        return sp.lambdify((self.eps_vars[0], self.T_var, 
                            self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('*args',), 
                           self.dDiss_dEps_, numpy_dirac, cse=True)

    dDiss_dEps_ = tr.Property()
    @tr.cached_property
    def _get_dDiss_dEps_(self):
        dFG_dEps_explicit_ = self.dF_dEps_.as_explicit()
        return (self.T_var * dFG_dEps_explicit_.diff(self.T_var) - dFG_dEps_explicit_)

    def get_dot_Eps_bounds(self, dot_eps: Any, dot_Eps: Any, *args: Any) -> Any:
        """Get bounds for internal variable rates"""
        dot_eps_sp_ = np.moveaxis(np.atleast_1d(dot_eps), -1, 0)
        dot_Eps_sp_ = np.moveaxis(dot_Eps, -1, 0)
        dot_Eps_bounds_sp = self._dot_Eps_bounds_lambdified(dot_eps_sp_, dot_Eps_sp_, *args)
        return np.moveaxis(dot_Eps_bounds_sp, 0, -1)
    
    _dot_Eps_bounds_lambdified = tr.Property()
    @tr.cached_property
    def _get__dot_Eps_bounds_lambdified(self):
        dot_eps = self.dot_eps_a[0]
        return sp.lambdify((dot_eps, 
                            self.dot_Eps.as_explicit()) + self.m_params + ('*args',), 
                           self.dot_Eps_bounds_, numpy_dirac, cse=True)

    dot_Eps_bounds_ = tr.Property()
    @tr.cached_property
    def _get_dot_Eps_bounds_(self):
        return self.dot_Eps_bounds_expr

    def get_Sig(self, eps: Any, Eps: Any, *args: Any) -> Any:
        """Calculate the stress based on the given inputs"""
        eps_sp_ = np.moveaxis(np.atleast_1d(eps), -1, 0)
        Eps_sp_ = np.moveaxis(Eps, -1, 0)
        Sig_sp = self._Sig_lambdified(eps_sp_, Eps_sp_, *args)
        Sig_sp_ = Sig_sp.reshape(Eps_sp_.shape)
        return np.moveaxis(Sig_sp_, 0, -1)
    
    _Sig_lambdified = tr.Property()
    @tr.cached_property
    def _get__Sig_lambdified(self):
        return sp.lambdify((self.eps_vars[0], 
                            self.Eps.as_explicit()) + self.m_params + ('*args',), 
                           self.Sig_.as_explicit(), numpy_dirac, cse=True)

    Sig_ = tr.Property()
    @tr.cached_property
    def _get_Sig_(self):
        return self.Y_ * self.dF_dEps_.as_explicit()

    # SymPy expression properties - minimal typing
    phi_ = tr.Property()
    @tr.cached_property  
    def _get_phi_(self):
        return (self.f_expr + self.phi_ext_expr).subs(self.m_param_subs)

    Y_ = tr.Property()
    @tr.cached_property
    def _get_Y_(self):
        Y_list = [sign_i_ * sp.eye(len(Sig_i_)) for sign_i_, Sig_i_ 
                    in zip(self.Sig_signs, self.Sig.blocks)]
        return sp.BlockDiagMatrix(*Y_list).as_explicit()

    Phi_ = tr.Property()
    @tr.cached_property
    def _get_Phi_(self):
        return -self.Y * self.phi_.diff(self.Sig.as_explicit())

    subs_Sig_Eps = tr.Property()
    @tr.cached_property
    def _get_subs_Sig_Eps(self):
        return dict(zip(self.Sig.as_explicit(), self.Sig_.as_explicit()))  # type: ignore[arg-type]

    Phi_Eps_ = tr.Property()
    @tr.cached_property
    def _get_Phi_Eps_(self):
        return self.Phi_.subs(self.subs_Sig_Eps)

    def get_Sig_f_R_dR_n1(self, eps_n: Any, d_eps: Any, Eps_n: Any, d_A: Any, d_t: Any, *args: Any) -> Tuple[Any, Any, Any, Any]:
        eps_n_sp_ = np.moveaxis(np.atleast_1d(eps_n), -1, 0)
        d_eps_sp_ = np.moveaxis(np.atleast_1d(d_eps), -1, 0)
        O_ = np.zeros_like(eps_n_sp_)
        I_ = np.ones_like(eps_n_sp_)
        d_A_sp_ = np.moveaxis(d_A, -1, 0)
        Eps_n_sp_ = np.moveaxis(Eps_n, -1, 0)
        Sig_sp_, f_sp_, R_sp_, d_R_sp_ = self._get_Sig_f_R_dR_n1_lambdified(eps_n_sp_, d_eps_sp_, Eps_n_sp_, d_A_sp_, d_t, O_, I_, *args)
        if self.phi_ == sp.S.Zero:
            f_sp_ = -np.ones_like(eps_n_sp_)
        Sig_sp_ = Sig_sp_.reshape(Eps_n_sp_.shape)
        return np.moveaxis(Sig_sp_, 0, -1), np.moveaxis(f_sp_, 0, -1), np.moveaxis(R_sp_[:, 0], 0, -1), np.moveaxis(d_R_sp_, (0, 1), (-2, -1))

    _get_Sig_f_R_dR_n1_lambdified = tr.Property()
    @tr.cached_property
    def _get__get_Sig_f_R_dR_n1_lambdified(self) -> Callable[..., Any]:
        _, (eps_n, delta_eps, Eps_n, delta_A, delta_t, self.Ox, self.Ix), Sig_n1, f_n1, R_n1, dR_dA_n1_OI = self.Sig_f_R_dR_n1
        # Convert traits tuple to regular tuple explicitly
        m_params_tuple = tuple(self.m_params)  # Force conversion
        return sp.lambdify((eps_n, delta_eps, Eps_n, delta_A, delta_t, self.Ox, self.Ix) + m_params_tuple + ('*args',), 
                            (Sig_n1, f_n1, R_n1, dR_dA_n1_OI), numpy_dirac, cse=True)
    
    Ox = sp.Symbol('O', real=True)
    Ix = sp.Symbol('I', real=True)

    def replace_zeros_and_ones_with_symbolic(self, dR_dA_: SymMatrix, delta_A: SymMatrix) -> SymMatrix:
        """Function to replace zero elements with a symbolic variable"""
        new_matrix = sp.Matrix(dR_dA_)
        rows, cols = new_matrix.shape
        for i in range(rows):
            for j in range(cols):
                if dR_dA_[i, j] == 0:
                    new_matrix[i, j] = self.Ox
                else:
                    grad_entry = dR_dA_[i, j].diff(delta_A)[:, 0]
                    ones_vector = sp.Matrix([1 for i in range(len(grad_entry))])
                    depends_on_Eps = (ones_vector.T * grad_entry)[0]
                    if depends_on_Eps == 0:
                        new_matrix[i, j] = dR_dA_[i, j] * self.Ix
                    else:
                        new_matrix[i, j] = dR_dA_[i, j]
        return new_matrix
    
    diff_along_rates = tr.Bool(True)

    Sig_f_R_dR_n1 = tr.Property()
    @tr.cached_property
    def _get_Sig_f_R_dR_n1(self) -> Tuple[Any, Any, Any, Any, Any, Any]:
        ## Manual construction of the residuum
        Eps = self.Eps.as_explicit()
        eps = self.eps_a[0]
        dot_Eps = self.dot_Eps.as_explicit()
        dot_eps = self.dot_eps_a[0]
        lam, lam_phi, f_ = self.lam_phi_f_
        dot_Lam, delta_Lam, dot_Lam_H_ = self.H_Lam

        # time symbols
        t = sp.Symbol(r't', real=True)
        delta_t = sp.Symbol(r'\Delta t', real=True)

        # fundamental state
        Eps_n = sp.Matrix([sp.Symbol(f'{var.name}_{{(n)}}', real=True) for var in Eps])
        eps_n = sp.Symbol(r'\varepsilon_{(n)}', real=True)

        # increment
        delta_Eps = sp.Matrix([sp.Symbol(f'\\Delta{{{var.name}}}', real=True) for var in Eps])
        delta_eps = sp.Symbol(r'\Delta{\varepsilon}', real=True)
        delta_lam = sp.Matrix([sp.Symbol(f'\\Delta{{{var.name}}}', real=True) for var in lam])
        
        # updated state
        Eps_n1 = Eps_n + delta_Eps
        eps_n1 = eps_n + delta_eps

        # rate of change
        dot_Eps_n = delta_Eps / delta_t
        dot_eps_n = delta_eps / delta_t
        dot_lam_n = delta_lam / delta_t


        # derive substitutions
        subs_dot_Eps = dict(zip(dot_Eps, dot_Eps_n))  # type: ignore[arg-type]
        subs_dot_eps = {dot_eps: dot_eps_n}
        subs_delta_lam = dict(zip(lam, delta_lam))  # type: ignore[arg-type]
        subs_dot_Lam = dict(zip(dot_Lam, delta_Lam))  # type: ignore[arg-type]
        subs_Eps_n1 = dict(zip(Eps, Eps_n1))  # type: ignore[arg-type]
        subs_eps_n1 = {eps: eps_n1}
        subs_dt = {self.d_t: delta_t}

        subs_n1 = {**subs_dot_Eps, **subs_dot_eps, **subs_delta_lam, 
                   **subs_dot_Lam, **subs_Eps_n1, **subs_eps_n1, **subs_dt}

        Sig = self.Sig.as_explicit()
        Sig_ = self.Sig_.as_explicit()

        f_ = self.f_expr

        gamma_mech = self.gamma_mech_sign * ((self.Y * Sig).T * self.dot_Eps.as_explicit())[0]

        # Full Lagrangian for the minimum principle of dissipation potential
        L_ = -gamma_mech - dot_Lam_H_ - lam_phi 

        # Generalized forces and flux increments
        S = sp.Matrix([Sig, sp.Matrix(dot_Lam), sp.Matrix(lam)])
        delta_A = sp.Matrix([delta_Eps, sp.Matrix(delta_Lam), sp.Matrix(delta_lam)])

        # Derivative of the Lagrangian with respect to the generalized forces
        dL_dS_ = L_.diff(S)
        dL_dS_ = sp.Matrix(dL_dS_)
        if self.f_expr != sp.S.Zero:
            dL_dS_[-1] = f_

        dL_dS_A_ = dL_dS_.subs(self.subs_Sig_Eps).subs(self.subs_sig_eps)
        R_n1 = dL_dS_A_.subs(subs_n1)

        # Jacobian of the residuum
        dR_dA_n1 = R_n1.jacobian(delta_A)
        dR_dA_n1 = dR_dA_n1.replace(sp.Derivative, lambda *args: 0)
        dR_dA_n1 = dR_dA_n1.replace(sp.DiracDelta, lambda *args: 0)

        # replace zeros and constant terms with symbolic variables to broadcast properly
        dR_dA_n1_OI = self.replace_zeros_and_ones_with_symbolic(dR_dA_n1, delta_A)

        # Values of the static threshold function
        f_Eps_ = f_.subs(self.subs_Sig_Eps)
        f_n1 = f_Eps_.subs(subs_n1)

        # External stress
        Sig_n1 = Sig_.subs(subs_n1)

        return (gamma_mech, L_, dL_dS_, dL_dS_A_, dR_dA_n1), (eps_n, delta_eps, Eps_n, delta_A, delta_t, self.Ox, self.Ix), Sig_n1, f_n1, R_n1, dR_dA_n1_OI

    def get_state_n1(self, eps_n: Any, d_eps: Any, d_t: Any, Eps_n: Any, *args: Any) -> Tuple[Any, Any, Any, Any]:
        """Calculate the state at time n+1 based on the given inputs using an iterative algorithm"""
        n_I = np.atleast_1d(eps_n).shape[0]
        tol = 1e-8
        k_I = np.zeros((n_I,), dtype=np.int_)
        d_A = np.zeros((n_I, self.n_Eps + self.n_Lam + self.n_lam), dtype=np.float64)
        
        Sig_n1, f_n1, R_n1, dR_dA_n1 = self.get_Sig_f_R_dR_n1(
            eps_n, d_eps, Eps_n, d_A, d_t, *args)

        I = f_n1 >= 0
        I_inel = np.copy(I)
        I_el = ~I_inel

        # Inelastic state update - only when inequality constraint is present
        if self.n_lam > 0:
            for k in range(self.k_max):
                if np.all(I == False):
                    break
                try:
                    d_A[I] -= np.linalg.solve(dR_dA_n1[I], R_n1[I][..., np.newaxis])[..., 0]
                except np.linalg.LinAlgError as e:
                    print("SingularMatrix encountered in dR_dA_n1[I]:", dR_dA_n1[I])
                    print(f"eps = {eps_n}, d_eps = {d_eps}, Eps_n = {Eps_n}, d_A = {d_A}, d_t = {d_t}")
                    raise
                
                Sig_n1[I], f_n1[I], R_n1[I], dR_dA_n1[I] = self.get_Sig_f_R_dR_n1(eps_n[I], d_eps[I], Eps_n[I], d_A[I], d_t, *args)
                norm_R_n1 = np.linalg.norm(R_n1, axis=-1)
                I[norm_R_n1 <= tol] = False
                k_I[I] += 1

            # If internal variables are out of bounds set the inelastic state to elastic
            if self.dot_Eps_bounds_expr != sp.S.Zero:
                I_el[self.get_dot_Eps_bounds(d_eps, d_A[..., :self.n_Eps], *args) > 0] = True

        # Elastic state update
        if self.n_Lam > 0:
            for k in range(self.k_max):
                if np.all(I_el == False):
                    break
                try:
                    if self.n_lam == 0:
                        i1 = None
                    else:
                        i1 = -1
                        d_A[I_el, i1] = 0
                    d_A[I_el, :i1] -= np.linalg.solve(dR_dA_n1[I_el, :i1, :i1], R_n1[I_el, :i1, np.newaxis])[..., 0]
                except np.linalg.LinAlgError as e:
                    print("SingularMatrix encountered in dR_dA_n1[I]:", dR_dA_n1[I_el])
                    print(f"eps = {eps_n}, d_eps = {d_eps}, Eps_n = {Eps_n}, d_A = {d_A}, d_t = {d_t}")
                    raise
                
                Sig_n1[I_el], f_n1[I_el], R_n1[I_el], dR_dA_n1[I_el] = self.get_Sig_f_R_dR_n1(eps_n[I_el], d_eps[I_el], Eps_n[I_el], d_A[I_el], d_t, *args)
                norm_R_n1 = np.linalg.norm(R_n1[...,:i1], axis=-1)
                k_I[I_el] += 1
                I_el[norm_R_n1 <= tol] = False

        lam_k = d_A[..., self.n_Eps:]
        Eps_n1 = Eps_n + d_A[..., :self.n_Eps]

        return Eps_n1, Sig_n1, lam_k, k_I

    def get_response(self, eps_ta: npt.NDArray[np.float64], t_t: npt.NDArray[np.float64], *args: float) -> Tuple[npt.NDArray[np.float64], ...]:
        """Time integration procedure"""
        if eps_ta.ndim == 2:
            eps_ta = eps_ta[:,np.newaxis,:]
        if eps_ta.ndim == 1:
            eps_ta = eps_ta[:, np.newaxis]

        d_eps_ta = np.diff(eps_ta, axis=0)
        d_t_t = np.diff(t_t, axis=0)
        Eps_n1 = np.zeros(eps_ta.shape[1:] + (self.n_Eps,), dtype=np.float64)
        Sig_n1 = np.zeros_like(Eps_n1)
        lam_n1 = np.zeros(eps_ta.shape[1:] + (self.n_lam + self.n_Lam,), dtype=np.float64)
        lam_n1 = np.zeros(eps_ta.shape[1:] + (self.n_lam + self.n_Lam,), dtype=np.float64)
        Sig_record = [Sig_n1]
        Eps_record = [Eps_n1]
        iter_record = [np.zeros(eps_ta.shape[1:])]
        lam_record = [lam_n1]
        for n, dt in enumerate(d_t_t):
            print('increment', n+1, end='\r')
            try:
                Eps_n1, Sig_n1, lam_n1, k = self.get_state_n1(
                    eps_ta[n], d_eps_ta[n], dt, Eps_n1, *args
                )
            except RuntimeError as e:
                print(f'{n+1}({k}) ... {str(e)}', end='\r')
                break
            Sig_record.append(Sig_n1)
            Eps_record.append(Eps_n1)
            lam_record.append(lam_n1)
            iter_record.append(k)
        Sig_t = np.array(Sig_record, dtype=np.float64)
        Eps_t = np.array(Eps_record, dtype=np.float64)
        iter_t = np.array(iter_record,dtype=np.int_)
        lam_t = np.array(lam_record,dtype=np.float64)
        n_t = len(Eps_t)
        eps_ta = eps_ta[:n_t]
        sig_ta = self.get_sig_a(eps_ta[..., np.newaxis], Eps_t, *args)
        return (t_t[:n_t], eps_ta, sig_ta, Eps_t, Sig_t, iter_t, lam_t, (d_t_t[:n_t], d_eps_ta[:n_t]))

    def save_to_disk(self) -> None:
        """Serialize this instance to disk."""
        filepath = os.path.join(CACHE_DIR, f"{self.name}.pkl")
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(filepath, 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load_from_disk(name: str) -> 'GSMEngine':
        """Deserialize an instance from disk."""
        filepath = os.path.join(CACHE_DIR, f"{name}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No cached data found for {name}.")
        with open(filepath, 'rb') as file:
            return cast('GSMEngine', dill.load(file))
