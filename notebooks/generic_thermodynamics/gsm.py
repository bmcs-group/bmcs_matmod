"""Generalized standard material

Formulation of a generic framework for the derivation of constitutive models capturing 
dissipative processes in response of inelastic materials to mechanical and environmental 
loading. This framework aspires to combine the high-level mathematical formulation of the
thermodynamic potentials with an efficient execution of the automatically derived evolution
equations in the computational environment.
"""

import traits.api as tr
import bmcs_utils.api as bu
import sympy as sp
import numpy as np

def get_dirac_delta(x):
    return 0
numpy_dirac =[{'DiracDelta': get_dirac_delta }, 'numpy']

class GSM(bu.Model):
    """Generalized Standard Material

    The class definition consists of 

    F = Free energy potential
    u = external variable
    T = temperature
    Eps = symbolic definition of the internal state (sympy.BlockMatrix)
    Sig = symbolic definition of the internal state (sympy.BlockMatrix)
    mp = list of material parameters
    f = threshold function
    phi_ext = flow potential extension
    """

    u_vars = tr.Any
    """External variable
    """

    T_var = tr.Any
    """Temperature
    """

    Eps_vars = tr.Tuple
    """Symbolic definition of the internal state (sympy.BlockMatrix)
    """

    Sig_vars = tr.Tuple
    """Symbolic definition of the internal state (sympy.BlockMatrix)
    """

    m_params = tr.Tuple
    """List of material parameters
    """

    F_expr = tr.Any
    """Free energy potential
    """

    Sig_signs = tr.Tuple
    """Signs of the derivatives of the free energy potential with respect 
    to the internal variables
    """

    f_expr = tr.Any 
    """Threshold function delineating the reversible and reversible state 
    domains. The function can consist of subdomains within the state space
    domain. 
    """

    phi_ext_expr = tr.Any
    """Extension of the threshold function detailing the trajectories
    within the state domain that govern the irreversible state evolution   
    """ 

    # Derived expressions 
    Eps = tr.Property()
    @tr.cached_property
    def _get_Eps(self):
        return sp.BlockMatrix([Eps_i.T for Eps_i in self.Eps_vars]).T
    
    Sig = tr.Property()
    @tr.cached_property
    def _get_Sig(self):
        return sp.BlockMatrix([Sig_i.T for Sig_i in self.Sig_vars]).T

    dF_dEps_ = tr.Property()
    @tr.cached_property
    def _get_dF_dEps_(self):
        return sp.BlockMatrix([sp.diff(self.F_expr, var).T for var in self.Eps.blocks]).T

    Sig_ = tr.Property()
    @tr.cached_property
    def _get_Sig_(self):
        return sp.BlockMatrix([(sign_i_ * dF_dEps_i_).T for sign_i_, dF_dEps_i_ 
                               in zip(self.Sig_signs, self.dF_dEps_.blocks)]).T
    
    get_Sig = tr.Property()
    @tr.cached_property
    def _get_get_Sig(self):
        return sp.lambdify((self.u_vars, self.T_var, self.Eps.blocks, self.Sig.blocks) + self.m_params + ('**kw',), 
                           list(self.Sig_.blocks), cse=True)

    lambdified_operators = tr.Property()
    @tr.cached_property
    def _get_lambdified_operators(self):
        # symbolic derivation
        subs_Sig_Eps_ = dict(zip(self.Sig.as_explicit(), self.Sig_.as_explicit()))
        f_exprs, f_conds = zip(*self.f_expr.args)
        df_dEps_exprs = [f_expr.subs(subs_Sig_Eps_).diff(self.Eps.as_explicit()) 
                         for f_expr in f_exprs]
        phi_exprs = [(f_expr + self.phi_ext_expr) for f_expr in f_exprs]
        Phi_exprs = []
        for phi_expr in phi_exprs:
            Phi_expr = sp.BlockMatrix([[-sign_i_ * phi_expr.diff(Sig_i_)] 
                                       for sign_i_, Sig_i_ 
                                       in zip(self.Sig_signs, self.Sig.blocks)]).as_explicit()
            Phi_exprs.append(Phi_expr)

        # common subexpression eliminitation
        n_exprs = len(f_exprs)
        all_expr = (f_exprs + tuple(df_dEps_exprs) + tuple(Phi_exprs) + f_conds)
        f_reduced_, f_cse_list_ = sp.cse(all_expr)
        f_cse_ = f_cse_list_[:n_exprs]
        df_dEps_cse_ = f_cse_list_[n_exprs:n_exprs*2]
        Phi_cse_ = f_cse_list_[n_exprs*2:n_exprs*3]
        f_conds_cse_ = f_cse_list_[n_exprs*3:-1]
        df_dlambda_cse_ = [(df_dEps_cse_[i].T * Phi_cse_[i])[0,0] for i in range(n_exprs)]
        args_ = f_cse_ + df_dlambda_cse_ + f_conds_cse_

        # labmdify the reduced expressions
        cs_vars, get_common = self._lambdify_cs(
            (self.u_vars, self.T_var, self.Eps.blocks, self.Sig.blocks), self.m_params, f_reduced_
            )
        get_f_cse_list = [sp.lambdify(
            (self.u_vars, self.T_var, self.Eps.blocks, self.Sig.blocks) + cs_vars + self.m_params + ('**kw',), 
                                    f_cse_, numpy_dirac)
                                    for f_cse_ in args_]
        get_f = get_f_cse_list[:n_exprs]
        get_df_dlambda = get_f_cse_list[n_exprs:n_exprs*2]
        get_conds = get_f_cse_list[n_exprs*2:]
        return [get_common, get_f, get_df_dlambda, get_conds]

    @staticmethod
    def _lambdify_cs(vars, sp_vars, cs_list):
        """Subsidiary method generating an operator evaluating all common 
        subexpressions"""
        cs_vars, cs_exprs = zip(*cs_list)
        get_x_exprs = [sp.lambdify(vars + cs_vars[:i] + sp_vars + ('**kw',), cs_expr)
                            for i, cs_expr in enumerate(cs_exprs)]
        def get_interim_vars(*params, **kw):
            parameters = list(params)
            results = []
            for get_x_i in get_x_exprs:
                result = get_x_i(*parameters, **kw)
                results.append(result)
                parameters.append(result)
            return results
        return tuple(cs_vars), get_interim_vars

    def get_f_df(self, _u_a, _T, _Eps, _Sig, **_mp):
        get_common, get_f, get_df_dlambda, get_conds = self.lambdified_operators

        _x_i_cse = get_common(_u_a, _T, _Eps, _Sig, **_mp)
        args = (_u_a, _T, _Eps, _Sig, *_x_i_cse)

        # Initialize the result array with the default value
        result = np.full((1,11), np.nan)

        # Initialize the mask to cover all grid elements
        mask = np.ones_like(result, dtype=bool)

        # Efficient cascading condition evaluation and index recovery
        for i, get_f_cond in enumerate(get_conds):
            # Apply condition only where the result is undefined and mask is valid
            mask_update = np.logical_and(get_f_cond(*args, **_mp), mask)
            # call the function only on the elements where the condition is satisfied 
            _f_i = get_f[i](*args, **_mp)
            # Evaluate conditions only on valid masked elements and update results
            result[mask_update] = _f_i[mask_update]
            # Update the mask by excluding elements where condition is satisfied
            mask[mask_update] = False

        result[mask_update] = get_f[-1](*args, **_mp)[mask_update]
        # result now contains the output after applying the corresponding functions based on conditions
        return result

