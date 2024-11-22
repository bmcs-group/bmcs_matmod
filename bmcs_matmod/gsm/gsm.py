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
import dill
import os
import functools

def get_dirac_delta(x, x_0=0):
    return 0
numpy_dirac =[{'DiracDelta': get_dirac_delta }, 'numpy']


def lambdify_and_cache(func):
    @functools.wraps(func)
    def wrapper(self):
        # Generate the filename based on class name and property name
        class_name = self.__class__.__name__
        object_name = self.name
        property_name = func.__name__
        cache_dir = '_lambdified_cache'
        filename = os.path.join(cache_dir, f"{class_name}_{object_name}_{property_name}.pkl")

        # Check if the cache directory exists, if not, create it
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Check if the file already exists
        if os.path.exists(filename):
            # Load the lambdified function from the file
            with open(filename, 'rb') as f:
                lambdified_func = dill.load(f)
        else:
            # Call the original function to get the symbolic expression
            lambdified_func = func(self)

            # Save the lambdified function to a file
            with open(filename, 'wb') as f:
                dill.dump(lambdified_func, f)

        return lambdified_func

    return wrapper

class GSM(bu.Model):
    """Generalized Standard Material

    The class definition consists of 

    F_expr   = Free energy potential

    u_vars   = external variable
    
    T_var    = temperature
    
    Eps_vars = symbolic definition of the internal state (sympy.BlockMatrix)
    
    Sig_vars = symbolic definition of the internal state (sympy.BlockMatrix)
    
    m_params = list of material parameters
    
    m_param_subs = material parameter substitutions. A dictionary of sympy substitutions 
                   that can be used to simplify the potentials before they are submitted 
                   to differentiation. This feature has been introduce to handle the algebraic
                   limiting configurations, like for example the case of (Y**(r+1) / Y) that 
                   occurs in some flow potentials. For Y = 0 and r > 0, the value of this expression is
                   zero. However, as sympy cannot apply the assumption r > 0 yet, it falsely recognizes
                   this case as division by zero and returns nan. Using this attribute the substitution
                   {r : 2} avoids this problem. If the variable r is a subject of a parametric study
                   or calibration, the depending derivations must be re-rendered. Currently, this means 
                   that all expressions are rerendered. However, as it takes less than a second for the 
                   currently implemented potentials, this does not impose any significant limitation.   
    
    f_expr       = threshold function
    
    phi_ext_expr = flow potential extension


    """

    u_vars = tr.Any
    """External variable
    """

    sig_vars = tr.Any
    """External conjugate variables
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

    m_param_subs = tr.Dict
    """Substitutions of hard-wired parameters
    """

    F_expr = tr.Any
    """Free energy potential
    """

    Sig_signs = tr.Tuple
    """Signs of the derivatives of the free energy potential with respect 
    to the internal variables
    """

    dF_sign = bu.Float(1)
    """Sign relating the rate of free energy to the dissipation.
    For Helmholtz free energy it is negative, for the Gibbs free energy it is positive
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

    vp_on = bu.Bool(True)
    """Viscoplasticity on or off   
    """ 

    update_at_k = bu.Bool(True)
    """Debugging for type of viscoplastic regularization update
    """ 

    # Internal variable representations and conversions
    Eps_list = tr.Property()
    @tr.cached_property
    def _get_Eps_list(self):
        return [Eps_i.T for Eps_i in self.Eps_vars]

    Eps = tr.Property()
    @tr.cached_property
    def _get_Eps(self):
        return sp.BlockMatrix(self.Eps_list).T
    
    n_Eps_explicit = tr.Property
    @tr.cached_property
    def _get_n_Eps_explicit(self):
        return len(self.Eps.as_explicit())

    _Eps_as_array_lambdified = tr.Property
    @tr.cached_property
    def _get__Eps_as_array_lambdified(self):
        return sp.lambdify(self.Eps.blocks, self.Eps.as_explicit())

    def Eps_as_array(self, arr):
        return self._Eps_as_array_lambdified(arr)[:,0]

    _Eps_as_blocks_lambdified = tr.Property
    @tr.cached_property
    def _get__Eps_as_blocks_lambdified(self):
        return sp.lambdify(self.Eps.as_explicit(), self.Eps_list)

    def Eps_as_blocks(self, arr):
        return [Eps_i[0] for Eps_i in self._Eps_as_blocks_lambdified(arr)]

    # Conjugate state variable representations and conversions
    Sig_list = tr.Property()
    @tr.cached_property
    def _get_Sig_list(self):
        return [Sig_i.T for Sig_i in self.Sig_vars]

    Sig = tr.Property()
    @tr.cached_property
    def _get_Sig(self):
        return sp.BlockMatrix([Sig_i.T for Sig_i in self.Sig_vars]).T

    dF_dEps_ = tr.Property()
    @tr.cached_property
    def _get_dF_dEps_(self):
        return sp.BlockMatrix([sp.simplify(sp.diff(self.F_expr, var)).T for var in self.Eps.blocks]).T.subs(self.m_param_subs)

    ######################################

    def get_sig(self, u, T, Eps, Sig, **m_params):
        """
        Calculates the displacement for a given stress level

        Args:
            sig: Control stress.
            T: Temperature.
            Eps: Strain.
            Sig: Stress.
            **m_params: Additional model parameters.

        Returns:
            Calculated displacement for a stress level and control stress.
        """
        return self._sig_lambdified(u, T, Eps, Sig, **m_params)[:, 0]

    _sig_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__sig_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, 
                            self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), 
                           self.sig_, numpy_dirac, cse=True)

    sig_ = tr.Property()
    @tr.cached_property
    def _get_sig_(self):
        return self.F_expr.diff(self.u_vars)

    ######################################
    def get_dDiss_dEps(self, u, T, Eps, Sig, **m_params):
        """
        Calculates the derivative of the dissipation rate with respect 
        to internal variables.

        Args:
            u: Displacement.
            T: Temperature.
            Eps: Strain.
            Sig: Stress.
            **m_params: Additional model parameters.

        Returns:
            Calculated derivative of the dissipation rate with respect to strain.
        """
        return self._dDiss_dEps_lambdified(u, T, Eps, Sig, **m_params)[:, 0]

    _dDiss_dEps_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__dDiss_dEps_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, 
                            self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), 
                           self.dDiss_dEps_, numpy_dirac, cse=True)

    dDiss_dEps_ = tr.Property()
    @tr.cached_property
    def _get_dDiss_dEps_(self):
        "For Gibbs the energy sign is swapped using the dF_sign parameter = -1"
        dFG_dEps_explicit_ = self.dF_sign * self.dF_dEps_.as_explicit()
        return (self.T_var * dFG_dEps_explicit_.diff(self.T_var) - dFG_dEps_explicit_)

    
    ######################################
    def get_Sig(self, u, T, Eps, Sig, **m_params):
        """
        Calculates the stress based on the given inputs.

        Args:
            u: Displacement.
            T: Temperature.
            Eps: Strain.
            Sig: Stress.
            **m_params: Additional model parameters.

        Returns:
            Calculated stress.
        """
        _Sig = self._Sig_lambdified(u, T, Eps, Sig, **m_params)
        return _Sig.reshape(Sig.shape)
    
    _Sig_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__Sig_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, 
                            self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), 
                           self.Sig_.as_explicit(), numpy_dirac, cse=True)

    Sig_ = tr.Property()
    @tr.cached_property
    def _get_Sig_(self):
        return sp.BlockMatrix([(sign_i_ * dF_dEps_i_).T for sign_i_, dF_dEps_i_ 
                               in zip(self.Sig_signs, self.dF_dEps_.blocks)]).T    

    ######################################

    phi_ = tr.Property()
    @tr.cached_property
    def _get_phi_(self):
        return (self.f_expr + self.phi_ext_expr).subs(self.m_param_subs)


    def get_Phi(self, u, T, Eps, Sig, **m_params):
        """
        Calculates the gradient Phi of the flow potential phi based on the 
        given inputs.

        Args:
            u: Displacement.
            T: Temperature.
            Eps: Strain.
            Sig: Stress.
            **m_params: Additional model parameters.

        Returns:
            Calculated state variable Phi.
        """
        return self._Phi_lambdified(u, T, Eps, Sig, **m_params).reshape(Eps.shape)# [:, 0]

    _Phi_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__Phi_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, 
                            self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), 
                           self.Phi_, numpy_dirac, cse=True)

    Phi_ = tr.Property()
    @tr.cached_property
    def _get_Phi_(self):
        Phi_list = [-sign_i_ * self.phi_.diff(Sig_i_) for sign_i_, Sig_i_ 
                    in zip(self.Sig_signs, self.Sig.blocks)]
        return sp.BlockMatrix([[Phi_i] for Phi_i in Phi_list]).as_explicit()
    
    ######################################

    def get_Phi_Eps(self, u, T, Eps, Sig, **m_params):
        """
        Calculates the gradient Phi of the flow potential phi based on the 
        given inputs.

        Args:
            u: Displacement.
            T: Temperature.
            Eps: Strain.
            Sig: Stress.
            **m_params: Additional model parameters.

        Returns:
            Calculated state variable Phi.
        """
        return self._Phi_Eps_lambdified(u, T, Eps, Sig, **m_params)[:, 0]

    _Phi_Eps_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__Phi_Eps_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, 
                            self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), 
                           self.Phi_Eps_, numpy_dirac, cse=True)

    Phi_Eps_ = tr.Property()
    @tr.cached_property
    def _get_Phi_Eps_(self):
        subs_Sig_Eps = dict(zip(self.Sig.as_explicit(), self.Sig_.as_explicit()))
        return self.Phi_.subs(subs_Sig_Eps)

    ######################################

    def get_DScale(self, u, T, lam, Eps, **m_params):
        return self._DScale_lambdified(u, T, lam, Eps, **m_params)

    _DScale_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__DScale_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var,
                            self.lambda_var,
                            self.Eps.as_explicit()) + self.m_params + ('**kw',),
                           self.DScale_, numpy_dirac,
                           cse=True)

    DScale_ = tr.Property()
    @tr.cached_property
    def _get_DScale_(self):
        return (self.dF_dEps_.as_explicit().T * self.Phi_Eps_)[0,0]

    ######################################

    def get_dDScale_dEps(self, u, T, Eps, **m_params):
        """
        Calculates the gradient Phi of the flow potential phi based on the 
        given inputs.

        Args:
            u: Displacement.
            T: Temperature.
            Eps: Strain.
            Sig: Stress.
            **m_params: Additional model parameters.

        Returns:
            Calculated state variable Phi.
        """
        return self._dDScale_dEps_lambdified(u, T, Eps, **m_params)[:, 0]

    _dDScale_dEps_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__dDScale_dEps_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, 
                            self.Eps.as_explicit()) + self.m_params + ('**kw',), 
                           self.dDScale_dEps_, numpy_dirac, 
                           cse=True)

    dDScale_dEps_ = tr.Property()
    @tr.cached_property
    def _get_dDScale_dEps_(self):
        return self.DScale_.diff(self.Eps.as_explicit())

    ######################################

    def get_ddDScale_ddEps(self, u, T, Eps, **m_params):
        """
        Calculates the Hessian of the dissipation potential for the
        given inputs.

        Args:
            u: Displacement.
            T: Temperature.
            Eps: Strain.
            Sig: Stress.
            **m_params: Additional model parameters.

        Returns:
            Calculated Hessian of the dissipation potential.
        """
        O_expand = np.zeros_like(T)
        return self._ddDScale_ddEps_lambdified(O_expand, u, T, Eps, **m_params)

    O = bu.Cymbol('O')

    _ddDScale_ddEps_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__ddDScale_ddEps_lambdified(self):
        return sp.lambdify((self.O, self.u_vars, self.T_var, 
                            self.Eps.as_explicit()) + self.m_params + ('**kw',), 
                           self.ddDScale_ddEps_, numpy_dirac, 
                           cse=True)

    ddDScale_ddEps_ = tr.Property()
    @tr.cached_property
    def _get_ddDScale_ddEps_(self):
        return self.dDScale_dEps_.diff(self.Eps.as_explicit())[:, 0, :, 0].subs(0, self.O)

    ######################################

    def get_f(self, u, T, Eps, Sig, **m_params):
        """
        Calculates the stress increment based on the given inputs.

        Args:
            u: Displacement.
            T: Temperature.
            Eps: Strain.
            Sig: Stress.
            **m_params: Additional model parameters.

        Returns:
            Calculated stress increment.
        """
        _Sig = self.get_Sig(u, T, Eps, Sig, **m_params)
        return self._f_lambdified(u, T, Eps, _Sig, **m_params)

    _f_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__f_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), 
                           self.f_, numpy_dirac, cse=True)

    f_ = tr.Property()
    @tr.cached_property
    def _get_f_(self):
        return self.f_expr.subs(self.m_param_subs)

    ######################################
    def get_df_dlambda(self, u, T, Eps, Sig, **m_params):
        """
        Calculates the derivative of the stress increment with respect 
        to the load parameter lambda.

        Args:
            u: Displacement.
            T: Temperature.
            Eps: Strain.
            Sig: Stress.
            **m_params: Additional model parameters.

        Returns:
            Calculated derivative of the stress increment with respect to lambda.
        """
        _Sig = self.get_Sig(u, T, Eps, Sig, **m_params)
        return self._df_dlambda_lambdified(u, T, Eps, _Sig, **m_params)
    
    _df_dlambda_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__df_dlambda_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), self.df_dlambda_, 
                            numpy_dirac, cse=True)

    _f_df_dlambda_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__f_df_dlambda_lambdified(self):
        f_df_ = [self.f_, self.df_dlambda_]
        return sp.lambdify((self.u_vars, self.T_var, 
                            self.Eps.as_explicit(), self.Sig.as_explicit()) + self.m_params + ('**kw',), f_df_, 
                            numpy_dirac, cse=True)
    
    df_dlambda_ = tr.Property()
    @tr.cached_property
    def _get_df_dlambda_(self):
#        return sp.simplify((sp.simplify(self.df_dEps_.T) * sp.simplify(self.Phi_))[0,0])
        return (self.df_dEps_.T * self.Phi_)[0,0]
        # return sp.simplify((self.df_dEps_.T * self.Phi_)[0, 0])

    ######################################

    def get_f_df_Sig(self, u, T, Eps, Sig, **m_params):
        """
        Calculates the threshold function and its derivative 
        with respect to the load parameter lambda.

        Args:
            u: Displacement.
            T: Temperature.
            Eps: Strain.
            Sig: Stress.
            **m_params: Additional model parameters.

        Returns:
            List containing the stress increment, its derivative 
            with respect to lambda, and the updated stress.
        """
        _Sig = self.get_Sig(u, T, Eps, Sig, **m_params)
        _f, _df_dlambda = self._f_df_dlambda_lambdified(u, T, Eps, _Sig, **m_params)
        return _f, _df_dlambda, _Sig

    ######################################

    def get_dSig_dEps(self, u, T, Eps, Sig, **m_params):
        """
        Gradient of thermodynamic forces with respect to the kinematic variables
        """
        return self._dSig_dEps_lambdified(u, T, Eps, Sig, **m_params)
    
    _dSig_dEps_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__dSig_dEps_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), self.dSig_dEps_, 
                            numpy_dirac, cse=True)

    dSig_dEps_ = tr.Property()
    @tr.cached_property
    def _get_dSig_dEps_(self):
        dSig_dEps_ = sp.Matrix([[sp.simplify(Sig_i.diff(Eps_i))
                                for Sig_i in self.Sig_.as_explicit()]
                                for Eps_i in self.Eps.as_explicit()])
        return dSig_dEps_

    ######################################

    def get_df_dSig(self, u, T, Eps, Sig, **m_params):
        return self._df_dSig_lambdified(u, T, Eps, Sig, **m_params)
    
    _df_dSig_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__df_dSig_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), self.df_dSig_, 
                            numpy_dirac, cse=True)

    df_dSig_ = tr.Property()
    @tr.cached_property
    def _get_df_dSig_(self):
        # gradient of threshold function w.r.t. thermodynamic forces
        return self.f_.diff(self.Sig.as_explicit())

    ######################################

    def get_df_dEps(self, u, T, Eps, Sig, **m_params):
        return self._df_dEps_lambdified(u, T, Eps, Sig, **m_params)
    
    _df_dEps_lambdified = tr.Property()
    @tr.cached_property
    @lambdify_and_cache
    def _get__df_dEps_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), self.df_dEps_, 
                            numpy_dirac, cse=True)

    df_dEps_ = tr.Property()
    @tr.cached_property
    def _get_df_dEps_(self):
        subs_Sig_Eps = dict(zip(self.Sig.as_explicit(), self.Sig_.as_explicit()))
        return self.f_.subs(subs_Sig_Eps).diff(self.Eps.as_explicit())


    ######################################

    # Evolution equations
    def get_Eps_k1(self, u_n1, T_n1, Eps_n, lambda_k, Eps_k, Sig_k, **m_params):
        """
        Calculates the strain increment Eps_k1 based on the given inputs.

        Args:
            u_n1: Displacement at time n+1.
            T_n1: Temperature at time n+1.
            Eps_n: Strain at time n.
            lambda_k: Load parameter.
            Eps_k: Strain at time k.
            Sig_k: Stress at time n.
            **m_params: Additional model parameters.

        Returns:
            Calculated strain increment Eps_k1.
        """
        Phi_k = self.get_Phi(u_n1, T_n1, Eps_k, Sig_k, **m_params)
        Eps_k1 = Eps_n + lambda_k * Phi_k
        return Eps_k1

    def get_t_relax(self, **m_params):
        """
        Calculates the relaxation time based on the given model parameters.

        Args:
            **m_params: Additional model parameters.

        Returns:
            Calculated relaxation time.
        """
        return self._t_relax_lambdified(**m_params)[0]

    _t_relax_lambdified = tr.Property
    @tr.cached_property
    def _get__t_relax_lambdified(self):
        return sp.lambdify(self.m_params + ('**kw',), self.t_relax.T, cse=True)
    
    def get_state_n1(self, u_n, d_u_n1, T_n, d_T_n, d_t, Sig_n, Eps_n, k_max, **kw):
        """
        Calculates the state at time n+1 based on the given inputs using an iterative algorithm.

        Args:
            u_n: Displacement at time n.
            du_n1: Displacement increment from time n to n+1.
            T_n: Temperature at time n.
            dt: Time step size.
            Sig_n: Stress at time n.
            Eps_n: Strain at time n.
            k_max: Maximum number of iterations.
            **kw: Additional keyword arguments.

        Returns:
            Tuple containing the updated strain Eps_k, stress Sig_k, temperature T_n+1, number of iterations k, 
            and dissipation rate gradient dDiss_dEps.
        """
        u_n, d_u_n1, Sig_n, Eps_n = [
            np.moveaxis(arg, -1, 0) for arg in (u_n, d_u_n1, Sig_n, Eps_n)
        ]

        u_n1 = u_n + d_u_n1

        relax_t = self.get_t_relax(**kw)
        n_vk = len(relax_t)
        d_t_tau = d_t / relax_t
        inv_1_d_t_tau = 1 / (np.ones_like(d_t_tau) + d_t_tau)

        Eps_k = np.copy(Eps_n)
        Sig_k = np.copy(Sig_n)
        f_k, df_k, Sig_k = self.get_f_df_Sig(u_n1, T_n, Eps_k, Sig_k, **kw)
        f_k = np.atleast_1d(f_k)
        df_k = np.atleast_1d(df_k)
        f_k_norm = np.fabs(f_k)
        f_k_trial = np.copy(f_k)
        lam_k = np.zeros_like(f_k)
        k = 0
        while k < k_max:
            I_ = np.logical_and(f_k_trial > 0, f_k_norm >= 1e-4)
            if np.all(I_ == False):
                break # convergence reached
            I = np.where(I_)
            bI = (slice(None), *I)
            lam_k[I] -= f_k[I] / df_k[I] # increment of lambda with delta_lambda = -f / df
            Eps_k[bI] = self.get_Eps_k1(u_n1[bI], T_n[I], Eps_n[bI], lam_k[I], Eps_k[bI], Sig_k[bI], **kw)

            f_k[I], df_k[I], Sig_k[bI] = self.get_f_df_Sig(u_n1[bI], T_n[I], Eps_k[bI], Sig_k[bI], **kw)

            if np.any(np.isnan(f_k[I])):
                print('there is nan in f_k')
                raise RuntimeError(f'there is nan in f_k {I}')

            if np.any(np.isnan(Eps_k[bI])):
                print('there is nan in Eps_k')
                raise RuntimeError(f'there is nan in Eps_k {I}')

            f_k_norm[I] = np.fabs(f_k[I])
            k += 1
        else:
            raise RuntimeError(f'no convergence for indexes {I}')


        # viscoplastic regularization
        if self.vp_on and np.any(f_k_trial > 0):
            I = np.where(f_k_trial > 0)
            bI = (slice(None), *I)
            ### Perzyna type model - exploiting that \gamma = f / eta corresponds to \lambda above ???
            gamma_vk_bI = lam_k[I][np.newaxis,...] * np.ones_like(Eps_k[bI])
            gamma_vk_bI[:n_vk] *= (d_t_tau * inv_1_d_t_tau)[:, np.newaxis]
            # Check the singularity emerging upon update from zero state directly to the inelastic range
            if self.update_at_k:
                Eps_kbI = self.get_Eps_k1(u_n1[bI], T_n[I], Eps_k[bI], -lam_k[I], Eps_k[bI], Sig_k[bI], **kw)
                _, _, Sig_kbI = self.get_f_df_Sig(u_n1[bI], T_n[I], Eps_kbI, Sig_k[bI], **kw)
                Eps_k[bI] = self.get_Eps_k1(u_n1[bI], T_n[I], Eps_kbI, gamma_vk_bI, Eps_kbI, Sig_kbI, **kw)
            else:
                Eps_k[bI] = self.get_Eps_k1(u_n1[bI], T_n[I], Eps_n[bI], gamma_vk_bI, Eps_n[bI], Sig_n[bI], **kw)
            Sig_k[bI] = self.get_Sig(u_n1[bI], T_n[I], Eps_k[bI], Sig_k[bI], **kw)

        dEps_k = Eps_k - Eps_n
        dDiss_dEps = self.get_dDiss_dEps(u_n1, T_n, Eps_k, Sig_k, **kw)
        # dissipation rate
        dDiss_dt = np.einsum('b...,b...->...', dDiss_dEps, dEps_k)
        C_v_ = kw['C_v_']
        d_T = d_T_n + d_t * (dDiss_dt / C_v_ )# / rho_'

        return np.moveaxis(Eps_k, 0, -1), np.moveaxis(Sig_k, 0, -1), T_n + d_T, k, np.moveaxis(dDiss_dEps, 0, -1), lam_k

    def get_response(self, u_ta, T_t, t_t, k_max=20, **kw):
        """Time integration procedure 
        """
        if u_ta.ndim == 2:
            u_ta = u_ta[:,np.newaxis,:]

        d_u_ta = np.diff(u_ta, axis=0)
        d_t_t = np.diff(t_t, axis=0)
        d_T_t = np.diff(T_t, axis=0)
        T_0 = T_t[0]
        T_0 = np.atleast_1d(T_0)

        Eps_n1 = np.zeros(u_ta.shape[1:-1] + (self.n_Eps_explicit,), dtype=np.float_)
        Sig_n1 = np.zeros_like(Eps_n1)
        Sig_record = [Sig_n1]
        Eps_record = [Eps_n1]
        dDiss_dEps_record = [Sig_n1]
        T_record = [T_0]
        iter_record = [0]
        lam_record = [0]

        k = 0
        T_n = T_0 # initial condition
        for n, dt in enumerate(d_t_t):
            try:
                Eps_n1, Sig_n1, T_n1, k, dDiss_dEps, lam = self.get_state_n1(
                    u_ta[n], d_u_ta[n], T_n, d_T_t[n], dt, Sig_n1, Eps_n1, k_max, **kw
                )
            except RuntimeError as e:
                print(f'{n+1}({k}) ... {str(e)}', end='\r')
                break
            Sig_record.append(Sig_n1)
            Eps_record.append(Eps_n1)
            dDiss_dEps_record.append(dDiss_dEps)
            T_record.append(T_n1)
            iter_record.append(k)
            lam_record.append(lam)
            T_n = T_n1
        Sig_t = np.array(Sig_record, dtype=np.float_)
        Eps_t = np.array(Eps_record, dtype=np.float_)
        dDiss_dEps_t = np.array(dDiss_dEps_record, dtype=np.float_)
        T_t = np.array(T_record, dtype=np.float_)
        iter_t = np.array(iter_record,dtype=np.int_)
        lam_t = np.array(lam_record,dtype=np.float_)
        n_t = len(Eps_t)
        return t_t[:n_t], u_ta[:n_t], T_t, Eps_t, Sig_t, iter_t, dDiss_dEps_t, lam_t

