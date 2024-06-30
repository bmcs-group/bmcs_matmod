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
import sys

def get_dirac_delta(x):
    return 0
numpy_dirac =[{'DiracDelta': get_dirac_delta }, 'numpy']

class GSM(bu.Model):
    """Generalized Standard Material

    The class definition consists of 

    F_expr   = Free energy potential
    u_vars   = external variable
    T_var    = temperature
    Eps_vars = symbolic definition of the internal state (sympy.BlockMatrix)
    Sig_vars = symbolic definition of the internal state (sympy.BlockMatrix)
    m_params = list of material parameters
    f_expr       = threshold function
    phi_ext_expr = flow potential extension
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
        return sp.BlockMatrix([sp.diff(self.F_expr, var).T for var in self.Eps.blocks]).T

    dDiss_dEps_ = tr.Property()
    @tr.cached_property
    def _get_dDiss_dEps_(self):
        dF_dEps_explicit_ = self.dF_dEps_.as_explicit()
        return (self.T_var * dF_dEps_explicit_.diff(self.T_var) - dF_dEps_explicit_)

    _dDiss_dEps_lambdified = tr.Property()
    @tr.cached_property
    def _get__dDiss_dEps_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, 
                            self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), 
                           self.dDiss_dEps_, cse=True)

    def get_dDiss_dEps(self, u, T, Eps, Sig, **m_params):
        """
        Calculates the derivative of the dissipation rate with respect 
        to strain.

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

    def get_Sig_bI(self, u_aI, T, Eps_bI, Sig_bI, **m_params):
        return self._Sig_lambdified(u_aI, T, Eps_bI, Sig_bI, **m_params).reshape(Sig_bI.shape)

    def get_Sig_Ib(self, u_Ia, T, Eps_Ib, Sig_Ib, **m_params):
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
        u_aI, Eps_bI, Sig_bI = [np.moveaxis(arg, -1, 0) for arg in (u_Ia, Eps_Ib, Sig_Ib)]
        _Sig_bI = self._Sig_lambdified(u_aI, T, Eps_bI, Sig_bI, **m_params)
        return np.moveaxis(_Sig_bI.reshape(Sig_bI.shape), 0, -1)
    
    Sig_ = tr.Property()
    @tr.cached_property
    def _get_Sig_(self):
        return sp.BlockMatrix([(sign_i_ * dF_dEps_i_).T for sign_i_, dF_dEps_i_ 
                               in zip(self.Sig_signs, self.dF_dEps_.blocks)]).T
    
    _Sig_lambdified = tr.Property()
    @tr.cached_property
    def _get__Sig_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, 
                            self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), 
                           self.Sig_.as_explicit(), numpy_dirac, cse=True)

    def get_Phi_bI(self, u_aI, T, Eps_bI, Sig_bI, **m_params):
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
        return self._Phi_lambdified(u_aI, T, Eps_bI, Sig_bI, **m_params)[:, 0]

    Phi_ = tr.Property()
    @tr.cached_property
    def _get_Phi_(self):
        phi_ = self.f_expr + self.phi_ext_expr 
        Phi_list = [-sign_i_ * phi_.diff(Sig_i_) for sign_i_, Sig_i_ 
                    in zip(self.Sig_signs, self.Sig.blocks)]
        return sp.BlockMatrix([[Phi_i] for Phi_i in Phi_list])
    
    _Phi_lambdified = tr.Property()
    @tr.cached_property
    def _get__Phi_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, 
                            self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), 
                           self.Phi_.as_explicit(), numpy_dirac, cse=True)

    ## lambdified functions

    def get_f(self, u_aI, T, Eps_bI, Sig_bI, **m_params):
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
        _Sig_bI = self.get_Sig_bI(u_aI, T, Eps_bI, Sig_bI, **m_params)
        return self._f_lambdified(u_aI, T, Eps_bI, _Sig_bI, **m_params)

    _f_lambdified = tr.Property()
    @tr.cached_property
    def _get__f_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, self.Eps.as_explicit(), self.Sig.as_explicit()) + self.m_params + ('**kw',), self.f_expr, 
                              numpy_dirac, cse=True)

    def get_df_dlambda(self, u_aI, T, Eps_bI, Sig_bI, **m_params):
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
        _Sig_bI = self.get_Sig_bI(u_aI, T, Eps_bI, Sig_bI, **m_params)
        return self._df_dlambda_lambdified(u_aI, T, Eps_bI, _Sig_bI, **m_params)
    
    _df_dlambda_lambdified = tr.Property()
    @tr.cached_property
    def _get__df_dlambda_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), self.df_dlambda_, 
                            numpy_dirac, cse=True)

    df_dlambda_ = tr.Property()
    @tr.cached_property
    def _get_df_dlambda_(self):
        # gradient of thermodynamic forces with respect to the kinematic variables
        dSig_dEps_ = sp.BlockMatrix([[sp.Matrix(Sig_i.diff(Eps_i)[:,0,:,0]) 
                                      for Sig_i in self.Sig_.blocks] 
                                      for Eps_i in self.Eps.blocks])
        # gradient of threshold function w.r.t. thermodynamic forces
        df_dSig_list = [self.f_expr.diff(Sig_i) for Sig_i in self.Sig.blocks]
        df_dSig_ = sp.BlockMatrix([[df_dSig_i] for df_dSig_i in df_dSig_list])
        # gradient of threshold function w.r.t. kinematic variables
        df_dEps_list = [self.f_expr.diff(Eps_i) for Eps_i in self.Eps.blocks]
        df_dEps_ = sp.BlockMatrix([[df_dEps_i] for df_dEps_i in df_dEps_list])
        # derivative of threshold function w.r.t. irreversibility parameter $\lambda$
        return ((dSig_dEps_.as_explicit() * df_dSig_.as_explicit() + 
                 df_dEps_.as_explicit()).T * self.Phi_.as_explicit())[0, 0]

    _f_df_dlambda_lambdified = tr.Property()
    @tr.cached_property
    def _get__f_df_dlambda_lambdified(self):
        f_df_ = [self.f_expr, self.df_dlambda_]
        return sp.lambdify((self.u_vars, self.T_var, 
                            self.Eps.as_explicit(), self.Sig.as_explicit()) + self.m_params + ('**kw',), f_df_, 
                            numpy_dirac, cse=True)
    
    def get_f_df_Sig_bI(self, u_aI, T, Eps_bI, Sig_bI, **m_params):
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
        _Sig_bI = self.get_Sig_bI(u_aI, T, Eps_bI, Sig_bI, **m_params)
        _f, _df_dlambda = self._f_df_dlambda_lambdified(u_aI, T, Eps_bI, _Sig_bI, **m_params)
        return _f, _df_dlambda, _Sig_bI

    def get_f_df_Sig_Ib(self, u_Ia, T, Eps_Ib, Sig_Ib, **m_params):
        """
        Calculates the threshold function and its derivative 
        with respect to the load parameter lambda.
        """
        u_aI, Eps_bI, Sig_bI = [np.moveaxis(v, -1, 0) for v in (u_Ia, Eps_Ib, Sig_Ib)]
        _f, _df_dlambda, _Sig_bI = self.get_f_df_Sig_bI(u_aI, T, Eps_bI, Sig_bI, **m_params)
        return _f, _df_dlambda, np.moveaxis(_Sig_bI, 0, -1)

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
    
    # Evolution equations
    def get_Eps_k1_bI(self, u_n1_aI, T_n1, Eps_n_bI, lambda_k, Eps_k_bI, Sig_k_bI, **m_params):
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
        Phi_k_bI = self.get_Phi_bI(u_n1_aI, T_n1, Eps_k_bI, Sig_k_bI, **m_params)
        Eps_k1_bI = Eps_n_bI + lambda_k * Phi_k_bI
        return Eps_k1_bI
    
    def get_Eps_k1_Ib(self, u_n1_Ia, T_n1, Eps_n_Ib, lambda_k, Eps_k_Ib, Sig_k_Ib, **m_params):
        u_n1_aI, Eps_n_bI, Eps_k_bI, Sig_k_bI = [
            np.moveaxis(arg, -1, 0) for arg in (u_n1_Ia, Eps_n_Ib, Eps_k_Ib, Sig_k_Ib)
        ]
        Eps_k1_bI = self.get_Eps_k1_bI(u_n1_aI, T_n1, Eps_n_bI, lambda_k, Eps_k_bI, Sig_k_bI, **m_params)
        np.moveaxis(Eps_k1_bI, 0, -1)


    def get_state_n1(self, u_n_Ia, du_n1_Ia, T_n, dt, Sig_n_Ib, Eps_n_Ib, k_max, **kw):
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
            Tuple containing the updated strain Eps_k, stress Sig_k, temperature T_n+1, number of iterations k, and dissipation rate gradient dDiss_dEps.
        """
        u_n1_Ia = u_n_Ia + du_n1_Ia

        u_n1_aI, Eps_n_bI, Sig_n_bI = [np.moveaxis(var, -1, 0) 
                                     for var in (u_n1_Ia, Eps_n_Ib, Sig_n_Ib)]

        Eps_k_bI = np.copy(Eps_n_bI)
        Sig_k_bI = np.copy(Sig_n_bI)
        Eps_k_Ib, Sig_k_Ib = [np.moveaxis(var, -1, 0) for var in (Eps_k_bI, Sig_k_bI)]

        relax_t = self.get_t_relax(**kw)
        n_vk = len(relax_t)
        dt_tau = dt / relax_t
        inv_1_dt_tau = 1 / (np.ones_like(dt_tau) + dt_tau)

        f_k, df_k, Sig_k_bI = self.get_f_df_Sig_bI(u_n1_aI, T_n, Eps_k_bI, Sig_k_bI, **kw)
        f_k = np.atleast_1d(f_k)
        df_k = np.atleast_1d(df_k)
        f_k_norm = np.fabs(f_k)
        f_k_trial = np.copy(f_k)
        lam_k = np.zeros_like(f_k)
        k = 0
        while k < k_max:
            I = np.logical_and(f_k_trial > 0, f_k_norm >= 1e-4)
            if np.all(I == False):
                break # convergence reached
            lam_k[I] -= f_k[I] / df_k[I] # increment of lambda with delta_lambda = -f / df
            Eps_k_Ib[I] = self.get_Eps_k1_Ib(u_n1_Ia[I], T_n[I], Eps_n_Ib[I], lam_k[I], Eps_k_Ib[I], Sig_k_Ib[I], **kw)
            f_k[I], df_k[I], Sig_k_Ib[I] = self.get_f_df_Sig_Ib(u_n1_Ia[I], T_n[I], Eps_k_Ib[I], Sig_k_Ib[I], **kw)
            f_k_norm[I] = np.fabs(f_k[I])
            k += 1
        else:
            raise ValueError('no convergence')

        # u_n1, Eps_n, Eps_k, Sig_k = [np.moveaxis(var, 0, -1) for var in (u_n1, Eps_n, Eps_k, Sig_k)]

        # viscoplastic regularization
        print('f_k', f_k_trial)
        if np.any(f_k_trial > 0):
            I = np.where(f_k_trial > 0)
            ### Perzyna type model - exploiting that \gamma = f / eta corresponds to \lambda above ???
            print(lam_k[I].shape, Eps_k_Ib.shape )
            gamma_vk_Ib = np.einsum('b,b...->b...', lam_k[I],np.ones_like(Eps_k_Ib[I]))
            print(gamma_vk_Ib[...,:n_vk].shape, (dt_tau * inv_1_dt_tau)[np.newaxis,:].shape)
            gamma_vk_Ib[...,:n_vk] *= (dt_tau * inv_1_dt_tau)[np.newaxis,:]
            print('u_n1', u_n1_Ia[I].shape)
            print('Eps', Eps_n_Ib[I].shape)
            print('gamma_vk_Ib', gamma_vk_Ib.shape)
            Eps_k_Ib[I] = self.get_Eps_k1_Ib(u_n1_Ia[I], T_n[I], Eps_n_Ib[I], gamma_vk_Ib, Eps_n_Ib[I], Sig_n_Ib[I], **kw)
            Sig_k_Ib[I] = self.get_Sig_Ib(u_n1_Ia[I], T_n[I], Eps_k_Ib[I], Sig_k_Ib[I], **kw)

        return Eps_k_Ib, Sig_k_Ib, T_n, k, None

        # dEps_k = Eps_k - Eps_n
        # dDiss_dEps = self.get_dDiss_dEps(u_n1, T_n, Eps_k, Sig_k, **kw)
        # # dissipation rate
        # dDiss_dt = np.einsum('i,i', dDiss_dEps, dEps_k)
        # C_v_ = kw['C_v_']
        # dT = dt * (dDiss_dt) / C_v_ # / rho_

        # return Eps_k, Sig_k, T_n + dT, k, dDiss_dEps

    def get_response(self, u_ta, t_t, T_0, k_max=20, **kw):
        if u_ta.ndim == 2:
            u_ta = u_ta[:,np.newaxis,:]
        T_0 = np.atleast_1d(T_0)

        du_ta = np.diff(u_ta, axis=0)
        dt_t = np.diff(t_t, axis=0)

        Eps_n1 = np.zeros(u_ta.shape[1:-1] + (self.n_Eps_explicit,), dtype=np.float_)
        Sig_n1 = np.zeros_like(Eps_n1)
        Sig_record = [Sig_n1]
        Eps_record = [Eps_n1]
        dF_dEps_record = [Sig_n1]
        T_record = [T_0]
        iter_record = [0]
        T_n = T_0 # initial condition
        for n, dt in enumerate(dt_t):
#            try:
            Eps_n1, Sig_n1, T_n1, k, dF_dEps_ = self.get_state_n1(u_ta[n], du_ta[n], T_n, dt, 
                                            Sig_n1, Eps_n1, k_max, **kw)
            # except ValueError:
            #     break
            Sig_record.append(Sig_n1)
            Eps_record.append(Eps_n1)
            dF_dEps_record.append(dF_dEps_)
            T_record.append(T_n1)
            iter_record.append(k)
            print(f'{n}({k})', end='\r')
            T_n = T_n1
        Sig_t = np.array(Sig_record, dtype=np.float_)
        Eps_t = np.array(Eps_record, dtype=np.float_)
#        dF_dEps_t = np.array(dF_dEps_record, dtype=np.float_)
        T_t = np.array(T_record, dtype=np.float_)
        iter_t = np.array(iter_record,dtype=np.int_)
        n_t = len(Eps_t)
        # return t_t[:n_t], u_ta[:n_t], T_t, Eps_t, Sig_t, iter_t, dF_dEps_t
        return t_t[:n_t], u_ta[:n_t], T_t, Eps_t, Sig_t, iter_t, None

    ####################################################################

    def get_dSig_dEps(self, u, T, Eps, Sig, **m_params):
        return self._dSig_dEps_lambdified(u, T, Eps, Sig, **m_params)
    
    _dSig_dEps_lambdified = tr.Property()
    @tr.cached_property
    def _get__dSig_dEps_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), self.dSig_dEps_, 
                            numpy_dirac, cse=True)

    dSig_dEps_ = tr.Property()
    @tr.cached_property
    def _get_dSig_dEps_(self):
        # gradient of thermodynamic forces with respect to the kinematic variables
        dSig_dEps_ = sp.Matrix([[Sig_i.diff(Eps_i) 
                                for Sig_i in self.Sig_.as_explicit()]
                                for Eps_i in self.Eps.as_explicit()])
        return dSig_dEps_

    ######################################

    def get_df_dSig(self, u, T, Eps, Sig, **m_params):
        return self._df_dSig_lambdified(u, T, Eps, Sig, **m_params)
    
    _df_dSig_lambdified = tr.Property()
    @tr.cached_property
    def _get__df_dSig_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), self.df_dSig_, 
                            numpy_dirac, cse=True)

    df_dSig_ = tr.Property()
    @tr.cached_property
    def _get_df_dSig_(self):
        # gradient of threshold function w.r.t. thermodynamic forces
        df_dSig_list = [self.f_expr.diff(Sig_i) for Sig_i in self.Sig.blocks]
        df_dSig_ = sp.BlockMatrix([[df_dSig_i] for df_dSig_i in df_dSig_list])
        return df_dSig_.as_explicit()

    ######################################

    def get_df_dEps(self, u, T, Eps, Sig, **m_params):
        return self._df_dEps_lambdified(u, T, Eps, Sig, **m_params)
    
    _df_dEps_lambdified = tr.Property()
    @tr.cached_property
    def _get__df_dEps_lambdified(self):
        return sp.lambdify((self.u_vars, self.T_var, self.Eps.as_explicit(), 
                            self.Sig.as_explicit()) + self.m_params + ('**kw',), self.df_dEps_, 
                            numpy_dirac, cse=True)

    df_dEps_ = tr.Property()
    @tr.cached_property
    def _get_df_dEps_(self):
        # gradient of threshold function w.r.t. kinematic variables
        df_dEps_list = [self.f_expr.diff(Eps_i) for Eps_i in self.Eps.blocks]
        df_dEps_ = sp.BlockMatrix([[df_dEps_i] for df_dEps_i in df_dEps_list])
        return df_dEps_.as_explicit()

    ###################################################################


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

