"""Generalized standard material

Formulation of a generic framework for the derivation of constitutive models capturing 
dissipative processes in response of inelastic materials to mechanical and environmental 
loading. This framework aspires to combine the high-level mathematical formulation of the
thermodynamic potentials with an efficient execution of the automatically derived evolution
equations in the computational environment.
"""

from re import M
import traits.api as tr
#import bmcs_utils.api as bu
from bmcs_utils.api import Cymbol
import sympy as sp
import numpy as np
import dill
import os
import functools

def get_dirac_delta(x, x_0=0):
    return 0
numpy_dirac =[{'DiracDelta': get_dirac_delta }, 'numpy']

# Directory to store the serialized symbolic instances
CACHE_DIR = '_lambdified_cache'

def lambdify_and_cache(func):
    @functools.wraps(func)
    def wrapper(self):
        # Generate the filename based on class name and property name
        class_name = self.__class__.__name__
        object_name = self.name
        property_name = func.__name__
        cache_dir = CACHE_DIR
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

class GSMNR(tr.HasTraits):
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

    name = tr.Str('unnamed')

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

    sig_sign = tr.Float(1)
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

    vp_on = tr.Bool(True)
    """Viscoplasticity on or off   
    """ 

    update_at_k = tr.Bool(True)
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
       
    dot_Eps_list = tr.Property
    @tr.cached_property
    def _get_dot_Eps_list(self):
        """
        Compute the time derivative of internal state variables.

        This cached property method generates a list of symbolic matrices representing
        the time derivatives of the internal state variables (`Eps_vars`). Each element in the
        list corresponds to an internal state variable matrix from `Eps_vars`, where each entry
        in the matrix is replaced by its time derivative.

        Returns:
            list: A list of `sympy.Matrix` objects, where each matrix contains the
                time derivatives of the corresponding internal state variable matrix from
                `Eps_vars`. The time derivatives are represented as `Cymbol` objects
                with names and codenames indicating the time derivative.
        """
        return [
            sp.Matrix(eps_var.shape[0], eps_var.shape[1], 
                    lambda i, j: Cymbol(name=f'\\dot{{{eps_var[i, j].name}}}', 
                                        codename=f'dot_{eps_var[i, j].codename}'))
            for eps_var in self.Eps_list
        ]
    
    dot_Eps = tr.Property()
    @tr.cached_property
    def _get_dot_Eps(self):
        return sp.BlockMatrix(self.dot_Eps_list).T

    # Conjugate state variable representations and conversions
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
        """
        Compute the time derivative of thermodynamic forces.

        This cached property method generates a list of symbolic matrices representing
        the time derivatives of the thermodynamic forces (`Sig_vars`). Each element in the
        list corresponds to a thermodynamic force matrix from `Sig_vars`, where each entry
        in the matrix is replaced by its time derivative.

        Returns:
            list: A list of `sympy.Matrix` objects, where each matrix contains the
                time derivatives of the corresponding thermodynamic force matrix from
                `Sig_vars`. The time derivatives are represented as `Cymbol` objects
                with names and codenames indicating the time derivative.
        """
        return [
            sp.Matrix(sig_var.shape[0], sig_var.shape[1], 
                    lambda i, j: Cymbol(name=f'\\dot{{{sig_var[i, j].name}}}', 
                                        codename=f'dot_{sig_var[i, j].codename}'))
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
        "For Gibbs for external strain the sign is swapped using the sig_sign parameter = -1"
        return self.sig_sign * self.F_expr.diff(self.u_vars)

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
        dFG_dEps_explicit_ = self.dF_dEps_.as_explicit()
        return (self.T_var * dFG_dEps_explicit_.diff(self.T_var) - dFG_dEps_explicit_)

    
    ######################################
    def get_Sig(self, eps, Eps, **m_params):
        """
        Calculates the stress based on the given inputs.

        Args:
            eps: Strain.
            T: Temperature.
            Eps: Strain.
            Sig: Stress.
            **m_params: Additional model parameters.

        Returns:
            Calculated stress.
        """
        eps_sp_ = np.moveaxis(np.atleast_1d(eps), -1, 0)
        Eps_sp_ = np.moveaxis(Eps, -1, 0)
        Sig_sp = self._Sig_lambdified(eps_sp_, Eps_sp_, **m_params)
        Sig_sp_ = Sig_sp.reshape(Eps_sp_.shape)
        return np.moveaxis(Sig_sp_, 0, -1)

        # Eps_sp_ = np.moveaxis(Eps, 0, -1)
        # Sig_sp = self._Sig_lambdified(eps, Eps_sp_, **m_params)
        # return np.moveaxis(Sig_sp.reshape(Eps.shape), 0, -1)
    
    _Sig_lambdified = tr.Property()
    @tr.cached_property
#    @lambdify_and_cache
    def _get__Sig_lambdified(self):
        return sp.lambdify((self.u_vars[0], 
                            self.Eps.as_explicit()) + self.m_params + ('**kw',), 
                           self.Sig_.as_explicit(), numpy_dirac, cse=True)

    Sig_ = tr.Property()
    @tr.cached_property
    def _get_Sig_(self):
        return sp.BlockMatrix([(sign_i_ * dF_dEps_i_).T for sign_i_, dF_dEps_i_ 
                               in zip(self.Sig_signs, self.dF_dEps_.blocks)]).T    

    ######################################


    ######################################

    phi_ = tr.Property()
    @tr.cached_property
    def _get_phi_(self):
        return (self.f_expr + self.phi_ext_expr).subs(self.m_param_subs)

    Phi_ = tr.Property()
    @tr.cached_property
    def _get_Phi_(self):
        Phi_list = [-sign_i_ * self.phi_.diff(Sig_i_) for sign_i_, Sig_i_ 
                    in zip(self.Sig_signs, self.Sig.blocks)]
        return sp.BlockMatrix([[Phi_i] for Phi_i in Phi_list]).as_explicit()

    subs_Sig_Eps = tr.Property()
    @tr.cached_property
    def _get_subs_Sig_Eps(self):
        return dict(zip(self.Sig.as_explicit(), self.Sig_.as_explicit()))

    Phi_Eps_ = tr.Property()
    @tr.cached_property
    def _get_Phi_Eps_(self):
        return self.Phi_.subs(self.subs_Sig_Eps)

    ######################################

    def get_f_R_dR_n1(self, eps_n, d_eps, Eps_n, d_A, d_t, **kw):
        eps_n_sp_ = np.moveaxis(np.atleast_1d(eps_n), -1, 0)
        d_eps_sp_ = np.moveaxis(np.atleast_1d(d_eps), -1, 0)
        O_ = np.zeros_like(eps_n_sp_)
        I_ = np.ones_like(eps_n_sp_)
        d_A_sp_ = np.moveaxis(d_A, -1, 0)
        Eps_n_sp_ = np.moveaxis(Eps_n, -1, 0)
        f_sp_, R_sp_, d_R_sp_ = self._get_f_R_dR_n1_lambdified(eps_n_sp_, d_eps_sp_, Eps_n_sp_, d_A_sp_, d_t, O_, I_, **kw)
        return np.moveaxis(f_sp_, 0, -1), np.moveaxis(R_sp_[:, 0], 0, -1), np.moveaxis(d_R_sp_, (0, 1), (-2, -1))

    _get_f_R_dR_n1_lambdified = tr.Property()
    @tr.cached_property
#    @lambdify_and_cache
    def _get__get_f_R_dR_n1_lambdified(self):

        def get_dirac_delta(x, x_0=0):
            return 0
        numpy_dirac =[{'DiracDelta': get_dirac_delta }, 'numpy']

        _, (eps_n, delta_eps, Eps_n, delta_A, delta_t, self.Ox, self.Ix), f_n1, R_n1, dR_dA_n1_OI = self.f_R_dR_n1
        return sp.lambdify((eps_n, delta_eps, Eps_n, delta_A, delta_t, self.Ox, self.Ix)  + self.m_params + ('**kw',), 
                            (f_n1, R_n1, dR_dA_n1_OI), numpy_dirac, cse=True)

    
    Ox = Cymbol(name='O', codename='O', real=True)
    Ix = Cymbol(name='I', codename='I', real=True)
    # Function to replace zero elements with a symbolic variable
    def replace_zeros_and_ones_with_symbolic(self, dR_dA_, delta_A):
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
    
    f_R_dR_n1 = tr.Property()
    @tr.cached_property
    def _get_f_R_dR_n1(self):
        ## Manual construction of the residuum
        Eps = self.Eps.as_explicit()
        eps = self.u_vars[0]
        dot_Eps = sp.Matrix([Cymbol(name=f'\\dot{{{var.name}}}',
                                    codename=f'dot_{var.codename}') for var in list(Eps)])
        dot_eps = Cymbol(name=f'\\dot{{{eps.name}}}', codename=f'dot_{eps.codename}')
        dot_lam = Cymbol(r'\dot{\lambda}', codename='dot_lam', real=True)

        Sig = self.Sig.as_explicit()
        Sig_ = self.Sig_.as_explicit()

        f_ = self.f_expr

        # sp.diag(*Sig_signs) * 
        dot_Eps_implicit = -dot_Eps + dot_lam * self.Phi_Eps_
        R_ = dot_Eps_implicit.row_insert(dot_Eps_implicit.shape[0], -sp.Matrix([f_.subs(dict(zip(Sig, Sig_)))]))

        # time
        t = Cymbol(r't', codename='t', real=True)
        delta_t = Cymbol(r'\Delta t', codename='delta_t', real=True)

        # fundamental state
        Eps_n = sp.Matrix([Cymbol(f'{var.name}_{{(n)}}', codename=f'{var.name}_{{(n)}}', real=True) for var in Eps])
        eps_n = Cymbol(r'\varepsilon_n', codename='eps_n', real=True)

        # increment
        delta_Eps = sp.Matrix([Cymbol(f'\\Delta{{{var.name}}}', codename=f'delta_{{{var.name}}}',
                                    real=True) for var in Eps])
        delta_eps = Cymbol(r'\Delta{\varepsilon}_n', codename='delta_eps_n', real=True)
        delta_lam = Cymbol(r'\Delta{\lambda}_n', codename='delta_lam_n', real=True)

        # updated state
        Eps_n1 = Eps_n + delta_Eps
        eps_n1 = eps_n + delta_eps

        # rate of change
        dot_Eps_n = delta_Eps / delta_t
        dot_eps_n = delta_eps / delta_t
        dot_lam_n = delta_lam / delta_t

        # derive substitutions
        subs_dot_Eps = dict(zip(dot_Eps, dot_Eps_n))
        subs_dot_eps = {dot_eps: dot_eps_n}
        subs_dot_lam = {dot_lam: dot_lam_n}
        subs_Eps = dict(zip(Eps, Eps_n1))
        subs_eps = {eps: eps_n1}

        subs_n1 = {**subs_dot_Eps, **subs_dot_eps, **subs_dot_lam, **subs_Eps, **subs_eps}

        # residuum vector in n+1 step
        R_n1 = R_.subs(subs_n1)
        f_n1 = f_.subs(self.subs_Sig_Eps).subs(subs_n1)

        # construct the jacobian of the residuum
        delta_A = sp.Matrix([delta_Eps, delta_lam])
        dR_dA_n1_ = R_n1.jacobian(delta_A)
        dR_dA_n1 = dR_dA_n1_.replace(sp.Derivative, lambda *args: 0)

        # replace zeros and constant terms with symbolic variables to broadcast properly
        dR_dA_n1_OI = self.replace_zeros_and_ones_with_symbolic(dR_dA_n1, delta_A)

        return R_, (eps_n, delta_eps, Eps_n, delta_A, delta_t, self.Ox, self.Ix), f_n1, R_n1, dR_dA_n1_OI
    
    ######################################

    def get_state_n1(self, eps_n, d_eps, d_t, Eps_n, k_max, **kw):
        """
        Calculates the state at time n+1 based on the given inputs using an iterative algorithm.

        Args:
            eps_n: strain at time n.
            d_eps: strain increment from time n to n+1.
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
        n_I = np.atleast_1d(eps_n).shape[0]
        d_A = np.zeros((n_I, self.n_Eps_explicit+1), dtype=np.float_)
        tol = 1e-8
        k_I = np.zeros((n_I,), dtype=np.int_)
        f_n1, R_n1, dR_dA_n1 = self.get_f_R_dR_n1(eps_n, d_eps, Eps_n, d_A, d_t, **kw)
        I_inel = f_n1 > 0
        I = np.copy(I_inel)
        for k in range(k_max):
            if np.all(I == False):
                break
            d_A[I] += np.linalg.solve(dR_dA_n1[I], -R_n1[I])
            d_A[d_A[..., 2] > 1] = 0.9999
            # TODO include thermodynamic stresses in get_f_R_dR_n1
            # print(f'eps_n:{eps_n.shape}, d_eps:{d_eps.shape}, Eps_n:{Eps_n.shape}, d_A:{d_A.shape}, d_t:{d_t}')
            f_n1[I], R_n1[I], dR_dA_n1[I] = self.get_f_R_dR_n1(eps_n[I], d_eps[I], Eps_n[I], d_A[I], d_t, **kw)
            k_I[I] += 1
            # This contains redundancy - only the inelastic strains need to be considered.
            # However, the implementation is kept simple for now.
            norm_R_n1 = np.linalg.norm(R_n1, axis=-1)
            I[norm_R_n1 <= tol] = False

        lam_k = d_A[..., -1]
        Eps_n1 = np.where(I_inel[:, np.newaxis], Eps_n + d_A[..., :-1], Eps_n)
        eps_n1 = eps_n + d_eps
        Sig_n1 = self.get_Sig(eps_n1, Eps_n1, **kw)

        return Eps_n1, Sig_n1, lam_k, k_I

    def get_response(self, eps_ta, t_t, k_max=20, **kw):
        """Time integration procedure 
        """
        if eps_ta.ndim == 2:
            eps_ta = eps_ta[:,np.newaxis,:]

        if eps_ta.ndim == 1:
            eps_ta = eps_ta[:, np.newaxis]

        d_eps_ta = np.diff(eps_ta, axis=0)
        d_t_t = np.diff(t_t, axis=0)

        Eps_n1 = np.zeros(eps_ta.shape[1:] + (self.n_Eps_explicit,), dtype=np.float_)
        Sig_n1 = np.zeros_like(Eps_n1)

        Sig_record = [Sig_n1]
        Eps_record = [Eps_n1]
        iter_record = [0]
        lam_record = [0]

        for n, dt in enumerate(d_t_t):
            print('increment', n+1, end='\r')
            try:
                Eps_n1, Sig_n1, lam, k = self.get_state_n1(
                    eps_ta[n], d_eps_ta[n], dt, Eps_n1, k_max, **kw
                )
            except RuntimeError as e:
                print(f'{n+1}({k}) ... {str(e)}', end='\r')
                break
            Sig_record.append(Sig_n1)
            Eps_record.append(Eps_n1)
            iter_record.append(k)
            lam_record.append(lam)
        Sig_t = np.array(Sig_record, dtype=np.float_)
        Eps_t = np.array(Eps_record, dtype=np.float_)
        iter_t = np.array(iter_record,dtype=np.int_)
        lam_t = np.array(lam_record,dtype=np.float_)
        n_t = len(Eps_t)
        return (t_t[:n_t], eps_ta[:n_t], Eps_t, Sig_t, iter_t, 
                lam_t, (d_t_t[:n_t], d_eps_ta[:n_t]))
    
    def save_to_disk(self):
        """Serialize this instance to disk."""
        filepath = os.path.join(CACHE_DIR, f"{self.name}.pkl")
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(filepath, 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load_from_disk(name):
        """Deserialize an instance from disk."""
        filepath = os.path.join(CACHE_DIR, f"{name}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No cached data found for {name}.")
        with open(filepath, 'rb') as file:
            return dill.load(file)