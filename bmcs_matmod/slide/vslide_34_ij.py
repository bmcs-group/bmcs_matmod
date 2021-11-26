
from ibvpy.tmodel.mats3D.mats3D_eval import MATS3DEval
import bmcs_utils.api as bu
import numpy as np
import traits.api as tr

from .vslide_34 import Slide34

#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
class MATS3DSlideStrain(MATS3DEval):
    r'''
    Isotropic damage model.
    '''
    node_name = 'Slide3D'

    n_a = tr.Array(np.float_, value=[0,1,0])

    slide_displ = bu.Instance(Slide34, ())

    tree = ['slide_displ']

    ipw_view = bu.View(
        bu.Item('n_a'),
        bu.Item('slide_displ')
    )

    state_var_shapes = tr.Property
    @tr.cached_property
    def _get_var_shapes(self):
        return self.slide_displ.state_var_shapes
    r'''
    Shapes of the state variables
    to be stored in the global array at the level 
    of the domain.
    '''

    def get_eps_N(self, eps_ij, n_i):
        eps_N = np.einsum('...ij,...i,...j->...', eps_ij, n_i, n_i)
        return eps_N

    def get_eps_T(self, eps_ij, n_i):
        delta_ij = np.identity(3)
        eps_T = 0.5 * (np.einsum('...i,...jk,...ij->...k', n_i, delta_ij, eps_ij)
                       + np.einsum('...j,...ik,...ij->...k', n_i, delta_ij, eps_ij)
                       - 2 * np.einsum('...i,...j,...k,...ij->...k', n_i, n_i, n_i, eps_ij))
        return eps_T

    def get_eps_T_p(self, eps_T_p, eps_T):
        director_vector = [0, 0, 1]
        eps_T_p = np.einsum('...,...i->...i', eps_T_p, director_vector)
        return eps_T_p

    def get_E_T(self, E, nu, n_i):
        delta_ij = np.identity(3)
        D_ijkl = self.D_abef
        operator = 0.5 * (np.einsum('i,jk,l->ijkl', n_i, delta_ij, n_i)
                          + np.einsum('j,ik,l->jikl', n_i, delta_ij, n_i)
                          - 2 * np.einsum('i,j,k,l->ijkl', n_i, n_i, n_i, n_i))
        E_T = np.einsum('ijkl,ijkl->', D_ijkl, operator)
        return E_T

    def get_corr_pred(self, eps_Emab_n1, tn1, **state):
        r'''
        Corrector predictor computation.
        '''
        n_i = self.n_a
        eps_ij = eps_Emab_n1
        eps_N = np.einsum('...ij,...i,...j->...', eps_ij, n_i, n_i)
        eps_T = self.get_eps_T(eps_ij, n_i)

        se = self.slide_displ
        eps_NT_Ema = np.concatenate([eps_N, eps_T], axis=-1)
        sig_NT_Ema, D_Emab = se.get_corr_pred(eps_NT_Ema, tn1, **state)

        eps_N_p, eps_T_p_x, eps_T_p_y = state['w_pi'], state['s_pi_x'], state['s_pi_y']
        eps_T = self.get_eps_T(eps_ij, n_i)
        eps_T_p_i = self.get_eps_T_p(eps_T_p_x, eps_T)
        omega_N_Em, omega_T_Em = state['omega_N'], state['omega_T']

        phi_Emab = np.zeros_like(eps_ij)
        phi_Emab[1, 1] = 0.
        phi_Emab[2, 2] = np.sqrt(1 - omega_T_Em)
        phi_Emab[0, 0] = np.sqrt(1 - omega_N_Em)

        beta_Emijkl = np.einsum('...ik,...jl->...ijkl', phi_Emab,
                              np.transpose(phi_Emab, (1, 0)))

        eps_ij_p = (np.einsum('i,...j->...ij', n_i, eps_T_p_i) +
                    np.einsum('...i,j->...ij', eps_T_p_i,n_i)) + \
                    np.einsum('...,i,j->...ij',eps_N_p, n_i, n_i)

        D_abef = self.D_abef

        D_Emabcd = np.einsum('...ijkl,klrs,...rstu->...ijtu', beta_Emijkl, D_abef, beta_Emijkl)

        sigma_Emab = np.einsum('...ijkl,...kl->...ij', D_Emabcd, (eps_Emab_n1 - eps_ij_p))

        return sigma_Emab, D_Emabcd


