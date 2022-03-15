'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model 3D

(compressive plasticity (CP) + tensile damage (TD)
+ cumulative damage sliding (CSD))

Using Jirasek homogenization approach [1999]
'''
import bmcs_utils.view
import numpy as np
import traits.api as tr
from bmcs_utils.api import Float, Instance, EitherType, View, Item
from .i_ntim import INTIM
from ibvpy.tmodel.mats3D.mats3D_eval import MATS3DEval

@tr.provides(INTIM)
class VDNTIM(MATS3DEval):
    """
    Vectorized uncoupled normal tngential interface model
    """
    # -------------------------------------------------------------------------
    # Elasticity
    # -------------------------------------------------------------------------
    name = 'equiv damage NTIM'

    E_N = Float(10000, MAT=True)

    E_T = Float(1000, MAT=True)

    epsilon_0 = Float(59.0e-6, MAT=True,
                      label="a",
                      desc="Lateral pressure coefficient",
                      enter_set=True,
                      auto_set=False)

    epsilon_f = Float(250.0e-6, MAT=True,
                      label="a",
                      desc="Lateral pressure coefficient",
                      enter_set=True,
                      auto_set=False)

    c_T = Float(0.01, MAT=True,
                label="a",
                desc="Lateral pressure coefficient",
                enter_set=True,
                auto_set=False)

    ipw_view = View(
        Item('E_T', readonly=True),
        Item('E_N', readonly=True),
        Item('epsilon_0'),
        Item('epsilon_f'),
        Item('c_T'),
        Item('eps_max'),
        Item('n_eps')
    )

    n_D = 3

    state_var_shapes = dict(
        kappa=(),
        omega_N=(),
        omega_T=()
    )

    def get_e_equiv(self, eps_N, eps_T):
        r"""
        Returns a list of the microplane equivalent strains
        based on the list of microplane strain vectors
        """
        # positive part of the normal strain magnitude for each microplane
        e_N_pos = (np.abs(eps_N) + eps_N) / 2.0
        # tangent strain ratio
        c_T = self.c_T
        # equivalent strain for each microplane
        eps_equiv = np.sqrt(e_N_pos * e_N_pos + c_T * eps_T)
        return eps_equiv

    def get_normal_law(self, eps_N, eps_T_a, kappa, omega_N, omega_T):
        E_N = self.E_N
        E_T = self.E_T
        eps_T = np.sqrt(np.einsum('...a,...a->...', eps_T_a, eps_T_a))
        eps_equiv = self.get_e_equiv(eps_N, eps_T)
        kappa[...] = np.max(np.array([kappa, eps_equiv]), axis=0)
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        I = np.where(kappa >= epsilon_0)
        omega_N[I] = (
            1.0 - (epsilon_0 / kappa[I] *
                   np.exp(-1.0 * (kappa[I] - epsilon_0) / (epsilon_f - epsilon_0))
            )
        )
        sig_N = (1 - omega_N) * E_N * eps_N
        sig_T_a = (1 - omega_N[..., np.newaxis]) * E_T * eps_T_a
        sig_a = np.concatenate([sig_N[...,np.newaxis], sig_T_a], axis=-1)
        return sig_a

    def get_corr_pred(self, eps_a, t_n1, **Eps):
        eps_a_ = np.einsum('...a->a...',eps_a)
        eps_N = eps_a_[0,...]
        eps_T_a = np.einsum('a...->...a', eps_a_[1:,...])
        sig_a = self.get_normal_law(eps_N, eps_T_a, **Eps)

        D_ = np.zeros(eps_a.shape + (eps_a.shape[-1],))
        D_[..., 0, 0] = self.E_N# * (1 - omega_N)
        D_[..., 1, 1] = self.E_T# * (1 - omega_T)
        D_[..., 2, 2] = self.E_T# * (1 - omega_T)
        D_[..., 3, 3] = self.E_T# * (1 - omega_T)

        return sig_a, D_

    def get_eps_NT_p(self, **Eps):
        """Plastic strain tensor
        """
        return None

    def plot_idx(self, ax_sig, ax_d_sig, idx=0):
        eps_max = self.eps_max
        n_eps = self.n_eps
        eps1_range = np.linspace(1e-9,eps_max,n_eps)
        Eps = { var : np.zeros( (1,) + shape )
            for var, shape in self.state_var_shapes.items()
        }
        eps_range = np.zeros((n_eps, 4))
        eps_range[:,idx] = eps1_range

        # monotonic load in the normal direction
        sig1_range, d_sig11_range = [], []
        for eps_a in eps_range:
            sig_a, D_range = self.get_corr_pred(eps_a[np.newaxis, ...], 1, **Eps)
            sig1_range.append(sig_a[0, idx])
            d_sig11_range.append(D_range[0, idx, idx])
        sig1_range = np.array(sig1_range, dtype=np.float_)
        eps1_range = eps1_range[:len(sig1_range)]

        ax_sig.plot(eps1_range, sig1_range,color='blue')
        d_sig11_range = np.array(d_sig11_range, dtype=np.float_)
        ax_d_sig.plot(eps1_range, d_sig11_range, linestyle='dashed', color='gray')
        ax_sig.set_xlabel(r'$\varepsilon_{11}$ [-]')
        ax_sig.set_ylabel(r'$\sigma_{11}$ [MPa]')
        ax_d_sig.set_ylabel(r'$\mathrm{d} \sigma_{11} / \mathrm{d} \varepsilon_{11}$ [MPa]')
        ax_d_sig.plot(eps1_range[:-1],
                    (sig1_range[:-1]-sig1_range[1:])/(eps1_range[:-1]-eps1_range[1:]),
                    color='orange', linestyle='dashed')

    def subplots(self, fig):
        ax_sig_N, ax_sig_T = fig.subplots(1,2)
        ax_d_sig_N = ax_sig_N.twinx()
        ax_d_sig_T = ax_sig_T.twinx()
        return ax_sig_N, ax_d_sig_N, ax_sig_T, ax_d_sig_T

    def update_plot(self, axes):
        ax_sig_N, ax_d_sig_N, ax_sig_T, ax_d_sig_T = axes
        self.plot_idx(ax_sig_N, ax_d_sig_N, 0)
        self.plot_idx(ax_sig_T, ax_d_sig_T, 1)