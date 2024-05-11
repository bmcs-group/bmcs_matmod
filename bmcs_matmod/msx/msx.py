'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model 3D

(compressive plasticity (CP) + tensile damage (TD) 
+ cumulative damage sliding (CSD))

Using Jirasek homogenization approach [1999]
'''
import bmcs_utils.trait_types
import bmcs_utils.view
import numpy as np
import traits.api as tr
from bmcs_utils.api import \
    Float, Instance, EitherType, View, Item, Model, Bool
from bmcs_matmod.ntim import INTIM, VCoNTIM, VUNTIM, VUNTIM_M, VDNTIM, ReturnMappingError
from ibvpy.tmodel.mats3D.mats3D_eval import MATS3DEval
from bmcs_matmod.msx.ms_integ_schemes import MSIntegScheme, MSIS3DM28
from bmcs_matmod.msx.energy_dissipation import EnergyDissipation
from ibvpy.api import XDomainSinglePoint, TStepBC, BCDof


class MSX(MATS3DEval):

    name = 'MSX'

    double_pvw = Bool(True, MAT=True)

    ipw_view = View(
        Item('E'),
        Item('nu'),
        Item('double_pvw'),
        Item('eps_max'),
        Item('n_eps')
    )

    mic = EitherType(options=[('contim', VCoNTIM),
                              ('untim', VUNTIM),
                              ('untim_m', VUNTIM_M),
                              ('dntim', VDNTIM)],
                     on_option_change='reset_mic')

    integ_scheme = EitherType(options=[('3DM28', MSIS3DM28)])

    @tr.on_trait_change('E, nu')
    def _set_E(self, event):
        self.reset_mic()

    def reset_mic(self):
        self.mic_.E_N = self.E / (1.0 - 2.0 * self.nu)
        self.mic_.E_T = self.E * (1.0 - 4 * self.nu) / \
                 ((1.0 + self.nu) * (1.0 - 2 * self.nu))

    tree = ['mic', 'integ_scheme']
    depends_on = ['mic', 'integ_scheme']

    state_var_shapes = tr.Property(depends_on='mic_, integ_scheme_')
    @tr.cached_property
    def _get_state_var_shapes(self):
        sv_shapes = {
            name: (self.integ_scheme_.n_mp,) + shape
            for name, shape
            in self.mic_.state_var_shapes.items()
        }
        return sv_shapes

    def _get_e_a(self, eps_ab):
        """
        Get the microplane projected strains
        """
        # get the normal strain array for each microplane
        e_N = np.einsum('nij,...ij->...n', self.integ_scheme_.MPNN, eps_ab)
        # get the tangential strain vector array for each microplane
        MPTT_ijr = self.integ_scheme_.MPTT
        e_T_a = np.einsum('nija,...ij->...na', MPTT_ijr, eps_ab)
        return np.concatenate([e_N[..., np.newaxis], e_T_a], axis=-1)

    def _get_beta_abcd(self, eps_ab, omega_N, omega_T, **Eps):
        """
        Returns the 4th order damage tensor 'beta4' using
        (cf. [Baz99], Eq.(63))
        """
        MPW = self.integ_scheme_.MPW
        MPN = self.integ_scheme_.MPN

        delta = np.identity(3)
        beta_N = np.sqrt(1. - omega_N)
        beta_T = np.sqrt(1. - omega_T)

        beta_ijkl = (
            np.einsum('n,...n,ni,nj,nk,nl->...ijkl',
                MPW, beta_N, MPN, MPN, MPN, MPN)
            + 0.25 *
            (
                np.einsum('n,...n,ni,nk,jl->...ijkl',
                          MPW, beta_T, MPN, MPN, delta) +
                np.einsum('n,...n,ni,nl,jk->...ijkl',
                          MPW, beta_T, MPN, MPN, delta) +
                np.einsum('n,...n,nj,nk,il->...ijkl',
                          MPW, beta_T, MPN, MPN, delta) +
                np.einsum('n,...n,nj,nl,ik->...ijkl',
                          MPW, beta_T, MPN, MPN, delta) -
                4.0 *
                np.einsum('n,...n,ni,nj,nk,nl->...ijkl',
                          MPW, beta_T, MPN, MPN, MPN, MPN)
            )
        )
        return beta_ijkl

    def NT_to_ab(self, v_N, v_T_a):
        """
        Integration of the (inelastic) strains for each microplane
        """
        MPW = self.integ_scheme_.MPW
        MPN = self.integ_scheme_.MPN

        delta = np.identity(3)
        # 2-nd order plastic (inelastic) tensor
        tns_ab = (
                np.einsum('n,...n,na,nb->...ab',
                          MPW, v_N, MPN, MPN) +
                0.5 * (
                        np.einsum('n,...nf,na,fb->...ab',
                                  MPW, v_T_a, MPN, delta) +
                        np.einsum('n,...nf,nb,fa->...ab',
                                  MPW, v_T_a, MPN, delta)
                )
        )
        return tns_ab

    def get_corr_pred(self, eps_ab, t_n1, **Eps):
        """
        Corrector predictor computation.
        """
        # ------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        # ------------------------------------------------------------------
        eps_a = self._get_e_a(eps_ab)
        sig_a, D_ab = self.mic_.get_corr_pred(eps_a, t_n1, **Eps)
        beta_abcd = self._get_beta_abcd(eps_ab, **Eps)
        # ------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        # ------------------------------------------------------------------
        D_abcd = np.einsum('...ijab, abef, ...cdef->...ijcd',
                           beta_abcd, self.D_abef, beta_abcd)

        if self.double_pvw:
            if eps_p_a := self.mic_.get_eps_NT_p(**Eps):
                eps_N_p, eps_T_p_a = eps_p_a
                eps_p_ab = self.NT_to_ab(eps_N_p, eps_T_p_a)
                eps_e_ab = eps_ab - eps_p_ab
            else:
                eps_e_ab = eps_ab
            sig_ab = np.einsum('...abcd,...cd->...ab', D_abcd, eps_e_ab)
        else:
            sig_N, sig_T_a = sig_a[...,0], sig_a[...,1:]
            sig_ab = self.NT_to_ab(sig_N, sig_T_a)

        return sig_ab, D_abcd

    def update_plot(self, axes):
        ax_sig, ax_d_sig = axes
        ax_work = ax_sig.twinx()
        eps_max = self.eps_max
        n_eps = self.n_eps
        n_eps = 100
        xmodel = XDomainSinglePoint()
        m = TStepBC(
            domains=[(xmodel, self), ],
            bc=[BCDof(
                var='u', dof=0, value=eps_max,
            )]
        )
        m.sim.tline.trait_set(step=float(1.0/n_eps))
        m.sim.run()
        energydissipation = EnergyDissipation()
        # W_arr_micro, W_arr_macro = energydissipation.plot_work(m)

        eps_t = m.hist.U_t[:, 0]
        sig_t = m.hist.F_t[:, 0]
        ax_sig.plot(eps_t, sig_t, linestyle='solid', color='blue',label=r'$\sigma - \varepsilon$')
        # ax_work.plot(eps_t, W_arr_micro, linestyle='dashed', color='red', label='microplane work')
        # ax_work.plot(eps_t, W_arr_macro, linestyle='solid', color='red', label='macroscopic work')
        ax_sig.set_xlabel(r'$\varepsilon_{11}$ [-]')
        ax_sig.set_ylabel(r'$\sigma_{11}$ [MPa]')
        ax_sig.legend()
        ax_work.legend()

    def xupdate_plot(self, axes):
        ax_sig, ax_d_sig = axes
        eps_max = self.eps_max
        n_eps = self.n_eps
        eps11_range = np.linspace(1e-9, eps_max, n_eps)
        eps_range = np.zeros((n_eps, 3, 3))
        eps_range[:, 0, 0] = eps11_range
        state_vars = {
            var: np.zeros((1,) + shape)
            for var, shape in self.state_var_shapes.items()
        }
        sig11_range, d_sig1111_range, state_vars_record = [], [], []
        for eps_ab in eps_range:
            try:
                sig_ab, D_abcd = self.get_corr_pred(eps_ab[np.newaxis, ...], 1, **state_vars)
            except ReturnMappingError:
                break
            sig11_range.append(sig_ab[0, 0, 0])
            d_sig1111_range.append(D_abcd[0, 0, 0, 0, 0])
        sig11_range = np.array(sig11_range, dtype=np.float_)
        eps11_range = eps11_range[:len(sig11_range)]
        ax_sig.plot(eps11_range, sig11_range, color='blue')
        d_sig1111_range = np.array(d_sig1111_range, dtype=np.float_)
        ax_d_sig.plot(eps11_range, d_sig1111_range, linestyle='dashed', color='gray')
        ax_sig.set_xlabel(r'$\varepsilon_{11}$ [-]')
        ax_sig.set_ylabel(r'$\sigma_{11}$ [MPa]')
        ax_d_sig.set_ylabel(r'$\mathrm{d} \sigma_{11} / \mathrm{d} \varepsilon_{11}$ [MPa]')
        ax_d_sig.plot(eps11_range[:-1],
                    (sig11_range[:-1]-sig11_range[1:])/(eps11_range[:-1]-eps11_range[1:]),
                    color='orange', linestyle='dashed')

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     mic = MSX(E=28000, nu=0.2,
#           mic='untim', eps_max=0.01, n_eps=100, double_pvw=False)
#     fig = plt.figure()
#     ax_sig = fig.subplots(1, 1)
#     ax_d_sig = ax_sig.twinx()
#     axes = ax_sig, ax_d_sig
#     m = mic.update_plot(axes)
#     energydissipation = EnergyDissipation()
#     fig = energydissipation.plot_energy_dissp(m,mic.mic_)
