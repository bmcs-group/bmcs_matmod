
import bmcs_utils.api as bu
import traits.api as tr
from scipy.integrate import cumtrapz
import numpy as np
from types import SimpleNamespace
from bmcs_matmod.msx.ms_integ_schemes import MSIS3DM28
import matplotlib.pyplot as plt

class EnergyDissipation(bu.InteractiveModel):
    name='Energy'

    colors = dict( # color associations
        stored_energy = 'darkgreen', # recoverable
        free_energy_kin = 'darkcyan', # freedom - sky
        free_energy_iso = 'darkslateblue', # freedom - sky
        plastic_diss_s = 'darkorange', # fire - heat
        plastic_diss_w = 'red', # fire - heat
        damage_diss_s = 'darkgray', # ruined
        damage_diss_w = 'black'  # ruined
    )
    slider_exp = tr.WeakRef(bu.InteractiveModel)

    t_arr = tr.DelegatesTo('slider_exp')
    Sig_arr = tr.DelegatesTo('slider_exp')
    Eps_arr = tr.DelegatesTo('slider_exp')
    s_x_t = tr.DelegatesTo('slider_exp')
    s_y_t = tr.DelegatesTo('slider_exp')
    w_t = tr.DelegatesTo('slider_exp')
    iter_t = tr.DelegatesTo('slider_exp')

    show_iter = bu.Bool(False)
    E_plastic_work = bu.Bool(False)
    E_iso_free_energy = bu.Bool(True)
    E_kin_free_energy = bu.Bool(True)
    E_plastic_diss = bu.Bool(True)
    E_damage_diss = bu.Bool(True)

    ipw_view = bu.View(
        bu.Item('show_iter'),
        bu.Item('E_damage_diss'),
        bu.Item('E_plastic_work'),
        bu.Item('E_iso_free_energy'),
        bu.Item('E_kin_free_energy'),
        bu.Item('E_plastic_diss'),
    )

    WUG_t = tr.Property
    def _get_W_t(self, **Eps_k):
        W_arr = (
                cumtrapz(self.Sig_arr[:, 0], self.s_x_t, initial=0) +
                cumtrapz(self.Sig_arr[:, 1], self.s_y_t, initial=0) +
                cumtrapz(self.Sig_arr[:, 2], self.w_t, initial=0)
        )
        s_x_el_t = (self.s_x_t - self.Eps_arr[:, 0])
        s_y_el_t = (self.s_y_t - self.Eps_arr[:, 1])
        w_el_t = (self.w_t - self.Eps_arr[:, 2])
        U_arr = (
                self.Sig_arr[:, 0] * s_x_el_t / 2.0 +
                self.Sig_arr[:, 1] * s_y_el_t / 2.0 +
                self.Sig_arr[:, 2] * w_el_t / 2.0
        )
        G_arr = W_arr - U_arr
        return W_arr, U_arr, G_arr

    Eps = tr.Property
    """Energy dissipated in associatiation with individual internal variables 
    """
    def _get_Eps(self):
        Eps_names = self.slider_exp.slide_model.Eps_names
        E_i = cumtrapz(self.Sig_arr, self.Eps_arr, initial=0, axis=0)
        return SimpleNamespace(**{Eps_name: E for Eps_name, E in zip(Eps_names, E_i.T)})

    mechanisms = tr.Property
    """Energy in association with mechanisms (damage and plastic dissipation)
    or free energy
    """
    def _get_mechanisms(self):
        E_i = cumtrapz(self.Sig_arr, self.Eps_arr, initial=0, axis=0)
        E_T_x_pi_, E_T_y_pi_, E_N_pi_, E_z_, E_alpha_x_, E_alpha_y_, E_omega_T_, E_omega_N_ = E_i.T
        E_plastic_work_T = E_T_x_pi_ + E_T_y_pi_
        E_plastic_work_N = E_N_pi_
        E_plastic_work = E_plastic_work_T + E_plastic_work_N
        E_iso_free_energy = E_z_
        E_kin_free_energy = E_alpha_x_ + E_alpha_y_
        E_plastic_diss_T = E_plastic_work_T - E_iso_free_energy - E_kin_free_energy
        E_plastic_diss_N = E_plastic_work_N
        E_plastic_diss = E_plastic_diss_T + E_plastic_diss_N
        E_damage_diss = E_omega_T_ + E_omega_N_

        return SimpleNamespace(**{'plastic_work_N': E_plastic_work_N,
                                  'plastic_work_T': E_plastic_work_T,
                                  'plastic_work': E_plastic_work,
                                  'iso_free_energy': E_iso_free_energy,
                                  'kin_free_energy': E_kin_free_energy,
                                  'plastic_diss_N': E_plastic_diss_N,
                                  'plastic_diss_T': E_plastic_diss_T,
                                  'plastic_diss': E_plastic_diss,
                                  'damage_diss_N': E_omega_N_,
                                  'damage_diss_T': E_omega_T_,
                                  'damage_diss': E_damage_diss})


    def get_eps_ab(self, eps_O):

        DELTA = np.identity(3)

        EPS = np.zeros((3, 3, 3), dtype='f')
        EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
        EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1

        DD = np.hstack([DELTA, np.zeros_like(DELTA)])
        EEPS = np.hstack([np.zeros_like(EPS), EPS])

        GAMMA = np.einsum(
            'ik,jk->kij', DD, DD
        ) + np.einsum(
            'ikj->kij', np.fabs(EEPS)
        )
        return np.einsum(
            'Oab,...O->...ab', GAMMA, eps_O
        )[np.newaxis, ...]

    def get_sig_O(self, sig_ab):
        DELTA = np.identity(3)

        EPS = np.zeros((3, 3, 3), dtype='f')
        EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
        EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1

        DD = np.hstack([DELTA, np.zeros_like(DELTA)])
        EEPS = np.hstack([np.zeros_like(EPS), EPS])

        GAMMA_inv = np.einsum(
            'aO,bO->Oab', DD, DD
        ) + 0.5 * np.einsum(
            'aOb->Oab', np.fabs(EEPS)
        )

        return np.einsum(
            'Oab,...ab->...O', GAMMA_inv, sig_ab
        )[0, ...]



    def get_K_OP(self, D_abcd):

        DELTA = np.identity(3)

        EPS = np.zeros((3, 3, 3), dtype='f')
        EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
        EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1

        DD = np.hstack([DELTA, np.zeros_like(DELTA)])
        EEPS = np.hstack([np.zeros_like(EPS), EPS])

        GAMMA_inv = np.einsum(
            'aO,bO->Oab', DD, DD
        ) + 0.5 * np.einsum(
            'aOb->Oab', np.fabs(EEPS)
        )
        GG = np.einsum(
            'Oab,Pcd->OPabcd', GAMMA_inv, GAMMA_inv
        )

        return np.einsum(
            'OPabcd,abcd->OP', GG, D_abcd
        )

    def _get_MPN(self):
        return np.array([[.577350259, .577350259, .577350259],
                         [.577350259, .577350259, -.577350259],
                         [.577350259, -.577350259, .577350259],
                         [.577350259, -.577350259, -.577350259],
                         [.935113132, .250562787, .250562787],
                         [.935113132, .250562787, -.250562787],
                         [.935113132, -.250562787, .250562787],
                         [.935113132, -.250562787, -.250562787],
                         [.250562787, .935113132, .250562787],
                         [.250562787, .935113132, -.250562787],
                         [.250562787, -.935113132, .250562787],
                         [.250562787, -.935113132, -.250562787],
                         [.250562787, .250562787, .935113132],
                         [.250562787, .250562787, -.935113132],
                         [.250562787, -.250562787, .935113132],
                         [.250562787, -.250562787, -.935113132],
                         [.186156720, .694746614, .694746614],
                         [.186156720, .694746614, -.694746614],
                         [.186156720, -.694746614, .694746614],
                         [.186156720, -.694746614, -.694746614],
                         [.694746614, .186156720, .694746614],
                         [.694746614, .186156720, -.694746614],
                         [.694746614, -.186156720, .694746614],
                         [.694746614, -.186156720, -.694746614],
                         [.694746614, .694746614, .186156720],
                         [.694746614, .694746614, -.186156720],
                         [.694746614, -.694746614, .186156720],
                         [.694746614, -.694746614, -.186156720]])

    def _get_MPW(self):
        return np.array([.0160714276, .0160714276, .0160714276, .0160714276, .0204744730,
                         .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                         .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                         .0204744730, .0158350505, .0158350505, .0158350505, .0158350505,
                         .0158350505, .0158350505, .0158350505, .0158350505, .0158350505,
                         .0158350505, .0158350505, .0158350505]) * 6.0

    def _get_e_a(self, eps_ab):
        """
        Get the microplane projected strains
        """
        # get the normal strain array for each microplane
        delta = np.identity(3)
        MPN = self._get_MPN()
        MPNN_nij = np.einsum('ni,nj->nij', MPN, MPN)
        MPTT_nijr = 0.5 * (
                np.einsum('ni,jr -> nijr', MPN, delta) +
                np.einsum('nj,ir -> njir', MPN, delta) - 2 *
                np.einsum('ni,nj,nr -> nijr', MPN, MPN, MPN)
        )
        e_N = np.einsum('nij,...ij->...n', MPNN_nij, eps_ab)
        # get the tangential strain vector array for each microplane
        e_T_a = np.einsum('nija,...ij->...na', MPTT_nijr, eps_ab)
        return np.concatenate([e_N[..., np.newaxis], e_T_a], axis=-1)

    def plot_work(self, m):

        eps_Emab_hist = self.get_eps_ab(m.hist.U_t).squeeze()
        sig_Emab_hist = self.get_eps_ab(m.hist.F_t).squeeze()
        delta_eps_Emab = np.concatenate((np.zeros((3,3))[np.newaxis,...],np.diff(eps_Emab_hist,axis=0)),axis=0)
        eps_a = self._get_e_a(eps_Emab_hist)
        eps_a_ = np.einsum('...a->a...', eps_a)
        eps_N = eps_a_[0, ...]
        eps_T_a = np.einsum('a...->...a', eps_a_[1:, ...])
        delta_eps_N = np.concatenate((np.zeros(28,)[np.newaxis,...],np.diff(eps_N,axis=0)),axis=0)
        delta_eps_T_a = np.concatenate((np.zeros((28,3))[np.newaxis, ...], np.diff(eps_T_a, axis=0)), axis=0)
        sigma_N_Emn, sigma_T_Emna = [], []
        for i in range(len(m.hist.state_vars)):
            sigma_N_Emn.append(m.hist.state_vars[i][0]['sig_N'])
            sigma_T_Emna.append(m.hist.state_vars[i][0]['sig_T_a'])
        sigma_N_Emn = np.array(sigma_N_Emn).squeeze()
        sigma_T_Emna = np.array(sigma_T_Emna).squeeze()

        work_microplane = np.einsum('...n,...n->...n', sigma_N_Emn, delta_eps_N) + np.einsum('...na,...na->...n',
                                                                                                 sigma_T_Emna,
                                                                                                 delta_eps_T_a)

        W_arr_micro = cumtrapz(np.einsum('...n,...n->...', self._get_MPW(), work_microplane), initial=0)
        # W_arr_micro = np.einsum('...n,...n->...', self._get_MPW(), work_microplane)


        W_arr_macro = cumtrapz(np.einsum('...ij,...ij->...', sig_Emab_hist, delta_eps_Emab), initial=0)

        return W_arr_micro, W_arr_macro

    def plot_energy_dissp(self, m, MSX):

        sigma_ab, eps_ab = [], []
        for i in range(len(m.hist.F_t)):
            sigma_ab.append(m.fe_domain[0].xmodel.map_U_to_field(m.hist.F_t[i]) * 2)
            eps_ab.append(m.fe_domain[0].xmodel.map_U_to_field(m.hist.U_t[i]))
        sigma_ab = np.array(sigma_ab)
        eps_ab = np.array(eps_ab)

        fig_list = []

        for i in range(8):

            eps_Emab_hist = eps_ab[:, :, i, :, :].squeeze()
            delta_eps_Emab = np.concatenate((np.zeros((3, 3))[np.newaxis, ...], np.diff(eps_Emab_hist, axis=0)), axis=0)
            eps_a = self._get_e_a(eps_Emab_hist)
            eps_a_ = np.einsum('...a->a...', eps_a)
            eps_N = eps_a_[0, ...]
            eps_T_a = np.einsum('a...->...a', eps_a_[1:, ...])
            delta_eps_N = np.concatenate((np.zeros(28, )[np.newaxis, ...], np.diff(eps_N, axis=0)), axis=0)
            delta_eps_T_a = np.concatenate((np.zeros((28, 3))[np.newaxis, ...], np.diff(eps_T_a, axis=0)), axis=0)

            omega_N, z_N, alpha_N, r_N, eps_N_p, sig_N, omega_T, z_T, alpha_T_a, eps_T_p_a, sig_T_a = \
                [], [], [], [], [], [], [], [], [], [], []
            for j in range(len(m.hist.state_vars)):
                omega_N.append(m.hist.state_vars[j][0]['omega_N'][0][i])
                z_N.append(m.hist.state_vars[j][0]['z_N'][0][i])
                alpha_N.append(m.hist.state_vars[i][0]['alpha_N'][0][i])
                # r_N.append(m.hist.state_vars[j][0]['r_N'][0][i])
                eps_N_p.append(m.hist.state_vars[j][0]['eps_N_p'][0][i])
                sig_N.append(m.hist.state_vars[j][0]['sig_N'][0][i])
                omega_T.append(m.hist.state_vars[j][0]['omega_T'][0][i])
                z_T.append(m.hist.state_vars[j][0]['z_T'][0][i])
                alpha_T_a.append(m.hist.state_vars[j][0]['alpha_T_a'][0][i])
                eps_T_p_a.append(m.hist.state_vars[j][0]['eps_T_p_a'][0][i])
                sig_T_a.append(m.hist.state_vars[j][0]['sig_T_a'][0][i])

            omega_N = np.array(omega_N).squeeze()
            z_N = np.array(z_N).squeeze()
            alpha_N = np.array(alpha_N).squeeze()
            eps_N_p = np.array(eps_N_p).squeeze()
            sig_N = np.array(sig_N).squeeze()
            omega_T = np.array(omega_T).squeeze()
            z_T = np.array(z_T).squeeze()
            alpha_T_a = np.array(alpha_T_a).squeeze()
            eps_T_p_a = np.array(eps_T_p_a).squeeze()
            sig_T_a = np.array(sig_T_a).squeeze()
            eps_N_e = eps_N - eps_N_p
            eps_T_e_a = eps_T_a - eps_T_p_a
            sig_Emab_hist = MSX.NT_to_ab(sig_N, sig_T_a)

            work_microplane = np.einsum('...n,...n->...n', sig_N, delta_eps_N) + np.einsum('...na,...na->...n',
                                                                                           sig_T_a,
                                                                                           delta_eps_T_a)
            W_arr_micro = cumtrapz(np.einsum('...n,...n->...', self._get_MPW(), work_microplane), initial=0)
            W_arr_macro = cumtrapz(np.einsum('...ij,...ij->...', sig_Emab_hist, delta_eps_Emab), initial=0)

            delta_eps_N_p = np.concatenate((np.zeros(28, )[np.newaxis, ...], np.diff(eps_N_p, axis=0)), axis=0)
            delta_eps_N_e = np.concatenate((np.zeros(28, )[np.newaxis, ...], np.diff(eps_N_e, axis=0)), axis=0)
            delta_alpha_N = np.concatenate((np.zeros(28, )[np.newaxis, ...], np.diff(alpha_N, axis=0)), axis=0)
            delta_z_N = np.concatenate((np.zeros(28, )[np.newaxis, ...], np.diff(z_N, axis=0)), axis=0)
            delta_omega_N = np.concatenate((np.zeros(28, )[np.newaxis, ...], np.diff(omega_N, axis=0)), axis=0)

            Z_N = MSX.mic_.K_N * z_N
            X_N = MSX.mic_.gamma_N * alpha_N
            Y_N = 0.5 * MSX.mic_.E_N * (eps_N - eps_N_p) ** 2.0

            plastic_work_N = np.einsum('...n,...n->...n', sig_N, delta_eps_N_p)
            elastic_work_N = np.einsum('...n,...n->...n', sig_N, delta_eps_N_e)
            kin_free_energy_N = np.einsum('...n,...n->...n', X_N, delta_alpha_N)
            iso_free_energy_N = np.einsum('...n,...n->...n', Z_N, delta_z_N)
            damage_dissip_N = np.einsum('...n,...n->...n', Y_N, delta_omega_N)

            E_plastic_work_N = cumtrapz(np.einsum('...n,...n->...', self._get_MPW(), plastic_work_N), initial=0)
            E_elastic_work_N = cumtrapz(np.einsum('...n,...n->...', self._get_MPW(), elastic_work_N), initial=0)
            E_iso_free_energy_N = cumtrapz(np.einsum('...n,...n->...', self._get_MPW(), iso_free_energy_N), initial=0)
            E_kin_free_energy_N = cumtrapz(np.einsum('...n,...n->...', self._get_MPW(), kin_free_energy_N), initial=0)
            E_plastic_diss_N = E_plastic_work_N - E_iso_free_energy_N - E_kin_free_energy_N
            E_damage_N = cumtrapz(np.einsum('...n,...n->...', self._get_MPW(), damage_dissip_N), initial=0)

            delta_eps_T_p_a = np.concatenate((np.zeros((28, 3))[np.newaxis, ...], np.diff(eps_T_p_a, axis=0)), axis=0)
            delta_eps_T_e_a = np.concatenate((np.zeros((28, 3))[np.newaxis, ...], np.diff(eps_T_e_a, axis=0)), axis=0)
            delta_alpha_T_a = np.concatenate((np.zeros((28, 3))[np.newaxis, ...], np.diff(alpha_T_a, axis=0)), axis=0)
            delta_z_T = np.concatenate((np.zeros(28, )[np.newaxis, ...], np.diff(z_T, axis=0)), axis=0)
            delta_omega_T = np.concatenate((np.zeros(28, )[np.newaxis, ...], np.diff(omega_T, axis=0)), axis=0)

            Z_T = MSX.mic_.K_T * z_T
            X_T = MSX.mic_.gamma_T * alpha_T_a
            Y_T = 0.5 * MSX.mic_.E_T * np.einsum('...na,...na->...n', (eps_T_a - eps_T_p_a), (eps_T_a - eps_T_p_a))

            plastic_work_T = np.einsum('...na,...na->...n', sig_T_a, delta_eps_T_p_a)
            elastic_work_T = np.einsum('...na,...na->...n', sig_T_a, delta_eps_T_e_a)
            kin_free_energy_T = np.einsum('...na,...na->...n', X_T, delta_alpha_T_a)
            iso_free_energy_T = np.einsum('...n,...n->...n', Z_T, delta_z_T)
            damage_dissip_T = np.einsum('...n,...n->...n', Y_T, delta_omega_T)

            E_plastic_work_T = cumtrapz(np.einsum('...n,...n->...', self._get_MPW(), plastic_work_T), initial=0)
            E_elastic_work_T = cumtrapz(np.einsum('...n,...n->...', self._get_MPW(), elastic_work_T), initial=0)
            E_iso_free_energy_T = cumtrapz(np.einsum('...n,...n->...',self._get_MPW(), iso_free_energy_T), initial=0)
            E_kin_free_energy_T = cumtrapz(np.einsum('...n,...n->...', self._get_MPW(), kin_free_energy_T), initial=0)
            E_plastic_diss_T = E_plastic_work_T - E_iso_free_energy_T - E_kin_free_energy_T
            E_damage_T = cumtrapz(np.einsum('...n,...n->...', self._get_MPW(), damage_dissip_T), initial=0)

            E_kin_free_energy = E_kin_free_energy_T + E_kin_free_energy_N
            E_iso_free_energy = E_iso_free_energy_T + E_iso_free_energy_N
            E_plastic_diss = E_plastic_diss_T + E_plastic_diss_N
            E_damage_diss = E_damage_T + E_damage_N
            E_plastic_work = E_plastic_work_T + E_plastic_work_N
            E_elastic_work = E_elastic_work_T + E_elastic_work_N

            t_arr = np.linspace(0, 1, len(E_plastic_work))

            fig = plt.figure()
            ax = fig.subplots(1, 1)
            E_level = 0

            #ax2.plot(eps_Emab_hist[:, 0, 0], sig_Emab_hist[:, 0, 0])
            #ax2.plot(eps_Emab_hist[:, 0, 1], sig_Emab_hist[:, 0, 1])
            #ax2.plot(eps_Emab_hist[:, 0, 2], sig_Emab_hist[:, 0, 2])

            ax.plot(t_arr, E_damage_diss + E_level, color='black', lw=2)
            ax.fill_between(t_arr, E_damage_N + E_level, E_level, color='black',
                            hatch='|', label=r'$W$ - damage N diss');
            E_d_level = E_level + abs(E_damage_N)
            ax.fill_between(t_arr, abs(E_damage_T) + E_d_level, E_d_level, color='gray',
                            alpha=0.3, label=r'$W$ - damage T diss');

            E_level = abs(E_damage_diss)

            ax.plot(t_arr, E_plastic_diss + E_level, lw=1., color='red')
            ax.fill_between(t_arr, E_plastic_diss_N + E_level, E_level, color='red',
                            hatch='-', label=r'$W$ - plastic N diss')
            E_d_level = E_level + E_plastic_diss_N
            ax.fill_between(t_arr, E_plastic_diss_T + E_d_level, E_d_level, color='red',
                            alpha=0.3, label=r'$W$ - plastic T diss')
            E_level += E_plastic_diss

            ax.plot(t_arr, abs(E_iso_free_energy) + E_level, '-.', lw=0.5, color='black')
            ax.fill_between(t_arr, abs(E_iso_free_energy) + E_level, E_level, color='royalblue',
                            hatch='|', label=r'$W$ - iso free energy')

            E_level += abs(E_iso_free_energy)
            ax.plot(t_arr, abs(E_kin_free_energy) + E_level, '-.', color='black', lw=0.5)
            ax.fill_between(t_arr, abs(E_kin_free_energy) + E_level, E_level, color='royalblue', alpha=0.2,
                            label=r'$W$ - kin free energyy')

            E_level += abs(E_kin_free_energy)

            ax.plot(t_arr, W_arr_macro, lw=2.5, color='black', label=r'$W$ - Input work')
            ax.plot(t_arr, W_arr_micro, lw=2.5, color='red', label=r'$W$ - Input work - micro')
            # ax.plot(t_arr, G_arr, '--', color='black', lw = 0.5, label=r'$W^\mathrm{inel}$ - Inelastic work')
            ax.fill_between(t_arr, W_arr_micro, E_level, color='green', alpha=0.2, label=r'$W$ - stored energy')
            ax.set_xlabel('$t$ [-]');
            ax.set_ylabel(r'$E$ [Nmm]')
            ax.legend()
            fig_list.append(fig)

        return fig_list


    def plot_energy(self, ax, ax_i):

        W_arr = (
                cumtrapz(self.Sig_arr[:, 0], self.s_x_t, initial=0) +
                cumtrapz(self.Sig_arr[:, 1], self.s_y_t, initial=0) +
                cumtrapz(self.Sig_arr[:, 2], self.w_t, initial=0)
        )

        s_x_el_t = (self.s_x_t - self.Eps_arr[:, 0])
        s_y_el_t = (self.s_y_t - self.Eps_arr[:, 1])
        w_el_t = (self.w_t - self.Eps_arr[:, 2])
        U_arr = (
                self.Sig_arr[:, 0] * s_x_el_t / 2.0 +
                self.Sig_arr[:, 1] * s_y_el_t / 2.0 +
                self.Sig_arr[:, 2] * w_el_t / 2.0
        )
        G_arr = W_arr - U_arr
        ax.plot(self.t_arr, W_arr, lw=0.5, color='black', label=r'$W$ - Input work')
        ax.plot(self.t_arr, G_arr, '--', color='black', lw = 0.5, label=r'$W^\mathrm{inel}$ - Inelastic work')
        ax.fill_between(self.t_arr, W_arr, G_arr,
                        color=self.colors['stored_energy'], alpha=0.2)
        ax.set_xlabel('$t$ [-]');
        ax.set_ylabel(r'$E$ [Nmm]')
        ax.legend()

        E_i = cumtrapz(self.Sig_arr, self.Eps_arr, initial=0, axis=0)
        E_T_x_pi_, E_T_y_pi_, E_N_pi_, E_z_, E_alpha_x_, E_alpha_y_, E_omega_T_, E_omega_N_ = E_i.T
        E_plastic_work_T = E_T_x_pi_ + E_T_y_pi_
        E_plastic_work_N = E_N_pi_
        E_plastic_work = E_plastic_work_T + E_plastic_work_N
        E_iso_free_energy = E_z_
        E_kin_free_energy = E_alpha_x_ + E_alpha_y_
        E_plastic_diss_T = E_plastic_work_T - E_iso_free_energy - E_kin_free_energy
        E_plastic_diss_N = E_plastic_work_N
        E_plastic_diss = E_plastic_diss_T + E_plastic_diss_N
        E_damage_diss = E_omega_T_ + E_omega_N_

        E_level = 0
        if self.E_damage_diss:
            ax.plot(self.t_arr, E_damage_diss + E_level, color='black', lw=1)
            ax_i.plot(self.t_arr, E_damage_diss, color='gray', lw=2,
                      label=r'damage diss.: $Y\dot{\omega}$')
            ax.fill_between(self.t_arr, E_omega_N_ + E_level, E_level, color='black',
                            hatch='|');
            E_d_level = E_level + E_omega_N_
            ax.fill_between(self.t_arr, E_omega_T_ + E_d_level, E_d_level, color='gray',
                            alpha=0.3);
        E_level = E_damage_diss
        if self.E_plastic_work:
            ax.plot(self.t_arr, E_plastic_work + E_level, lw=0.5, color='black')
            # ax.fill_between(self.t_arr, E_plastic_work + E_level, E_level, color='red', alpha=0.3)
            label = r'plastic work: $\sigma \dot{\varepsilon}^\pi$'
            ax_i.plot(self.t_arr, E_plastic_work, color='red', lw=2,label=label)
            ax.fill_between(self.t_arr, E_plastic_work_N + E_level, E_level, color='orange',
                            alpha=0.3);
            E_p_level = E_level + E_plastic_work_N
            ax.fill_between(self.t_arr, E_plastic_work_T + E_p_level, E_p_level, color='red',
                            alpha=0.3);
        if self.E_plastic_diss:
            ax.plot(self.t_arr, E_plastic_diss + E_level, lw=.4, color='black')
            label = r'apparent pl. diss.: $\sigma \dot{\varepsilon}^\pi - X\dot{\alpha} - Z\dot{z}$'
            ax_i.plot(self.t_arr, E_plastic_diss, color='red', lw=2, label=label)
            ax.fill_between(self.t_arr, E_plastic_diss_N + E_level, E_level, color='red',
                            hatch='-');
            E_d_level = E_level + E_plastic_diss_N
            ax.fill_between(self.t_arr, E_plastic_diss_T + E_d_level, E_d_level, color='red',
                            alpha=0.3);
            E_level += E_plastic_diss
        if self.E_iso_free_energy:
            ax.plot(self.t_arr, E_iso_free_energy + E_level, '-.', lw=0.5, color='black')
            ax.fill_between(self.t_arr, E_iso_free_energy + E_level, E_level, color='royalblue',
                            hatch='|')
            ax_i.plot(self.t_arr, -E_iso_free_energy, '-.', color='royalblue', lw=2,
                      label=r'iso. diss.: $Z\dot{z}$')
        E_level += E_iso_free_energy
        if self.E_kin_free_energy:
            ax.plot(self.t_arr, E_kin_free_energy + E_level, '-.', color='black', lw=0.5)
            ax.fill_between(self.t_arr, E_kin_free_energy + E_level, E_level, color='royalblue', alpha=0.2);
            ax_i.plot(self.t_arr, -E_kin_free_energy, '-.', color='blue', lw=2,
                      label=r'free energy: $X\dot{\alpha}$')

        ax_i.legend()
        ax_i.set_xlabel('$t$ [-]');
        ax_i.set_ylabel(r'$E$ [Nmm]')

    @staticmethod
    def subplots(fig):
        ax_work, ax_energies = fig.subplots(1, 2)
        ax_iter = ax_work.twinx()
        return ax_work, ax_energies, ax_iter

    def update_plot(self, axes):
        ax_work, ax_energies, ax_iter = axes
        self.plot_energy(ax_work, ax_energies)
        if self.show_iter:
            ax_iter.plot(self.t_arr, self.iter_t)
            ax_iter.set_ylabel(r'$n_\mathrm{iter}$')

    def xsubplots(self, fig):
        ((ax1, ax2), (ax3, ax4)) = fig.subplots(2, 2, figsize=(10, 5), tight_layout=True)
        ax11 = ax1.twinx()
        ax22 = ax2.twinx()
        ax33 = ax3.twinx()
        ax44 = ax4.twinx()
        return ax1, ax11, ax2, ax22, ax3, ax33, ax4, ax44

    def xupdate_plot(self, axes):
        ax1, ax11, ax2, ax22, ax3, ax33, ax4, ax44 = axes
        self.get_response([6, 0, 0])
        # plot_Sig_Eps(s_x_t, Sig_arr, Eps_arr, iter_t, *axes)
        s_x_pi_, s_y_pi_, w_pi_, z_, alpha_x_, alpha_y_, omega_s_, omega_w_ = self.Eps_arr.T
        tau_x_pi_, tau_y_pi_, sig_pi_, Z_, X_x_, X_y_, Y_s_, Y_w_ = self.Sig_arr.T
        ax1.plot(self.w_t, sig_pi_, color='green')
        ax11.plot(self.s_x_t, tau_x_pi_, color='red')
        ax2.plot(self.w_t, omega_w_, color='green')
        ax22.plot(self.w_t, omega_s_, color='red')
