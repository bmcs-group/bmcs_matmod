
import bmcs_utils.api as bu
import traits.api as tr
from scipy.integrate import cumtrapz
import numpy as np
from types import SimpleNamespace

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
    def _get_W_t(self):
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
