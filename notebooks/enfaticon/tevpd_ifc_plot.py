import matplotlib.pylab as plt
from scipy.integrate import cumtrapz
import numpy as np

colors = dict( # color associations
     stored_energy = 'darkgreen', # recoverable
     free_energy_kin = 'darkcyan', # freedom - sky
        free_energy_iso = 'darkslateblue', # freedom - sky
        plastic_diss_s = 'darkorange', # fire - heat
        plastic_diss_w = 'red', # fire - heat
        damage_diss_s = 'darkgray', # ruined
        damage_diss_w = 'black'  # ruined
)

class TEVPDIfcPlot(object):
    """
    Thermo-elasto-visco-plastic damage model
    """
        
    @staticmethod
    def param_study_plot(param_name, response_values, get_Gamma=None, **mp):
        """
        The function is used for visualizing the simulation results. This function accepts four arguments: `param_name`, `s_arr` (array of slip), `t_arr` (array of companion timestamps), and `response_values` (dictionary containing responses).

        Four subplots are generated using this function:
        1. **Loading scenario**: This plot displays how slip changes with time for different values of the parameter specified by `param_name`. This information is overlaid with the viscoplastic stress (`tau_vp`) change over time.
        2. **Stress-Slip relation**: This plot illustrates how viscoplastic stress (`tau_vp`) changes with slip for different parameter values.
        3. **Evolution of temperature**: This plot displays how temperature (`s_vp`) evolves along the time for different parameter values.
        4. **Damage evolution**: This shows how the scalar damage variable (`w`) evolves with slip over time for different parameter values.

        Each subplot includes a legend depicting the parameter name and its corresponding value, and zero lines for easy reference.
        """
        fig, ((ax1,  ax2, ax5), (ax3,  ax4, ax6)) = plt.subplots(2,3, tight_layout=True, figsize=(10.5, 7))
        fig.canvas.header_visible = False
        ax1_twin = ax1.twinx()
        ax5_twin = ax5.twinx()

        for (param, rv), color in zip(response_values.items(), ['black', 'red', 'green']):
            t_t, u_ta, T_t, Eps_t, Sig_t, iter_t, _ = rv
            u_p_Tx_t, u_p_Ty_t, u_p_N_t, z_T_t, alpha_Tx_t, alpha_Ty_t, omega_T_t, omega_N_t = Eps_t.T
            sig_Tx_t, sig_Ty_t, sig_N_t, Z_T_t, X_Tx_t, X_Ty_t, Y_T_t, Y_N_t = Sig_t.T
            ax1.plot(
                t_t,
                u_ta[:, 0],
                color=color,
                linewidth=1,
                label=f"{param_name} = {param}",
            )
            ax1_twin.plot(t_t, sig_Tx_t, linestyle='dashed', color=color, linewidth=1)
            ax2.plot(
                u_ta[:, 0],
                sig_Tx_t,
                color=color,
                linewidth=1,
                label=f"{param_name} = {param}",
            )
            ax3.plot(t_t, T_t, color=color, linewidth=1, label=f"{param_name} = {param}")
            ax4.plot(
                u_ta[:, 0],
                omega_T_t,
                color=color,
                linewidth=1,
                label=f"{param_name} = {param}",
            )
            ax5.plot(
                t_t,
                z_T_t,
                color=color,
                linewidth=1,
                label=f"{param_name} = {param}",
            )
            if get_Gamma is not None:
                ax5_twin.plot(
                        t_t, (mp['f_s_'] + Z_T_t) * get_Gamma(T_t,**mp),
                        color=color,
                        linewidth=1,
                        linestyle='dashed'
                )
                ax5_twin.plot(
                        t_t, Z_T_t,
                        color=color,
                        linewidth=1,
                        linestyle='dotted'
                )

        ax1.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax1.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        ax1.set_title('loading scenario')
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('slip [mm]')
        ax1.legend()

        ax2.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax2.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        ax2.set_title('stress-slip')
        ax2.set_xlabel('slip [mm]')
        ax2.set_ylabel('stress [MPa]')
        ax2.legend()

        ax3.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax3.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        ax3.set_title('evolution of temperature')
        ax3.set_xlabel('time [sec]')
        ax3.set_ylabel('temperature [$^{\circ}$C]')
        ax3.legend()

        ax4.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax4.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        ax4.set_title('damage evolution')
        ax4.set_xlabel('slip [mm]')
        ax4.set_ylabel('damage [-]')
        ax4.legend()

        ax5.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax5.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        ax5.set_title('isotropic hardening')
        ax5.set_xlabel('time [-]')
        ax5.set_ylabel('z [mm]')
        ax5.legend()

    @staticmethod
    def plot_Sig_Eps(rv, ax1, ax11, ax2, ax22, ax3, ax33, ax4, ax44):
        colors = ['blue','red', 'green', 'black', 'magenta' ]
        t_t, u_ta, T_t, Eps_t, Sig_t, iter_t, _ = rv
        s_x_t = u_ta[:,0]
        u_Tx_pi_, u_Ty_pi_, u_N_pi_, z_, alpha_Tx_, alpha_Ty_, omega_T_, omega_N_ = Eps_t.T
        tau_x_pi_, tau_y_pi_, sig_pi_, Z_, X_x_, X_y_, Y_T_, Y_N_ = Sig_t.T
        n_step = len(u_Tx_pi_)
        ax1.plot(s_x_t, tau_x_pi_, color='black', 
                label='n_steps = %g' % n_step)
        ax1.set_xlabel('$s$'); ax1.set_ylabel(r'$\tau$')
        ax1.legend()
        ax11.plot(s_x_t, iter_t, '-.')
        ax2.plot(s_x_t, omega_T_, color='red', 
                label='n_steps = %g' % n_step)
        ax2.set_xlabel('$s$'); ax2.set_ylabel(r'$\omega$')
        ax2.plot(s_x_t, omega_N_, color='green', )
        ax22.plot(s_x_t, Y_T_, '-.', color='red', 
                label='n_steps = %g' % n_step)
        ax22.set_ylabel('$Y$')
        ax3.plot(s_x_t, z_, color='green', 
                label='n_steps = %g' % n_step)
        ax3.set_xlabel('$s$'); ax3.set_ylabel(r'$z$')
        ax33.plot(s_x_t, Z_, '-.', color='green')
        ax33.set_ylabel(r'$Z$')
        ax4.plot(s_x_t, alpha_Tx_, color='blue', 
                label='n_steps = %g' % n_step)
        ax4.set_xlabel('$s$'); ax4.set_ylabel(r'$\alpha$')
        ax44.plot(s_x_t, X_x_, '-.', color='blue')
        ax44.set_ylabel(r'$X$')

    @staticmethod
    def plot_work(ax, rv):
        """Plot input work and stored energy on top of the dissipated energy.
        """
        t_t, u_ta, T_t, Eps_t, Sig_t, iter_t, _ = rv
        u_Tx_, u_Ty_, u_N_, = u_ta.T
        u_Tx_p_, u_Ty_p_, u_N_p_, z_, alpha_Tx_, alpha_Ty_, omega_T_, omega_N_ = Eps_t.T
        sig_Tx_p_, sig_Ty_p_, sig_N_p_, Z_, X_x_, X_y_, Y_T_, Y_N_ = Sig_t.T

        W_t = (
                cumtrapz(sig_Tx_p_, u_Tx_, initial=0) +
                cumtrapz(sig_Ty_p_, u_Ty_, initial=0) +
                cumtrapz(sig_N_p_, u_N_, initial=0)
        )

        u_Tx_el_ = (u_Tx_ - u_Tx_p_)
        u_Ty_el_ = (u_Ty_ - u_Ty_p_)
        u_N__el_ = (u_N_ - u_N_p_)
        U_t = (
                sig_Tx_p_ * u_Tx_el_ / 2.0 +
                sig_Ty_p_ * u_Ty_el_ / 2.0 +
                sig_N_p_ * u_N__el_ / 2.0
        )
        G_t = W_t - U_t
        ax.plot(t_t, W_t, lw=0.5, color='black', label=r'$W$ - Input work')
        ax.plot(t_t, G_t, '--', color='black', lw = 0.5, label=r'$W^\mathrm{inel}$ - Inelastic work')
        ax.fill_between(t_t, W_t, G_t,
                        color=colors['stored_energy'], alpha=0.2)
        ax.set_xlabel('$t$ [-]');
        ax.set_ylabel(r'$E$ [Nmm]')
        ax.legend()

    @staticmethod
    def plot_dissipation(ax, rv, ax_i=None):
        """Stapled and absolute plots of energy dissipation.
        """
        t_t, u_ta, T_t, Eps_t, Sig_t, iter_t, drho_psi_dEps_t = rv

        E_i = cumtrapz(Sig_t, Eps_t, initial=0, axis=0)
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

        if ax_i is not None:
            ax_i.plot(t_t, E_damage_diss, color='gray', lw=2,
                    label=r'damage diss.: $Y\dot{\omega}$')
            ax_i.plot(t_t, E_plastic_work, color='red', lw=2,
                    label=r'plastic work: $\sigma \dot{\varepsilon}^\pi$')
            ax_i.plot(t_t, E_plastic_diss, color='red', lw=2, 
                    label=r'apparent pl. diss.: $\sigma \dot{\varepsilon}^\pi - X\dot{\alpha} - Z\dot{z}$')
            ax_i.plot(t_t, -E_iso_free_energy, '-.', color='royalblue', lw=2,
                    label=r'iso. diss.: $Z\dot{z}$')
            ax_i.plot(t_t, -E_kin_free_energy, '-.', color='blue', lw=2,
                    label=r'free energy: $X\dot{\alpha}$')
            ax_i.legend()
            ax_i.set_xlabel('$t$ [-]');
            ax_i.set_ylabel(r'$E$ [Nmm]')

        # stapled plot
        E_level = 0
        # E_damage_diss
        ax.plot(t_t, E_damage_diss + E_level, color='black', lw=1)
        ax.fill_between(t_t, E_omega_N_ + E_level, E_level, color='black',
                        hatch='|');
        E_d_level = E_level + E_omega_N_
        ax.fill_between(t_t, E_omega_T_ + E_d_level, E_d_level, color='gray',
                        alpha=0.3);
        E_level = E_damage_diss
        # E_plastic_work:
        ax.plot(t_t, E_plastic_work + E_level, lw=0.5, color='black')
        # ax.fill_between(self.t_arr, E_plastic_work + E_level, E_level, color='red', alpha=0.3)
        ax.fill_between(t_t, E_plastic_work_N + E_level, E_level, color='orange',
                        alpha=0.3);
        E_p_level = E_level + E_plastic_work_N
        ax.fill_between(t_t, E_plastic_work_T + E_p_level, E_p_level, color='red',
                        alpha=0.3);
        # E_plastic_diss:
        ax.plot(t_t, E_plastic_diss + E_level, lw=.4, color='black')
        ax.fill_between(t_t, E_plastic_diss_N + E_level, E_level, color='red',
                        hatch='-');
        E_d_level = E_level + E_plastic_diss_N
        ax.fill_between(t_t, E_plastic_diss_T + E_d_level, E_d_level, color='red',
                        alpha=0.3);
        E_level += E_plastic_diss
        # E_iso_free_energy
        ax.plot(t_t, E_iso_free_energy + E_level, '-.', lw=0.5, color='black')
        ax.fill_between(t_t, E_iso_free_energy + E_level, E_level, color='royalblue',
                        hatch='|')
        E_level += E_iso_free_energy
        # E_kin_free_energy:
        ax.plot(t_t, E_kin_free_energy + E_level, '-.', color='black', lw=0.5)
        ax.fill_between(t_t, E_kin_free_energy + E_level, E_level, color='royalblue', alpha=0.2);

        D_ti = cumtrapz(drho_psi_dEps_t, Eps_t, initial=0, axis=0)
        D_t = np.sum(D_ti, axis=1)
        ax.plot(t_t, D_t, linestyle='dashed', color='red', lw=2)

    @classmethod
    def plot_energy_breakdown(cls, ax_i, t_arr, s_x_t, s_y_t, w_t, Eps_arr, Sig_arr):
        """Plot the stapled and absolute breakdown of energies
        """
        TEVPDIfcPlot.plot_work(cls, t_arr, s_x_t, s_y_t, w_t, Eps_arr, Sig_arr)
        TEVPDIfcPlot.plot_dissipation(cls, t_arr, Eps_arr, Sig_arr, ax_i)