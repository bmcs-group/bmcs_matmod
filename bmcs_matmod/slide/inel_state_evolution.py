import bmcs_utils.api as bu
import traits.api as tr
from scipy.integrate import cumtrapz
import numpy as np
from scipy.integrate import cumtrapz

class InelStateEvolution(bu.InteractiveModel):
    name = 'State evolution'

    slider_exp = tr.WeakRef(bu.InteractiveModel)

    t_arr = tr.DelegatesTo('slider_exp')
    Sig_arr = tr.DelegatesTo('slider_exp')
    Eps_arr = tr.DelegatesTo('slider_exp')
    s_x_t = tr.DelegatesTo('slider_exp')
    s_y_t = tr.DelegatesTo('slider_exp')
    w_t = tr.DelegatesTo('slider_exp')
    iter_t = tr.DelegatesTo('slider_exp')

    s_x = bu.Bool(True)
    s_y = bu.Bool(True)
    w = bu.Bool(True)

    ipw_view = bu.View(
        bu.Item('s_x'),
        bu.Item('s_y'),
        bu.Item('w'),
    )

    def plot_Sig_Eps(self, axes):
        ax1, ax11, ax2, ax22, ax3, ax33, ax4, ax44 = axes
        colors = ['blue', 'red', 'green', 'black', 'magenta']
        s_x_pi_, s_y_pi_, w_pi_, z_, alpha_x_, alpha_y_, omega_s_, omega_w_ = self.Eps_arr.T
        tau_x_pi_, tau_y_pi_, sig_pi_, Z_, X_x_, X_y_, Y_s_, Y_w_ = self.Sig_arr.T
        n_step = len(s_x_pi_)

        # slip path in 2d
        s_x, s_y = self.s_x_t, self.s_y_t
        d_s_x, d_s_y = s_x[1:] - s_x[:-1], s_y[1:] - s_y[:-1]
        d_s = np.hstack([0, np.sqrt(d_s_x**2 + d_s_y**2)])
        s_t = cumtrapz(d_s, initial=0)
        w_t = self.w_t
        tau_pi = np.sqrt(tau_x_pi_**2 + tau_y_pi_**2)

        ax1.plot(s_t, tau_pi, color='green',
                 label=r'$||\tau(s)||$')
        ax1.set_xlabel('$s,w$');
        ax1.set_ylabel(r'$|| \tau ||, \sigma$')
        ax1.plot(self.w_t, sig_pi_, color='red',
                label = r'$\sigma(w)$')
        ax1.legend()

        ax2.plot(s_t, omega_s_, color='green',
                 label=r'$\omega_s(s)$')
        ax2.set_xlabel('$s, w$');
        ax2.set_ylabel(r'$\omega$')
        ax2.plot(w_t, omega_w_, color='red',
                 label=r'$\omega_w(w)$')
        ax22.plot(s_t, Y_s_, '-.', color='green',
                  label=r'$Y(s)$')
        ax22.plot(w_t, Y_w_, '-.', color='red',
                  label=r'$Y(w)$')
        ax22.set_ylabel('$Y$')

        ax3.plot(s_t, z_, color='green',
                 label=r'$z(s)$')
        ax3.plot(w_t, z_, color='red',
                 label=r'$z(w)$')
        ax3.set_xlabel('$s, w$');
        ax3.set_ylabel(r'$z$')
        ax33.plot(s_t, Z_, '-.', color='green')
        ax33.plot(w_t, Z_, '-.', color='red')
        ax33.set_ylabel(r'$Z$')

        alpha_t = np.sqrt(alpha_x_**2 + alpha_y_**2)
        X_t = np.sqrt(X_x_**2 + X_y_**2)
        ax4.plot(s_t, alpha_t, color='green',
                 label=r'$\alpha(s)$')
        ax4.plot(w_t, alpha_t, color='red',
                 label=r'$\alpha(w)$')
        ax4.set_xlabel('$s, w$');
        ax4.set_ylabel(r'$\alpha$')
        ax44.plot(s_t, X_t, '-.', color='green')
        ax44.plot(w_t, X_t, '-.', color='red')
        ax44.set_ylabel(r'$X$')

    def plot_dissipation2(self, ax):
        colors = ['blue', 'red', 'green', 'black', 'magenta']
        E_i = cumtrapz(self.Sig_arr, self.Eps_arr, initial=0, axis=0)
        E_s_x_pi_, E_s_y_pi_, E_w_pi_, E_z_, E_alpha_x_, E_alpha_y_, E_omega_s_, E_omega_w_ = E_i.T
        c = 'brown'
        E_plastic_work = E_s_x_pi_ + E_s_y_pi_ + E_w_pi_
        ax.plot(self.t_arr, E_plastic_work, '-.', lw=1, color=c)
        c = 'blue'
        E_isotropic_diss = E_z_
        ax.plot(self.t_arr, E_isotropic_diss, '-.', lw=1, color='black')
        ax.fill_between(self.t_arr, E_isotropic_diss, 0, color=c, alpha=0.3)
        c = 'blue'
        E_free_energy = E_alpha_x_ + E_alpha_y_
        ax.plot(self.t_arr, E_free_energy, color='black', lw=1)
        ax.fill_between(self.t_arr, E_free_energy, E_isotropic_diss,
                        color=c, alpha=0.2);
        E_plastic_diss = E_plastic_work - E_free_energy
        ax.plot(self.t_arr, E_plastic_diss, color='black', lw=1)
        ax.fill_between(self.t_arr, E_plastic_diss, 0,
                        color='orange', alpha=0.3);
        c = 'magenta'
        E_damage_diss = E_omega_s_ + E_omega_w_
        ax.plot(self.t_arr, E_plastic_diss + E_damage_diss, color=c, lw=1)
        ax.fill_between(self.t_arr, E_plastic_diss + E_damage_diss,
                        E_plastic_work,
                        color=c, alpha=0.2);
        ax.fill_between(self.t_arr, E_free_energy + E_plastic_diss + E_damage_diss,
                        E_plastic_diss + E_damage_diss,
                        color='yellow', alpha=0.3);

    @staticmethod
    def subplots(fig):
        ((ax1, ax2), (ax3, ax4)) = fig.subplots(2, 2)
        ax11 = ax1.twinx()
        ax22 = ax2.twinx()
        ax33 = ax3.twinx()
        ax44 = ax4.twinx()
        return ax1, ax11, ax2, ax22, ax3, ax33, ax4, ax44

    def update_plot(self, axes):
        self.plot_Sig_Eps(axes)
