
import bmcs_utils.api as bu
import numpy as np
import traits.api as tr
from bmcs_matmod.matmod.slide.slide_32 import Slide32
from bmcs_matmod.time_fn.time_function import TimeFunction
from scipy.integrate import cumtrapz

class Explorer(bu.InteractiveModel):
    pass

class EnergyDissipation(bu.InteractiveModel):
    name='Energy'

    slider_exp = tr.WeakRef(Explorer)

    t_arr = tr.DelegatesTo('slider_exp')
    Sig_arr = tr.DelegatesTo('slider_exp')
    Eps_arr = tr.DelegatesTo('slider_exp')
    s_x_t = tr.DelegatesTo('slider_exp')
    s_y_t = tr.DelegatesTo('slider_exp')
    w_t = tr.DelegatesTo('slider_exp')
    iter_t = tr.DelegatesTo('slider_exp')

    def plot_work(self, ax):
        W_arr = (
                cumtrapz(self.Sig_arr[:, 0], self.s_x_t, initial=0) +
                cumtrapz(self.Sig_arr[:, 1], self.s_y_t, initial=0) +
                cumtrapz(self.Sig_arr[:, 2], self.w_t, initial=0)
        )
        U_arr = (
                self.Sig_arr[:, 0] * (self.s_x_t - self.Eps_arr[:, 0]) / 2.0 +
                self.Sig_arr[:, 1] * (self.s_y_t - self.Eps_arr[:, 1]) / 2.0 +
                self.Sig_arr[:, 2] * (self.w_t - self.Eps_arr[:, 2]) / 2.0
        )
        G_arr = W_arr - U_arr
        ax.plot(self.t_arr, W_arr, lw=2, color='black', label=r'$W$ - Input work')
        ax.plot(self.t_arr, G_arr, color='red', label=r'$G$ - Plastic work')
        ax.fill_between(self.t_arr, W_arr, G_arr, color='green', alpha=0.2)
        ax.set_xlabel('$t$ [-]');
        ax.set_ylabel(r'$E$ [Nmm]')
        ax.legend()

    E_plastic_work = bu.Bool(True)
    E_iso_free_energy = bu.Bool(True)
    E_kin_free_energy = bu.Bool(True)
    E_app_plastic_diss = bu.Bool(True)
    E_damage_diss = bu.Bool(True)

    ipw_view = bu.View(
        bu.Item('E_damage_diss'),
        bu.Item('E_plastic_work'),
        bu.Item('E_iso_free_energy'),
        bu.Item('E_kin_free_energy'),
        bu.Item('E_app_plastic_diss'),
    )

    def plot_dissipation(self, ax, ax_i):
        colors = ['blue', 'red', 'green', 'black', 'magenta']
        E_i = cumtrapz(self.Sig_arr, self.Eps_arr, initial=0, axis=0)
        E_s_x_pi_, E_s_y_pi_, E_w_pi_, E_z_, E_alpha_x_, E_alpha_y_, E_omega_s_, E_omega_w_ = E_i.T

        E_plastic_work = E_s_x_pi_ + E_s_y_pi_ + E_w_pi_
        E_isotropic_diss = E_z_
        E_free_energy = E_alpha_x_ + E_alpha_y_
        E_app_plastic_diss = E_plastic_work - E_free_energy - E_isotropic_diss
        E_damage_diss = E_omega_s_ + E_omega_w_

        E_level = 0
        if self.E_damage_diss:
            ax.plot(self.t_arr, E_damage_diss + E_level, color='black', lw=1)
            ax.fill_between(self.t_arr, E_damage_diss + E_level, E_level, color='gray', alpha=0.3);
        E_level = E_damage_diss
        if self.E_app_plastic_diss:
            ax.plot(self.t_arr, E_app_plastic_diss + E_level, lw=1, color='magenta')
            ax.fill_between(self.t_arr, E_app_plastic_diss + E_level, E_level, color='magenta', alpha=0.3)
        E_level += E_app_plastic_diss
        if self.E_iso_free_energy:
            ax.plot(self.t_arr, E_isotropic_diss + E_level, '-.', lw=1, color='black')
            ax.fill_between(self.t_arr, E_isotropic_diss + E_level, E_level, color='orange', alpha=0.3)
        E_level += E_isotropic_diss
        if self.E_kin_free_energy:
            ax.plot(self.t_arr, E_free_energy + E_level, '-.', color='black', lw=1)
            ax.fill_between(self.t_arr, E_free_energy + E_level, E_level, color='blue', alpha=0.2);

        if self.E_plastic_work:
            ax_i.plot(self.t_arr, E_plastic_work, color='red', lw=2,
                      label=r'plast. work: $\sigma\dot{\varepsilon}^\pi$')
        if self.E_damage_diss:
            ax_i.plot(self.t_arr, E_damage_diss, color='gray', lw=2,
                      label=r'damage diss.: $Y\dot{\omega}$')
        if self.E_app_plastic_diss:
            label = r'apparent pl. diss.: $\sigma \dot{\varepsilon}^\pi - X\dot{\alpha} - Z\dot{z}$'
            ax_i.plot(self.t_arr, E_app_plastic_diss, color='magenta', lw=2,
                      label=label)
        if self.E_iso_free_energy:
            ax_i.plot(self.t_arr, -E_isotropic_diss, '-.', color='orange', lw=2,
                      label=r'iso. diss.: $Z\dot{z}$')
        if self.E_kin_free_energy:
            ax_i.plot(self.t_arr, -E_free_energy, '-.', color='blue', lw=2,
                      label=r'free energy: $X\dot{\alpha}$')

        ax_i.legend()
        ax_i.set_xlabel('$t$ [-]');
        ax_i.set_ylabel(r'$E$ [Nmm]')

    def subplots(self, fig):
        ax_work, ax_energies = fig.subplots(1, 2)
        ax_iter = ax_work.twinx()
        return ax_work, ax_energies, ax_iter

    def update_plot(self, axes):
        ax_work, ax_energies, ax_iter = axes
        self.plot_work(ax_work)
        self.plot_dissipation(ax_work, ax_energies)
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


class SlideExplorer(Explorer):
    name = 'Explorer'

    slide_model = tr.Instance(Slide32, ())

    energy_dissipation = tr.Instance(EnergyDissipation)
    '''Viewer to the energy dissipation'''
    def _energy_dissipation_default(self):
        return EnergyDissipation(slider_exp=self)

    time_fn = tr.Instance(TimeFunction, ())

    def __init__(self, *args, **kw):
        super(SlideExplorer, self).__init__(*args, **kw)
        self.reset_i()

    n_Eps = tr.Property()

    def _get_n_Eps(self):
        return len(self.slide_model.symb.Eps)

    s_x_1 = bu.Float(0, INC=True)
    s_y_1 = bu.Float(0, INC=True)
    w_1 = bu.Float(0, INC=True)

    n_steps = tr.Int(10, ALG=True)
    k_max = tr.Int(20, ALG=True)

    Sig_arr = tr.Array
    Eps_arr = tr.Array

    ipw_view = bu.View(
        bu.Item('s_x_1', latex=r's_x', minmax=(-4, 4)),
        bu.Item('s_y_1', latex=r's_y', minmax=(-4, 4)),
        bu.Item('w_1', latex=r'w', minmax=(-4, 4))
    )

    def reset_i(self):
        self.s_x_0, self.s_y_0, self.w_0 = 0, 0, 0
        self.t0 = 0
        self.Sig_arr = np.zeros((0, self.n_Eps))
        self.Eps_arr = np.zeros((0, self.n_Eps))
        self.Sig_record = []
        self.Eps_record = []
        self.iter_record = []
        self.t_arr = []
        self.s_x_t, self.s_y_t, self.w_t = [], [], []
        self.Eps_n1 = np.zeros((self.n_Eps,), dtype=np.float_)
        self.s_x_1 = 0
        self.s_y_1 = 0
        self.w_1 = 0

    def get_response_i(self):
        # global Eps_record, Sig_record, iter_record
        # global t_arr, s_x_t, s_y_t, w_t, s_x_0, s_y_0, w_0, t0, Eps_n1
        n_steps = self.n_steps
        t1 = self.t0 + 1
        ti_arr = np.linspace(self.t0, t1, n_steps + 1)
        si_x_t = np.linspace(self.s_x_0, self.s_x_1, n_steps + 1) + 1e-9
        si_y_t = np.linspace(self.s_y_0, self.s_y_1, n_steps + 1) + 1e-9
        wi_t = np.linspace(self.w_0, self.w_1, n_steps + 1) + 1e-9
        for s_x_n1, s_y_n1, w_n1 in zip(si_x_t, si_y_t, wi_t):
            self.Eps_n1, Sig_n1, k = self.slide_model.get_sig_n1(
                s_x_n1, s_y_n1, w_n1, self.Eps_n1, self.k_max
            )
            self.Sig_record.append(Sig_n1)
            self.Eps_record.append(self.Eps_n1)
            self.iter_record.append(k)

        self.Sig_arr = np.array(self.Sig_record, dtype=np.float_)
        self.Eps_arr = np.array(self.Eps_record, dtype=np.float_)
        self.iter_t = np.array(self.iter_record, dtype=np.int_)
        self.t_arr = np.hstack([self.t_arr, ti_arr])
        self.s_x_t = np.hstack([self.s_x_t, si_x_t])
        self.s_y_t = np.hstack([self.s_y_t, si_y_t])
        self.w_t = np.hstack([self.w_t, wi_t])
        self.t0 = t1
        self.s_x_0, self.s_y_0, self.w_0 = self.s_x_1, self.s_y_1, self.w_1
        return

    # ## Plotting functions
    # To simplify postprocessing examples, here are two aggregate plotting functions, one for the state and force variables, the other one for the evaluation of energies

    def plot_sig_w(self, ax):
        sig_t = self.Sig_arr.T[2, ...]
        ax.plot(self.w_t, sig_t, color='orange', lw=3)

    def plot3d_Sig_Eps(self, ax3d):
        tau_x, tau_y = self.Sig_arr.T[:2, ...]
        tau = np.sqrt(tau_x ** 2 + tau_y ** 2)
        ax3d.plot3D(self.s_x_t, self.s_y_t, tau, color='orange', lw=3)

    def subplots(self, fig):
        ax_sxy = fig.add_subplot(1, 2, 1, projection='3d')
        ax_sig = fig.add_subplot(1, 2, 2)

        return ax_sxy, ax_sig

    def update_plot(self, axes):
        ax_sxy, ax_sig = axes
        try:
            self.get_response_i()
        except ValueError:
            print('No convergence reached')
            return

        self.plot_sig_w(ax_sig)
        ax_sig.set_xlabel(r'$w$ [mm]');
        ax_sig.set_ylabel(r'$\sigma$ [MPa]');

        #    plot_tau_s(ax1, Eps_arr[-1,...],s_max,500,get_g3,**kw)
        self.plot3d_Sig_Eps(ax_sxy)
        ax_sxy.plot(self.s_x_t, self.s_y_t, 0, color='red')
        ax_sxy.set_xlabel(r'$s_x$ [mm]');
        ax_sxy.set_ylabel(r'$s_y$ [mm]');
        ax_sxy.set_zlabel(r'$\| \tau \| = \sqrt{\tau_x^2 + \tau_y^2}$ [MPa]');

    def plot_Sig_Eps(self, axes):
        ax1, ax11, ax2, ax22, ax3, ax33, ax4, ax44 = axes
        colors = ['blue', 'red', 'green', 'black', 'magenta']
        s_x_pi_, s_y_pi_, w_pi_, z_, alpha_x_, alpha_y_, omega_s_, omega_w_ = self.Eps_arr.T
        tau_x_pi_, tau_y_pi_, sig_pi_, Z_, X_x_, X_y_, Y_s_, Y_w_ = self.Sig_arr.T
        n_step = len(s_x_pi_)
        ax1.plot(self.s_x_t, self.tau_x_pi_, color='black',
                 label='n_steps = %g' % n_step)
        ax1.set_xlabel('$s$');
        ax1.set_ylabel(r'$\tau$')
        ax1.legend()
        ax11.plot(self.s_x_t, self.iter_t, '-.')
        ax2.plot(self.s_x_t, omega_s_, color='red',
                 label='n_steps = %g' % n_step)
        ax2.set_xlabel('$s$');
        ax2.set_ylabel(r'$\omega$')
        ax2.plot(self.s_x_t, omega_w_, color='green', )
        #    ax22.plot(s_x_t, Y_s_, '-.', color='red',
        #             label='n_steps = %g' % n_step)
        #    ax22.set_ylabel('$Y$')
        ax3.plot(self.s_x_t, z_, color='green',
                 label='n_steps = %g' % n_step)
        ax3.set_xlabel('$s$');
        ax3.set_ylabel(r'$z$')
        ax33.plot(self.s_x_t, Z_, '-.', color='green')
        ax33.set_ylabel(r'$Z$')
        ax4.plot(self.s_x_t, alpha_x_, color='blue',
                 label='n_steps = %g' % n_step)
        ax4.set_xlabel('$s$');
        ax4.set_ylabel(r'$\alpha$')
        ax44.plot(self.s_x_t, X_x_, '-.', color='blue')
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


