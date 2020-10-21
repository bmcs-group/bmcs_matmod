
import bmcs_utils.api as bu
import numpy as np
import traits.api as tr
from bmcs_matmod.matmod.slide.slide_32 import Slide32
from bmcs_matmod.time_fn.time_function import TimeFunction
from scipy.integrate import cumtrapz

class SlideExplorer(bu.InteractiveModel):

    slide_model = tr.Instance(Slide32, ())

    time_fn = tr.Instance(TimeFunction, ())

    def __init__(self, *args, **kw):
        super(SlideExplorer,self).__init__(*args,**kw)
        self.reset_i()

    n_Eps = tr.Property()
    def _get_n_Eps(self):
        return len(self.slide_model.symb.Eps)

    s_x_1 = tr.Float(0, INC=True)
    s_y_1 = tr.Float(0, INC=True)
    w_1 = tr.Float(0, INC=True)

    n_steps = tr.Int(10, ALG=True)
    k_max = tr.Int(20, ALG=True)

    ipw_view = bu.View(
        bu.Item('s_x_1', latex=r's_x', minmax=(-4,4)),
        bu.Item('s_y_1', latex=r's_y', minmax=(-4,4)),
        bu.Item('w_1', latex=r'w', minmax=(-4,4))
    )

    def reset_i(self):
        self.s_x_0, self.s_y_0, self.w_0 = 0, 0, 0
        self.t0 = 0
        self.Sig_record = []
        self.Eps_record = []
        self.iter_record = []
        self.t_arr = []
        self.s_x_t, self.s_y_t, self.w_t = [], [], []
        self.Eps_n1 = np.zeros((self.n_Eps,), dtype=np.float_)
        self.s_x_1 = 0
        self.s_y_1 = 0
        self.w_1 = 0

    def get_response_i(self, s_x_1, s_y_1, w_1):
        # global Eps_record, Sig_record, iter_record
        # global t_arr, s_x_t, s_y_t, w_t, s_x_0, s_y_0, w_0, t0, Eps_n1
        n_steps = self.n_steps
        t1 = self.t0 + n_steps + 1
        ti_arr = np.linspace(self.t0, t1, n_steps + 1)
        si_x_t = np.linspace(self.s_x_0, self.s_x_1, n_steps + 1)
        si_y_t = np.linspace(self.s_y_0, self.s_y_1, n_steps + 1)
        wi_t = np.linspace(self.w_0, self.w_1, n_steps + 1)
        for s_x_n1, s_y_n1, w_n1 in zip(si_x_t, si_y_t, wi_t):
            self.Eps_n1, Sig_n1, k = self.slide_model.get_sig_n1(
                s_x_n1, s_y_n1, w_n1, self.Eps_n1, self.k_max
            )
            self.Sig_record.append(Sig_n1)
            self.Eps_record.append(self.Eps_n1)
            self.iter_record.append(k)

        self.Sig_arr = np.array(self.Sig_record, dtype=np.float_)
        self.Eps_arr = np.array(self.Eps_record, dtype=np.float_)
        self.iter_arr = np.array(self.iter_record, dtype=np.int_)
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
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax_sig = fig.add_subplot(1, 3, 2)
        ax2 = fig.add_subplot(1, 3, 3)
        ax_iter = ax2.twinx()
        return ax1, ax2, ax_sig, ax_iter

    def update_plot(self, axes):
        ax1, ax2, ax_sig, ax_iter = axes
        self.get_response_i(self.s_x_1, self.s_y_1, self.w_1)
        #    plot_tau_s(ax1, Eps_arr[-1,...],s_max,500,get_g3,**kw)
        self.plot3d_Sig_Eps(ax1)
        ax1.plot(self.s_x_t, self.s_y_t, 0, color='red')
        ax1.set_xlabel(r'$s_x$ [mm]');
        ax1.set_ylabel(r'$s_y$ [mm]');
        ax1.set_zlabel(r'$\| \tau \| = \sqrt{\tau_x^2 + \tau_y^2}$ [MPa]');

        self.plot_sig_w(ax_sig)
        ax_sig.set_xlabel(r'$w$ [mm]');
        ax1.set_ylabel(r'$\sigma$ [MPa]');

        # self.plot_work(ax2, t_arr, s_x_t, s_y_t, w_t, Eps_arr, Sig_arr)
        # plot_dissipation(ax2, t_arr, Eps_arr, Sig_arr)
        ax_iter.plot(self.t_arr, self.iter_arr)
        ax_iter.set_ylabel(r'$n_\mathrm{iter}$')

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
        ax11.plot(self.s_x_t, self.iter_arr, '-.')
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
        ax.set_xlabel('$t$');
        ax.set_ylabel(r'$E$')
        ax.legend()

    def plot_dissipation(self, ax):
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

    def plot_dissipation(self, ax):
        colors = ['blue', 'red', 'green', 'black', 'magenta']
        E_i = cumtrapz(self.Sig_arr, self.Eps_arr, initial=0, axis=0)
        E_s_x_pi_, E_s_y_pi_, E_w_pi_, E_z_, E_alpha_x_, E_alpha_y_, E_omega_s_, E_omega_w_ = E_i.T

        E_plastic_work = E_s_x_pi_ + E_s_y_pi_ + E_w_pi_
        E_isotropic_diss = E_z_
        E_free_energy = E_alpha_x_ + E_alpha_y_
        E_my_plastic_diss = E_plastic_work - E_free_energy - E_isotropic_diss
        E_damage_diss = E_omega_s_ + E_omega_w_

            # ax2.plot(self.t_arr, E_damage_diss, color='gray', lw=1, label='E damage diss')
            # ax2.plot(self.t_arr, E_my_plastic_diss, color='magenta', lw=1, label='E plast diss')
            # ax2.plot(self.t_arr, E_isotropic_diss, color='red', lw=1, label='E iso')
            # ax2.plot(self.t_arr, E_free_energy, color='blue', lw=1, label='E free')
            # ax2.legend()

        E_level = 0
        ax.plot(self.t_arr, E_damage_diss + E_level, color='black', lw=1)
        ax.fill_between(self.t_arr, E_damage_diss + E_level, E_level, color='gray', alpha=0.3);
        E_level = E_damage_diss
        ax.plot(self.t_arr, E_my_plastic_diss + E_level, '-.', lw=1, color='magenta')
        ax.fill_between(self.t_arr, E_my_plastic_diss + E_level, E_level, color='magenta', alpha=0.3)
        E_level += E_my_plastic_diss
        ax.plot(self.t_arr, E_isotropic_diss + E_level, '-.', lw=1, color='black')
        ax.fill_between(self.t_arr, E_isotropic_diss + E_level, E_level, color='red', alpha=0.3)
        E_level += E_isotropic_diss
        ax.plot(self.t_arr, E_free_energy + E_level, color='black', lw=1)
        ax.fill_between(self.t_arr, E_free_energy + E_level, E_level, color='blue', alpha=0.2);

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
        # plot_Sig_Eps(s_x_t, Sig_arr, Eps_arr, iter_arr, *axes)
        s_x_pi_, s_y_pi_, w_pi_, z_, alpha_x_, alpha_y_, omega_s_, omega_w_ = self.Eps_arr.T
        tau_x_pi_, tau_y_pi_, sig_pi_, Z_, X_x_, X_y_, Y_s_, Y_w_ = self.Sig_arr.T
        ax1.plot(self.w_t, sig_pi_, color='green')
        ax11.plot(self.s_x_t, tau_x_pi_, color='red')
        ax2.plot(self.w_t, omega_w_, color='green')
        ax22.plot(self.w_t, omega_s_, color='red')

    def xget_response(self, s_max=[3,0,0]):
        Eps_n1 = np.zeros((self.n_Eps), dtype=np.float_)
        t_arr = np.linspace(0,1,self.n_steps+1)
        s_x_max, s_y_max, w_max = s_max
        s_x_t = s_x_max * self.time_fn(t_arr) + 1e-9 # @todo: check the reason for 1e-9
        s_y_t = s_y_max * self.time_fn(t_arr) + 1e-9
        w_t = w_max * self.time_fn(t_arr) + 1e-9
        for s_x_n1, s_y_n1, w_n1 in zip(s_x_t, s_y_t, w_t):
            Eps_n1, Sig_n1, k = self.slide_model.get_sig_n1(
                s_x_n1, s_y_n1, w_n1, Eps_n1, self.k_max
            )
            self.Sig_record.append(Sig_n1)
            self.Eps_record.append(Eps_n1)
            self.iter_record.append(k)

        self.Sig_arr = np.array(self.Sig_record, dtype=np.float_)
        self.Eps_arr = np.array(self.Eps_record, dtype=np.float_)
        self.iter_arr = np.array(self.iter_record,dtype=np.int_)
        return t_arr, s_x_t, s_y_t, w_t, Eps_arr, Sig_arr, iter_arr
