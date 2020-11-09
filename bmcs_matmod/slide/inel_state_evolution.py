import bmcs_utils.api as bu
import traits.api as tr
from bmcs_utils.api import mpl_align_yaxis
import numpy as np
from scipy.integrate import cumtrapz

class InelStateEvolution(bu.InteractiveModel):
    name = 'State evolution'

    slider_exp = tr.WeakRef(bu.InteractiveModel)

    t_slider = bu.Float(0)
    t_max = bu.Float(1.001)

    t_arr = tr.DelegatesTo('slider_exp')
    Sig_arr = tr.DelegatesTo('slider_exp')
    Eps_arr = tr.DelegatesTo('slider_exp')
    s_x_t = tr.DelegatesTo('slider_exp')
    s_y_t = tr.DelegatesTo('slider_exp')
    w_t = tr.DelegatesTo('slider_exp')
    iter_t = tr.DelegatesTo('slider_exp')

    ipw_view = bu.View(
        bu.Item('t_slider', latex=r't',
                editor=bu.FloatRangeEditor(low=0, high_name='t_max', n_steps=50)),
        bu.Item('t_max', latex=r't_{\max}'),
    )

    def plot_Sig_Eps(self, axes):
        ax1, ax11, ax2, ax22, ax3, ax33, ax4, ax44 = axes
        colors = ['blue', 'red', 'green', 'black', 'magenta']
        t = self.t_arr
        s_x_pi_, s_y_pi_, w_pi_, z_, alpha_x_, alpha_y_, omega_s_, omega_w_ = self.Eps_arr.T
        tau_x_pi_, tau_y_pi_, sig_pi_, Z_, X_x_, X_y_, Y_s_, Y_w_ = self.Sig_arr.T
        n_step = len(s_x_pi_)

        idx = np.argmax(self.t_slider < self.t_arr)

        # slip path in 2d
        def get_cum_s(s_x, s_y):
            d_s_x, d_s_y = s_x[1:] - s_x[:-1], s_y[1:] - s_y[:-1]
            d_s = np.hstack([0, np.sqrt(d_s_x**2 + d_s_y**2)])
            return cumtrapz(d_s, initial=0)

        s_t = get_cum_s(self.s_x_t, self.s_y_t)
        s_pi_t = get_cum_s(s_x_pi_, s_y_pi_)
        w_t = self.w_t
        w_pi_t = w_pi_
        tau_pi = np.sqrt(tau_x_pi_**2 + tau_y_pi_**2)

        ax1.set_title('stress - displacement');
        ax1.plot(t, tau_pi, '--', color='darkgreen', label=r'$||\tau||$')
        ax1.fill_between(t, tau_pi, 0, color='limegreen', alpha=0.1)
        ax1.plot(t, sig_pi_, '--', color='olivedrab', label = r'$\sigma$')
        ax1.fill_between(t, sig_pi_, 0, color='olivedrab', alpha=0.1)
        ax1.set_ylabel(r'$|| \tau ||, \sigma$')
        ax1.set_xlabel('$t$');
        ax1.plot(t[idx], 0, marker='H', color='red')
        ax1.legend()
        ax11.plot(t, s_t, color='darkgreen', label=r'$||s||$')
        ax11.plot(t, s_pi_t, '--', color='orange', label=r'$||s^\pi||$')
        ax11.plot(t, w_t, color='olivedrab', label=r'$w$')
        ax11.plot(t, w_pi_t, '--', color='chocolate', label=r'$w^\pi$')
        ax11.set_ylabel(r'$|| s ||, w$')
        ax11.legend()
        mpl_align_yaxis(ax1,0,ax11,0)

        ax2.set_title('energy release rate - damage');
        ax2.plot(t, Y_w_, '--', color='darkgray', label=r'$Y_w$')
        ax2.fill_between(t, Y_w_, 0, color='darkgray', alpha=0.15)
        ax2.plot(t, Y_s_, '--', color='darkslategray', label=r'$Y_s$')
        ax2.fill_between(t, Y_s_, 0, color='darkslategray', alpha=0.05)
        ax2.set_xlabel('$t$');
        ax2.set_ylabel('$Y$')
        ax2.plot(t[idx], 0, marker='H', color='red')
        ax2.legend()
        ax22.plot(t, omega_w_, color='darkgray', label=r'$\omega_w$')
        ax22.plot(t, omega_s_, color='darkslategray', label=r'$\omega_s$')
        ax22.set_ylim(ymax=1)
        ax22.set_ylabel(r'$\omega$')
        ax22.legend()

        ax3.set_title('hardening force - displacement');
        alpha_t = np.sqrt(alpha_x_**2 + alpha_y_**2)
        X_t = np.sqrt(X_x_**2 + X_y_**2)
        ax3.plot(t, Z_, '--', color='darkcyan', label=r'$Z$')
        ax3.fill_between(t, Z_, 0, color='darkcyan', alpha=0.05)
        ax3.plot(t, X_t, '--', color='darkslateblue', label=r'$X$')
        ax3.fill_between(t, X_t, 0, color='darkslateblue', alpha=0.05)
        ax3.set_ylabel(r'$Z, X$')
        ax3.set_xlabel('$t$');
        ax3.plot(t[idx], 0, marker='H', color='red')
        ax3.legend()
        ax33.plot(t, z_, color='darkcyan', label=r'$z$')
        ax33.plot(t, alpha_t, color='darkslateblue', label=r'$\alpha$')
        ax33.set_ylabel(r'$z, \alpha$')
        ax33.legend(loc='lower left')

        slide_model = self.slider_exp.slide_model
        slide_model.plot_f_state(ax4, self.Eps_arr[idx,:], self.Sig_arr[idx,:] )

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
