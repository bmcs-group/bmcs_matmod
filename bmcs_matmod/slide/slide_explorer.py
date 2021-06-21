
import bmcs_utils.api as bu
import numpy as np
import traits.api as tr
from bmcs_matmod.slide.slide_32 import Slide32, ConvergenceError
from bmcs_matmod.slide.vslide_34 import Slide34, ConvergenceError
from bmcs_matmod.slide.energy_dissipation import EnergyDissipation
from bmcs_matmod.slide.inel_state_evolution import InelStateEvolution
#from bmcs_matmod.time_fn.time_function import TimeFunction
from ibvpy.tfunction import TimeFunction, TFSelector

class SlideExplorer(bu.Model):
    name = 'Explorer'

    tree = ['slide_model', 'inel_state_evolution', 'energy_dissipation', 'tf_s_x', 'tf_s_y', 'tf_w']

    slide_model = bu.Instance(Slide32, (), tree=True)
#    slide_model = bu.Instance(Slide34, (), tree=True)

    energy_dissipation = bu.Instance(EnergyDissipation, tree=True)
    '''Viewer to the energy dissipation'''
    def _energy_dissipation_default(self):
        return EnergyDissipation(slider_exp=self)

    inel_state_evolution = bu.Instance(InelStateEvolution, tree=True)
    '''Viewer to the inelastic state evolution'''
    def _inel_state_evolution_default(self):
        return InelStateEvolution(slider_exp=self)

    time_fn = bu.Instance(TimeFunction, (), tree=True)

    def __init__(self, *args, **kw):
        super(SlideExplorer, self).__init__(*args, **kw)
        self.reset_i()

    n_Eps = tr.Property()

    def _get_n_Eps(self):
        return len(self.slide_model.symb.Eps)

    s_x_1 = bu.Float(0, INC=True)
    s_y_1 = bu.Float(0, INC=True)
    w_1 = bu.Float(0, INC=True)

    tf_s_x = bu.Instance(TimeFunction, TIME=True)
    def _tf_s_x_default(self):
        return TFSelector()

    tf_s_y = bu.Instance(TimeFunction, TIME=True)
    def _tf_s_y_default(self):
        return TFSelector()

    tf_w = bu.Instance(TimeFunction, TIME=True)
    def _tf_w_default(self):
        return TFSelector()

    n_steps = bu.Int(10, ALG=True)
    k_max = bu.Int(20, ALG=True)

    Sig_arr = tr.Array
    Eps_arr = tr.Array

    Sig_t = tr.Property
    def _get_Sig_t(self):
        return self.Sig_arr

    Eps_t = tr.Property
    def _get_Eps_t(self):
        return self.Eps_arr

    ipw_view = bu.View(
        bu.Item('s_x_1', latex=r's_x'),
        bu.Item('s_y_1', latex=r's_y'),
        bu.Item('w_1', latex=r'w'),
        bu.Item('n_steps'),
        bu.Item('k_max'),
        bu.Item('t_max', readonly=True),
        time_editor=bu.ProgressEditor(run_method='run',
                                   reset_method='reset',
                                   interrupt_var='sim_stop',
                                   time_var='t',
                                   time_max='t_max',
                                   )
    )

    def reset_i(self):
        self.s_x_0, self.s_y_0, self.w_0 = 0, 0, 0
        self.t0 = 0
        self.t = 0
        self.t_max = 1
        self.Sig_arr = np.zeros((0, self.n_Eps))
        self.Eps_arr = np.zeros((0, self.n_Eps))
        self.Sig_record = []
        self.Eps_record = []
        self.iter_record = []
        self.t_arr = []
        self.s_x_t, self.s_y_t, self.w_t = [], [], []
        self.Eps_n1 = np.zeros((self.n_Eps,), dtype=np.float_)
        self.Sig_n1 = np.zeros((self.n_Eps,), dtype=np.float_)
        self.s_x_1 = 0
        self.s_y_1 = 0
        self.w_1 = 0

    t = bu.Float(0)
    t_max = bu.Float(1)
    def _t_max_changed(self):
        self.inel_state_evolution.t_max = self.t_max

    def get_response_i(self, update_progress=lambda t: t):
        # global Eps_record, Sig_record, iter_record
        # global t_arr, s_x_t, s_y_t, w_t, s_x_0, s_y_0, w_0, t0, Eps_n1
        n_steps = self.n_steps
        t_i = np.linspace(0,1, n_steps+1)
        t1 = self.t0 + 1
        self.t_max = t1
        ti_arr = np.linspace(self.t0, t1, n_steps + 1)
        delta_t =  t1 - self.t0
        tf_s_x = self.tf_s_x(np.linspace(0,delta_t,n_steps + 1))
        tf_s_y = self.tf_s_y(np.linspace(0,delta_t,n_steps + 1))
        tf_w = self.tf_w(np.linspace(0,delta_t,n_steps + 1))
        # si_x_t = tf_s_x * np.linspace(self.s_x_0, self.s_x_1, n_steps + 1) + 1e-9
        # si_y_t = tf_s_y * np.linspace(self.s_y_0, self.s_y_1, n_steps + 1) + 1e-9
        # wi_t = tf_w * np.linspace(self.w_0, self.w_1, n_steps + 1) + 1e-9
        si_x_t = self.s_x_0 + tf_s_x * (self.s_x_1 - self.s_x_0) + 1e-9
        si_y_t = self.s_y_0 + tf_s_y * (self.s_y_1 - self.s_y_0) + 1e-9
        wi_t = self.w_0 + tf_w * (self.w_1 - self.w_0) + 1e-9
        for t, s_x_n1, s_y_n1, w_n1 in zip(t_i, si_x_t, si_y_t, wi_t):
            try: self.Eps_n1, self.Sig_n1, k = self.slide_model.get_sig_n1(
                    s_x_n1, s_y_n1, w_n1, self.Sig_n1, self.Eps_n1, self.k_max
                )
            except ConvergenceError as e:
                print(e)
                break
            self.Sig_record.append(self.Sig_n1)
            self.Eps_record.append(self.Eps_n1)
            self.iter_record.append(k)
            self.t = t

        self.Sig_arr = np.array(self.Sig_record, dtype=np.float_)
        self.Eps_arr = np.array(self.Eps_record, dtype=np.float_)
        self.iter_t = np.array(self.iter_record, dtype=np.int_)
        n_i = len(self.iter_t)
        self.t_arr = np.hstack([self.t_arr, ti_arr])[:n_i]
        self.s_x_t = np.hstack([self.s_x_t, si_x_t])[:n_i]
        self.s_y_t = np.hstack([self.s_y_t, si_y_t])[:n_i]
        self.w_t = np.hstack([self.w_t, wi_t])[:n_i]
        self.t0 = t1
        self.s_x_0, self.s_y_0, self.w_0 = self.s_x_1, self.s_y_1, self.w_1
        # set the last step index in the response browser
        self.inel_state_evolution.t_max = self.t_arr[-1]
        return

    # ## Plotting functions
    # To simplify postprocessing examples, here are two aggregate plotting functions, one for the state and force variables, the other one for the evaluation of energies

    def plot_sig_w(self, ax):
        sig_t = self.Sig_arr.T[2, ...]
        ax.plot(self.w_t, sig_t, color='orange', lw=3)

    def plot3d_Sig_Eps(self, ax3d):
        tau_x, tau_y = self.Sig_arr.T[:2, ...]
        tau = np.sqrt(tau_x ** 2 + tau_y ** 2)
        ax3d.plot3D(self.s_x_t, self.s_y_t, tau, color='orange', lw=2)

    def run(self, update_progress=lambda t: t):
        try:
            self.get_response_i(update_progress)
        except ValueError:
            print('No convergence reached')
            return

    def reset(self):
        self.reset_i()

    def subplots(self, fig):
        ax = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(ax[0, 0:2], projection='3d')
        ax2 = fig.add_subplot(ax[0:, -1])
        return ax1, ax2

    def update_plot(self, axes):
        ax_sxy, ax_sig = axes
        self.plot_sig_w(ax_sig)
        ax_sig.set_xlabel(r'$w$ [mm]');
        ax_sig.set_ylabel(r'$\sigma$ [MPa]');

        #    plot_tau_s(ax1, Eps_arr[-1,...],s_max,500,get_g3,**kw)
        ax_sxy.plot(self.s_x_t, self.s_y_t, 0, color='red', lw=1)
        self.plot3d_Sig_Eps(ax_sxy)
        ax_sxy.set_xlabel(r'$s_x$ [mm]');
        ax_sxy.set_ylabel(r'$s_y$ [mm]');
        ax_sxy.set_zlabel(r'$\| \tau \| = \sqrt{\tau_x^2 + \tau_y^2}$ [MPa]');

