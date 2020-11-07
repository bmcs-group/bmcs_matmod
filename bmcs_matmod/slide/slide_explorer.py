
import bmcs_utils.api as bu
import numpy as np
import traits.api as tr
from bmcs_matmod.slide.slide_32 import Slide32, ConvergenceError
from bmcs_matmod.slide.energy_dissipation import EnergyDissipation
from bmcs_matmod.slide.inel_state_evolution import InelStateEvolution
from bmcs_matmod.time_fn.time_function import TimeFunction

class SlideExplorer(bu.InteractiveModel):
    name = 'Explorer'

    slide_model = tr.Instance(Slide32, ())

    energy_dissipation = tr.Instance(EnergyDissipation)
    '''Viewer to the energy dissipation'''
    def _energy_dissipation_default(self):
        return EnergyDissipation(slider_exp=self)

    inel_state_evolution = tr.Instance(InelStateEvolution)
    '''Viewer to the inelastic state evolution'''
    def _inel_state_evolution_default(self):
        return InelStateEvolution(slider_exp=self)

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
        bu.Item('s_x_1', latex=r's_x', minmax=(-4, 4)),
        bu.Item('s_y_1', latex=r's_y', minmax=(-4, 4)),
        bu.Item('w_1', latex=r'w', minmax=(-4, 4)),
        bu.Item('n_steps'),
        bu.Item('k_max'),
        simulator='run',
        reset_simulator='reset'
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

    def get_response_i(self, update_progress=lambda t: t):
        # global Eps_record, Sig_record, iter_record
        # global t_arr, s_x_t, s_y_t, w_t, s_x_0, s_y_0, w_0, t0, Eps_n1
        n_steps = self.n_steps
        t_i = np.linspace(0,1, n_steps+1)
        t1 = self.t0 + 1
        ti_arr = np.linspace(self.t0, t1, n_steps + 1)
        si_x_t = np.linspace(self.s_x_0, self.s_x_1, n_steps + 1) + 1e-9
        si_y_t = np.linspace(self.s_y_0, self.s_y_1, n_steps + 1) + 1e-9
        wi_t = np.linspace(self.w_0, self.w_1, n_steps + 1) + 1e-9
        for t, s_x_n1, s_y_n1, w_n1 in zip(t_i, si_x_t, si_y_t, wi_t):
            try: self.Eps_n1, Sig_n1, k = self.slide_model.get_sig_n1(
                    s_x_n1, s_y_n1, w_n1, self.Eps_n1, self.k_max
                )
            except ConvergenceError as e:
                print(e)
                break
            self.Sig_record.append(Sig_n1)
            self.Eps_record.append(self.Eps_n1)
            self.iter_record.append(k)
            update_progress(t)

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
        ax3d.plot3D(self.s_x_t, self.s_y_t, tau, color='orange', lw=3)

    def subplots(self, fig):
        ax_sxy = fig.add_subplot(1, 2, 1, projection='3d')
        ax_sig = fig.add_subplot(1, 2, 2)

        return ax_sxy, ax_sig

    def run(self, update_progress=lambda t: t):
        try:
            self.get_response_i(update_progress)
        except ValueError:
            print('No convergence reached')
            return

    def reset(self):
        self.reset_i()

    def update_plot(self, axes):
        ax_sxy, ax_sig = axes
        self.plot_sig_w(ax_sig)
        ax_sig.set_xlabel(r'$w$ [mm]');
        ax_sig.set_ylabel(r'$\sigma$ [MPa]');

        #    plot_tau_s(ax1, Eps_arr[-1,...],s_max,500,get_g3,**kw)
        self.plot3d_Sig_Eps(ax_sxy)
        ax_sxy.plot(self.s_x_t, self.s_y_t, 0, color='red')
        ax_sxy.set_xlabel(r'$s_x$ [mm]');
        ax_sxy.set_ylabel(r'$s_y$ [mm]');
        ax_sxy.set_zlabel(r'$\| \tau \| = \sqrt{\tau_x^2 + \tau_y^2}$ [MPa]');

