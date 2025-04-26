from turtle import color
from urllib import response
from bmcs_matmod.api import GSMRM
import matplotlib.pylab as plt 
import sympy as sp
import numpy as np
from bmcs_utils.api import Cymbol, Model, mpl_align_yaxis_to_zero, Float, SymbExpr, InjectSymbExpr
# from bmcs_matmod.gsm.potentials.potential1d_t_e_vp_d import Potential1D_T_E_VP_D_SymbExpr
import traits.api as tr
from scipy.integrate import cumtrapz
from pathlib import Path
from scipy.interpolate import interp1d

class MatStudy_T_E_VP_D(Model):

    gsm_F = GSMRM.load_from_disk('gsm_F_1d_t_e_vp_d')
    gsm_G = GSMRM.load_from_disk('gsm_G_1d_t_e_vp_d')

    @staticmethod
    def gsm_run(gsm_, u_ta, T_t, t_t, **material_params):
        response = gsm_.get_response(u_ta, T_t, t_t, **material_params)
        _t_t, _u_tIa, _T_t, _Eps_tIb, _Sig_tIb, _iter_t, _dF_dEps_t, lam_t = response
        _u_atI, _Eps_btI, _Sig_btI, _dF_dEps_btI = [np.moveaxis(v_, -1, 0) for v_ in (_u_tIa, _Eps_tIb, _Sig_tIb, _dF_dEps_t)]
        _sig_atI = gsm_.get_sig(_u_atI, _T_t, _Eps_btI, _Sig_btI, **material_params )
        return _t_t, _u_atI, _sig_atI, _T_t, _Eps_btI, _Sig_btI, _dF_dEps_btI 

    gsm_F.vp_on = True
    gsm_F.update_at_k = False
    gsm_G.vp_on = True
    gsm_G.update_at_k = False

    material_params = tr.Dict
    def _material_params_default(self):
        _f_c = 44
        _f_t = -0.1 * _f_c
        _X_0 = (_f_c + _f_t) / 2
        _f_s = (_f_c - _f_t) / 2
        _E = 50000
        _KH_factor = 4
        _KH = _E * _KH_factor
        _K_ratio = 0.01 # 0.015
        _K = _KH * _K_ratio
        _H = _KH * (1 - _K_ratio)
        
        return dict(
            E=_E, 
            gamma_lin=_H, # _E * 10, 
            gamma_exp=0.5,
            alpha_0=0.5,
            K_lin=_K, # _E / 5,
            k_exp=10,
            z_0=10,
            S=0.008,
            c=2.5,
            r=2.7,
            f_c=_f_s,
            X_0=_X_0,  
            eta=500,
            T_0=20,
            C_v=0.01, # 0.0001, 
            beta=0.0001,
            alpha_therm=0, # 1.2e-5,
            d_N=1
        )

    dot_eps = Float(0.01, TIME=True)
    eps_max = Float(0.0035, TIME=True)

    monotonic_response = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_monotonic_response(self):
        # params
        print('recalculating')
        n_t = 151
        n_I = 1
        t_max = self.eps_max / self.dot_eps
        t_t_F = np.linspace(0, t_max, n_t)
        u_ta_F = (self.dot_eps * t_t_F).reshape(-1, 1)
        T_t = 20 + t_t_F * 0
        response_monotonic_F = self.gsm_run(self.gsm_F, u_ta_F, T_t, t_t_F, **self.material_params)
        _t_t_F, _u_atI_F, _sig_atI_F, _T_t_F, _Eps_btI_F, _Sig_btI_F, _dF_dEps_btI_F = response_monotonic_F
        _argmax_sig = np.argmax(_sig_atI_F)
        print('argmax_max_sig', _argmax_sig)
        print('shape', t_t_F.shape, _sig_atI_F.shape)
#        _max_sig = _sig_atI_F[_argmax_t]
        _max_sig = _sig_atI_F[0, _argmax_sig, 0]
        print('max_sig', _max_sig)
#        t_t_G = np.linspace(0, _t_t_F[_argmax_sig], n_t)
#        u_ta_G = (_max_sig / t_t).reshape(-1, 1)
        # u_ta_G = np.linspace(0, _max_sig, n_t).reshape(-1, 1)
        t_t_G = _t_t_F[:_argmax_sig+1]
        print('history', t_t_F[_argmax_sig], t_t_G[-1])
        u_ta_G = _sig_atI_F[0, :_argmax_sig+1, 0].reshape(-1, 1)
        T_t = 20 + t_t_G * 0
        # print('max_sig', _max_sig)
        # print('shape', len(t_t), t_t[-1], u_ta_G[-1])
        response_monotonic_G = self.gsm_run(self.gsm_G, u_ta_G, T_t, t_t_G, **self.material_params)
        _t_t_G, _u_atI_G, _sig_atI_G, _T_t_G, _Eps_btI_G, _Sig_btI_G, _dF_dEps_btI_G = response_monotonic_G
        print(_t_t_F[_argmax_sig], _t_t_G[-1])
        Diss_btI_F = cumtrapz(_dF_dEps_btI_F, _Eps_btI_F, initial=0, axis=1)
        Diss_btI_G = cumtrapz(_dF_dEps_btI_G, _Eps_btI_G, initial=0, axis=1)
        return response_monotonic_F, response_monotonic_G, (Diss_btI_F, Diss_btI_G, _max_sig)

    def get_fig_monotonic(self):
        fig, axes = plt.subplots(2,3, figsize=(12,6), tight_layout=True)
        return fig, axes

    def plot_monotonic(self, axes):
        (ax, ax_T, ax_Diss), (ax_omega, ax_3, ax_4) = axes
        response_F, response_G, (Diss_btI_F, Diss_btI_G, _max_sig) = self.monotonic_response
        _t_t_F, _u_atI_F, _sig_atI_F, _T_t_F, _Eps_btI_F, _Sig_btI_F, _dF_dEps_btI_F = response_F
        _t_t_G, _u_atI_G, _sig_atI_G, _T_t_G, _Eps_btI_G, _Sig_btI_G, _dF_dEps_btI_G = response_G
        _u_p_atI, _z_atI, _alpha_atI, _omega_atI = self.gsm_G.Eps_as_blocks(_Eps_btI_G)
        _, _Z_atI, _X_atI, _Y_atI = self.gsm_G.Eps_as_blocks(_Sig_btI_G)
        _arg_t_F = _t_t_F[np.argmax(_sig_atI_F)]
        _t_F_scale = _arg_t_F * _t_t_F[-1]

        ax_3_a = ax_3.twinx()
        ax_3.plot(_t_t_F, _u_atI_F[0, :, 0], label='Helmholtz', color='blue');
        ax_3_a.plot(_t_t_G, _u_atI_G[0, :, 0], label='Gibbs', color='orange');
        self._decorate_plot(ax_3, r'loading history', r'$t$')
        ax.plot(_u_atI_F[0, :, 0], _sig_atI_F[0, :, 0], label='Helmholtz', color='blue');
        ax.plot(_sig_atI_G[0, :, 0], _u_atI_G[0, :, 0], label='Gibbs', color='orange');
        self._decorate_plot(ax, r'stress-strain', r'$\varsigma$')
        ax_T.plot(_u_atI_F[0, :, 0], _T_t_F, label='Helmholtz', color='blue');
        ax_T.plot(_sig_atI_G[0, :, 0], _T_t_G, label='Gibbs', color='orange');
        self._decorate_plot(ax_T, r'temperature', r'$\vartheta$')
        ax_Diss.plot(_t_t_F, np.sum(Diss_btI_F[...,0], axis=0), alpha=1, label='F', color='blue')   
        ax_Diss.plot(_t_t_G * _t_F_scale, np.sum(Diss_btI_G[...,0], axis=0), alpha=1, label='G', color='orange')    

        ax_omega.plot(_sig_atI_G[0, :, 0], _omega_atI[0, :, 0], color='orange', lw=2, label='Gibbs')
        ax_omega.set_xlabel(r'$-\varepsilon$/-')
        ax_omega.set_ylabel(r'$\omega$/-')
        return

    # TODO Rename this here and in `plot_monotonic`
    def _decorate_plot(self, arg0, arg1, arg2):
        # ax_T.plot(_t_t_F, _T_t_F);
        # ax_T.plot(_t_t_G * _t_F_scale, _T_t_G);
        arg0.legend()
        arg0.set_title(arg1)
        arg0.set_ylabel(arg2)
        arg0.set_xlabel(r'$-\varepsilon$')

    @staticmethod
    def generate_cyclic_load(max_s, min_s, freq, total_cycles, points_per_cycle):
        # Calculate the time for one cycle
        total_time = total_cycles / freq

        # Calculate the mean value and amplitude
        mean_value = (max_s + min_s) / 2
        amplitude = (max_s - min_s) / 2

        # Calculate the initial loading slope
        slope = 2 * np.pi * freq * amplitude
        
        # Time arrays for linear increase and sinusoidal part
        initial_duration = mean_value / slope
        initial_points = int(initial_duration * freq * points_per_cycle)
        total_points = int(total_time * freq * points_per_cycle)
        
        # Generate the initial linear increase
        initial_t = np.linspace(0, initial_duration, initial_points, endpoint=False)
        initial_loading = slope * initial_t

        # Generate the sinusoidal loading
        sinusoidal_t = np.linspace(0, total_time, total_points, endpoint=False)
        sinusoidal_loading = mean_value + amplitude * np.sin(2 * np.pi * freq * sinusoidal_t)

        # Combine the initial linear increase with the sinusoidal loading
        t_full = np.concatenate((initial_t, sinusoidal_t + initial_duration))
        s_full = np.concatenate((initial_loading, sinusoidal_loading))
        
        return t_full, s_full

    @staticmethod    
    def arg_max_min(data):
        # Find local maxima and minima
        maxima_indexes = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1
        minima_indexes = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1
        return maxima_indexes, minima_indexes
    
    get_step_loading = tr.Property
    tr.cached_property
    def _get_get_step_loading(self):
        s_1 = sp.Symbol('s_1')
        t_1 = sp.Symbol('t_1')
        t = sp.Symbol('t')
        fn_s_t = sp.Piecewise((t * s_1/t_1, t < t_1),(s_1, True))
        return sp.lambdify((t, s_1, t_1), fn_s_t)

    def plot_loading(self):
        fig, (ax, ax_N) = plt.subplots(2,1, figsize=(8,6))
        t_t, s_t = self.generate_cyclic_load(1, 0.1, self.freq, 10, 30)
        ax.plot(t_t, s_t, '-o')
        arg_max, arg_min = self.arg_max_min(s_t)
        ax.plot(t_t[arg_max], s_t[arg_max], 'o', color='red')
        ax.plot(t_t[arg_min], s_t[arg_min], 'o', color='orange')
        ax_N.plot(s_t[arg_max], 'o-')
        ax_N.plot(s_t[arg_min], 'o-')

        u_t_fat = self.get_step_loading(t_t, 1, 50)
        ax.plot(t_t, u_t_fat)
        return
    
    get_sig_p_0 = tr.Property
    @tr.cached_property
    def _get_get_sig_p_0(self):
        sig_p = self.gsm_G.Sig_vars[0][0]
        sig_p_solved_ = sp.solve((self.gsm_G.f_), sig_p)
        return sp.lambdify((self.gsm_G.T_var, 
                            self.gsm_G.Eps.as_explicit(), 
                            self.gsm_G.Sig.as_explicit()) + self.gsm_G.m_params + ('**kw',),
                            sig_p_solved_, cse=True)

    S_max_levels = tr.Array(value = [1, 0.95, 0.85, 0.75, 0.65], dtype=np.float_, TIME=True)

    S_min = Float(0.1, TIME=True)

    freq = Float(5, TIME=True)

    total_cycles = Float(1000, TIME=True)

    points_per_cycle = Float(66, TIME=True)

    fatigue_responses = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_fatigue_responses(self):
        print(f'calculating fatigue for {self.freq}')
        response_F, response_G, (Diss_btI_F, _, _max_sig) = self.monotonic_response
        S_max_levels = self.S_max_levels
        responses = {}
        N_S_min = {}

        N_S_max = {1: 1}
        Diss_Sbt = {1: Diss_btI_F[:,:,0]}
        _delta_T = 20
        for S_max in S_max_levels[1:]:

            print('S_max', S_max)
            # params
            t_t, s_t = self.generate_cyclic_load(S_max, self.S_min, self.freq, self.total_cycles, self.points_per_cycle)
            u_ta_fat = (_max_sig * s_t).reshape(-1, 1)
            T_t = 20 + t_t * 0

            response = self.gsm_run(self.gsm_G, u_ta_fat, T_t, t_t, **self.material_params)
            responses[S_max] = response

            _t_t_fat, _u_atI_fat, _sig_atI_fat, _T_t_fat, _Eps_btI_fat, _Sig_btI_fat, _dF_dEps_btI_fat = response

            _sig_t_fat = _sig_atI_fat[0,:,0]
            arg_max_u, arg_min_u = self.arg_max_min(_sig_t_fat)
            _N_S_max, _N_S_min = len(arg_max_u), len(arg_min_u)
            Diss_btI_fat = cumtrapz(_dF_dEps_btI_fat, _Eps_btI_fat, initial=0, axis=1)
            N_S_max[S_max] = _N_S_max
            N_S_min[S_max] = _N_S_min
            Diss_Sbt[S_max] = Diss_btI_fat[:, :, 0]

        _Diss_plastic_St = {S_max : np.sum(Diss_Sbt[S_max][:-1, :], axis=0) for S_max in S_max_levels}
        _Diss_damage_St = {S_max : Diss_Sbt[S_max][-1, :] for S_max in S_max_levels}

        _Diss_plastic_S = np.array([_Diss_plastic_St[S_max][-1] for S_max in S_max_levels])
        _Diss_damage_S = np.array([_Diss_damage_St[S_max][-1] for S_max in S_max_levels])
        return responses, N_S_max, N_S_min, _Diss_plastic_S, _Diss_damage_S

    def get_fig_SN(self):
        if self.include_last_col:
            fig, ((ax_sig_u, ax_el, ax_T), (ax_omega, ax_u, ax_SN)) = plt.subplots(2,3, figsize=(12,7), tight_layout=True)
        else:
            fig, ((ax_sig_u, ax_el), (ax_omega, ax_u)) = plt.subplots(2,2, figsize=(12,7), tight_layout=True)

        fig.canvas.header_visible=False

        ax_sig_u.set_title('stress-strain')
        ax_omega.set_title('damage')
        ax_el.set_title('elastic domain')
        ax_u.set_title('fatigue strain')

        ax_sig_u.set_xlabel(r'$-\varepsilon$/-')
        ax_sig_u.set_ylabel(r'$-\sigma$/MPa')
        ax_omega.set_xlabel(r'$\eta$/-')
        ax_omega.set_ylabel(r'$\omega$/-')
        ax_el.set_xlabel(r'$\eta$/-')
        ax_el.set_ylabel(r'$-\sigma$/MPa')
        ax_u.set_xlabel(r'$\eta$/-')
        ax_u.set_ylabel(r'$-\varepsilon$/-')

        if self.include_last_col:
            ax_T.set_title(r'temperature')
            ax_SN.set_title(r'S-N \& dissipation')

            ax_T.set_xlabel(r'$\eta$/-')
            ax_T.set_ylabel(r'$\vartheta$/$^{\circ}$C')

            ax_SN.set_xlabel(r'log$N$/-')
            ax_SN.set_ylabel(r'$S_\mathrm{max}$/-')

            ax_SN_twin = ax_SN.twinx()
            ax_SN_twin.set_ylabel(r'$\mathcal{D}/$J/mm$^{{2}}$')

            return fig, (ax_sig_u, ax_el, ax_T, ax_u, ax_omega, ax_SN, ax_SN_twin) 
        else:
            return fig, (ax_sig_u, ax_el, None, ax_u, ax_omega, None, None) 

    include_last_col = tr.Bool(True)

    def plot_fatigue_load(self, axes, S_max, c, alpha_line):
        responses, N_S_max, N_S_min, _Diss_plastic_S, _Diss_damage_S = self.fatigue_responses
        response = responses[S_max]
        _t_t_fat, _u_atI_fat, _sig_atI_fat, _T_t_fat, _Eps_btI_fat, _Sig_btI_fat, _dF_dEps_btI_fat = response
        ax_load, = axes
        ax_load.plot(_t_t_fat, _u_atI_fat[0,:,0], color=c, alpha=alpha_line, label=f'$S_{{\max}} = {S_max}$' )

    def plot_fatigue_response(self, axes, S_max, c, alpha_line):
        # params

        responses, N_S_max, N_S_min, _Diss_plastic_S, _Diss_damage_S = self.fatigue_responses
        response = responses[S_max]

        ax_sig_u, ax_el, ax_T, ax_u, ax_omega, ax_SN, ax_SN_twin = axes

        _t_t_fat, _u_atI_fat, _sig_atI_fat, _T_t_fat, _Eps_btI_fat, _Sig_btI_fat, _dF_dEps_btI_fat = response
        _sig_atI_top, _sig_atI_bot = self.get_sig_p_0(_T_t_fat, _Eps_btI_fat, _Sig_btI_fat, **self.material_params )
        _u_p_atI, _z_atI, _alpha_atI, _omega_atI = self.gsm_G.Eps_as_blocks(_Eps_btI_fat)
        _, _Z_atI, _X_atI, _Y_atI = self.gsm_G.Eps_as_blocks(_Sig_btI_fat)

        _sig_t_fat = _sig_atI_fat[0,:,0]
        _eta_t_fat = _t_t_fat / _t_t_fat[-1]

        _sig_t_fat = _sig_atI_fat[0,:,0]
        arg_max_u, arg_min_u = self.arg_max_min(_sig_t_fat)

        _sig_t_fat = _sig_atI_fat[0,:,0]
        _u_t_fat = _u_atI_fat[0,:,0]
        ax_sig_u.plot(_sig_t_fat, _u_t_fat, color=c, alpha=alpha_line, label=f'$S_{{\max}} = {S_max}$' )

        ax_omega.plot(_eta_t_fat, _omega_atI[0, :, 0], color=c, lw=2, label=f'$N = {N_S_max[S_max]}$' )

        ax_el.plot(_eta_t_fat, _sig_atI_top[:, 0], color=c, alpha=alpha_line)
        ax_el.plot(_eta_t_fat, _sig_atI_bot[:, 0], color=c, alpha=alpha_line)
        ax_el.fill_between(_eta_t_fat, _sig_atI_bot[:, 0], _sig_atI_top[:, 0], color=c, alpha=0.05)

        _eta_max_n = np.linspace(0, 1, N_S_max[S_max])
        _eta_min_n = np.linspace(0, 1, N_S_min[S_max])
        ax_u.plot(_eta_max_n, _sig_t_fat[arg_max_u], '-', lw=2, color=c)
        ax_u.plot(_eta_min_n, _sig_t_fat[arg_min_u], '--', lw=2, color=c)
        # if N_S_min[S_max] == N_S_max[S_max]:
        #     ax_el.fill_between(_eta_max_n, _sig_t_fat[arg_min_u], _sig_t_fat[arg_max_u], color=c, alpha=0.1)
        # else:
        if len(_eta_min_n) > 0 and len(_eta_max_n) > 0:
            f_min = interp1d(_eta_min_n, _sig_t_fat[arg_min_u], kind='linear', fill_value="extrapolate")
            f_max = interp1d(_eta_max_n, _sig_t_fat[arg_max_u], kind='linear', fill_value="extrapolate")
            _eta_common = np.linspace(0, 1, max(N_S_max[S_max], N_S_min[S_max]))
            ax_u.fill_between(_eta_common, f_min(_eta_common), f_max(_eta_common), color=c, alpha=0.05)

        if ax_T:
            ax_T.plot(_eta_t_fat, _T_t_fat, lw=2, color=c)

        arg_dissip = np.where( _dF_dEps_btI_fat[-1,:,0] > 1e-4 )
        # ax_sig_u.plot(_sig_t_fat[arg_dissip], _u_t_fat[arg_dissip], 'o', color='yellow', markersize=2)
        # ax_el.plot(_eta_t_fat[arg_dissip], _u_t_fat[arg_dissip], 'o', color='orange', markersize=3)
        # ax_el.plot(_eta_t_fat, _u_t_fat, 'o', color='yellow', markersize=2)

    def plot_S_max_study(self, S_max_select, include_last_col):
        # def plot_S_max_study(S_max_levels, responses, N_S_max, S_max_select=None, include_last_col=True):
        if S_max_select is None:
            S_max_select = np.arange(len(self.S_max_levels))

        fig_ax = self.get_fig_SN()
        fig, axes = fig_ax
        ax_sig_u, ax_el, ax_T, ax_u, ax_omega, ax_SN, ax_SN_twin = axes
        alpha_line = 0.5

        response_F, response_G, _ = self.monotonic_response
        _t_t_F, _u_atI_F, _sig_atI_F, _T_t_F, _Eps_btI_F, _Sig_btI_F, _dF_dEps_btI_F = response_F
        _t_t_G, _u_atI_G, _sig_atI_G, _T_t_G, _Eps_btI_G, _Sig_btI_G, _dF_dEps_btI_G = response_G

        ax_sig_u.plot(_u_atI_F[0,:,0], _sig_atI_F[0,:,0], color='black', ls='dashed')
        ax_sig_u.plot(_sig_atI_G[0,:,0], _u_atI_G[0,:,0], color='black')

        colors = ['black', 'red', 'darkslategrey', 'maroon', 'darkblue', 'magenta']

        for i in S_max_select:
            S_max = self.S_max_levels[i+1]
            c = colors[i+1]
            self.plot_fatigue_response(axes, S_max, c, alpha_line)

        if ax_SN:
            responses, N_S_max, N_S_min, _Diss_plastic_S, _Diss_damage_S = self.fatigue_responses
            self.plot_SN_curve(ax_SN, N_S_max, self.S_max_levels, ax_SN_twin,
                                                     _Diss_plastic_S, _Diss_damage_S)
        ax_u.set_ylim(ymin=0)

        ax_sig_u.legend()
        ax_omega.legend()
        #ax_el.set_ylim(ymax=_max_sig)
        return fig

    # TODO Rename this here and in `plot_S_max_study`
    def plot_SN_curve(self, ax_SN, N_S_max, S_max_levels, ax_SN_twin,
                                            _Diss_plastic_S, _Diss_damage_S):
        _N_S_max = np.array([N_S_max[key] for key in S_max_levels])
        ax_SN.semilogx(_N_S_max, S_max_levels, 'o-', lw=2, label='S-N')
        ax_SN.legend()
        ax_SN_twin.semilogx(_N_S_max, _Diss_plastic_S, 'o-', alpha=0.4, color='red', label='plastic')
        ax_SN_twin.semilogx(_N_S_max, _Diss_damage_S, 'o-', alpha=0.4, color='gray', label='damage')
        ax_SN_twin.legend()
        ax_SN_twin.set_xscale('log')

    path = Path().home() / 'simdb' / 'data'
    plot_config = {
    #     'one': ([0], True),
        # 'two': ([1,3], False),
        # 'endurance' : ([1,4], False), 
        # 'three': ([1,2, 3], False),
        # 'four': ([0, 1,2, 3], False),
        'all': ([0, 1, 2, 3], True),
    }

    def plot_fatigue_config(self):
        for name, config in self.plot_config.items():
            S_max_select, include_last_col = config
            fig = self.plot_S_max_study(S_max_select=S_max_select, include_last_col=include_last_col);
            fname = f'GSM_demo_fatigue_uniaxial_stress_{name}.png'
            fig.savefig(self.path / fname)
            fname = f'GSM_demo_fatigue_uniaxial_stress_{name}.pdf'
            fig.savefig(self.path / fname)
        return