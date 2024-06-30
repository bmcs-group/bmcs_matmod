#!/usr/bin/env python
# coding: utf-8

# Thermo elasto visco-plastic damage interface model

import sympy as sp
from bmcs_matmod.gsm import GSM
from bmcs_utils.api import Cymbol
import bmcs_utils.api as bu
from bmcs_matmod.slide.f_double_cap import FDoubleCap, FDoubleCapExpr
import traits.api as tr
import numpy as np
from ibvpy.tmodel import MATSEval
from .i_ntim import INTIM
from .exceptions import ReturnMappingError

class TEVPDIfcSymb(bu.SymbExpr):

    # switch on expression reduction
    cse = True

    H = sp.Heaviside

    # Material parameters
    E_T = Cymbol(r'E_{\mathrm{T}}', codename='E_T_', real=True, nonnegative=True)
    gamma_T = Cymbol(r'\gamma_{\mathrm{T}}', codename='gamma_T_', real=True)
    K_T = Cymbol(r'K_{\mathrm{T}}', codename='K_T_', real=True)
    S_T = Cymbol(r'S_{\mathrm{T}}', codename='S_T_', real=True, nonnegative=True)
    r_T = Cymbol(r'r_{\mathrm{T}}', codename='r_T_', real=True, nonnegative=True)
    c_T = Cymbol(r'c_{\mathrm{T}}', codename='c_T_', real=True, nonnegative=True)
    eta_T = Cymbol(r'\eta_{\mathrm{T}}', codename='eta_T_', real=True, nonnegative=True)

    E_N = Cymbol(r'E_{\mathrm{N}}', codename='E_N_', real=True, nonnegative=True)
    gamma_N = Cymbol(r'\gamma_{\mathrm{N}}', codename='gamma_N_', real=True)
    K_N = Cymbol(r'K_{\mathrm{N}}', codename='K_N_', real=True)
    S_N = Cymbol(r'S_{\mathrm{N}}', codename='S_N_', real=True, nonnegative=True)
    r_N = Cymbol(r'r_{\mathrm{N}}', codename='r_N_', real=True, nonnegative=True)
    c_N = Cymbol(r'c_{\mathrm{N}}', codename='c_N_', real=True, nonnegative=True)

    eta_N = Cymbol(r'\eta_{\mathrm{N}}', codename='eta_N_', real=True, nonnegative=True)
    zeta = Cymbol('zeta', codename='zeta_', real=True, nonnegative=True)

    d_N = Cymbol(r'd_{\mathrm{N}}', codename='d_N_', real=True, nonnegative=True)
    alpha_therm = Cymbol(r'\alpha_{\vartheta}', codename='alpha_therm_', real=True, nonnegative=True)
    # temperature 
    C_v = Cymbol(r'C_{\mathrm{v}}', codename='C_v_', real=True, nonnegative=True)
    T_0 = Cymbol(r'\vartheta_0', codename='T_0_', real=True, nonnegative=True)
    beta = Cymbol(r'\beta', codename='beta_', real=True, nonnegative=True)

    f_t = Cymbol(r'f_\mathrm{Nt}', codename='f_t_')
    f_c = Cymbol(r'f_\mathrm{Nc}', codename='f_c_')
    f_c0 = Cymbol(r'f_\mathrm{Nc0}', codename='f_c0_')
    f_s = Cymbol(r'f_\mathrm{T}', codename='f_s_')
    m = Cymbol(r'm', codename='m_', real=True, nonnegative=True)

    # External state variables
    u_N = Cymbol(r'u_\mathrm{N}', codename='u_N_', real=True)
    u_Tx = Cymbol(r'u_\mathrm{Tx}', codename='u_Tx_', real=True)
    u_Ty = Cymbol(r'u_\mathrm{Ty}', codename='u_Ty_', real=True)
    u_Ta = sp.Matrix([u_Tx, u_Ty])
    u_a = sp.Matrix([u_N, u_Tx, u_Ty])
    u = u_a
    sig_N = Cymbol(r'\sigma_\mathrm{N}', codename='sig_N_', real=True)
    sig_Tx = Cymbol(r'\sigma_\mathrm{Tx}', codename='sig_Tx_', real=True)
    sig_Ty = Cymbol(r'\sigma_\mathrm{Ty}', codename='sig_Ty_', real=True)
    sig_Ta = sp.Matrix([sig_Tx, sig_Ty])
    sig_a = sp.Matrix([sig_N, sig_Tx, sig_Ty])

    # Temperature effect on elastic domain function
    T = Cymbol(r'\vartheta', codename='T_', real=True)
    Gamma = sp.exp(-beta * (T - T_0))

    # Internal state variables
    ## plasticity
    u_p_N = Cymbol(r'u_\mathrm{N}^\mathrm{p}', codename='u_p_N_', real=True)
    u_p_Tx = Cymbol(r'u_\mathrm{Tx}^\mathrm{p}', codename='u_p_Tx_', real=True)
    u_p_Ty = Cymbol(r'u_\mathrm{Ty}^\mathrm{p}', codename='u_p_Ty_', real=True)
    u_p_Ta = sp.Matrix([u_p_Tx, u_p_Ty])
    u_p_a = sp.Matrix([u_p_N, u_p_Tx, u_p_Ty])
    sig_p_N = Cymbol(r'\sigma^\mathrm{p}_\mathrm{N}', codename='sig_p_N_', real=True)
    sig_p_Tx = Cymbol(r'\sigma^\mathrm{p}_\mathrm{Tx}', codename='sig_p_Tx_', real=True)
    sig_p_Ty = Cymbol(r'\sigma^\mathrm{p}_\mathrm{Ty}', codename='sig_p_Ty_', real=True)
    sig_p_Ta = sp.Matrix([sig_p_Tx, sig_p_Ty])
    sig_p_a = sp.Matrix([sig_p_N, sig_p_Tx, sig_p_Ty])
    ## damage
    omega_N = Cymbol(r'\omega_\mathrm{N}', codename='omega_N_', real=True)
    omega_T = Cymbol(r'\omega_\mathrm{T}', codename='omega_T_', real=True)
    u_e_N = u_N - u_p_N
    omega_ab = sp.Matrix([[H(u_e_N) * omega_N, 0, 0],
                        [0, omega_T, 0],
                        [0, 0, omega_T]])
    omega_a = sp.Matrix([omega_N, omega_T])
    Y_N = Cymbol(r'Y_\mathrm{N}', codename='Y_N_', real=True)
    Y_T = Cymbol(r'Y_\mathrm{T}', codename='Y_T_', real=True)
    Y_a = sp.Matrix([Y_N, Y_T])
    ## isotropic hardening
    z_N = Cymbol(r'z_\mathrm{N}', codename='z_N_', real=True)
    z_T = Cymbol(r'z_\mathrm{T}', codename='z_T_', real=True)
    K_ab = sp.Matrix([[K_T]])
    z_a = sp.Matrix([z_T])
    Z_N = Cymbol(r'Z_\mathrm{N}', codename='Z_N_', real=True)
    Z_T = Cymbol(r'Z_\mathrm{T}', codename='Z_T_', real=True)
    Z_a = sp.Matrix([Z_T])
    ## kinematic hardening
    alpha_N = Cymbol(r'\alpha_\mathrm{N}', codename='alpha_N_', real=True, nonnegative=True)
    alpha_Tx = Cymbol(r'\alpha_\mathrm{Tx}', codename='alpha_Tx_', real=True, nonnegative=True)
    alpha_Ty = Cymbol(r'\alpha_\mathrm{Ty}', codename='alpha_Ty_', real=True, nonnegative=True)
    gamma_ab = sp.Matrix([[0, 0, 0],
                        [0, gamma_T, 0],
                        [0, 0, gamma_T]])
    alpha_Ta = sp.Matrix([alpha_Tx, alpha_Ty])
    alpha_a = sp.Matrix([alpha_N, alpha_Tx, alpha_Ty])
    X_N = Cymbol(r'X_\mathrm{N}', codename='X_N_', real=True, nonnegative=True)
    X_Tx = Cymbol(r'X_\mathrm{Tx}', codename='X_Ty_', real=True, nonnegative=True)
    X_Ty = Cymbol(r'X_\mathrm{Ty}', codename='X_Tx_', real=True, nonnegative=True)
    X_Ta = sp.Matrix([X_Tx, X_Ty])
    X_a = sp.Matrix([X_N, X_Tx, X_Ty])
    ## Effective stiffness including damage
    E_ab = sp.Matrix([[E_N, 0, 0],
                    [0, E_T, 0],
                    [0, 0, E_T]])
    u_el_a = u_a - u_p_a
    E_eff_ab = (sp.eye(3) - omega_ab) * E_ab
    # HFE
    U_T_ = ( (1 - omega_N) * E_N * alpha_therm * (T - T_0) * (u_N - u_p_N) * d_N )
    U_e_ = sp.Rational(1,2) * (u_el_a.T * E_eff_ab * u_el_a)[0]
    U_p_ = sp.Rational(1,2) * (z_a.T * K_ab * z_a + alpha_a.T * gamma_ab * alpha_a)[0]
    TS_ = C_v * (T - T_0) **2 / (2 * T_0)
    F_ = U_e_ + U_p_ + U_T_ - TS_
    # Threshold function
    sig_eff_Tx = sp.Function(r'\sigma^{\mathrm{eff}}_{\mathrm{T}x}')(sig_p_Tx, omega_T)
    sig_eff_Ty = sp.Function(r'\sigma^{\mathrm{eff}}_{\mathrm{T}y}')(sig_p_Ty, omega_T)
    sig_eff_N = sp.Function(r'\sigma^{\mathrm{eff}}_{\mathrm{N}}')(sig_p_N, omega_N)
    q_Tx = sp.Function(r'q_Tx')(sig_eff_Tx,X_Tx)
    q_Ty = sp.Function(r'q_Ty')(sig_eff_Ty,X_Ty)
    q_N = sp.Function(r'q_N')(sig_eff_N)
    norm_q_T = sp.sqrt(q_Tx*q_Tx + q_Ty*q_Ty)
    subs_q_T = {q_Tx: (sig_eff_Tx - X_Tx), q_Ty: (sig_eff_Ty - X_Ty)}
    subs_q_N = {q_N: sig_eff_N}
    subs_sig_eff = {sig_eff_Tx: sig_p_Tx / (1-omega_T),
                    sig_eff_Ty: sig_p_Ty / (1-omega_T),
                    sig_eff_N: sig_p_N / (1-omega_N)
                    }
    f_ = (FDoubleCapExpr.f_solved
        .subs({FDoubleCapExpr.x: q_N, FDoubleCapExpr.y: norm_q_T})
        .subs(subs_q_T)
        .subs(subs_q_N)
        .subs(subs_sig_eff)
        .subs(f_s, ((f_s+Z_T) * Gamma))
        )
    # Flow potential extension
    S_NT = sp.sqrt(S_N*S_T)
    c_NT = sp.sqrt(c_N*c_T)
    r_NT = sp.sqrt(r_N*r_T)
    omega_NT = 1 - sp.sqrt((1-omega_N)*(1-omega_T))
    phi_N = (1 - omega_N)**c_N * S_N / (r_N+1) * (Y_N / S_N)**(r_N+1) 
    phi_T = (1 - omega_T)**c_T * S_T / (r_T+1) * (Y_T / S_T)**(r_T+1)
    phi_NT = (1 - omega_NT)**c_NT * S_NT / (r_NT+1) * ((Y_N + Y_T)/(S_NT))**(r_NT+1)
    phi_ext_ = ((1 - zeta)*(phi_N + phi_T) + zeta*phi_NT)
    # Relaxation time
    t_relax_N_ = eta_N / (E_N)
    t_relax_T_ = eta_T / (E_T + K_T + gamma_T)
    t_relax_ = sp.Matrix([
                        t_relax_N_,
                        t_relax_T_,
                        t_relax_T_,
                        t_relax_T_,
                        t_relax_T_,
                        t_relax_T_,
                        ] 
                )

    Eps_vars = (u_p_a, z_a, alpha_Ta, omega_a)
    Sig_vars = (sig_p_a, Z_a, X_Ta, Y_a)
    Sig_signs = (-1, 1, 1, -1)

    ## BEGIN GSM

    mparams = (E_T, gamma_T, K_T, S_T, c_T, f_s, E_N, S_N, c_N, m, f_t, f_c, f_c0, 
            eta_N, eta_T, zeta, C_v, T_0, d_N, alpha_therm, beta)
    m_param_subs = {r_N:1, r_T:1}

    u_vars = u_a
    T_var = T
    m_params = mparams
    m_param_subs = m_param_subs
    Eps_vars = Eps_vars
    Sig_vars = Sig_vars
    Sig_signs = Sig_signs
    F_expr = F_
    f_expr = f_
    phi_ext_expr = phi_ext_
    t_relax = t_relax_

    # Internal variable representations and conversions
    Eps_list = tr.Property()
    @tr.cached_property
    def _get_Eps_list(self):
        return [Eps_i.T for Eps_i in self.Eps_vars]

    Eps = tr.Property()
    @tr.cached_property
    def _get_Eps(self):
        return sp.BlockMatrix(self.Eps_list).T
    
    n_Eps_explicit = tr.Property
    @tr.cached_property
    def _get_n_Eps_explicit(self):
        return len(self.Eps.as_explicit())

    _Eps_as_array_lambdified = tr.Property
    @tr.cached_property
    def _get__Eps_as_array_lambdified(self):
        return sp.lambdify(self.Eps.blocks, self.Eps.as_explicit())

    def Eps_as_array(self, arr):
        return self._Eps_as_array_lambdified(arr)[:,0]

    _Eps_as_blocks_lambdified = tr.Property
    @tr.cached_property
    def _get__Eps_as_blocks_lambdified(self):
        return sp.lambdify(self.Eps.as_explicit(), self.Eps_list)

    def Eps_as_blocks(self, arr):
        return [Eps_i[0] for Eps_i in self._Eps_as_blocks_lambdified(arr)]

    # Conjugate state variable representations and conversions
    Sig_list = tr.Property()
    @tr.cached_property
    def _get_Sig_list(self):
        return [Sig_i.T for Sig_i in self.Sig_vars]

    Sig = tr.Property()
    @tr.cached_property
    def _get_Sig(self):
        return sp.BlockMatrix([Sig_i.T for Sig_i in self.Sig_vars]).T

    dF_dEps_ = tr.Property()
    @tr.cached_property
    def _get_dF_dEps_(self):
        return sp.BlockMatrix([sp.diff(self.F_expr, var).T for var in self.Eps.blocks]).T.subs(self.m_param_subs)

    dDiss_dEps_ = tr.Property()
    @tr.cached_property
    def _get_dDiss_dEps_(self):
        dF_dEps_explicit_ = self.dF_dEps_.as_explicit()
        return (self.T_var * dF_dEps_explicit_.diff(self.T_var) - dF_dEps_explicit_)

    Sig_ = tr.Property()
    @tr.cached_property
    def _get_Sig_(self):
        return sp.BlockMatrix([(sign_i_ * dF_dEps_i_).T for sign_i_, dF_dEps_i_ 
                               in zip(self.Sig_signs, self.dF_dEps_.blocks)]).T    

    Phi_ = tr.Property()
    @tr.cached_property
    def _get_Phi_(self):
        Phi_list = [-sign_i_ * self.phi_.diff(Sig_i_) for sign_i_, Sig_i_ 
                    in zip(self.Sig_signs, self.Sig.blocks)]
        return sp.BlockMatrix([[Phi_i] for Phi_i in Phi_list]).as_explicit()
    
    f_ = tr.Property()
    @tr.cached_property
    def _get_f_(self):
        return self.f_expr.subs(self.m_param_subs)
    
    df_dlambda_ = tr.Property()
    @tr.cached_property
    def _get_df_dlambda_(self):
        return (self.df_dEps_.T * self.Phi_)[0, 0]

    dSig_dEps_ = tr.Property()
    @tr.cached_property
    def _get_dSig_dEps_(self):
        dSig_dEps_ = sp.Matrix([[Sig_i.diff(Eps_i) 
                                for Sig_i in self.Sig_.as_explicit()]
                                for Eps_i in self.Eps.as_explicit()])
        return dSig_dEps_

    df_dSig_ = tr.Property()
    @tr.cached_property
    def _get_df_dSig_(self):
        # gradient of threshold function w.r.t. thermodynamic forces
        return self.f_.diff(self.Sig.as_explicit())

    df_dEps_ = tr.Property()
    @tr.cached_property
    def _get_df_dEps_(self):
        subs_Sig_Eps = dict(zip(self.Sig.as_explicit(), self.Sig_.as_explicit()))
        return self.f_.subs(subs_Sig_Eps).diff(self.Eps.as_explicit())

    ## EOF GSM

    # symb_model_params = [
    #     'E_T_', 'gamma_T_', 'K_T_', 'S_T_', 'c_T_', 'f_s_', 
    #     'E_N_', 'S_N_', 'c_N_', 'm_', 'f_t-', 'f_c_', 'f_c0_', 
    #     'eta_N_', 'eta_T_', 'zeta_', 'C_v_', 'T_0_', 'd_N_', 'alpha_therm_', 'beta_' ]

    # # List of expressions for which the methods `get_`
    # symb_expressions = [
    #     ('Sig_', ('u', 'T', 'Sig', 'Eps')),
    #     ('dF_dEps_', ('u', 'T', 'Sig', 'Eps')),
    #     ('dSig_dEps_', ('u', 'T', 'Sig', 'Eps')),
    #     ('dDiss_dEps_', ('u', 'T', 'Sig', 'Eps')),
    #     ('f_', ('u', 'T', 'Sig', 'Eps')),
    #     ('df_dlambda_', ('u', 'T', 'Sig', 'Eps')),
    #     ('dSig_dEps_', ('u', 'T', 'Sig', 'Eps')),
    #     ('Phi_', ('u', 'T', 'Sig', 'Eps')),
    #     ('df_dEps_', ('u', 'T', 'Sig', 'Eps')),
    #     ('df_dSig_', ('u', 'T', 'Sig', 'Eps')),
    # ]

@tr.provides(INTIM)
class TEVPDIfc(MATSEval,bu.InjectSymbExpr):

    name = 'TEVPDIfc 3.4'
    symb_class = TEVPDIfcSymb

    gsm = tr.Property(bu.Instance(GSM))
    @tr.cached_property
    def _get_gsm(self):
        symb = self.symb_class
        return GSM(
            u_vars = symb.u_vars,
            sig_vars = symb.sig_a,
            T_var = symb.T_var,
            m_params = symb.mparams,
            m_param_subs = symb.m_param_subs,
            Eps_vars = symb.Eps_vars,
            Sig_vars = symb.Sig_vars,
            Sig_signs = symb.Sig_signs,
            F_expr = symb.F_expr,
            f_expr = symb.f_expr,
            phi_ext_expr = symb.phi_ext_expr,
            t_relax = symb.t_relax_
        )

    Eps_names = tr.Property
    @tr.cached_property
    def _get_Eps_names(self):
        return [eps.codename for eps in self.symb.Eps]

    Sig_names = tr.Property
    @tr.cached_property
    def _get_Sig_names(self):
        return [sig.codename for sig in self.symb.Sig]

    state_var_shapes = tr.Property
    @tr.cached_property
    def _get_state_var_shapes(self):
        '''State variables shapes:
        variables are using the codename string in the Cymbol definition
        Since the same string is used in the lambdify method via print_Symbol
        method defined in Cymbol as well'''
        return {eps_name: () for eps_name in self.Eps_names + self.Sig_names}

    k_max = bu.Int(100, ALG=True)
    '''Maximum number of iterations'''

    m_param_values = tr.Dict

    def get_corr_pred(self, u_n, du_n1, T_n, dt, **state):
        '''Return mapping iteration:
        This function represents a user subroutine in a finite element
        code or in a lattice model. The input is $s_{n+1}$ and the state variables
        representing the state in the previous solved step $\boldsymbol{\mathcal{E}}_n$.
        The procedure returns the stresses and state variables of
        $\boldsymbol{\mathcal{S}}_{n+1}$ and $\boldsymbol{\mathcal{E}}_{n+1}$
        '''
        
        Eps_n = np.array([ state[eps_name] for eps_name in self.Eps_names], dtype=np.float_)
        Sig_n = np.array([state[sig_name] for sig_name in self.Sig_names], dtype=np.float_)
        Eps_n1, Sig_n1, T_n1, k, dF_dEps = self.gsm.get_state_n1(u_n, du_n1, T_n, dt, Sig_n, Eps_n, self.k_max, **self.m_param_values)

        # empty inelastic entries - accept state
        #return Eps_k, Sig_k, k + 1
        dSig_dEps_k = self.gsm.get_dSig_dEps_(u_n+du_n1, T_n, Sig_n1, Eps_n1)
        select_idx = (0, 1, 2)
        ix1, ix2 = np.ix_(select_idx, select_idx)
        D_ = np.einsum('ab...->...ab',dSig_dEps_k[ix1, ix2, ...])
        sig_ = np.einsum('a...->...a',Sig_n1[select_idx,...])
        # quick fix
        _, _, _, _, _, _, _, _, omega_T, omega_N = Eps_n1
        D_ = np.zeros(sig_.shape + (sig_.shape[-1],))
        D_[...,0,0] = self.E_N * (1 - omega_N)
        D_[...,1,1] = self.E_T * (1 - omega_T)
        D_[...,2,2] = self.E_T * (1 - omega_T)
        for eps_name, Eps_ in zip(self.Eps_names, Eps_n1):
            state[eps_name][...] = Eps_[...]
        for sig_name, Sig_ in zip(self.Sig_names, Sig_n1):
            state[sig_name][...] = Sig_[...]
        return sig_, D_

    def get_eps_NT_p(self, **Eps):
        # plastic strain tensor
        eps_N_p = Eps['w_pi']
        eps_T_p_a = np.einsum('a...->...a',
                              np.array([Eps['s_pi_x'], Eps['s_pi_y'], Eps['s_pi_z']])
                              )
        return eps_N_p, eps_T_p_a

    def plot_f_state(self, ax, Eps, Sig):
        lower = -self.f_c * 1.05
        upper = self.f_t + 0.05 * self.f_c
        lower_tau = -self.bartau * 2
        upper_tau = self.bartau * 2
        lower_tau = 0
        upper_tau = 10
        sig, tau_x, tau_y = Sig[:3]
        tau = np.sqrt(tau_x**2 + tau_y**2)
        sig_ts, tau_x_ts  = np.mgrid[lower:upper:201j,lower_tau:upper_tau:201j]
        Sig_ts = np.zeros((len(self.symb.Eps),) + tau_x_ts.shape)
        Eps_ts = np.zeros_like(Sig_ts)
        Sig_ts[0,...] = sig_ts
        Sig_ts[1,...] = tau_x_ts
        Sig_ts[3:,...] = Sig[3:,np.newaxis,np.newaxis]
        Eps_ts[...] = Eps[:,np.newaxis,np.newaxis]
        H_sig_pi = self.symb.get_H_sig_pi_(Sig_ts)
        f_ts = np.array([self.symb.get_f_(Eps_ts, Sig_ts, H_sig_pi)])

        #phi_ts = np.array([self.symb.get_phi_(Eps_ts, Sig_ts)])
        ax.set_title('threshold function');

        omega_N = Eps_ts[-1,:]
        omega_T = Eps_ts[-2,:]
        sig_ts_eff = sig_ts / (1 - H_sig_pi*omega_N)
        tau_x_ts_eff = tau_x_ts / (1 - omega_T)
        ax.contour(sig_ts_eff, tau_x_ts_eff, f_ts[0,...], levels=0, colors=('green',))

        ax.contour(sig_ts, tau_x_ts, f_ts[0,...], levels=0, colors=('red',))
        #ax.contour(sig_ts, tau_x_ts, phi_ts[0, ...])
        ax.plot(sig, tau, marker='H', color='red')
        ax.plot([lower, upper], [0, 0], color='black', lw=0.4)
        ax.plot([0, 0], [lower_tau, upper_tau], color='black', lw=0.4)
        ax.set_ylim(ymin=0, ymax=10)

    def plot_f(self, ax):
        lower = -self.f_c * 1.05
        upper = self.f_t + 0.05 * self.f_c
        lower_tau = -self.bartau * 2
        upper_tau = self.bartau * 2
        sig_ts, tau_x_ts  = np.mgrid[lower:upper:201j,lower_tau:upper_tau:201j]
        Sig_ts = np.zeros((len(self.symb.Eps),) + tau_x_ts.shape)
        Sig_ts[0,:] = sig_ts
        Sig_ts[1,:] = tau_x_ts
        Eps_ts = np.zeros_like(Sig_ts)
        H_sig_pi = self.symb.get_H_sig_pi_(Sig_ts)
        f_ts = np.array([self.symb.get_f_(Eps_ts, Sig_ts, H_sig_pi)])
        phi_ts = np.array([self.get_phi_(Eps_ts, Sig_ts, H_sig_pi)])
        ax.set_title('threshold function');
        ax.contour(sig_ts, tau_x_ts, f_ts[0,...], levels=0)
        ax.contour(sig_ts, tau_x_ts, phi_ts[0, ...])
        ax.plot([lower, upper], [0, 0], color='black', lw=0.4)
        ax.plot([0, 0], [lower_tau, upper_tau], color='black', lw=0.4)

    def plot_sig_w(self, ax):
        pass

    def plot_tau_s(self, ax):
        pass

    def subplots(self, fig):
        return fig.subplots(2,2)

    def update_plot(self, axes):
        (ax_sig_w, ax_tau_s), (ax_f, _) = axes
        self.plot_sig_w(ax_sig_w)
        self.plot_tau_s(ax_tau_s)
        self.plot_f(ax_f)
