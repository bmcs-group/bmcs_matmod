'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model 3D

(compressive plasticity (CP) + tensile damage (TD)
+ cumulative damage sliding (CSD))

Using Jirasek homogenization approach [1999]
'''
import numpy as np
from bmcs_utils.api import Float, View, Item
from ibvpy.tmodel.mats3D.mats3D_eval import MATS3DEval
from scipy.integrate import cumtrapz


# @tr.provides(INTIM)
class VUNTIM_M(MATS3DEval):
    """
    Vectorized uncoupled normal tngential interface model
    """
    # -------------------------------------------------------------------------
    # Elasticity
    # -------------------------------------------------------------------------

    E_N = Float(46666.6666666667, MAT=True)

    E_T = Float(7777.7777777777765, MAT=True)

    gamma_T = Float(1000., MAT=True)

    K_T = Float(1000., MAT=True)

    S_T = Float(0.005, MAT=True)

    r_T = Float(9., MAT=True)

    e_T = Float(12., MAT=True)

    c_T = Float(4.6, MAT=True)

    sigma_T_0 = Float(1.7, MAT=True)

    a = Float(0.003, MAT=True)

    # ------------------------------------------------------------------------------
    # Normal_Tension constitutive law parameters (without cumulative normal strain)
    # ------------------------------------------------------------------------------
    Ad = Float(500.0, MAT=True)

    eps_0 = Float(0.00008, MAT=True)

    # -----------------------------------------------
    # Normal_Compression constitutive law parameters
    # -----------------------------------------------
    K_N = Float(10000., MAT=True)

    gamma_N = Float(5000., MAT=True)

    sig_0 = Float(30., MAT=True)

    ipw_view = View(
        Item('E_N'),
        Item('E_T'),
        Item('Ad_tension'),
        Item('Ad_compression'),
        Item('Ad_shear'),
        Item('eps_0_tension'),
        Item('eps_0_compression'),
        Item('eps_0_shear'),
        Item('K_N'),
        Item('gamma_N'),
        Item('sig_0'),
        Item('gamma_T', latex=r'\gamma_\mathrm{T}', minmax=(10, 100000)),
        Item('K_T', latex=r'K_\mathrm{T}', minmax=(10, 10000)),
        Item('S_T', latex=r'S_\mathrm{T}', minmax=(0.001, 0.01)),
        Item('r_T', latex=r'r_\mathrm{T}', minmax=(1, 3)),
        Item('e_T', latex=r'e_\mathrm{T}', minmax=(1, 40)),
        Item('c_T', latex=r'c_\mathrm{T}', minmax=(1, 10)),
        Item('sigma_T_0', latex=r'\bar{\sigma}^\pi_{T}', minmax=(1, 10)),
        Item('a', latex=r'a_\mathrm{T}', minmax=(0.001, 3)),
    )

    n_D = 3

    state_var_shapes = dict(
        omega_N=(),
        omega_N_compression=(),
        omega_N_tension=(),# damage N
        z_N=(),
        alpha_N =(),
        r_N_compression=(),
        r_N_tension=(),
        eps_N_p=(),
        sig_N=(),
        omega_T=(),
        z_T=(),
        alpha_T_a=(n_D,),
        eps_T_p_a=(n_D,),
        sig_T_a = (n_D,),
    )

    # --------------------------------------------------------------
    # microplane constitutive law (normal behavior CP + TD)
    # --------------------------------------------------------------
    def get_normal_law(self, eps_N, **Eps):
        omega_N, omega_N_compression, omega_N_tension, z_N, alpha_N, r_N_compression, r_N_tension, eps_N_p, sig_N = [
            Eps[key] for key in ['omega_N', 'omega_N_compression', 'omega_N_tension', 'z_N', 'alpha_N',
                                 'r_N_compression', 'r_N_tension','eps_N_p','sig_N']
        ]
        E_N = self.E_N

        # When deciding if a microplane is in tensile or compression, we define a strain boundary such that that
        # sigN <= 0 if eps_N < 0, avoiding entering in the quadrant of compressive strains and traction

        sigma_N_Emn_tilde = E_N * (eps_N - eps_N_p)

        pos = sigma_N_Emn_tilde > 1e-6  # microplanes under tension
        pos2 = sigma_N_Emn_tilde < -1e-6  # microplanes under compression
        tension = 1.0 * pos
        compression = 1.0 * pos2

        # thermo forces

        Z = self.K_N * z_N * compression
        X = self.gamma_N * alpha_N * compression
        h = (self.sig_0 + Z) * compression

        f_trial = (abs(sigma_N_Emn_tilde - X) - h) * compression

        # threshold plasticity

        thres_1 = f_trial > 1e-10

        delta_lamda = f_trial / \
                      (E_N / (1 - omega_N) + abs(self.K_N) + self.gamma_N) * thres_1
        eps_N_p += delta_lamda * \
                       np.sign(sigma_N_Emn_tilde - X)
        z_N += delta_lamda
        alpha_N += delta_lamda * \
                       np.sign(sigma_N_Emn_tilde - X)



        # threshold damage compression

        def R_N_compression(r_N_Emn_compression): return (1.0 / self.Ad_compression) * (-r_N_Emn_compression / (1.0 + r_N_Emn_compression))

        Y_N = 0.5 * E_N * ((eps_N - eps_N_p) ** 2.0) * compression
        Y_0_compression = 0.5 * E_N * self.eps_0_compression ** 2.0

        f_compression = (Y_N - (Y_0_compression + R_N_compression(r_N_compression))) * compression

        def f_w_compression(Y): return 1.0 - 1.0 / (1.0 + self.Ad_compression * (Y - Y_0_compression))

        omega_N_compression[f_compression > 1e-6] = f_w_compression(Y_N)[f_compression > 1e-6]
        omega_N_compression[...] = np.clip(omega_N_compression, 0, 1.0)
        r_N_compression[f_compression > 1e-6] = -omega_N_compression[f_compression > 1e-6]

        def R_N_tension(r_N_Emn_tension): return (1.0 / self.Ad_tension) * (-r_N_Emn_tension / (1.0 + r_N_Emn_tension))

        Y_N = 0.5 * E_N * ((eps_N - eps_N_p) ** 2.0) * tension
        Y_0_tension = 0.5 * E_N * self.eps_0_tension ** 2.0

        f_tension = (Y_N - (Y_0_tension + R_N_tension(r_N_tension))) * tension

        # threshold damage

        def f_w_tension(Y): return 1.0 - 1.0 / (1.0 + self.Ad_tension * (Y - Y_0_tension))

        omega_N_tension[f_tension > 1e-6] = f_w_tension(Y_N)[f_tension > 1e-6]
        omega_N_tension[...] = np.clip(omega_N_tension, 0, 1.0)
        r_N_tension[f_tension > 1e-6] = -omega_N[f_tension > 1e-6]

        omega_N[...] = omega_N_tension + omega_N_compression
        omega_N[...] = np.clip(omega_N, 0, 1.0)

        sig_N[...] = (1.0 - omega_N) * E_N * (eps_N - eps_N_p)
        Z = self.K_N * z_N * compression
        X = self.gamma_N * alpha_N * compression

        return sig_N

    # -------------------------------------------------------------------------
    # microplane constitutive law (Tangential CSD)-(Pressure sensitive cumulative damage)
    # -------------------------------------------------------------------------
    def get_tangential_law(self, eps_T_a, eps_N, **Eps):

        omega_T, omega_N, z_T, alpha_T_a, eps_T_p_a, eps_N_p, sig_T_a= [
            Eps[key] for key in ['omega_T', 'omega_N', 'z_T', 'alpha_T_a', 'eps_T_p_a', 'eps_N_p','sig_T_a']
        ]

        E_T = self.E_T
        E_N = self.E_N

        # thermodynamic forces
        sig_pi_trial = E_T * (eps_T_a - eps_T_p_a)

        Z = self.K_T * z_T
        X = self.gamma_T * alpha_T_a
        norm_1 = np.sqrt(
            np.einsum(
                '...na,...na->...n',
                (sig_pi_trial - X), (sig_pi_trial - X))
        )
        Y = 0.5 * E_T * \
            np.einsum(
                '...na,...na->...n',
                (eps_T_a - eps_T_p_a),
                (eps_T_a - eps_T_p_a))
        sig_N = (1.0 - omega_N) * E_N * (eps_N - eps_N_p)

        f = norm_1 - self.sigma_T_0 - Z - self.a * sig_N

        plas_1 = f > 1e-15
        elas_1 = f < 1e-15

        delta_lamda = (f + self.a * sig_N) / \
                      (E_T / (1.0 - omega_T) + self.gamma_T + self.K_T) * plas_1
        norm_2 = 1.0 * elas_1 + np.sqrt(
            np.einsum(
                '...na,...na->...n',
                (sig_pi_trial - X), (sig_pi_trial - X))) * plas_1

        eps_T_p_a[..., 0] += plas_1 * delta_lamda * \
                                 ((sig_pi_trial[..., 0] - X[..., 0]) /
                                  (1.0 - omega_T)) / norm_2
        eps_T_p_a[..., 1] += plas_1 * delta_lamda * \
                                 ((sig_pi_trial[..., 1] - X[..., 1]) /
                                  (1.0 - omega_T)) / norm_2

        eps_T_p_a[..., 2] += plas_1 * delta_lamda * \
                                 ((sig_pi_trial[..., 2] - X[..., 2]) /
                                  (1.0 - omega_T)) / norm_2
        omega_T += plas_1 * ((1 - omega_T) ** self.c_T) * \
                       (delta_lamda * (Y / self.S_T) ** self.r_T) * \
                       (self.sigma_T_0 / (self.sigma_T_0 - self.a * sig_N)) ** self.e_T

        # omega_T_Emn[...] = np.clip(omega_T_Emn,0,1.0)

        alpha_T_a[..., 0] += plas_1 * delta_lamda * \
                                (sig_pi_trial[..., 0] - X[..., 0]) / norm_2
        alpha_T_a[..., 1] += plas_1 * delta_lamda * \
                                (sig_pi_trial[..., 1] - X[..., 1]) / norm_2

        alpha_T_a[..., 2] += plas_1 * delta_lamda * \
                                (sig_pi_trial[..., 2] - X[..., 2]) / norm_2

        z_T += plas_1 * delta_lamda

        sig_T_a[...] = np.einsum(
            '...n,...na->...na', (1 - omega_T), E_T * (eps_T_a - eps_T_p_a))

        Z = self.K_T * z_T
        X = self.gamma_T * alpha_T_a
        Y = 0.5 * E_T * \
            np.einsum(
                '...na,...na->...n',
                (eps_T_a - eps_T_p_a),
                (eps_T_a - eps_T_p_a))

        return sig_T_a


    # def get_tangential_law(self, eps_T_a, eps_N, **Eps):
    #
    #     omega_T, omega_N, z_T, alpha_T_a, eps_T_p_a, eps_N_p, sig_T_a= [
    #         Eps[key] for key in ['omega_T', 'omega_N', 'z_T', 'alpha_T_a', 'eps_T_p_a', 'eps_N_p','sig_T_a']
    #     ]
    #
    #     E_T = self.E_T
    #     E_N = self.E_N
    #
    #     # thermodynamic forces
    #     sig_pi_trial = E_T * (eps_T_a - eps_T_p_a)
    #
    #     Z = self.K_T * z_T
    #     X = self.gamma_T * alpha_T_a
    #     norm_1 = np.sqrt(
    #         np.einsum(
    #             '...na,...na->...n',
    #             (sig_pi_trial - X), (sig_pi_trial - X))
    #     )
    #     Y = 0.5 * E_T * \
    #         np.einsum(
    #             '...na,...na->...n',
    #             (eps_T_a - eps_T_p_a),
    #             (eps_T_a - eps_T_p_a))
    #     sig_N = (1.0 - omega_N) * E_N * (eps_N - eps_N_p)
    #
    #     # f = norm_1 - self.sigma_T_0 - Z - self.a * sig_N
    #     #
    #     # plas_1 = f > 1e-15
    #     # elas_1 = f < 1e-15
    #     #
    #     # delta_lamda = (f + self.a * sig_N) / \
    #     #               (E_T / (1.0 - omega_T) + self.gamma_T + self.K_T) * plas_1
    #     # norm_2 = 1.0 * elas_1 + np.sqrt(
    #     #     np.einsum(
    #     #         '...na,...na->...n',
    #     #         (sig_pi_trial - X), (sig_pi_trial - X))) * plas_1
    #     #
    #     # eps_T_p_a[..., 0] += plas_1 * delta_lamda * \
    #     #                          ((sig_pi_trial[..., 0] - X[..., 0]) /
    #     #                           (1.0 - omega_T)) / norm_2
    #     # eps_T_p_a[..., 1] += plas_1 * delta_lamda * \
    #     #                          ((sig_pi_trial[..., 1] - X[..., 1]) /
    #     #                           (1.0 - omega_T)) / norm_2
    #     #
    #     # eps_T_p_a[..., 2] += plas_1 * delta_lamda * \
    #     #                          ((sig_pi_trial[..., 2] - X[..., 2]) /
    #     #                           (1.0 - omega_T)) / norm_2
    #     # # omega_T += plas_1 * ((1 - omega_T) ** self.c_T) * \
    #     # #                (delta_lamda * (Y / self.S_T) ** self.r_T) * \
    #     # #                (self.sigma_T_0 / (self.sigma_T_0 - self.a * sig_N)) ** self.e_T
    #     #
    #     # # omega_T_Emn[...] = np.clip(omega_T_Emn,0,1.0)
    #     #
    #     # alpha_T_a[..., 0] += plas_1 * delta_lamda * \
    #     #                         (sig_pi_trial[..., 0] - X[..., 0]) / norm_2
    #     # alpha_T_a[..., 1] += plas_1 * delta_lamda * \
    #     #                         (sig_pi_trial[..., 1] - X[..., 1]) / norm_2
    #     #
    #     # alpha_T_a[..., 2] += plas_1 * delta_lamda * \
    #     #                         (sig_pi_trial[..., 2] - X[..., 2]) / norm_2
    #
    #     # z_T += plas_1 * delta_lamda
    #
    #
    #
    #     def R_N_shear(z_T): return (1.0 / self.Ad_shear) * (-z_T / (1.0 + z_T))
    #
    #     Y_T = 0.5 * E_T * (np.einsum('...na,...na->...n',(eps_T_a ), (eps_T_a )))
    #     Y_0 = 0.5 * E_T * self.eps_0_shear ** 2.0
    #
    #     f_shear = (Y_T - (Y_0 + R_N_shear(z_T)))
    #
    #     def f_w_shear(Y): return 1.0 - 1.0 / (1.0 + self.Ad_shear * (Y - Y_0))
    #
    #     omega_T[f_shear > 1e-6] = f_w_shear(Y_T)[f_shear > 1e-6]
    #     omega_T[...] = np.clip(omega_T, 0, 1.0)
    #     z_T[f_shear > 1e-6] = -z_T[f_shear > 1e-6]
    #
    #     sig_T_a[...] = np.einsum(
    #         '...n,...na->...na', (1 - omega_T), E_T * (eps_T_a))
    #
    #     # Z = self.K_T * z_T
    #     # X = self.gamma_T * alpha_T_a
    #     # Y = 0.5 * E_T * \
    #     #     np.einsum(
    #     #         '...na,...na->...n',
    #     #         (eps_T_a - eps_T_p_a),
    #     #         (eps_T_a - eps_T_p_a))
    #
    #     return sig_T_a

    def get_corr_pred(self, eps_a, t_n1, **Eps):
        eps_a_ = np.einsum('...a->a...',eps_a)
        eps_N_n1 = eps_a_[0,...]
        eps_T_a_n1 = np.einsum('a...->...a', eps_a_[1:,...])

        sig_N = self.get_normal_law(eps_N_n1, **Eps)
        sig_T_a = self.get_tangential_law(eps_T_a_n1,eps_N_n1, **Eps)

        D_ = np.zeros(eps_a.shape + (eps_a.shape[-1],))
        D_[..., 0, 0] = self.E_N # * (1 - omega_N)
        D_[..., 1, 1] = self.E_T # * (1 - omega_T)
        D_[..., 2, 2] = self.E_T # * (1 - omega_T)
        D_[..., 3, 3] = self.E_T # * (1 - omega_T)
        sig_a = np.concatenate([sig_N[...,np.newaxis], sig_T_a], axis=-1)
        return sig_a, D_

    def get_eps_NT_p(self, **Eps):
        """Plastic strain tensor
        """
        return Eps['eps_N_p'], Eps['eps_T_p_a']

    def plot_idx(self, ax_sig, ax_d_sig, ax_energy, idx=0):
        eps_max = self.eps_max
        n_eps = self.n_eps
        eps1_range = np.linspace(1e-9,eps_max,n_eps)
        Eps = { var : np.zeros( (1,) + shape )
            for var, shape in self.state_var_shapes.items()
        }
        eps_range = np.zeros((n_eps, 4))
        eps_range[:,idx] = eps1_range

        # monotonic load in the normal direction
        sig1_range, d_sig11_range, eps1_in_range = [], [], []
        for eps_a in eps_range:
            sig_a, D_range = self.get_corr_pred(eps_a[np.newaxis, ...], 1, **Eps)
            eps_in = [Eps['eps_N_p'], Eps['eps_T_p_a'][0][0]]
            eps1_in_range.append(eps_in[idx])
            sig1_range.append(sig_a[0, idx])
            d_sig11_range.append(D_range[0, idx, idx])
        sig1_range = np.array(sig1_range, dtype=np.float_)
        eps1_in_range = np.array(eps1_in_range, dtype=np.float_).squeeze()
        eps1_range = eps1_range[:len(sig1_range)]
        # print(eps1_in_range)

        W_arr, U_arr, G_arr = self.energy_calc(eps1_range,sig1_range,eps1_in_range)
        ax_sig.plot(eps1_range, sig1_range,color='blue')
        d_sig11_range = np.array(d_sig11_range, dtype=np.float_)
        ax_d_sig.plot(eps1_range, d_sig11_range, linestyle='dashed', color='gray')
        ax_sig.set_xlabel(r'$\varepsilon_{11}$ [-]')
        ax_sig.set_ylabel(r'$\sigma_{11}$ [MPa]')
        ax_d_sig.set_ylabel(r'$\mathrm{d} \sigma_{11} / \mathrm{d} \varepsilon_{11}$ [MPa]')
        ax_d_sig.plot(eps1_range[:-1],
                    (sig1_range[:-1]-sig1_range[1:])/(eps1_range[:-1]-eps1_range[1:]),
                    color='orange', linestyle='dashed')
        ax_energy.plot(eps1_range, W_arr, lw=0.5, color='black', label=r'$W$ - Input work')
        ax_energy.plot(eps1_range, G_arr, '--', color='black', lw=0.5, label=r'$W^\mathrm{inel}$ - Inelastic work')
        ax_energy.fill_between(eps1_range, W_arr, G_arr,
                        color='green', alpha=0.2)
        ax_energy.fill_between(eps1_range, G_arr, np.zeros_like(G_arr),
                               color='black', alpha=0.2)
        ax_energy.set_xlabel('$\varepsilon_{11}$ [-]');
        ax_energy.set_ylabel(r'$E$ [Nmm]')
        ax_energy.legend()

    def subplots(self, fig):
        ax_sig_N, ax_ener_N, ax_sig_T, ax_ener_T = fig.subplots(1,4)
        ax_d_sig_N = ax_sig_N.twinx()
        ax_d_sig_T = ax_sig_T.twinx()
        return ax_sig_N, ax_d_sig_N, ax_ener_N, ax_sig_T, ax_d_sig_T, ax_ener_T

    def update_plot(self, axes):
        ax_sig_N, ax_d_sig_N, ax_ener_N, ax_sig_T, ax_d_sig_T, ax_ener_T = axes
        self.plot_idx(ax_sig_N, ax_d_sig_N, ax_ener_N, 0)
        self.plot_idx(ax_sig_T, ax_d_sig_T,ax_ener_T, 1)

    def energy_calc(self,eps_a,sig_a,eps_in):
        W_arr = (
                cumtrapz(sig_a, eps_a, initial=0)
        )
        eps_el = (eps_a - eps_in)
        U_arr = (
                sig_a * eps_el / 2.0
        )
        G_arr = W_arr - U_arr
        return W_arr, U_arr, G_arr

# if __name__ == "__main__":
#     plane = VUNTIM()
#     fig = plt.figure()
#     ax_sig_N, ax_ener_N, ax_sig_T, ax_ener_T = fig.subplots(1, 4)
#     ax_d_sig_N = ax_sig_N.twinx()
#     ax_d_sig_T = ax_sig_T.twinx()
#     axes = ax_sig_N, ax_d_sig_N, ax_ener_N, ax_sig_T, ax_d_sig_T, ax_ener_T
#     plane.plot_idx(ax_sig_N, ax_d_sig_N, ax_ener_N, 0)
#     plane.plot_idx(ax_sig_T, ax_d_sig_T,ax_ener_T, 1)