'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model 3D

(compressive plasticity (CP) + tensile damage (TD) 
+ cumulative damage sliding (CSD))

Using Jirasek homogenization approach [1999]
'''
import bmcs_utils.view
import numpy as np
import traits.api as tr
from bmcs_utils.api import Float, Instance, EitherType, View, Item
from bmcs_matmod.slide.vcontim import VConTIM
from ibvpy.tmodel.mats3D.mats3D_eval import MATS3DEval

class MicroplaneNT(MATS3DEval):

    gamma_T = Float(100000.,
                       label="Gamma",
                       desc=" Tangential Kinematic hardening modulus",
                       enter_set=True,
                       auto_set=False)

    K_T = Float(10000.,
                   label="K",
                   desc="Tangential Isotropic harening",
                   enter_set=True,
                   auto_set=False)

    S_T = Float(0.005,
                   label="S",
                   desc="Damage strength",
                   enter_set=True,
                   auto_set=False)

    r_T = Float(9.,
                   label="r",
                   desc="Damage cumulation parameter",
                   enter_set=True,
                   auto_set=False)
    e_T = Float(12.,
                   label="e",
                   desc="Damage cumulation parameter",
                   enter_set=True,
                   auto_set=False)

    c_T = Float(4.6,
                   label="c",
                   desc="Damage cumulation parameter",
                   enter_set=True,
                   auto_set=False)

    tau_pi_bar = Float(1.7,
                          label="Tau_bar",
                          desc="Reversibility limit",
                          enter_set=True,
                          auto_set=False)

    a = Float(0.003,
                 label="a",
                 desc="Lateral pressure coefficient",
                 enter_set=True,
                 auto_set=False)

    # -------------------------------------------
    # Normal_Tension constitutive law parameters (without cumulative normal strain)
    # -------------------------------------------
    Ad = Float(100.0,
                  label="a",
                  desc="brittleness coefficient",
                  enter_set=True,
                  auto_set=False)

    eps_0 = Float(0.00008,
                     label="a",
                     desc="threshold strain",
                     enter_set=True,
                     auto_set=False)

    # -----------------------------------------------
    # Normal_Compression constitutive law parameters
    # -----------------------------------------------
    K_N = Float(10000.,
                   label="K_N",
                   desc=" Normal isotropic harening",
                   enter_set=True,
                   auto_set=False)

    gamma_N = Float(5000.,
                       label="gamma_N",
                       desc="Normal kinematic hardening",
                       enter_set=True,
                       auto_set=False)

    sigma_0 = Float(30.,
                       label="sigma_0",
                       desc="Yielding stress",
                       enter_set=True,
                       auto_set=False)

    # -------------------------------------------------------------------------
    # Cached elasticity tensors
    # -------------------------------------------------------------------------

    E = Float(35e+3,
                 label="E",
                 desc="Young's Modulus",
                 auto_set=False,
                 input=True)

    nu = Float(0.2,
                  label='nu',
                  desc="Poison ratio",
                  auto_set=False,
                  input=True)

    ipw_view = View(
        Item('nu'),
        Item('E'),
    )

    n_D = 3

    state_var_shapes = dict(
        omega_N=(),  # damage N
        z_N=(),
        alpha_N =(),
        r_N=(),
        eps_N_p=(),
        sig_N=(),
        omega_T=(),
        z_T=(),
        alpha_T=(n_D,),
        eps_T_p_a=(n_D,),
        sig_T = (n_D,),
    )

    # --------------------------------------------------------------
    # microplane constitutive law (normal behavior CP + TD)
    # --------------------------------------------------------------
    def get_normal_law(self, eps_N, omega_N, z_N, alpha_N, r_N, eps_N_p, **kw):

        E_N = self.E / (1.0 - 2.0 * self.nu)

        # When deciding if a microplane is in tensile or compression, we define a strain boundary such that that
        # sigN <= 0 if eps_N < 0, avoiding entering in the quadrant of compressive strains and traction

        sig_trial = E_N * (eps_N - eps_N_p)
        pos1 = [(eps_N < -1e-6) & (sig_trial > 1e-6)]  # looking for microplanes violating strain boundary
        sig_trial[pos1[0]] = 0
        pos = sig_trial > 1e-6  # microplanes under traction
        pos2 = sig_trial < -1e-6  # microplanes under compression
        H = 1.0 * pos
        H2 = 1.0 * pos2

        # thermo forces
        sig_N_tilde = E_N * (eps_N - eps_N_p)
        sig_N_tilde[pos1[0]] = 0  # imposing strain boundary

        Z = self.K_N * z_N
        X = self.gamma_N * alpha_N * H2
        h = (self.sig_0 + Z) * H2

        f_trial = (abs(sig_N_tilde - X) - h) * H2

        # threshold plasticity

        thres_1 = f_trial > 1e-6

        delta_lamda = f_trial / \
                      (E_N / (1 - omega_N) + abs(self.K_N) + self.gamma_N) * thres_1
        eps_N_p = eps_N_p + delta_lamda * \
                      np.sign(sig_N_tilde - X)
        z_N = z_N + delta_lamda
        alpha_N = alpha_N + delta_lamda * \
                      np.sign(sig_N_tilde - X)

        def R_N(r_N): return (1.0 / self.Ad) * (-r_N / (1.0 + r_N))

        Y_N = 0.5 * H * E_N * (eps_N - eps_N_p) ** 2.0
        Y_0 = 0.5 * E_N * self.eps_0 ** 2.0

        f = (Y_N - (Y_0 + R_N(r_N)))

        # threshold damage

        thres_2 = f > 1e-6

        def f_w(Y): return 1.0 - 1.0 / (1.0 + self.Ad * (Y - Y_0))

        omega_N[f > 1e-6] = f_w(Y_N)[f > 1e-6]
        r_N[f > 1e-6] = -omega_N[f > 1e-6]

        sig_N = (1.0 - H * omega_N) * E_N * (eps_N - eps_N_p)
        pos1 = [(eps_N < -1e-6) & (sig_trial > 1e-6)]  # looking for microplanes violating strain boundary
        sig_N[pos1[0]] = 0

        Z = self.K_N * z_N
        X = self.gamma_N * alpha_N * H2

        return omega_N, z_N, alpha_N, r_N, eps_N_p, sig_N, Z, X, Y_N

    # -------------------------------------------------------------------------
    # microplane constitutive law (Tangential CSD)-(Pressure sensitive cumulative damage)
    # -------------------------------------------------------------------------
    def get_tangential_law(self, eps_T_a, omega_T, z_T,
                           alpha_T_a, eps_T_p_a, sig_N, **kw):

        E_T = self.E * (1.0 - 4 * self.nu) / \
              ((1.0 + self.nu) * (1.0 - 2 * self.nu))

        # thermo forces

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

        # threshold

        f = norm_1 - self.tau_pi_bar - \
            Z + self.a * sig_N

        plas_1 = f > 1e-6
        elas_1 = f < 1e-6

        delta_lamda = f / \
                      (E_T / (1.0 - omega_T) + self.gamma_T + self.K_T) * plas_1

        norm_2 = 1.0 * elas_1 + np.sqrt(
            np.einsum(
                '...na,...na->...n',
                (sig_pi_trial - X), (sig_pi_trial - X))) * plas_1

        eps_T_p_a[..., 0] = eps_T_p_a[..., 0] + plas_1 * delta_lamda * \
                                ((sig_pi_trial[..., 0] - X[..., 0]) /
                                 (1.0 - omega_T)) / norm_2
        eps_T_p_a[..., 1] = eps_T_p_a[..., 1] + plas_1 * delta_lamda * \
                                ((sig_pi_trial[..., 1] - X[..., 1]) /
                                 (1.0 - omega_T)) / norm_2
        eps_T_p_a[..., 2] = eps_T_p_a[..., 2] + plas_1 * delta_lamda * \
                                ((sig_pi_trial[..., 2] - X[..., 2]) /
                                 (1.0 - omega_T)) / norm_2

        omega_T += ((1.0 - omega_T) ** self.c_T) * \
                       (delta_lamda * (Y / self.S_T) ** self.r_T) * \
                       (self.tau_pi_bar / (self.tau_pi_bar - self.a * sig_N)) ** self.e_T

        alpha_T_a[..., 0] = alpha_T_a[..., 0] + plas_1 * delta_lamda * \
                               (sig_pi_trial[..., 0] - X[..., 0]) / norm_2
        alpha_T_a[..., 1] = alpha_T_a[..., 1] + plas_1 * delta_lamda * \
                               (sig_pi_trial[..., 1] - X[..., 1]) / norm_2
        alpha_T_a[..., 2] = alpha_T_a[..., 2] + plas_1 * delta_lamda * \
                               (sig_pi_trial[..., 2] - X[..., 2]) / norm_2
        z_T = z_T + delta_lamda

        sig_T_a = np.einsum(
            '...n,...na->...na', (1 - omega_T), E_T * (eps_T_a - eps_T_p_a))

        Z = self.K_T * z_T
        X = self.gamma_T * alpha_T_a
        Y = 0.5 * E_T * \
            np.einsum(
                '...na,...na->...n',
                (eps_T_a - eps_T_p_a),
                (eps_T_a - eps_T_p_a))

        return omega_T, z_T, alpha_T_a, eps_T_p_a, sig_T_a, Z, X, Y


class MSGeneric(MATS3DEval):

    ipw_view = View(
        Item('E'),
        Item('nu'),
        Item('eps_max'),
    )

    mic = Instance(VConTIM)

    def _mic_default(self):
        mic = VConTIM()
        self.reset_mic(mic)
        return mic

    @tr.on_trait_change('E, nu')
    def _set_E(self, event):
        self.reset_mic(self.mic)

    def reset_mic(self, mic):
        mic.E_N = self.E / (1.0 - 2.0 * self.nu)
        mic.E_T = self.E * (1.0 - 4 * self.nu) / \
                 ((1.0 + self.nu) * (1.0 - 2 * self.nu))

    tree = ['mic']

    state_var_shapes = tr.Property(depends_on='mic')
    @tr.cached_property
    def _get_state_var_shapes(self):
        sv_shapes = {
            name: (self.n_mp,) + shape
            for name, shape
            in self.mic.state_var_shapes.items()
        }
        return sv_shapes

    # -------------------------------------------------------------------------
    # MICROPLANE-Kinematic constraints
    # -------------------------------------------------------------------------
    # get the operator of the microplane normals
    _MPNN = tr.Property(depends_on='n_mp')

    @tr.cached_property
    def _get__MPNN(self):
        MPNN_nij = np.einsum('ni,nj->nij', self._MPN, self._MPN)
        return MPNN_nij

    # get the third order tangential tensor (operator) for each microplane
    _MPTT = tr.Property(depends_on='n_mp')

    @tr.cached_property
    def _get__MPTT(self):
        delta = np.identity(3)
        MPTT_nijr = 0.5 * (
                np.einsum('ni,jr -> nijr', self._MPN, delta) +
                np.einsum('nj,ir -> njir', self._MPN, delta) - 2 *
                np.einsum('ni,nj,nr -> nijr', self._MPN, self._MPN, self._MPN)
        )
        return MPTT_nijr


    def _get_e_a(self, eps_ab):
        """
        Get the microplane projected strains
        """
        # get the normal strain array for each microplane
        e_N = np.einsum('nij,...ij->...n', self._MPNN, eps_ab)
        # get the tangential strain vector array for each microplane
        MPTT_ijr = self._get__MPTT()
        e_T_a = np.einsum('nija,...ij->...na', MPTT_ijr, eps_ab)
        return np.concatenate([e_N[..., np.newaxis], e_T_a], axis=-1)


    def _get_beta_abcd(self, eps_ab, omega_N, omega_T, **Eps):
        """
        Returns the 4th order damage tensor 'beta4' using
        (cf. [Baz99], Eq.(63))
        """
        delta = np.identity(3)
        beta_N = np.sqrt(1. - omega_N)
        beta_T = np.sqrt(1. - omega_T)

        beta_ijkl = (
            np.einsum('n,...n,ni,nj,nk,nl->...ijkl',
                self._MPW, beta_N, self._MPN, self._MPN, self._MPN, self._MPN)
            + 0.25 *
            (
                    np.einsum('n,...n,ni,nk,jl->...ijkl',
                              self._MPW, beta_T, self._MPN, self._MPN, delta) +
                    np.einsum('n,...n,ni,nl,jk->...ijkl',
                              self._MPW, beta_T, self._MPN, self._MPN, delta) +
                    np.einsum('n,...n,nj,nk,il->...ijkl',
                              self._MPW, beta_T, self._MPN, self._MPN, delta) +
                    np.einsum('n,...n,nj,nl,ik->...ijkl',
                              self._MPW, beta_T, self._MPN, self._MPN, delta) -
                    4.0 *
                    np.einsum('n,...n,ni,nj,nk,nl->...ijkl',
                              self._MPW, beta_T, self._MPN, self._MPN, self._MPN, self._MPN)
            )
        )
        return beta_ijkl


    def _get_eps_p_ab(self, eps_ab, eps_N_p, eps_T_p_a, **Eps):
        """
        Integration of the (inelastic) strains for each microplane
        """
        delta = np.identity(3)
        # 2-nd order plastic (inelastic) tensor
        eps_p_ab = (
                np.einsum('n,...n,na,nb->...ab',
                          self._MPW, eps_N_p, self._MPN, self._MPN) +
                0.5 * (
                        np.einsum('n,...nf,na,fb->...ab',
                                  self._MPW, eps_T_p_a, self._MPN, delta) +
                        np.einsum('n,...nf,nb,fa->...ab', self._MPW,
                                  eps_T_p_a, self._MPN, delta)
                )
        )
        return eps_p_ab


    def get_corr_pred(self, eps_ab, t_n1, **Eps):
        """
        Evaluation - get the corrector and predictor

        Definition internal variables / forces per column:  1) damage N, 2)iso N, 3)kin N, 4)consolidation N, 5) eps p N,
        6) sig N, 7) iso F N, 8) kin F N, 9) energy release N, 10) damage T, 11) iso T, 12-13) kin T, 14-15) eps p T,
        16-17) sig T, 18) iso F T, 19-20) kin F T, 21) energy release T

        Corrector predictor computation.
        """
        # ------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        # ------------------------------------------------------------------
        eps_a = self._get_e_a(eps_ab)
        sig_a, D_ab = self.mic.get_corr_pred(eps_a, t_n1, **Eps)
        beta_abcd = self._get_beta_abcd(eps_ab, **Eps)
        # ------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        # ------------------------------------------------------------------
        D_abcd = np.einsum('...ijab, abef, ...cdef->...ijcd',
                           beta_abcd, self.D_abef, beta_abcd)
        # ----------------------------------------------------------------------
        # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
        # ----------------------------------------------------------------------
        # plastic strain tensor
        eps_N_p = Eps['w_pi']
        eps_T_p_a = np.einsum('a...->...a',
                              np.array([Eps['s_pi_x'], Eps['s_pi_y'], Eps['s_pi_z']])
                              )
        sig_N = Eps['sig_pi']
        sig_T_a = np.einsum('a...->...a',
                              np.array([Eps['tau_pi_x'], Eps['tau_pi_y'], Eps['tau_pi_z']])
                              )

        eps_p_ab = self._get_eps_p_ab(eps_ab, eps_N_p, eps_T_p_a)
        # elastic strain tensor
        eps_e_ab = eps_ab - eps_p_ab
        delta = np.identity(3)
        # calculation of the stress tensor
        sig_ab = np.einsum('...abcd,...cd->...ab', D_abcd, eps_e_ab)
        sig_ab_int = (
                np.einsum('n,...n,na,nb->...ab',
                          self._MPW, sig_N, self._MPN, self._MPN) + \
                0.5 * np.einsum('n,...ne,na,eb->...ab',
                                self._MPW, sig_T_a, self._MPN, delta) + \
                0.5 * np.einsum('n,...ne,nb,ea->...ab',
                                self._MPW, sig_T_a, self._MPN, delta))

        return sig_ab, D_abcd

    # -----------------------------------------------
    # number of microplanes - currently fixed for 3D
    # -----------------------------------------------
    n_mp = tr.Constant(28)

    # -----------------------------------------------
    # get the normal vectors of the microplanes
    # -----------------------------------------------
    _MPN = tr.Property(depends_on='n_mp')

    @tr.cached_property
    def _get__MPN(self):
        return np.array([[.577350259, .577350259, .577350259],
                         [.577350259, .577350259, -.577350259],
                         [.577350259, -.577350259, .577350259],
                         [.577350259, -.577350259, -.577350259],
                         [.935113132, .250562787, .250562787],
                         [.935113132, .250562787, -.250562787],
                         [.935113132, -.250562787, .250562787],
                         [.935113132, -.250562787, -.250562787],
                         [.250562787, .935113132, .250562787],
                         [.250562787, .935113132, -.250562787],
                         [.250562787, -.935113132, .250562787],
                         [.250562787, -.935113132, -.250562787],
                         [.250562787, .250562787, .935113132],
                         [.250562787, .250562787, -.935113132],
                         [.250562787, -.250562787, .935113132],
                         [.250562787, -.250562787, -.935113132],
                         [.186156720, .694746614, .694746614],
                         [.186156720, .694746614, -.694746614],
                         [.186156720, -.694746614, .694746614],
                         [.186156720, -.694746614, -.694746614],
                         [.694746614, .186156720, .694746614],
                         [.694746614, .186156720, -.694746614],
                         [.694746614, -.186156720, .694746614],
                         [.694746614, -.186156720, -.694746614],
                         [.694746614, .694746614, .186156720],
                         [.694746614, .694746614, -.186156720],
                         [.694746614, -.694746614, .186156720],
                         [.694746614, -.694746614, -.186156720]])

    # -------------------------------------
    # get the weights of the microplanes
    # -------------------------------------
    _MPW = tr.Property(depends_on='n_mp')

    @tr.cached_property
    def _get__MPW(self):
        return np.array([.0160714276, .0160714276, .0160714276, .0160714276, .0204744730,
                         .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                         .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                         .0204744730, .0158350505, .0158350505, .0158350505, .0158350505,
                         .0158350505, .0158350505, .0158350505, .0158350505, .0158350505,
                         .0158350505, .0158350505, .0158350505]) * 6.0
