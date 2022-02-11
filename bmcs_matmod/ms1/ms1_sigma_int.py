'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model 2D

(compressive plasticity (CP) + tensile damage (TD) 
+ cumulative damage sliding (CSD))

Using Jirasek homogenization approach [1999]
'''
import copy

from ibvpy.tmodel.mats3D.mats3D_eval import MATS3DEval

from bmcs_utils.api import InteractiveModel, Item, View
import numpy as np
import traits.api as tr


class MS1_sigma_int(MATS3DEval, InteractiveModel):
    gamma_T = tr.Float(1000000.,
                       label="gamma_T",
                       desc=" Tangential Kinematic hardening modulus",
                       enter_set=True,
                       auto_set=False)

    K_T = tr.Float(10000.,
                   label="K_T",
                   desc="Tangential Isotropic harening",
                   enter_set=True,
                   auto_set=False)

    S_T = tr.Float(0.005,
                   label="S_T",
                   desc="Damage strength",
                   enter_set=True,
                   auto_set=False)

    r_T = tr.Float(9.,
                   label="r",
                   desc="Damage cumulation parameter",
                   enter_set=True,
                   auto_set=False)
    p_T = tr.Float(2.,
                   label="p_T",
                   desc="Damage cumulation parameter",
                   enter_set=True,
                   auto_set=False)

    c_T = tr.Float(3,
                   label="c_T",
                   desc="Damage cumulation parameter",
                   enter_set=True,
                   auto_set=False)

    sigma_T_0 = tr.Float(1.7,
                          label="sigma_T_0",
                          desc="Reversibility limit",
                          enter_set=True,
                          auto_set=False)

    m_T = tr.Float(0.1,
                 label="m_T",
                 desc="Lateral pressure coefficient",
                 enter_set=True,
                 auto_set=False)

    # -------------------------------------------
    # Normal_Tension constitutive law parameters (without cumulative normal strain)
    # -------------------------------------------
    Ad = tr.Float(10.0,
                  label="A_d",
                  desc="brittleness coefficient",
                  enter_set=True,
                  auto_set=False)

    eps_0 = tr.Float(.0001,
                     label="eps_N_0",
                     desc="threshold strain",
                     enter_set=True,
                     auto_set=False)

    # -----------------------------------------------
    # Normal_Compression constitutive law parameters
    # -----------------------------------------------
    K_N = tr.Float(10000.,
                   label="K_N",
                   desc=" Normal isotropic harening",
                   enter_set=True,
                   auto_set=False)

    gamma_N = tr.Float(5000.,
                       label="gamma_N",
                       desc="Normal kinematic hardening",
                       enter_set=True,
                       auto_set=False)

    sigma_N_0 = tr.Float(10.,
                       label="sigma_N_0",
                       desc="Yielding stress",
                       enter_set=True,
                       auto_set=False)

    # -------------------------------------------------------------------------
    # Cached elasticity tensors
    # -------------------------------------------------------------------------

    E = tr.Float(35e+3,
                 label="E",
                 desc="Young's Modulus",
                 auto_set=False,
                 input=True)

    nu = tr.Float(0.2,
                  label='nu',
                  desc="Poison ratio",
                  auto_set=False,
                  input=True)

    ipw_view = View(
        Item('gamma_T', latex=r'\gamma_\mathrm{T}', minmax=(10, 100000)),
        Item('K_T', latex=r'K_\mathrm{T}', minmax=(10, 10000)),
        Item('S_T', latex=r'S_\mathrm{T}', minmax=(0.001, 0.01)),
        Item('r_T', latex=r'r_\mathrm{T}', minmax=(1, 3)),
        Item('p_T', latex=r'e_\mathrm{T}', minmax=(1, 40)),
        Item('c_T', latex=r'c_\mathrm{T}', minmax=(1, 10)),
        Item('sigma_T_0', latex=r'\bar{sigma}^\pi_{T}', minmax=(1, 10)),
        Item('m_T', latex=r'm_\mathrm{T}', minmax=(0.001, 3)),
    )

    n_D = 3

    state_var_shapes = tr.Property
    @tr.cached_property
    def _get_state_var_shapes(self):
        return {
            name: (self.n_mp,) + shape
            for name, shape
            in self.mic_state_var_shapes.items()
        }

    mic_state_var_shapes = dict(
        omega_N_Emn=(),  # damage N
        z_N_Emn=(),
        alpha_N_Emn =(),
        r_N_Emn=(),
        eps_N_p_Emn=(),
        sigma_N_Emn=(),
        omega_T_Emn=(),
        z_T_Emn=(),
        alpha_T_Emna=(n_D,),
        eps_T_pi_Emna=(n_D,),
        plastic_dissip_T_Emn=(),
        damage_dissip_T_Emn =(),
        plastic_dissip_N_Emn=(),
        damage_dissip_N_Emn=(),
    )
    '''
    State variables
     1) damage N, 
     2) iso N, 
     3) kin N, 
     4) consolidation N, 
     5) eps p N,
     6) sigma N, 
     7) iso F N, 
     8) kin F N, 
     9) energy release N, 
     10) damage T, 
     11) iso T, 
     12-13) kin T, 
     14-15) eps p T,
     16-17) sigma T, 18) iso F T, 19-20) kin F T, 21) energy release T
    '''

    # --------------------------------------------------------------
    # microplane constitutive law (normal behavior CP + TD)
    # (without cumulative normal strain for fatigue under tension)
    # --------------------------------------------------------------
    def get_normal_law(self, eps_N_Emn, omega_N_Emn, z_N_Emn,
                           alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, omega_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna,plastic_dissip_T_Emn, damage_dissip_T_Emn,
                           plastic_dissip_N_Emn, damage_dissip_N_Emn):

        E_N = self.E / (1.0 - 2.0 * self.nu)

        sigma_N_Emn_tilde = E_N * (eps_N_Emn - eps_N_p_Emn)

        pos = sigma_N_Emn_tilde > 1e-6  # microplanes under tension
        pos2 = sigma_N_Emn_tilde < -1e-6  # microplanes under compression
        tension = 1.0 * pos
        compression = 1.0 * pos2

        # thermo forces

        Z = self.K_N * z_N_Emn * compression
        X = self.gamma_N * alpha_N_Emn * compression
        h = (self.sigma_N_0 + Z) * compression

        f_trial = (abs(sigma_N_Emn_tilde - X) - h) * compression

        # threshold plasticity

        thres_1 = f_trial > 1e-10

        delta_lamda = f_trial / \
                      (E_N / (1 - omega_N_Emn) + abs(self.K_N) + self.gamma_N) * thres_1
        eps_N_p_Emn += delta_lamda * \
                      np.sign(sigma_N_Emn_tilde - X)
        z_N_Emn += delta_lamda
        alpha_N_Emn += delta_lamda * \
                      np.sign(sigma_N_Emn_tilde - X)

        def R_N(r_N_Emn): return (1.0 / self.Ad) * (-r_N_Emn / (1.0 + r_N_Emn))

        Y_N = 0.5 * tension * E_N * (eps_N_Emn - eps_N_p_Emn) ** 2.0
        Y_0 = 0.5 * E_N * self.eps_0 ** 2.0

        f = (Y_N - (Y_0 + R_N(r_N_Emn)))*tension

        # threshold damage

        thres_2 = f > 1e-6

        def f_w(Y): return 1.0 - 1.0 / (1.0 + self.Ad * (Y - Y_0))

        omega_N_Emn[f > 1e-6] = f_w(Y_N)[f > 1e-6]
        omega_N_Emn[...] = np.clip(omega_N_Emn,0,1.0)
        r_N_Emn[f > 1e-6] = -omega_N_Emn[f > 1e-6]

        sigma_N_Emn = (1.0 - tension * omega_N_Emn) * E_N * (eps_N_Emn - eps_N_p_Emn)
        # sigma_N_Emn = E_N * (eps_N_Emn - eps_N_p_Emn)
        # pos1 = [(eps_N_Emn < -1e-6) & (sigma_trial > 1e-6)]  # looking for microplanes violating strain boundary
        # sigma_N_Emn[pos1[0]] = 0


        Z = self.K_N * z_N_Emn * compression
        X = self.gamma_N * alpha_N_Emn * compression


        return sigma_N_Emn, Z, X, Y_N

    # -------------------------------------------------------------------------
    # microplane constitutive law (Tangential CSD)-(Pressure sensitive cumulative damage)
    # -------------------------------------------------------------------------
    def get_tangential_law(self, eps_T_Emna, eps_Emab, omega_N_Emn, z_N_Emn,
                           alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, omega_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna,plastic_dissip_T_Emn, damage_dissip_T_Emn,
                           plastic_dissip_N_Emn, damage_dissip_N_Emn):


        E_T = self.E * (1.0 - 4 * self.nu) / \
            ((1.0 + self.nu) * (1.0 - 2 * self.nu))

        # thermo forces

        sig_pi_trial = E_T * (eps_T_Emna - eps_T_pi_Emna)

        Z = self.K_T * z_T_Emn
        X = self.gamma_T * alpha_T_Emna
        norm_1 = np.sqrt(
            np.einsum(
                '...na,...na->...n',
                (sig_pi_trial - X), (sig_pi_trial - X))
        )
        Y = 0.5 * E_T * \
            np.einsum(
                '...na,...na->...n',
                (eps_T_Emna - eps_T_pi_Emna),
                (eps_T_Emna - eps_T_pi_Emna))


        E_N = self.E / (1.0 - 2.0 * self.nu)

        eps_N_Emn = self._get_e_N_Emn(eps_Emab)
        sigma_N_Emn = (1.0 - omega_N_Emn) * E_N * (eps_N_Emn - eps_N_p_Emn)

        f = norm_1 - self.sigma_T_0 - Z + self.m_T * sigma_N_Emn

        plas_1 = f > 1e-15
        elas_1 = f < 1e-15

        delta_lamda = f / \
                      (E_T / (1.0 - omega_T_Emn) + self.gamma_T + self.K_T) * plas_1

        norm_2 = 1.0 * elas_1 + np.sqrt(
            np.einsum(
                '...na,...na->...n',
                (sig_pi_trial - X), (sig_pi_trial - X))) * plas_1

        eps_T_pi_Emna[..., 0] += plas_1 * delta_lamda * \
                                ((sig_pi_trial[..., 0] - X[..., 0]) /
                                 (1.0 - omega_T_Emn)) / norm_2
        eps_T_pi_Emna[..., 1] += plas_1 * delta_lamda * \
                                ((sig_pi_trial[..., 1] - X[..., 1]) /
                                 (1.0 - omega_T_Emn)) / norm_2

        eps_T_pi_Emna[..., 2] +=  plas_1 * delta_lamda * \
                                ((sig_pi_trial[..., 2] - X[..., 2]) /
                                 (1.0 - omega_T_Emn)) / norm_2
        omega_T_Emn += plas_1 * ((1 - omega_T_Emn) ** self.c_T) * \
                       (delta_lamda * (Y / self.S_T) ** self.r_T) * \
                       (self.sigma_T_0 / (self.sigma_T_0 - self.m_T * sigma_N_Emn)) ** self.p_T
        omega_T_Emn[...] = np.clip(omega_T_Emn,0,1.0)

        alpha_T_Emna[..., 0] += plas_1 * delta_lamda * \
                               (sig_pi_trial[..., 0] - X[..., 0]) / norm_2
        alpha_T_Emna[..., 1] += plas_1 * delta_lamda * \
                               (sig_pi_trial[..., 1] - X[..., 1]) / norm_2

        alpha_T_Emna[..., 2] += plas_1 * delta_lamda * \
                               (sig_pi_trial[..., 2] - X[..., 2]) / norm_2



        z_T_Emn += plas_1 * delta_lamda

        sigma_T_Emna = np.einsum(
            '...n,...na->...na', (1 - omega_T_Emn), E_T * (eps_T_Emna - eps_T_pi_Emna))

        Z = self.K_T * z_T_Emn
        X = self.gamma_T * alpha_T_Emna
        Y = 0.5 * E_T * \
            np.einsum(
                '...na,...na->...n',
                (eps_T_Emna - eps_T_pi_Emna),
                (eps_T_Emna - eps_T_pi_Emna))

        # Energy dissipation evaluation

        delta_eps_T_pi_Emna = np.zeros_like(eps_T_pi_Emna)
        delta_alpha_T_Emna = np.zeros_like(alpha_T_Emna)
        delta_z_T_Emn = np.zeros_like(z_T_Emn)
        delta_omega_T_Emn = np.zeros_like(omega_T_Emn)

        delta_eps_T_pi_Emna[..., 0] = plas_1 * delta_lamda * \
                                ((sig_pi_trial[..., 0] - X[..., 0]) /
                                 (1.0 - omega_T_Emn)) / norm_2
        delta_eps_T_pi_Emna[..., 1] = plas_1 * delta_lamda * \
                                      ((sig_pi_trial[..., 1] - X[..., 1]) /
                                       (1.0 - omega_T_Emn)) / norm_2
        delta_eps_T_pi_Emna[..., 2] = plas_1 * delta_lamda * \
                                      ((sig_pi_trial[..., 2] - X[..., 2]) /
                                       (1.0 - omega_T_Emn)) / norm_2

        delta_alpha_T_Emna[..., 0] = plas_1 * delta_lamda * \
                                (sig_pi_trial[..., 0] - X[..., 0]) / norm_2
        delta_alpha_T_Emna[..., 1] = plas_1 * delta_lamda * \
                                (sig_pi_trial[..., 1] - X[..., 1]) / norm_2

        delta_alpha_T_Emna[..., 2] = plas_1 * delta_lamda * \
                                (sig_pi_trial[..., 2] - X[..., 2]) / norm_2

        delta_z_T_Emn = plas_1 * delta_lamda

        delta_omega_T_Emn = plas_1 * ((1 - omega_T_Emn) ** self.c_T) * \
                       (delta_lamda * (Y / self.S_T) ** self.r_T) * \
                       (self.sigma_T_0 / (self.sigma_T_0 - self.m_T * sigma_N_Emn)) ** self.p_T

        plastic_dissip_T_Emn = np.einsum('...na,...na->...n', sigma_T_Emna, delta_eps_T_pi_Emna) - \
        np.einsum('...na,...na->...n', X, delta_alpha_T_Emna) - np.einsum('...n,...n->...n', Z, delta_z_T_Emn)

        damage_dissip_T_Emn = np.einsum('...n,...n->...n', Y, delta_omega_T_Emn)

        # if plastic_dissip_T_Emn.any() < -1e-15:
        #     print(sigma_T_Emna, delta_eps_T_pi_Emna)
        return sigma_T_Emna, Z, X, Y, plastic_dissip_T_Emn

    #     #-------------------------------------------------------------------------
    #     # MICROPLANE-Kinematic constraints
    #     #-------------------------------------------------------------------------

    # -------------------------------------------------

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
        delta = self.DELTA
        MPTT_nijr = 0.5 * (
                np.einsum('ni,jr -> nijr', self._MPN, delta) +
                np.einsum('nj,ir -> njir', self._MPN, delta) - 2 *
                np.einsum('ni,nj,nr -> nijr', self._MPN, self._MPN, self._MPN)
        )
        return MPTT_nijr

    def _get_e_N_Emn(self, eps_Emab):
        # get the normal strain array for each microplane
        return np.einsum('nij,...ij->...n', self._MPNN, eps_Emab)

    def _get_e_T_Emna(self, eps_Emab):
        # get the tangential strain vector array for each microplane
        MPTT_ijr = self._get__MPTT()
        return np.einsum('nija,...ij->...na', MPTT_ijr, eps_Emab)


    # ---------------------------------------------------------------------
    # Extra homogenization of damage tensor in case of two damage parameters
    # Returns the 4th order damage tensor 'beta4' using (ref. [Baz99], Eq.(63))
    # ---------------------------------------------------------------------

    def _get_beta_Emabcd(self, eps_Emab, omega_N_Emn, z_N_Emn,
                           alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, omega_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna,plastic_dissip_T_Emn, damage_dissip_T_Emn,
                           plastic_dissip_N_Emn, damage_dissip_N_Emn):
        # Returns the 4th order damage tensor 'beta4' using
        # (cf. [Baz99], Eq.(63))

        delta = self.DELTA
        beta_N = np.sqrt(1. - omega_N_Emn)
        beta_T = np.sqrt(1. -  omega_T_Emn)

        beta_ijkl = np.einsum('n, ...n,ni, nj, nk, nl -> ...ijkl', self._MPW, beta_N, self._MPN, self._MPN, self._MPN,
                              self._MPN) + \
                    0.25 * (np.einsum('n, ...n,ni, nk, jl -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) +
                            np.einsum('n, ...n,ni, nl, jk -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) +
                            np.einsum('n, ...n,nj, nk, il -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) +
                            np.einsum('n, ...n,nj, nl, ik -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) -
                            4.0 * np.einsum('n, ...n, ni, nj, nk, nl -> ...ijkl', self._MPW, beta_T, self._MPN,
                                            self._MPN, self._MPN, self._MPN))

        return beta_ijkl

    def _get_phi(self, eps_Emab, omega_N_Emn, z_N_Emn,
                         alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, omega_T_Emn, z_T_Emn,
                         alpha_T_Emna, eps_T_pi_Emna,plastic_dissip_T_Emn):

        phi_n = np.sqrt(1.0 - omega_N_Emn)*np.sqrt(1.0 - omega_T_Emn)
        phi_ab = np.einsum('...n,n,nab->...ab', phi_n, self._MPW, self._MPNN)
        return phi_ab

    DELTA = np.identity(n_D)

    # -----------------------------------------------------------
    # Integration of the (inelastic) strains for each microplane
    # -----------------------------------------------------------

    def _get_eps_p_Emab(self, eps_Emab, omega_N_Emn, z_N_Emn,
                           alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, omega_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna,plastic_dissip_T_Emn, damage_dissip_T_Emn,
                           plastic_dissip_N_Emn, damage_dissip_N_Emn):

        # 2-nd order plastic (inelastic) tensor
        eps_p_Emab = (
                np.einsum('n,...n,na,nb->...ab',
                          self._MPW, eps_N_p_Emn, self._MPN, self._MPN) +
                0.5 * (
                        np.einsum('n,...nf,na,fb->...ab',
                                  self._MPW, eps_T_pi_Emna, self._MPN, self.DELTA) +
                        np.einsum('n,...nf,nb,fa->...ab',
                                  self._MPW, eps_T_pi_Emna, self._MPN, self.DELTA)
                )
        )

        return eps_p_Emab

    def get_corr_pred(self, eps_Emab, t_n1, **Eps_k):
        """Evaluation - get the corrector and predictor
        """
        # Corrector predictor computation.

        eps_N_Emn = self._get_e_N_Emn(eps_Emab)
        eps_T_Emna = self._get_e_T_Emna(eps_Emab)

        sigma_N_Emn, Z, X, Y_N = self.get_normal_law(eps_N_Emn,
            **Eps_k)

        # sliding tangential strains
        sigma_T_Emna, Z_T, X_T, Y_T, plastic_dissip_T_Emn = self.get_tangential_law(eps_T_Emna,
            eps_Emab,**Eps_k)

        delta = np.identity(3)

        sig_Emab = (
                np.einsum('n,...n,na,nb->...ab',
                          self._MPW, sigma_N_Emn, self._MPN, self._MPN) +
                0.5 * (
                        np.einsum('n,...nf,na,fb->...ab',
                                  self._MPW, sigma_T_Emna, self._MPN, delta) +
                        np.einsum('n,...nf,nb,fa->...ab', self._MPW,
                                  sigma_T_Emna, self._MPN, delta)
                )
        )
        # ------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        # ------------------------------------------------------------------
        beta_Emabcd = self._get_beta_Emabcd(eps_Emab, **Eps_k)

        # ------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        # ------------------------------------------------------------------

        D_Emabcd = np.einsum(
            '...ijab, abef, ...cdef->...ijcd',
            beta_Emabcd, self.D_abef, beta_Emabcd
        )

        # phi_ab = self._get_phi(eps_Emab, **Eps_k)
        #
        #
        # D_Emabcd = 0.25 * (
        #        np.einsum('mjnl, ...im, ...kn -> ...ijkl', self.D_abef, phi_ab, phi_ab) +
        #        np.einsum('imnl, ...jm, ...kn -> ...ijkl', self.D_abef, phi_ab, phi_ab) +
        #        np.einsum('mjkn, ...im, ...ln -> ...ijkl', self.D_abef, phi_ab, phi_ab) +
        #        np.einsum('imkn, ...jm, ...ln -> ...ijkl', self.D_abef, phi_ab, phi_ab)
        # )


        return sig_Emab, D_Emabcd


class MS12DPVW(MS1_sigma_int):
    """Two dimensional version of the MS1 model
    """

    n_mp = tr.Constant(360)
    '''Number of microplanes
    '''

    # -----------------------------------------------
    # get the normal vectors of the microplanes
    # -----------------------------------------------
    _MPN = tr.Property(depends_on='n_mp')

    @tr.cached_property
    def _get__MPN(self):
        # microplane normals:
        alpha_list = np.linspace(0, 2 * np.pi, self.n_mp)

        MPN = np.array([[np.cos(alpha), np.sin(alpha)]
                        for alpha in alpha_list])

        return MPN

    # -------------------------------------
    # get the weights of the microplanes
    # -------------------------------------
    _MPW = tr.Property(depends_on='n_mp')

    @tr.cached_property
    def _get__MPW(self):
        MPW = np.ones(self.n_mp) / self.n_mp * 2

        return MPW


class MS13DPVW(MS1_sigma_int):
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

    # -------------------------------------------------------------------------
    # Cached elasticity tensors
    # -------------------------------------------------------------------------

    def _get_lame_params(self):
        la = self.E * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E / (2. + 2.* self.nu)
        return la, mu

    D_abef = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_abef(self):
        la = self._get_lame_params()[0]
        mu = self._get_lame_params()[1]
        delta = np.identity(3)
        D_abef = (np.einsum(',ij,kl->ijkl', la, delta, delta) +
                  np.einsum(',ik,jl->ijkl', mu, delta, delta) +
                  np.einsum(',il,jk->ijkl', mu, delta, delta))

        return D_abef



    # -------------------------------------------------------------------------
    # Response variables for postprocessing
    # -------------------------------------------------------------------------

    def get_max_omega(self, eps_Emab, t_n1, **Eps_k):
        max_omega_N = self.get_max_omega_N(eps_Emab, t_n1, **Eps_k)
        max_omega_T = self.get_max_omega_T(eps_Emab, t_n1, **Eps_k)
        max_omega_NT = np.array([max_omega_N, max_omega_T], np.float_)
        return np.max(max_omega_NT, axis=0)

    def get_max_omega_N(self, eps_Emab, t_n1, **Eps_k):
        return np.max( Eps_k['omega_N_Emn'], axis=-1)

    def get_max_omega_T(self, eps_Emab, t_n1, **Eps_k):
        return np.max( Eps_k['omega_T_Emn'], axis=-1)

    def get_omega_ab(self, eps_Emab, t_n1, **Eps_k):
        return np.identity(3) - self._get_phi(eps_Emab, **Eps_k)

    def get_eps_p_ab(self, eps_Emab, t_n1, **Eps_k):
        return self._get_eps_p_Emab(eps_Emab, **Eps_k)

    def _get_var_dict(self):
        var_dict = super()._get_var_dict()
        var_dict.update(
            eps_p_ab=self.get_eps_p_ab,
            max_omega=self.get_max_omega,
            max_omega_N=self.get_max_omega_N,
            max_omega_T=self.get_max_omega_T,
            omega_ab = self.get_omega_ab
        )
        return var_dict
