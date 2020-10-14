'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model 2D

(compressive plasticity (CP) + tensile damage (TD) 
+ cumulative damage sliding (CSD))

Using Jirasek homogenization approach [1999]
'''

from bmcs_matmod.matmod.mats_base import MATSBase
from bmcs_utils.api import InteractiveModel, Item, View
import numpy as np
import traits.api as tr


class MATSXDMplCSDEEQ(MATSBase, InteractiveModel):

    gamma_T = tr.Float(100000.,
                       label="Gamma",
                        desc=" Tangential Kinematic hardening modulus",
                        enter_set=True,
                        auto_set=False)

    K_T = tr.Float(10000.,
                label="K",
                desc="Tangential Isotropic harening",
                enter_set=True,
                auto_set=False)

    S_T = tr.Float(0.005,
                label="S",
                desc="Damage strength",
                enter_set=True,
                auto_set=False)

    r_T = tr.Float(9.,
                label="r",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)
    e_T = tr.Float(12.,
                label="e",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    c_T = tr.Float(4.6,
                label="c",
                desc="Damage cumulation parameter",
                enter_set=True,
                auto_set=False)

    tau_pi_bar = tr.Float(1.7,
                       label="Tau_bar",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    a = tr.Float(0.003,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    ipw_view = View(
        Item('gamma_T', latex=r'\gamma_\mathrm{T}', minmax=(10,100000)),
        Item('K_T', latex=r'K_\mathrm{T}', minmax=(10, 10000)),
        Item('S_T', latex=r'S_\mathrm{T}', minmax=(0.001, 0.01)),
        Item('r_T', latex=r'r_\mathrm{T}', minmax=(1, 3)),
        Item('e_T', latex=r'e_\mathrm{T}', minmax=(1, 40)),
        Item('c_T', latex=r'c_\mathrm{T}', minmax=(1, 10)),
        Item('tau_pi_bar', latex=r'\bar{\tau}', minmax=(1, 10)),
        Item('a', latex=r'a', minmax=(0.001, 3)),
    )

    # -------------------------------------------
    # Normal_Tension constitutive law parameters (without cumulative normal strain)
    # -------------------------------------------
    Ad = tr.Float(100.0,
               label="a",
               desc="brittleness coefficient",
               enter_set=True,
               auto_set=False)

    eps_0 = tr.Float(0.00008,
                  label="a",
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

    sigma_0 = tr.Float(30.,
                    label="sigma_0",
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

    def _get_lame_params(self):
        # la = self.E * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        # # second Lame parameter (shear modulus)
        # mu = self.E / (2. + 2. * self.nu)
        la = self.E * self.nu / ((1. + self.nu) * (1. - self.nu))
        mu = self.E / (2. + 2. * self.nu)
        return la, mu

    D_abef = tr.Property(tr.Array, depends_on='+input')

    @tr.cached_property
    def _get_D_abef(self):
        la = self._get_lame_params()[0]
        mu = self._get_lame_params()[1]
        delta = np.identity(2)
        D_abef = (np.einsum(',ij,kl->ijkl', la, delta, delta) +
                  np.einsum(',ik,jl->ijkl', mu, delta, delta) +
                  np.einsum(',il,jk->ijkl', mu, delta, delta))
        return D_abef

    @tr.cached_property
    def _get_state_var_shapes(self):
        return {'w_N_Emn': (self.n_mp,),
                'z_N_Emn': (self.n_mp,),
                'alpha_N_Emn': (self.n_mp,),
                'r_N_Emn': (self.n_mp,),
                'eps_N_p_Emn': (self.n_mp,),
                'sigma_N_Emn': (self.n_mp,),
                'w_T_Emn': (self.n_mp,),
                'z_T_Emn': (self.n_mp,),
                'alpha_T_Emna': (self.n_mp, 2),
                'eps_T_pi_Emna': (self.n_mp, 2),
                }

    #--------------------------------------------------------------
    # microplane constitutive law (normal behavior CP + TD)
    # (without cumulative normal strain for fatigue under tension)
    #--------------------------------------------------------------
    def get_normal_law(self, eps_N_Emn, omega_N_Emn, z_N_Emn,
                       alpha_N_Emn, r_N_Emn, eps_N_p_Emn, eps_aux):

        eps_N_Aux = self._get_e_N_Emn_2(eps_aux)

        E_N = self.E / (1.0 - 2.0 * self.nu)

        # E_N = self.E * (1.0 + 2.0 *  self.nu) / (1.0 - self.nu**2)


        # When deciding if a microplane is in tensile or compression, we define a strain boundary such that that
        # sigmaN <= 0 if eps_N < 0, avoiding entering in the quadrant of compressive strains and traction

        sigma_trial = E_N * (eps_N_Emn - eps_N_p_Emn)
        pos1 = [(eps_N_Emn < -1e-6) & (sigma_trial > 1e-6)] # looking for microplanes violating strain boundary
        sigma_trial [pos1[0]] = 0
        pos = eps_N_Emn > 1e-6                            # microplanes under traction
        pos2 = eps_N_Emn < -1e-6                          # microplanes under compression
        H = 1.0 * pos
        H2 = 1.0 * pos2

        # thermo forces
        sigma_N_Emn_tilde = E_N * (eps_N_Emn - eps_N_p_Emn)
        sigma_N_Emn_tilde[pos1[0]] = 0                      # imposing strain boundary

        Z = self.K_N * z_N_Emn
        X = self.gamma_N * alpha_N_Emn * H2
        h = (self.sigma_0 + Z) * H2

        f_trial = (abs(sigma_N_Emn_tilde - X) - h) * H2

        # threshold plasticity

        thres_1 = f_trial > 1e-6

        delta_lamda = f_trial / \
            (E_N / (1 - omega_N_Emn) + abs(self.K_N) + self.gamma_N) * thres_1
        eps_N_p_Emn = eps_N_p_Emn + delta_lamda * \
            np.sign(sigma_N_Emn_tilde - X)
        z_N_Emn = z_N_Emn + delta_lamda
        alpha_N_Emn = alpha_N_Emn + delta_lamda * \
            np.sign(sigma_N_Emn_tilde - X)

        def R_N(r_N_Emn): return (1.0 / self.Ad) * (-r_N_Emn / (1.0 + r_N_Emn))

        Y_N = 0.5 * H * E_N * (eps_N_Emn - eps_N_p_Emn) ** 2.0
        Y_0 = 0.5 * E_N * self.eps_0 ** 2.0

        f = (Y_N - (Y_0 + R_N(r_N_Emn)))

        # threshold damage

        thres_2 = f > 1e-6

        def f_w(Y): return 1.0 - 1.0 / (1.0 + self.Ad * (Y - Y_0))

        omega_N_Emn[f > 1e-6] = f_w(Y_N)[f > 1e-6]
        r_N_Emn[f > 1e-6] = -omega_N_Emn[f > 1e-6]

        sigma_N_Emn = (1.0 - H * omega_N_Emn) * E_N * (eps_N_Emn - eps_N_p_Emn)
        pos1 = [(eps_N_Emn < -1e-6) & (sigma_trial > 1e-6)] # looking for microplanes violating strain boundary
        sigma_N_Emn[pos1[0]] = 0


        Z = self.K_N * z_N_Emn
        X = self.gamma_N * alpha_N_Emn * H2

        return omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z, X, Y_N

    #-------------------------------------------------------------------------
    # microplane constitutive law (Tangential CSD)-(Pressure sensitive cumulative damage)
    #-------------------------------------------------------------------------
    def get_tangential_law(self, eps_T_Emna, omega_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn):

        E_T = self.E / (1.0 + self.nu)

        # E_T = self.E * (1.0 - 4 * self.nu) / \
        #     ((1.0 + self.nu) * (1.0 - 2 * self.nu))

        # E_T = self.E * (1.0 - 3.0 *  self.nu) / (1.0 - self.nu**2)


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

        # threshold

        f = norm_1 - self.tau_pi_bar - \
            Z + self.a * sigma_N_Emn

        plas_1 = f > 1e-6
        elas_1 = f < 1e-6

        delta_lamda = f / \
            (E_T / (1.0 - omega_T_Emn) + self.gamma_T + self.K_T) * plas_1

        norm_2 = 1.0 * elas_1 + np.sqrt(
            np.einsum(
                '...na,...na->...n',
                (sig_pi_trial - X), (sig_pi_trial - X))) * plas_1

        eps_T_pi_Emna[..., 0] = eps_T_pi_Emna[..., 0] + plas_1 * delta_lamda * \
            ((sig_pi_trial[..., 0] - X[..., 0]) /
             (1.0 - omega_T_Emn)) / norm_2
        eps_T_pi_Emna[..., 1] = eps_T_pi_Emna[..., 1] + plas_1 * delta_lamda * \
            ((sig_pi_trial[..., 1] - X[..., 1]) /
             (1.0 - omega_T_Emn)) / norm_2

        omega_T_Emn += ((1 - omega_T_Emn) ** self.c_T) * \
            (delta_lamda * (Y / self.S_T) ** self.r_T) * \
            (self.tau_pi_bar / (self.tau_pi_bar + self.a * sigma_N_Emn)) ** self.e_T

        alpha_T_Emna[..., 0] = alpha_T_Emna[..., 0] + plas_1 * delta_lamda * \
            (sig_pi_trial[..., 0] - X[..., 0]) / norm_2
        alpha_T_Emna[..., 1] = alpha_T_Emna[..., 1] + plas_1 * delta_lamda * \
            (sig_pi_trial[..., 1] - X[..., 1]) / norm_2

        z_T_Emn = z_T_Emn + delta_lamda

        sigma_T_Emna = np.einsum(
            '...n,...na->...na', (1 - omega_T_Emn), E_T * (eps_T_Emna - eps_T_pi_Emna))

        Z = self.K_T * z_T_Emn
        X = self.gamma_T * alpha_T_Emna
        Y = 0.5 * E_T * \
            np.einsum(
                '...na,...na->...n',
                (eps_T_Emna - eps_T_pi_Emna),
                (eps_T_Emna - eps_T_pi_Emna))

        return omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z, X, Y

#     #-------------------------------------------------------------------------
#     # MICROPLANE-Kinematic constraints
#     #-------------------------------------------------------------------------

    #-------------------------------------------------

    # get the operator of the microplane normals
    _MPNN = tr.Property(depends_on='n_mp')

    @tr.cached_property
    def _get__MPNN(self):
        print(self._MPN)
        MPNN_nij = np.einsum('ni,nj->nij', self._MPN, self._MPN)
        return MPNN_nij

    # get the third order tangential tensor (operator) for each microplane
    _MPTT = tr.Property(depends_on='n_mp')

    @tr.cached_property
    def _get__MPTT(self):
        delta = np.identity(2)
        MPTT_nijr = 0.5 * (
            np.einsum('ni,jr -> nijr', self._MPN, delta) +
            np.einsum('nj,ir -> njir', self._MPN, delta) - 2 *
            np.einsum('ni,nj,nr -> nijr', self._MPN, self._MPN, self._MPN)
        )
        return MPTT_nijr

    def _get_e_N_Emn_2(self, eps_Emab):
        # get the normal strain array for each microplane
        return np.einsum('nij,...ij->...n', self._MPNN, eps_Emab)

    def _get_e_T_Emnar_2(self, eps_Emab):
        # get the tangential strain vector array for each microplane
        MPTT_ijr = self._get__MPTT()
        return np.einsum('nija,...ij->...na', MPTT_ijr, eps_Emab)

    #---------------------------------------------------------------------
    # Extra homogenization of damage tensor in case of two damage parameters
    # Returns the 4th order damage tensor 'beta4' using (ref. [Baz99], Eq.(63))
    #---------------------------------------------------------------------

    def _get_beta_Emabcd_2(self, eps_Emab, omega_N_Emn, z_N_Emn,
                           alpha_N_Emn, r_N_Emn, eps_N_p_Emn, omega_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna, eps_aux):

        # Returns the 4th order damage tensor 'beta4' using
        #(cf. [Baz99], Eq.(63))

        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emnar_2(eps_Emab)

        omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_n, X_n, Y_n = self.get_normal_law(
            eps_N_Emn, omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, eps_aux)

        omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z_T, X_T, Y_T = self.get_tangential_law(
            eps_T_Emna, omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

        delta = np.identity(2)
        beta_N = np.sqrt(1. - omega_N_Emn)
        beta_T = np.sqrt(1. - omega_T_Emn)

        beta_ijkl = np.einsum('n, ...n,ni, nj, nk, nl -> ...ijkl', self._MPW, beta_N, self._MPN, self._MPN, self._MPN, self._MPN) + \
            0.25 * (np.einsum('n, ...n,ni, nk, jl -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) +
                    np.einsum('n, ...n,ni, nl, jk -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) +
                    np.einsum('n, ...n,nj, nk, il -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) +
                    np.einsum('n, ...n,nj, nl, ik -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, delta) -
                    4.0 * np.einsum('n, ...n, ni, nj, nk, nl -> ...ijkl', self._MPW, beta_T, self._MPN, self._MPN, self._MPN, self._MPN))

        return beta_ijkl
    #-----------------------------------------------------------
    # Integration of the (inelastic) strains for each microplane
    #-----------------------------------------------------------

    def _get_eps_p_Emab(self, eps_Emab, omega_N_Emn, z_N_Emn,
                        alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
                        omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn, eps_aux):

        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emnar_2(eps_Emab)

        # plastic normal strains
        omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_n, X_n, Y_n = self.get_normal_law(
            eps_N_Emn, omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, eps_aux)

        # sliding tangential strains
        omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z_T, X_T, Y_T = self.get_tangential_law(
            eps_T_Emna, omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

        delta = np.identity(2)

        # 2-nd order plastic (inelastic) tensor
        eps_p_Emab = (
            np.einsum('n,...n,na,nb->...ab',
                      self._MPW, eps_N_p_Emn, self._MPN, self._MPN) +
            0.5 * (
                np.einsum('n,...nf,na,fb->...ab',
                          self._MPW, eps_T_pi_Emna, self._MPN, delta) +
                np.einsum('n,...nf,nb,fa->...ab', self._MPW,
                          eps_T_pi_Emna, self._MPN, delta)
            )
        )

        return eps_p_Emab

    def get_corr_pred(self, eps_Emab, t_n1, int_var, eps_aux, F):
        '''Evaluation - get the corrector and predictor
        '''

        # Definition internal variables / forces per column:  1) damage N, 2)iso N, 3)kin N, 4)consolidation N, 5) eps p N,
        # 6) sigma N, 7) iso F N, 8) kin F N, 9) energy release N, 10) damage T, 11) iso T, 12-13) kin T, 14-15) eps p T,
        # 16-17) sigma T, 18) iso F T, 19-20) kin F T, 21) energy release T

        # Corrector predictor computation.

        #------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        #------------------------------------------------------------------


        eps_N_Emn = self._get_e_N_Emn_2(eps_Emab)
        eps_T_Emna = self._get_e_T_Emnar_2(eps_Emab)

        omega_N_Emn = int_var[:, 0]
        z_N_Emn = int_var[:, 1]
        alpha_N_Emn = int_var[:, 2]
        r_N_Emn = int_var[:, 3]
        eps_N_p_Emn = int_var[:, 4]
        sigma_N_Emn = int_var[:, 5]

        omega_T_Emn = int_var[:, 9]
        z_T_Emn = int_var[:, 10]
        alpha_T_Emna = int_var[:, 11:13]
        eps_T_pi_Emna = int_var[:, 13:15]


        beta_Emabcd = self._get_beta_Emabcd_2(
            eps_Emab, omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, omega_T_Emn, z_T_Emn,
                           alpha_T_Emna, eps_T_pi_Emna, eps_aux
        )

        #------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        #------------------------------------------------------------------

        D_Emabcd = np.einsum(
            '...ijab, abef, ...cdef->...ijcd', beta_Emabcd, self.D_abef, beta_Emabcd)

        #----------------------------------------------------------------------
        # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
        #----------------------------------------------------------------------
        # plastic strain tensor
        eps_p_Emab = self._get_eps_p_Emab(
            eps_Emab, omega_N_Emn, z_N_Emn,
            alpha_N_Emn, r_N_Emn, eps_N_p_Emn,
            omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn, eps_aux)

        # elastic strain tensor
        eps_e_Emab = eps_Emab - eps_p_Emab

        # calculation of the stress tensor
        sig_Emab = np.einsum('...abcd,...cd->...ab', D_Emabcd, eps_e_Emab)

        return D_Emabcd, sig_Emab, eps_p_Emab


class MATS2DMplCSDEEQ(MATSXDMplCSDEEQ):
    """Two dimensional version of the MS1 model
    """

    n_mp = tr.Constant(360)
    '''Number of microplanes
    '''

    #-----------------------------------------------
    # get the normal vectors of the microplanes
    #-----------------------------------------------
    _MPN = tr.Property(depends_on='n_mp')

    @tr.cached_property
    def _get__MPN(self):
        # microplane normals:
        alpha_list = np.linspace(0, 2 * np.pi, self.n_mp)

        MPN = np.array([[np.cos(alpha), np.sin(alpha)]
                        for alpha in alpha_list])

        return MPN

    #-------------------------------------
    # get the weights of the microplanes
    #-------------------------------------
    _MPW = tr.Property(depends_on='n_mp')

    @tr.cached_property
    def _get__MPW(self):
        MPW = np.ones(self.n_mp) / self.n_mp * 2

        return MPW

    #--------------------------------------------------------
    # return the state variables (Damage , inelastic strains)
    #--------------------------------------------------------
    def _get_state_variables(self, eps_Emab,
                             int_var, eps_aux):

        e_N_arr = self._get_e_N_Emn_2(eps_Emab)
        e_T_vct_arr = self._get_e_T_Emnar_2(eps_Emab)

        omega_N_Emn = int_var[:, 0]
        z_N_Emn = int_var[:, 1]
        alpha_N_Emn = int_var[:, 2]
        r_N_Emn = int_var[:, 3]
        eps_N_p_Emn = int_var[:, 4]
        sigma_N_Emn = int_var[:, 5]

        omega_T_Emn = int_var[:, 9]
        z_T_Emn = int_var[:, 10]
        alpha_T_Emna = int_var[:, 11:13]
        eps_T_pi_Emna = int_var[:, 13:15]

        omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_n, X_n, Y_n = self.get_normal_law(e_N_arr,  omega_N_Emn, z_N_Emn,
                                                                                                        alpha_N_Emn, r_N_Emn, eps_N_p_Emn, eps_aux)

        omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z_T, X_T, Y_T = self.get_tangential_law(e_T_vct_arr, omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_N_Emn)

        # Definition internal variables / forces per column:  1) damage N, 2)iso N, 3)kin N, 4)consolidation N, 5) eps p N,
        # 6) sigma N, 7) iso F N, 8) kin F N, 9) energy release N, 10) damage T, 11) iso T, 12-13) kin T, 14-15) eps p T,
        # 16-17) sigma T, 18) iso F T, 19-20) kin F T, 21) energy release T

        int_var[:, 0] = omega_N_Emn
        int_var[:, 1] = z_N_Emn
        int_var[:, 2] = alpha_N_Emn
        int_var[:, 3] = r_N_Emn
        int_var[:, 4] = eps_N_p_Emn
        int_var[:, 5] = sigma_N_Emn
        int_var[:, 6] = Z_n
        int_var[:, 7] = X_n
        int_var[:, 8] = Y_n

        int_var[:, 9] = omega_T_Emn
        int_var[:, 10] = z_T_Emn
        int_var[:, 11:13] = alpha_T_Emna
        int_var[:, 13:15] = eps_T_pi_Emna
        int_var[:, 15:17] = sigma_T_Emna
        int_var[:, 17] = Z_T
        int_var[:, 18:20] = X_T
        int_var[:, 20] = Y_T


        return int_var
