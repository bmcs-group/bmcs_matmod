import os

import matplotlib

from bmcs_matmod.ms1.vmats3D_mpl_csd_eeq import MATS3DMplCSDEEQ

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


DELTA = np.identity(3)

EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1


DD = np.hstack([DELTA, np.zeros_like(DELTA)])
EEPS = np.hstack([np.zeros_like(EPS), EPS])

GAMMA = np.einsum(
    'ik,jk->kij', DD, DD
) + np.einsum(
    'ikj->kij', np.fabs(EEPS)
)


def get_eps_ab(eps_O): return np.einsum(
    'Oab,...O->...ab', GAMMA, eps_O
)[np.newaxis, ...]


GAMMA_inv = np.einsum(
    'aO,bO->Oab', DD, DD
) + 0.5 * np.einsum(
    'aOb->Oab', np.fabs(EEPS)
)


def get_sig_O(sig_ab): return np.einsum(
    'Oab,...ab->...O', GAMMA_inv, sig_ab
)[0, ...]


GG = np.einsum(
    'Oab,Pcd->OPabcd', GAMMA_inv, GAMMA_inv
)


def get_K_OP(D_abcd):
    return np.einsum(
        'OPabcd,abcd->OP', GG, D_abcd
    )

# The above operators provide the three mappings
# map the primary variable to from vector to field
# map the residuum field to evctor (assembly operator)
# map the gradient of the residuum field to system matrix
MPW = np.array([.0160714276, .0160714276, .0160714276, .0160714276, .0204744730,
                     .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                     .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                     .0204744730, .0158350505, .0158350505, .0158350505, .0158350505,
                     .0158350505, .0158350505, .0158350505, .0158350505, .0158350505,
                     .0158350505, .0158350505, .0158350505]) * 6.0

def get_UF_t(F, n_t, load, S_max1, S_max2, S_min1, n_mp, loading_scenario):

    int_var = np.zeros((n_mp, 25))
    int_var_aux = np.zeros((n_mp, 25))
    dissip = np.zeros((n_mp, 8))
    # save = np.concatenate((int_var, dissip), axis=1)
    # df = pd.DataFrame(save)
    # df.to_hdf(path, 'first', mode='w', format='table')
    D = np.zeros((3, 3, 3, 3))
    D = D[np.newaxis, :, :, :, :]

    # total number of DOFs
    n_O = 6
    # Global vectors
    F_ext = np.zeros((n_O,), np.float_)
    F_O = np.zeros((n_O,), np.float_)
    F_O_int = np.zeros((n_O,), np.float_)
    U_k_O = np.zeros((n_O,), dtype=np.float_)
    U_P = np.zeros((n_O,), np.float_)
    eps_aux = get_eps_ab(U_k_O)
    dissip_energy = np.zeros(2, np.float_)
    # Setup the system matrix with displacement constraints
    # Time stepping parameters
    t_aux, t_n1, t_max, t_step = 0, 0, len(F), 1 / n_t
    # Iteration parameters
    k_max, R_acc = 1000, 1e-3
    # Record solutions
    U_t_list, F_t_list, F_t_int_list, U_P_list, dissip_energy_list = [np.copy(U_k_O)], [np.copy(F_O)], [np.copy(F_O_int)], [np.copy(U_P)], [np.copy(dissip_energy)]

    # Load increment loop
    while t_n1 <= t_max - 1:

        F_ext[0] = F[t_n1]
        F_ext[1] = 0. * F[t_n1]
        F_ext[2] = 0. * F[t_n1]

        k = 0
        # Equilibrium iteration loop
        while k < k_max:
            # Transform the primary vector to field
            eps_ab = get_eps_ab(U_k_O).reshape(3, 3)
            # Stress and material stiffness

            D_abcd, sig_ab, eps_p_Emab, sig_ab_int = m.get_corr_pred(
                eps_ab, 1, int_var, eps_aux, F_ext
            )
            # Internal force
            F_O = get_sig_O(sig_ab.reshape(1,3,3)).reshape(6,)
            F_O_int = get_sig_O(sig_ab_int.reshape(1, 3, 3)).reshape(6, )
            # Residuum
            R_O = F_ext - F_O
            # System matrix
            K_OP = get_K_OP(D_abcd)
            # Convergence criterion
            R_norm = np.linalg.norm(R_O)
            delta_U_O = np.linalg.solve(K_OP, R_O)
            U_k_O += delta_U_O
            if R_norm < R_acc:
                # Convergence reached
                break
            # Next iteration
            k += 1

        else:
            print('no convergence')

            break

        # Update states variables after convergence
        int_var = m._x_get_state_variables(eps_ab, int_var, eps_aux)

        # Definition internal variables / forces per column:  1) damage N, 2)iso N, 3)kin N, 4) consolidation N, 5) eps p N,
        # 6) sigma N, 7) iso F N, 8) kin F N, 9) energy release N, 10) damage T, 11) iso T, 12-14) kin T, 15-17) eps p T,
        # 18-20) sigma T, 21) iso F T, 22-24) kin F T, 25) energy release T

        # Definition dissipation components per column: 1) damage N, 2) damage T, 3) eps p N, 4) eps p T, 5) iso N
        # 6) iso T, 7) kin N, 8) kin T

        omega_N_Emn_dot = int_var[:, 0] - int_var_aux[:, 0]
        z_N_Emn_dot = int_var[:, 1] - int_var_aux[:, 1]
        alpha_N_Emn_dot = int_var[:, 2] - int_var_aux[:, 2]
        eps_N_p_Emn_dot = int_var[:, 4] - int_var_aux[:, 4]

        sigma_N_Emn = int_var[:, 5]
        Z_n = int_var[:, 6]
        X_n = int_var[:, 7]
        Y_n = int_var[:, 8]

        omega_T_Emn_dot = int_var[:, 9] - int_var_aux[:, 9]
        z_T_Emn_dot = int_var[:, 10] - int_var_aux[:, 10]
        alpha_T_Emna_dot = int_var[:, 11:14] - int_var_aux[:, 11:14]
        eps_T_pi_Emna_dot = int_var[:, 14:17] - int_var_aux[:, 14:17]

        sigma_T_Emna = int_var[:, 17:20]
        Z_T = int_var[:, 20]
        X_T = int_var[:, 21:24]
        Y_T = int_var[:, 24]

        iso_free_N = np.einsum('...,...->...', Z_n, z_N_Emn_dot)
        kin_free_N = np.einsum('...,...->...', X_n, alpha_N_Emn_dot)
        plast_work_N = np.einsum('...,...->...', sigma_N_Emn, eps_N_p_Emn_dot)
        plast_dissip_N = plast_work_N - iso_free_N - kin_free_N


        iso_free_T = np.einsum('...,...->...', Z_T, z_T_Emn_dot)
        kin_free_T = np.einsum('...n,...n->...', X_T, alpha_T_Emna_dot)
        plast_work_T = np.einsum('...n,...n->...', sigma_T_Emna, eps_T_pi_Emna_dot)
        plast_dissip_T = plast_work_T - iso_free_T - kin_free_T
        plast_diss = np.einsum('...n,...n->...', MPW,
                                            plast_dissip_N + plast_dissip_T)

        dissip_energy[0] += plast_diss

        damage_diss_N = np.einsum('...,...->...', Y_n, omega_N_Emn_dot)
        damage_diss_T = np.einsum('...,...->...', Y_T, omega_T_Emn_dot)
        damage_diss = np.einsum('...n,...n->...', MPW, damage_diss_N + damage_diss_T)

        dissip_energy[1] += damage_diss

        if plast_diss < -1e-5:
            print('second law violation - plast')

        if damage_diss < -1e-5:
            print('second law violation - damage')

        int_var_aux = int_var * 1

        if loading_scenario == 'constant':
            # Saving data just at min and max levels
            if F[t_n1] == 0 or F[t_n1] == S_max1 * load or F[t_n1] == S_min1 * load:
                save = np.concatenate((int_var, dissip), axis=1)

                df = pd.DataFrame(save)
                df.to_hdf(path, 'middle' + str(t_aux), append=True)

                U_t_list.append(np.copy(U_k_O))
                F_t_list.append(F_O)
                F_t_int_list.append(F_O_int)
                if F[t_n1] == S_max1 * load:
                    dissip_energy_list.append(dissip_energy)
                    dissip_energy = np.zeros(2, np.float_)
                eps_aux = get_eps_ab(U_k_O)
                t_aux += 1

        if loading_scenario == 'order':
            # Saving data just at min and max levels
            if F[t_n1] == 0 or F[t_n1] == S_max1 * load or F[t_n1] == S_max2 * load or F[t_n1] == S_min1 * load:
                save = np.concatenate((int_var, dissip), axis=1)

                df = pd.DataFrame(save)
                df.to_hdf(path, 'middle' + np.str(t_aux), append=True)

                U_t_list.append(np.copy(U_k_O))
                F_t_list.append(F_O)
                F_t_int_list.append(F_O_int)
                eps_aux = get_eps_ab(U_k_O)
                t_aux += 1
        if loading_scenario == 'increasing':
            # Saving data all points

            save = np.concatenate((int_var, dissip), axis=1)

            df = pd.DataFrame(save)
            df.to_hdf(path, 'middle' + np.str(t_aux), append=True)

            U_t_list.append(np.copy(U_k_O))
            F_t_list.append(F_O)
            F_t_int_list.append(F_O_int)
            eps_aux = get_eps_ab(U_k_O)
            t_aux += 1
        t_n1 += 1

    U_t, F_t, dissip_energy = np.array(U_t_list), np.array(F_t_list), np.array(dissip_energy_list)
    return U_t, F_t, dissip_energy, t_n1 / t_max, t_aux

def get_int_var(path, size, n_mp):  # unpacks saved data

    S = np.zeros((len(F), n_mp, 33))

    S[0] = np.array(pd.read_hdf(path, 'first'))

    for i in range(1, size):
        S[i] = np.array(pd.read_hdf(path, 'middle' + np.str(i - 1)))

    omega_N_Emn = S[:, :, 0]
    z_N_Emn = S[:, :, 1]
    alpha_N_Emn = S[:, :, 2]
    r_N_Emn = S[:, :, 3]
    eps_N_p_Emn = S[:, :, 4]
    sigma_N_Emn = S[:, :, 5]
    Z_N_Emn = S[:, :, 6]
    X_N_Emn = S[:, :, 7]
    Y_N_Emn = S[:, :, 8]

    omega_T_Emn = S[:, :, 9]
    z_T_Emn = S[:, :, 10]
    alpha_T_Emna = S[:, :, 11:14]
    eps_T_pi_Emna = S[:, :, 14:17]
    sigma_T_Emna = S[:, :, 17:20]
    Z_T_pi_Emn = S[:, :, 20]
    X_T_pi_Emna = S[:, :, 21:24]
    Y_T_pi_Emn = S[:, :, 24]

    Disip_omena_N_Emn = S[:, :, 25]
    Disip_omena_T_Emn = S[:, :, 26]
    Disip_eps_p_N_Emn = S[:, :, 27]
    Disip_eps_p_T_Emn = S[:, :, 28]
    Disip_iso_N_Emn = S[:, :, 29]
    Disip_iso_T_Emn = S[:, :, 30]
    Disip_kin_N_Emn = S[:, :, 31]
    Disip_kin_T_Emn = S[:, :, 32]

    return omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_N_Emn, X_N_Emn, Y_N_Emn, \
           omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z_T_pi_Emn, X_T_pi_Emna, Y_T_pi_Emn, \
           Disip_omena_N_Emn, Disip_omena_T_Emn, Disip_eps_p_N_Emn, Disip_eps_p_T_Emn, Disip_iso_N_Emn, \
           Disip_iso_T_Emn, Disip_kin_N_Emn, Disip_kin_T_Emn


concrete_type= 1        # 0:C40MA, 1:C80MA, 2:120MA, 3:Tensile, 4:Compressive, 5:Biaxial

Concrete_Type_string = ['C40MA', 'C80MA','C120MA', 'Tensile', 'Compressive', 'Biaxial']

loading_scenario = 'constant'   # constant, order, increasing

M_plot = 1  # Plot microplanes polar graphs. 1: yes, 0: no

t_steps_cycle = 100
n_mp = 28

S_max1 = 0.85          # maximum loading level
S_min1 = 0.20           # minimum loading level
n_cycles1 = 2000        # number of applied cycles

# For sequence order effect

eta1 = 0.15             # fatigue life fraction first level
cycles1 = 20           # fatigue life first level

S_max2 = 0.85            # maximum loading level second level
cycles2 = 221         # fatigue life second level
n_cycles2 = int(1e3 - np.floor(eta1*cycles1)) # number of applied cycles second level

# For increasing loading levels

cycles10 = 10           # number of cycles per loading level
S_min10 = 0.1


# Path saving data

home_dir = os.path.expanduser('~')

if not os.path.exists('Data Processing'):
    os.makedirs('Data Processing')

path = os.path.join(
   home_dir, 'Data Processing/' + '3D' + Concrete_Type_string[concrete_type] + loading_scenario + str(S_max1) + str(eta1) + '.hdf5')

# FINAL LOADINGS

load_options = [-60.58945931264715, -93.73515724992052]

load = load_options[concrete_type]

# LOADING SCENARIOS

first_load = np.concatenate((np.linspace(0, load * S_max1, t_steps_cycle), np.linspace(
    load * S_max1, load * S_min1, t_steps_cycle)[1:]))

if loading_scenario == 'constant':
    first_load = np.concatenate((np.linspace(0, load * S_max1, t_steps_cycle), np.linspace(
        load * S_max1, load * S_min1, t_steps_cycle)[1:]))

    cycle1 = np.concatenate(
        (np.linspace(load * S_min1, load * S_max1, t_steps_cycle)[1:], np.linspace(load * S_max1, load * S_min1, t_steps_cycle)[
                                                              1:]))
    cycle1 = np.tile(cycle1, n_cycles1 - 1)

    sin_load = np.concatenate((first_load, cycle1))


if loading_scenario == 'order':

    first_load = np.concatenate((np.linspace(0, load * S_max1, t_steps_cycle), np.linspace(
        load * S_max1, load * S_min1, t_steps_cycle)[1:]))

    cycle1 = np.concatenate(
        (np.linspace(load * S_min1, load * S_max1, t_steps_cycle)[1:], np.linspace(load * S_max1, load * S_min1, t_steps_cycle)[
                                                              1:]))
    cycle1 = np.tile(cycle1, np.int(np.floor(eta1*cycles1)) - 1)

    change_order = np.concatenate((np.linspace(load * S_min1, load * S_max2, 632)[1:], np.linspace(load * S_max2, load * S_min1, 632)[
                                                                              1:]))

    cycle2 = np.concatenate(
        (np.linspace(load * S_min1, load * S_max2, t_steps_cycle)[1:], np.linspace(load * S_max2, load * S_min1, t_steps_cycle)[
                                                              1:]))
    cycle2 = np.tile(cycle2, n_cycles2)

    sin_load = np.concatenate((first_load, cycle1, change_order, cycle2))

if loading_scenario == 'increasing':

    first_load = np.concatenate((np.linspace(0, load * 0.5, t_steps_cycle), np.linspace(
        load * 0.5, load * S_min10, t_steps_cycle)[1:]))

    cycle1 = np.concatenate(
        (np.linspace(load * S_min10, load * 0.5, t_steps_cycle)[1:], np.linspace(load * 0.5, load * S_min10, t_steps_cycle)[
                                                              1:]))
    cycle1 = np.tile(cycle1, cycles10 - 1)

    cycle2 = np.concatenate(
        (np.linspace(load * S_min10, load * 0.55, t_steps_cycle)[1:], np.linspace(load * 0.55, load * S_min10, t_steps_cycle)[
                                                              1:]))
    cycle2 = np.tile(cycle2, cycles10)

    cycle3 = np.concatenate(
        (np.linspace(load * S_min10, load * 0.6, t_steps_cycle)[1:],
         np.linspace(load * 0.6, load * S_min10, t_steps_cycle)[
         1:]))
    cycle3 = np.tile(cycle3, cycles10)

    cycle4 = np.concatenate(
        (np.linspace(load * S_min10, load * 0.65, t_steps_cycle)[1:],
         np.linspace(load * 0.65, load * S_min10, t_steps_cycle)[
         1:]))
    cycle4 = np.tile(cycle4, cycles10)

    cycle5 = np.concatenate(
        (np.linspace(load * S_min10, load * 0.7, t_steps_cycle)[1:],
         np.linspace(load * 0.7, load * S_min10, t_steps_cycle)[
         1:]))
    cycle5 = np.tile(cycle5, cycles10)

    cycle6 = np.concatenate(
        (np.linspace(load * S_min10, load * 0.75, t_steps_cycle)[1:],
         np.linspace(load * 0.75, load * S_min10, t_steps_cycle)[
         1:]))
    cycle6 = np.tile(cycle6, cycles10)

    cycle7 = np.concatenate(
        (np.linspace(load * S_min10, load * 0.8, t_steps_cycle)[1:],
         np.linspace(load * 0.8, load * S_min10, t_steps_cycle)[
         1:]))
    cycle7 = np.tile(cycle7, cycles10)

    cycle8 = np.concatenate(
        (np.linspace(load * S_min10, load * 0.85, t_steps_cycle)[1:],
         np.linspace(load * 0.85, load * S_min10, t_steps_cycle)[
         1:]))
    cycle8 = np.tile(cycle8, cycles10)

    cycle9 = np.concatenate(
        (np.linspace(load * S_min10, load * 0.9, t_steps_cycle)[1:],
         np.linspace(load * 0.9, load * S_min10, t_steps_cycle)[
         1:]))
    cycle9 = np.tile(cycle9, cycles10)

    cycle10 = np.concatenate(
        (np.linspace(load * S_min10, load * 0.95, t_steps_cycle)[1:],
         np.linspace(load * 0.95, load * S_min10, t_steps_cycle)[
         1:]))
    cycle10 = np.tile(cycle10, cycles10)

    cycle11 = np.concatenate(
        (np.linspace(load * S_min10, load * 1.0, t_steps_cycle)[1:],
         np.linspace(load * 1.0, load * S_min10, t_steps_cycle)[
         1:]))
    cycle11 = np.tile(cycle11, cycles10)


    sin_load = np.concatenate((first_load, cycle1, cycle2, cycle3, cycle4, cycle5, cycle6, cycle7, cycle8, cycle9,cycle11))



t_steps = len(sin_load)
T = 1 / n_cycles1
#t = np.linspace(0, 1, len(sin_load))

m = MATS3DMplCSDEEQ(concrete_type)

start = time.time()



U, F, dissip_energy, cyc, number_cyc = get_UF_t(
    sin_load, t_steps, load, S_max1, S_max2, S_min1, n_mp, loading_scenario)

end = time.time()
print(end - start, 'seconds')


# [omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_N_Emn, X_N_Emn, Y_N_Emn, omega_T_Emn, z_T_Emn,
#  alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z_T_pi_Emn, X_T_pi_Emna, Y_T_pi_Emn, Disip_omena_N_Emn, Disip_omena_T_Emn,
#  Disip_eps_p_N_Emn, Disip_eps_p_T_Emn, Disip_iso_N_Emn, Disip_iso_T_Emn, Disip_kin_N_Emn, Disip_kin_T_Emn] \
#     = get_int_var(path, len(F), n_mp)

font = {'family': 'DejaVu Sans',
        'size': 18}

matplotlib.rc('font', **font)

print(np.max(np.abs(F[:, 0])), 'sigma1')
print(np.max(np.abs(F[:, 1])), 'sigma2')

# Fig 1, stress-strain curve

f, (ax2) = plt.subplots(1, 1, figsize=(5, 4))

ax2.plot(np.abs(U[:, 0]), np.abs(F[:, 0]), linewidth=2.5)
ax2.set_xlabel(r'$|\varepsilon_{11}$|', fontsize=25)
ax2.set_ylabel(r'$|\sigma_{11}$| [-]', fontsize=25)
ax2.set_title('stress-strain' + ',' + 'N =' + str(cyc*n_cycles1) + str(S_max1) + 'Smin=' + str(S_min1))
plt.show()

if loading_scenario == 'constant':

    # Fig 2, creep fatigue curve

    f, (ax) = plt.subplots(1, 1, figsize=(5, 4))

    ax.plot((np.arange(len(U[2::2, 0])) + 1) / len(U[2::2, 0]),
            np.abs(U[2::2, 0]), linewidth=2.5)

    ax.set_xlabel(r'$N / N_f $|', fontsize=25)
    ax.set_ylabel(r'$|\varepsilon_{11}^{max}$|', fontsize=25)
    ax.set_title('creep fatigue Smax=' + str(S_max1) + 'Smin=' + str(S_min1))
    plt.show()

    f, (ax) = plt.subplots(1, 1, figsize=(5, 4))

    ax.plot((np.arange(len(dissip_energy))) / len(dissip_energy),
            dissip_energy, linewidth=2.5)

    ax.set_xlabel(r'$N / N_f $|', fontsize=25)
    ax.set_ylabel(r'$|\varepsilon_{11}^{max}$|', fontsize=25)
    ax.set_title('creep fatigue Smax=' + str(S_max1) + 'Smin=' + str(S_min1))
    plt.show()

    f, (ax) = plt.subplots(1, 1, figsize=(5, 4))

    ax.plot((np.arange(len(dissip_energy))) / len(dissip_energy),
            np.cumsum(dissip_energy[:,0]), linewidth=2.5)
    ax.plot((np.arange(len(dissip_energy))) / len(dissip_energy),
            np.cumsum(dissip_energy[:,1]), linewidth=2.5)

    ax.set_xlabel(r'$N / N_f $|', fontsize=25)
    ax.set_ylabel(r'$|\varepsilon_{11}^{max}$|', fontsize=25)
    ax.set_title('creep fatigue Smax=' + str(S_max1) + 'Smin=' + str(S_min1))
    plt.show()

if loading_scenario == 'order':

    # Fig 2, Creep Fatigue Curve

    f, (ax) = plt.subplots(1, 1, figsize=(5, 4))

    X_axis1 = np.array(np.arange(eta1 * cycles1) + 1)[:] / cycles1
    #X_axis1 = np.concatenate((np.array([0]), X_axis1))
    Y_axis1 = np.abs(U[3:np.int(2 * eta1 * cycles1) + 2:2, 0])
    # Y_axis1 = np.concatenate((np.array([Y_axis1[0]]), Y_axis1))


    X_axis2 = np.array((np.arange(len(U[2::2, 0]) - (eta1 * cycles1)) + 1) / (cycles2) + eta1)
    X_axis2 = np.concatenate((np.array([X_axis1[-1]]), X_axis2))
    Y_axis2 = np.abs(U[np.int(2 * eta1 * cycles1) + 1::2, 0])
    Y_axis2 = np.concatenate((np.array([Y_axis2[0]]), Y_axis2))

    ax.plot(X_axis1, Y_axis1, 'k', linewidth=2.5)
    ax.plot(X_axis2, Y_axis2, 'k', linewidth=2.5)
    ax.plot([X_axis1[-1], X_axis2[0]], [Y_axis1[-1], Y_axis2[0]], 'k', linewidth=2.5)

    ax.set_ylim(0.002, 0.0045)
    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel('N/Nf', fontsize=25)
    ax.set_ylabel('strain', fontsize=25)
    plt.title('creep fatigue Smax1=' + str(S_max1) + 'Smax2=' + str(S_max2) + 'Smin=' + str(S_min1))
    plt.show()
