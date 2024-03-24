import matplotlib.pylab as plt

class TEVPDIfcPlot(object):

    @staticmethod
    def tevpd_ifc_param_study_plot(param_name, response_values):
        '''
        thermo-elasto-visco-plastic damage model
        '''
        fig, ((ax1,  ax2), (ax3,  ax4)) = plt.subplots(2,2, tight_layout=True, figsize=(7,7))
        fig.canvas.header_visible = False
        ax1_twin = ax1.twinx()

        for (param, rv), color in zip(response_values.items(), ['black', 'red', 'green']):
            t_t, u_ta, Eps_t, Sig_t, iter_t = rv
            u_p_Tx_t, u_p_Ty_t, u_p_N_t, z_T_t, alpha_Tx_t, alpha_Ty_t, omega_T_t, omega_N_t, T_t = Eps_t.T
            sig_Tx_t, sig_Ty_t, sig_N_t, Z_T_t, X_Tx_t, X_Ty_t, Y_T_t, Y_N_t, S_E_t = Sig_t.T
            ax1.plot(t_t, u_ta[:, 0], color=color, linewidth=1, label="{} = {}".format(param_name, param))  # Loading scenario
            ax1_twin.plot(t_t, sig_Tx_t, linestyle='dashed', color=color, linewidth=1)
            ax2.plot(u_ta[:,0], sig_Tx_t, color=color, linewidth=1, label="{} = {}".format(param_name, param))    # Stress-slip relation
            ax3.plot(t_t, T_t, color=color, linewidth=1, label="{} = {}".format(param_name, param))      # Evolution of temp
            ax4.plot(u_ta[:, 0], omega_T_t, color=color, linewidth=1, label="{} = {}".format(param_name, param))         # Damage evolution

        ax1.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax1.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        ax1.set_title('loading scenario')
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('slip [mm]')
        ax1.legend()

        ax2.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax2.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        ax2.set_title('stress-slip')
        ax2.set_xlabel('slip [mm]')
        ax2.set_ylabel('stress [MPa]')
        ax2.legend()

        ax3.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax3.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        ax3.set_title('evolution of temperature')
        ax3.set_xlabel('time [sec]')
        ax3.set_ylabel('temperature [$^{\circ}$C]')
        ax3.legend()

        ax4.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax4.axvline(x=0, color='k', linewidth=1, alpha=0.5)
        ax4.set_title('damage evolution')
        ax4.set_xlabel('slip [mm]')
        ax4.set_ylabel('damage [-]')
        ax4.legend()

    @staticmethod
    def tevpd_ifc_plot_Sig_Eps(rv, 
                    ax1, ax11, ax2, ax22, ax3, ax33, ax4, ax44):
        colors = ['blue','red', 'green', 'black', 'magenta' ]
        t_t, u_ta, Eps_t, Sig_t, iter_t = rv
        s_x_t = u_ta[:,0]
        u_Tx_pi_, u_Ty_pi_, u_N_pi_, z_, alpha_Tx_, alpha_Ty_, omega_T_, omega_N_, T_ = Eps_t.T
        tau_x_pi_, tau_y_pi_, sig_pi_, Z_, X_x_, X_y_, Y_T_, Y_N_, S_E_ = Sig_t.T
        n_step = len(u_Tx_pi_)
        ax1.plot(s_x_t, tau_x_pi_, color='black', 
                label='n_steps = %g' % n_step)
        ax1.set_xlabel('$s$'); ax1.set_ylabel(r'$\tau$')
        ax1.legend()
        ax11.plot(s_x_t, iter_arr, '-.')
        ax2.plot(s_x_t, omega_T_, color='red', 
                label='n_steps = %g' % n_step)
        ax2.set_xlabel('$s$'); ax2.set_ylabel(r'$\omega$')
        ax2.plot(s_x_t, omega_N_, color='green', )
        ax22.plot(s_x_t, Y_T_, '-.', color='red', 
                label='n_steps = %g' % n_step)
        ax22.set_ylabel('$Y$')
        ax3.plot(s_x_t, z_, color='green', 
                label='n_steps = %g' % n_step)
        ax3.set_xlabel('$s$'); ax3.set_ylabel(r'$z$')
        ax33.plot(s_x_t, Z_, '-.', color='green')
        ax33.set_ylabel(r'$Z$')
        ax4.plot(s_x_t, alpha_Tx_, color='blue', 
                label='n_steps = %g' % n_step)
        ax4.set_xlabel('$s$'); ax4.set_ylabel(r'$\alpha$')
        ax44.plot(s_x_t, X_x_, '-.', color='blue')
        ax44.set_ylabel(r'$X$')