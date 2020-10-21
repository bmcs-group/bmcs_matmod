# In[39]:

import numpy as np


def get_f_df(s_x_n1, s_y_n1, w_n1, Eps_k, **kw):
    Sig_k = get_Sig(s_x_n1, s_y_n1, w_n1, Eps_k, **kw)[0]
    dSig_dEps_k = get_dSig_dEps(s_x_n1, s_y_n1, w_n1, Eps_k, **kw)
    f_k = np.array([get_f(Eps_k, Sig_k, **kw)])
    df_dSig_k = get_df_dSig(Eps_k, Sig_k, **kw)
    ddf_dEps_k = get_ddf_dEps(Eps_k, Sig_k, **kw)
    df_dEps_k = np.einsum(
        'ik,ji->jk', df_dSig_k, dSig_dEps_k) + ddf_dEps_k
    Phi_k = get_Phi(Eps_k, Sig_k, **kw)
    dEps_dlambda_k = Phi_k
    df_dlambda = np.einsum(
        'ki,kj->ij', df_dEps_k, dEps_dlambda_k)
    df_k = df_dlambda
    return f_k, df_k, Sig_k


# **Evolution equations:** The update of state variables for an updated $\lambda_k$ is performed using this procedure.

# In[40]:


def get_Eps_k1(s_x_n1, s_y_n1, w_n1, Eps_n, lam_k, Eps_k, **kw):
    Sig_k = get_Sig(s_x_n1, s_y_n1, w_n1, Eps_k, **kw)[0]
    Phi_k = get_Phi(Eps_k, Sig_k, **kw)
    Eps_k1 = Eps_n + lam_k * Phi_k[:, 0]
    return Eps_k1


# **Return mapping iteration:** This function represents a user subroutine in a finite element code or in a lattice model. The input is $s_{n+1}$ and the state variables representing the state in the previous solved step $\boldsymbol{\mathcal{E}}_n$. The procedure returns the stresses and state variables of $\boldsymbol{\mathcal{S}}_{n+1}$ and $\boldsymbol{\mathcal{E}}_{n+1}$

# In[41]:


def get_material_model(s_x_n1, s_y_n1, w_n1, Eps_n, k_max, **kw):
    Eps_k = np.copy(Eps_n)
    lam_k = 0
    f_k, df_k, Sig_k = get_f_df(s_x_n1, s_y_n1, w_n1, Eps_k, **kw)
    f_k_norm = np.linalg.norm(f_k)
    f_k_trial = f_k[0]
    k = 0
    while k < k_max:
        if f_k_trial < 0 or f_k_norm < 1e-6:
            return Eps_k, Sig_k, k + 1
        dlam = np.linalg.solve(df_k, -f_k)
        lam_k += dlam
        Eps_k = get_Eps_k1(s_x_n1, s_y_n1, w_n1, Eps_n, lam_k, Eps_k, **kw)
        f_k, df_k, Sig_k = get_f_df(s_x_n1, s_y_n1, w_n1, Eps_k, **kw)
        f_k_norm = np.linalg.norm(f_k)
        k += 1
    else:
        raise ValueError('no convergence')


# **Loop over the time increments** for a single material point. This loop emulates the  finite-element spatial integration or a lattice-assembly algorithm.

# In[42]:


def get_response(s_max=[3, 0, 0], n_steps=10, k_max=20, get_load_fn=lambda t: t, **kw):
    Eps_n1 = np.zeros((len(Eps),), dtype=np.float_)
    Sig_record = []
    Eps_record = []
    iter_record = []
    t_arr = np.linspace(0, 1, n_steps + 1)
    s_x_max, s_y_max, w_max = s_max
    s_x_t = s_x_max * get_load_fn(t_arr) + 1e-9
    s_y_t = s_y_max * get_load_fn(t_arr) + 1e-9
    w_t = w_max * get_load_fn(t_arr) + 1e-9
    for s_x_n1, s_y_n1, w_n1 in zip(s_x_t, s_y_t, w_t):
        Eps_n1, Sig_n1, k = get_material_model(s_x_n1, s_y_n1, w_n1, Eps_n1, k_max, **kw)
        Sig_record.append(Sig_n1)
        Eps_record.append(Eps_n1)
        iter_record.append(k)
    Sig_arr = np.array(Sig_record, dtype=np.float_)
    Eps_arr = np.array(Eps_record, dtype=np.float_)
    iter_arr = np.array(iter_record, dtype=np.int_)
    return t_arr, s_x_t, s_y_t, w_t, Eps_arr, Sig_arr, iter_arr


# In[43]:


material_params = dict(
    E_s=1, gamma_s=5, K_s=5, S_s=0.6, c_s=1, bartau=1,
    E_w=1, S_w=0.6, c_w=1, m=0.01, f_t=1, f_c=-20, f_c0=-10, eta=0.5
)

# In[44]:


t_arr, s_x_t, s_y_t, w_t, Eps_arr, Sigma_arr, iter_arr = get_response(
    s_max=[1.1, 0, 0], n_steps=1, k_max=50, **material_params
)

# In[45]:


iter_arr

# # Support functions
# To run some examples, let us define some infrastructure including a more complex loading history and postprocessing

# ## Loading history
# This implementation uses the symbolic machinery which is not necessary a simpler data point based implementation with `numpy.interp1d` would be better ... later

# In[46]:


t, theta = sp.symbols(r't, \theta')
n_cycles = 5
A = 2
ups = np.array([((theta - 2 * cycle) * A + (1 - A), theta - 2 * cycle <= 1)
                for cycle in range(n_cycles)])
downs = np.array([((1 - (theta - (2 * cycle + 1))) * A + (1 - A), (theta - (2 * cycle + 1)) <= 1)
                  for cycle in range(n_cycles)])
ups[0, 0] = theta
updowns = np.einsum('ijk->jik', np.array([ups, downs])).reshape(-1, 2)
load_fn = sp.Piecewise(*updowns).subs(theta, t * n_cycles)
get_load_fn = sp.lambdify(t, load_fn, 'numpy')
t_arr = np.linspace(0, 1, 600)
plt.plot(t_arr, get_load_fn(t_arr));


# ## Plotting functions
# To simplify postprocessing examples, here are two aggregate plotting functions, one for the state and force variables, the other one for the evaluation of energies

# In[47]:


def plot_Sig_Eps(s_x_t, Sig_arr, Eps_arr, iter_arr,
                 ax1, ax11, ax2, ax22, ax3, ax33, ax4, ax44):
    colors = ['blue', 'red', 'green', 'black', 'magenta']
    s_x_pi_, s_y_pi_, w_pi_, z_, alpha_x_, alpha_y_, omega_s_, omega_w_ = Eps_arr.T
    tau_x_pi_, tau_y_pi_, sig_pi_, Z_, X_x_, X_y_, Y_s_, Y_w_ = Sig_arr.T
    n_step = len(s_x_pi_)
    ax1.plot(s_x_t, tau_x_pi_, color='black',
             label='n_steps = %g' % n_step)
    ax1.set_xlabel('$s$');
    ax1.set_ylabel(r'$\tau$')
    ax1.legend()
    ax11.plot(s_x_t, iter_arr, '-.')
    ax2.plot(s_x_t, omega_s_, color='red',
             label='n_steps = %g' % n_step)
    ax2.set_xlabel('$s$');
    ax2.set_ylabel(r'$\omega$')
    ax2.plot(s_x_t, omega_w_, color='green', )
    #    ax22.plot(s_x_t, Y_s_, '-.', color='red',
    #             label='n_steps = %g' % n_step)
    #    ax22.set_ylabel('$Y$')
    ax3.plot(s_x_t, z_, color='green',
             label='n_steps = %g' % n_step)
    ax3.set_xlabel('$s$');
    ax3.set_ylabel(r'$z$')
    ax33.plot(s_x_t, Z_, '-.', color='green')
    ax33.set_ylabel(r'$Z$')
    ax4.plot(s_x_t, alpha_x_, color='blue',
             label='n_steps = %g' % n_step)
    ax4.set_xlabel('$s$');
    ax4.set_ylabel(r'$\alpha$')
    ax44.plot(s_x_t, X_x_, '-.', color='blue')
    ax44.set_ylabel(r'$X$')


# In[48]:


from scipy.integrate import cumtrapz


def plot_work(ax, t_arr, s_x_t, s_y_t, w_t, Eps_arr, Sig_arr):
    W_arr = (
            cumtrapz(Sig_arr[:, 0], s_x_t, initial=0) +
            cumtrapz(Sig_arr[:, 1], s_y_t, initial=0) +
            cumtrapz(Sig_arr[:, 2], w_t, initial=0)
    )
    U_arr = (
            Sig_arr[:, 0] * (s_x_t - Eps_arr[:, 0]) / 2.0 +
            Sig_arr[:, 1] * (s_y_t - Eps_arr[:, 1]) / 2.0 +
            Sig_arr[:, 2] * (w_t - Eps_arr[:, 2]) / 2.0
    )
    G_arr = W_arr - U_arr
    ax.plot(t_arr, W_arr, lw=2, color='black', label=r'$W$ - Input work')
    ax.plot(t_arr, G_arr, color='red', label=r'$G$ - Plastic work')
    ax.fill_between(t_arr, W_arr, G_arr, color='green', alpha=0.2)
    ax.set_xlabel('$t$');
    ax3.set_ylabel(r'$E$')
    ax.legend()


# In[49]:


def plot_dissipation(ax, t_arr, Eps_arr, Sig_arr):
    colors = ['blue', 'red', 'green', 'black', 'magenta']
    E_i = cumtrapz(Sig_arr, Eps_arr, initial=0, axis=0)
    E_s_x_pi_, E_s_y_pi_, E_w_pi_, E_z_, E_alpha_x_, E_alpha_y_, E_omega_s_, E_omega_w_ = E_i.T
    c = 'brown'
    E_plastic_work = E_s_x_pi_ + E_s_y_pi_ + E_w_pi_
    ax.plot(t_arr, E_plastic_work, '-.', lw=1, color=c)
    c = 'blue'
    E_isotropic_diss = E_z_
    ax.plot(t_arr, E_isotropic_diss, '-.', lw=1, color='black')
    ax.fill_between(t_arr, E_isotropic_diss, 0, color=c, alpha=0.3)
    c = 'blue'
    E_free_energy = E_alpha_x_ + E_alpha_y_
    ax.plot(t_arr, E_free_energy, color='black', lw=1)
    ax.fill_between(t_arr, E_free_energy, E_isotropic_diss,
                    color=c, alpha=0.2);
    E_plastic_diss = E_plastic_work - E_free_energy
    ax.plot(t_arr, E_plastic_diss, color='black', lw=1)
    ax.fill_between(t_arr, E_plastic_diss, 0,
                    color='orange', alpha=0.3);
    c = 'magenta'
    E_damage_diss = E_omega_s_ + E_omega_w_
    ax.plot(t_arr, E_plastic_diss + E_damage_diss, color=c, lw=1)
    ax.fill_between(t_arr, E_plastic_diss + E_damage_diss,
                    E_plastic_work,
                    color=c, alpha=0.2);
    ax.fill_between(t_arr, E_free_energy + E_plastic_diss + E_damage_diss,
                    E_plastic_diss + E_damage_diss,
                    color='yellow', alpha=0.3);


# In[59]:


def plot_dissipation(ax, t_arr, Eps_arr, Sig_arr, ax2=None):
    colors = ['blue', 'red', 'green', 'black', 'magenta']
    E_i = cumtrapz(Sig_arr, Eps_arr, initial=0, axis=0)
    E_s_x_pi_, E_s_y_pi_, E_w_pi_, E_z_, E_alpha_x_, E_alpha_y_, E_omega_s_, E_omega_w_ = E_i.T

    E_plastic_work = E_s_x_pi_ + E_s_y_pi_ + E_w_pi_
    E_isotropic_diss = E_z_
    E_free_energy = E_alpha_x_ + E_alpha_y_
    E_my_plastic_diss = E_plastic_work - E_free_energy - E_isotropic_diss
    E_damage_diss = E_omega_s_ + E_omega_w_

    if ax2:
        ax2.plot(t_arr, E_damage_diss, color='gray', lw=1, label='E damage diss')
        ax2.plot(t_arr, E_my_plastic_diss, color='magenta', lw=1, label='E plast diss')
        ax2.plot(t_arr, E_isotropic_diss, color='red', lw=1, label='E iso')
        ax2.plot(t_arr, E_free_energy, color='blue', lw=1, label='E free')
        ax2.legend()

    E_level = 0
    ax.plot(t_arr, E_damage_diss + E_level, color='black', lw=1)
    ax.fill_between(t_arr, E_damage_diss + E_level, E_level, color='gray', alpha=0.3);
    E_level = E_damage_diss
    ax.plot(t_arr, E_my_plastic_diss + E_level, '-.', lw=1, color='magenta')
    ax.fill_between(t_arr, E_my_plastic_diss + E_level, E_level, color='magenta', alpha=0.3)
    E_level += E_my_plastic_diss
    ax.plot(t_arr, E_isotropic_diss + E_level, '-.', lw=1, color='black')
    ax.fill_between(t_arr, E_isotropic_diss + E_level, E_level, color='red', alpha=0.3)
    E_level += E_isotropic_diss
    ax.plot(t_arr, E_free_energy + E_level, color='black', lw=1)
    ax.fill_between(t_arr, E_free_energy + E_level, E_level, color='blue', alpha=0.2);


# # Examples

# ## Monotonic load
# Let's first run the example with different size of the time step to see if there is any difference

# In[60]:


material_params = dict(
    E_s=1, gamma_s=5, K_s=5, S_s=0.6, c_s=1, bartau=1,
    E_w=1, S_w=0.6, c_w=1, m=0.01, f_t=1, f_c=-20, f_c0=-10, eta=1
)
material_params = dict(
    E_s=1, gamma_s=0, K_s=-0.1, S_s=10000, c_s=1, bartau=1,
    E_w=1, S_w=1000, c_w=1, m=0.01, f_t=1, f_c=-20, f_c0=-10, eta=1
)

# In[61]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5), tight_layout=True)
ax11 = ax1.twinx()
ax22 = ax2.twinx()
ax33 = ax3.twinx()
ax44 = ax4.twinx()
axes = ax1, ax11, ax2, ax22, ax3, ax33, ax4, ax44
for n_steps in [200]:  # 20, 40, 200, 2000]:
    t_arr, s_x_t, s_y_t, w_t, Eps_arr, Sig_arr, iter_arr = get_response(
        s_max=[6, 0, 0], n_steps=n_steps, k_max=100, **material_params
    )
    # plot_Sig_Eps(s_x_t, Sig_arr, Eps_arr, iter_arr, *axes)
    s_x_pi_, s_y_pi_, w_pi_, z_, alpha_x_, alpha_y_, omega_s_, omega_w_ = Eps_arr.T
    tau_x_pi_, tau_y_pi_, sig_pi_, Z_, X_x_, X_y_, Y_s_, Y_w_ = Sig_arr.T
    ax1.plot(w_t, sig_pi_, color='green')
    ax11.plot(s_x_t, tau_x_pi_, color='red')
    ax2.plot(w_t, omega_w_, color='green')
    ax22.plot(w_t, omega_s_, color='red')

# In[63]:


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plot_work(ax, t_arr, s_x_t, s_y_t, w_t, Eps_arr, Sig_arr)
plot_dissipation(ax, t_arr, Eps_arr, Sig_arr, ax2)

# ## Cyclic loading

# In[64]:


material_params = dict(
    E_s=1, gamma_s=-0.1, K_s=0, S_s=10000, c_s=1, bartau=1,
    E_w=1, S_w=100000, c_w=1, m=0.00001, f_t=100, f_c=-20000, f_c0=-10000, eta=0
)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6), tight_layout=True)
ax11 = ax1.twinx()
ax22 = ax2.twinx()
ax33 = ax3.twinx()
ax44 = ax4.twinx()
axes = ax1, ax11, ax2, ax22, ax3, ax33, ax4, ax44
t_arr, s_x_t, s_y_t, w_t, Eps_arr, Sig_arr, iter_arr = get_response(
    s_max=[10, 0, 0], n_steps=5000, k_max=100, get_load_fn=get_load_fn, **material_params
)
plot_Sig_Eps(s_x_t, Sig_arr, Eps_arr, iter_arr, ax1, ax11, ax2, ax22, ax3, ax33, ax4, ax44)

# In[65]:


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plot_work(ax, t_arr, s_x_t, s_y_t, w_t, Eps_arr, Sig_arr)
plot_dissipation(ax, t_arr, Eps_arr, Sig_arr, ax2)


# # Interactive application

# In[67]:


def init():
    global Eps_record, Sig_record, iter_record
    global t_arr, s_x_t, s_y_t, w_t, s_x_0, s_y_0, w_0, t0, Eps_n1
    s_x_0, s_y_0, w_0 = 0, 0, 0
    t0 = 0
    Sig_record = []
    Eps_record = []
    iter_record = []
    t_arr = []
    s_x_t, s_y_t, w_t = [], [], []
    Eps_n1 = np.zeros((len(Eps),), dtype=np.float_)


def get_response_i(s_x_1, s_y_1, w_1, n_steps=10, k_max=300, **kw):
    global Eps_record, Sig_record, iter_record
    global t_arr, s_x_t, s_y_t, w_t, s_x_0, s_y_0, w_0, t0, Eps_n1
    t1 = t0 + n_steps + 1
    ti_arr = np.linspace(t0, t1, n_steps + 1)
    si_x_t = np.linspace(s_x_0, s_x_1, n_steps + 1)
    si_y_t = np.linspace(s_y_0, s_y_1, n_steps + 1)
    wi_t = np.linspace(w_0, w_1, n_steps + 1)
    for s_x_n1, s_y_n1, w_n1 in zip(si_x_t, si_y_t, wi_t):
        Eps_n1, Sig_n1, k = get_material_model(s_x_n1, s_y_n1, w_n1, Eps_n1, k_max, **kw)
        Sig_record.append(Sig_n1)
        Eps_record.append(Eps_n1)
        iter_record.append(k)
    t_arr = np.hstack([t_arr, ti_arr])
    s_x_t = np.hstack([s_x_t, si_x_t])
    s_y_t = np.hstack([s_y_t, si_y_t])
    w_t = np.hstack([w_t, wi_t])
    t0 = t1
    s_x_0, s_y_0, w_0 = s_x_1, s_y_1, w_1
    return


import ipywidgets as ipw


def plot3d_Sig_Eps(ax3d, s_x_t, s_y_t, Sig_arr, Eps_arr):
    tau_x, tau_y = Sig_arr.T[:2, ...]
    tau = np.sqrt(tau_x ** 2 + tau_y ** 2)
    ax3d.plot3D(s_x_t, s_y_t, tau, color='orange', lw=3)


def plot_sig_w(ax, w_t, Sig_arr, Eps_arr):
    sig_t = Sig_arr.T[2, ...]
    ax.plot(w_t, sig_t, color='orange', lw=3)


fig = plt.figure(figsize=(10, 3), tight_layout=True)
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax_sig = fig.add_subplot(1, 3, 2)
ax2 = fig.add_subplot(1, 3, 3)
ax_iter = ax2.twinx()


def update(s_x_1, s_y_1, w_1):
    global Eps_record, Sig_record, iter_record
    global t_arr, s_x_t, s_y_t, w_t, s_x_0, s_y_0, w_0, t0, Eps_n
    global kwargs
    get_response_i(s_x_1, s_y_1, w_1, **kwargs)
    Sig_arr = np.array(Sig_record, dtype=np.float_)
    Eps_arr = np.array(Eps_record, dtype=np.float_)
    iter_arr = np.array(iter_record, dtype=np.int_)
    ax1.clear()
    ax_sig.clear()
    ax2.clear()
    ax_iter.clear()
    #    plot_tau_s(ax1, Eps_arr[-1,...],s_max,500,get_g3,**kw)
    plot3d_Sig_Eps(ax1, s_x_t, s_y_t, Sig_arr, Eps_arr)
    ax1.plot(s_x_t, s_y_t, 0, color='red')
    ax1.set_xlabel(r'$s_x$ [mm]');
    ax1.set_ylabel(r'$s_y$ [mm]');
    ax1.set_zlabel(r'$\| \tau \| = \sqrt{\tau_x^2 + \tau_y^2}$ [MPa]');

    plot_sig_w(ax_sig, w_t, Sig_arr, Eps_arr)
    ax_sig.set_xlabel(r'$w$ [mm]');
    ax1.set_ylabel(r'$\sigma$ [MPa]');

    plot_work(ax2, t_arr, s_x_t, s_y_t, w_t, Eps_arr, Sig_arr)
    plot_dissipation(ax2, t_arr, Eps_arr, Sig_arr)
    ax_iter.plot(t_arr, iter_arr)
    ax_iter.set_ylabel(r'$n_\mathrm{iter}$')


s_x_1_slider = ipw.FloatSlider(description='s_x', value=0, min=-4, max=+4, step=0.1,
                               continuous_update=False)
s_y_1_slider = ipw.FloatSlider(description='s_y', value=0, min=-4, max=+4, step=0.1,
                               continuous_update=False)
w_1_slider = ipw.FloatSlider(description='w', value=0, min=-4, max=+4, step=0.1,
                             continuous_update=False)


def reset(**kwargs_):
    global kwargs
    kwargs = kwargs_
    init()
    s_x_1_slider.value = 0
    s_y_1_slider.value = 0
    w_1_slider.value = 0


n_steps = 20

kwargs_ranges = [('E_s', 1, 0.5, 100),
                 ('S_s', 0.6, 0.00001, 100),
                 ('c_s', 1, 0.0001, 10),
                 ('gamma_s', 0, -20, 20),
                 ('K_s', 0, -20, 20),
                 ('bartau', 1, 0.5, 20),
                 ('E_w', 1, 0.5, 100),
                 ('S_w', 0.6, 0.0001, 100),
                 ('c_w', 1, 0.0001, 10),
                 ('m', 0.1, 0.0001, 0.4),
                 ('f_t', 1, 0.1, 10),
                 ('f_c', 10, 1, 200),
                 ('f_c0', 5, 1, 100),
                 ('eta', 0, 0, 1)]

kwargs = {key: val for (key, val, _, _) in kwargs_ranges}

kwargs_sliders = {
    name: ipw.FloatSlider(description=name, value=val,
                          min=minval, max=maxval, step=(maxval - minval) / n_steps,
                          continuous_update=False)
    for name, val, minval, maxval in kwargs_ranges
}

slip_sliders = {'s_x_1': s_x_1_slider,
                's_y_1': s_y_1_slider,
                'w_1': w_1_slider}


def slider_layout(out1, out2):
    layout = ipw.Layout(grid_template_columns='1fr 1fr')
    slider_list = tuple(kwargs_sliders.values())
    grid = ipw.GridBox(slider_list, layout=layout)
    slip_slider_list = tuple(slip_sliders.values())
    hbox = ipw.HBox(slip_slider_list)
    box = ipw.VBox([hbox, grid, out1, out2])
    display(box)


init()
out1 = ipw.interactive_output(update, slip_sliders)
out2 = ipw.interactive_output(reset, kwargs_sliders);
slider_layout(out1, out2)

# # Code generation
# See the docs for the code generation, the latexified  sympy symbols
# must be substituted such that they can act as standard C variable names.
# The issue with this substitution might have been partially fixed by the substitution
# code defined by this code.
#
# The methods
#  * `get_f_df()`, and
#  * `get_Eps_n1` must be rewritten in C.
#
# **How to transform `einsum` to C?**
#
# The state arrays `Eps` and `Sig` must be prepared by the callee, i.e. the function to be
# called from at a level of material point in a finite-element or lattice code.
# All the state variable matrices are flattened in the generated C code so that index access operators must be constructed with the correct convention, i.e.
# `i * n_row + j` or `i + n_col * j`.
#
# Let us consider the line
# ```Python
# df_dEps_k = np.einsum('ik,ji->jk', df_dSig_k, dSig_dEps_k) + ddf_dEps_k
# ````
#
# To transform this systematically into the C loop it is proposed define
# index directives specifying the size of the first dimension. Then, the
# indexes from the `einsum` call can be systematically transferred to the
# multi-loop running over the indexes and respecting their order. No more
# thinking needed. The index operators below are prepared for arrays with
# 2, 3, and 4 dimensions.
# ```C
# #define IJ(N_I,I,J) (N_I * J + I)
# #define IJK(N_I,N_J,I,J,K) (N_I * IJ(N_J,J,K) + I)
# #define IJKL(N_I,N_J,N_K,I,J,K,L) (N_I * IJK(N_J,N_K,J,K,L) + I)
# ````
# to perform the matrix multiplication
# ```C
# int k=0;
# for(int i=0;i<N_I;i++)
#     for(int j=0;j<N_J;j++)
#         df_dEps_1[IJ(N_J,j,k)] += df_dSig_k[IJ(N_I(i,k)] * dSig_dEps_k[IJ(N_J,j,i)]
#     df_dEps[IJ(N_I,i,k)] = df_dEps_1[IJ(N_J,j,k)] + ddSig_dEps(IJ(N_I,i,k)];
# ````
#
# The whole material model is represented by the material method
# that mimics the internal par of the `get_response` function defined above
#
#  1. Call `get_f_df()` to get the trial state
#  2. Start the return mapping iteration loop
#  3. If admissibility criterion fulfilled - admissible state found - return stress
#  4. Evaluate delta of plastic multiplier and update it
#  5. Update state variables using evolution equations `get_Eps_n1`
#  6. Evaluate the residuum using `get_f_df()` and continuum with point 3.
#

# In[ ]:


C_code = get_Sig_C, get_dSig_dEps_C, get_f_C, get_df_dSig_C, get_ddf_dEps_C, get_Phi_C

# In[ ]:


import os
import os.path as osp

code_dirname = 'sympy_codegen'
code_fname = 'SLIDE_1_3_2D'

home_dir = osp.expanduser('~')
code_dir = osp.join(home_dir, code_dirname)
if not osp.exists(code_dir):
    os.makedirs(code_dir)

code_file = osp.join(code_dir, code_fname)

print('code_file', code_file)
h_file = code_file + '.h'
c_file = code_file + '.c'

h_f = open(h_file, 'w')
c_f = open(c_file, 'w')

if True:
    for function_C in C_code:
        h_f.write(function_C[1][1])
        c_f.write(function_C[0][1])
h_f.close()
c_f.close()

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:




