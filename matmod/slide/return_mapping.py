# In[39]:

import numpy as np


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

