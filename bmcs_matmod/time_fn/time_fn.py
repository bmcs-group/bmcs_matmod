
import bmcs_utils.api as bu
import sympy as sp
import numpy as np

# # Support functions
# To run some examples, let us define some infrastructure including a more complex loading history and postprocessing

# ## Loading history
# This implementation uses the symbolic machinery which is not necessary a simpler data point based implementation with `numpy.interp1d` would be better ... later


class LoadFn(bu.InteractiveModel):

    def __call__(self,t):
        return t

class CyclicLoadFnExpr(bu.SymbExpr):
    pass

class CyclicLoadFn(LoadFn):

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


