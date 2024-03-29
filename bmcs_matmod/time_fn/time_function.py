import traits.api as tr
import numpy as np
from scipy import interpolate
import bmcs_utils.api as bu


class TimeFunction(bu.InteractiveModel):
    name = 'Time function'
    n_i = bu.Int(0, input=True, desc='number of data points between data points')
    '''Number of steps between two values
    '''

    values = tr.Array(np.float_, value=[0, 1])
    '''Values of the time function.
    '''

    ipw_view = bu.View(
        bu.Item('n_i', latex=r'n_i', minmax=(0, 100))
    )

    t_step = tr.Property(tr.Float, depends_on='+input')
    '''Step size.
    '''

    @tr.cached_property
    def _get_t_step(self):
        n_values = len(self.values)
        return (self.n_i + 1) * n_values

    tf = tr.Property(depends_on='data_points')

    def _get_tf(self):
        n_values = len(self.values)
        t_values = np.linspace(0, 1, n_values)
        return interpolate.interp1d(t_values, self.values)

    def __call__(self, arg):
        return self.tf(arg)

    def update_plot(self, ax):
        n_values = len(self.values)
        t = np.linspace(0, 1, (self.n_i) * (n_values - 1) + n_values)
        v = self.tf(t)
        ax.plot(t, v, '-o')
