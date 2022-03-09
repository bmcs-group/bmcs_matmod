
import traits.api as tr
from .i_ntim import INTIM
from ibvpy.tmodel.mats3D.mats3D_eval import MATS3DEval

class NTIM(MATS3DEval):

    def plot_sig_eps(self, ax):
        ax.plot()

    def update_plot(self, ax):
        self.plot_sig_eps(ax)