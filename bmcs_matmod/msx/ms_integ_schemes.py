"""
Integration schemes for microplane models
"""
import numpy as np
import traits.api as tr
from bmcs_utils.api import Model


class MSIntegScheme(Model):
    """Interface of an integration scheme
    """

    MPNN = tr.Property(depends_on='n_mp')
    """Get the operator of the microplane normals
    """
    @tr.cached_property
    def _get_MPNN(self):
        MPNN_nij = np.einsum('ni,nj->nij', self.MPN, self.MPN)
        return MPNN_nij

    MPTT = tr.Property(depends_on='n_mp')
    """Rank three tangential tensor (operator) for each microplane
    """
    @tr.cached_property
    def _get_MPTT(self):
        delta = np.identity(3)
        MPTT_nijr = 0.5 * (
                np.einsum('ni,jr -> nijr', self.MPN, delta) +
                np.einsum('nj,ir -> njir', self.MPN, delta) - 2 *
                np.einsum('ni,nj,nr -> nijr', self.MPN, self.MPN, self.MPN)
        )
        return MPTT_nijr

class MSIS3DM28(MSIntegScheme):
    """Integration scheme 3D with 28 Microplane
    """
    n_mp = tr.Constant(28)
    """Number of microplanes
    """

    MPN = tr.Property(depends_on='n_mp')
    """Normal vectors of the microplanes
    """
    @tr.cached_property
    def _get_MPN(self):
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

    MPW = tr.Property(depends_on='n_mp')
    """Get the weights of the microplanes
    """
    @tr.cached_property
    def _get_MPW(self):
        return np.array([.0160714276, .0160714276, .0160714276, .0160714276, .0204744730,
                         .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                         .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                         .0204744730, .0158350505, .0158350505, .0158350505, .0158350505,
                         .0158350505, .0158350505, .0158350505, .0158350505, .0158350505,
                         .0158350505, .0158350505, .0158350505]) * 6.0

