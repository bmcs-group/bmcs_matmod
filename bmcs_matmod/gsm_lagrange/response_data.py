import numpy as np
import traits.api as tr
import bmcs_utils.api as bu

class ResponseData(bu.Model):
    """
    Flexible container for simulation response data and metadata from GSMModel.

    This class encapsulates the time history of simulation results, including
    time, control variables, conjugate variables, internal variables, and their
    metadata (names, shapes, etc.). It provides methods to extract, reshape,
    and access the data in a form suitable for postprocessing, plotting, or storage.

    Attributes
    ----------
    t_t : np.ndarray
        Time array (1D).
    ctrl_var : np.ndarray
        Control variable history (e.g., strain or stress, shape: (n_steps, ...)).
    conj_var : np.ndarray
        Conjugate variable history (e.g., stress or strain, shape: (n_steps, ...)).
    Eps_t : np.ndarray
        Flattened internal variables history (shape: (n_steps, n_flat)).
    Sig_t : np.ndarray
        Flattened conjugate variables to internal variables (shape: (n_steps, n_flat)).
    Eps_vars : tuple
        Tuple of variable metadata (from GSMEngine.Eps_vars).
    Sig_vars : tuple
        Tuple of variable metadata (from GSMEngine.Sig_vars).
    iter_t : np.ndarray
        Iteration counts per time step (optional).
    lam_t : np.ndarray
        Inelastic multipliers per time step (optional).

    Methods
    -------
    get_internal_var(i)
        Return the i-th internal variable as a time series (reshaped).
    get_conj_var(i)
        Return the i-th conjugate variable as a time series (reshaped).
    to_dict()
        Return all data and metadata as a dictionary for storage or further processing.
    """

    t_t = tr.Array
    ctrl_var = tr.Array
    conj_var = tr.Array
    Eps_t = tr.Array
    Sig_t = tr.Array
    Eps_vars = tr.Tuple
    Sig_vars = tr.Tuple
    iter_t = tr.Array
    lam_t = tr.Array

    def get_internal_var(self, i):
        """
        Extract the i-th internal variable as a time series, reshaped according to its definition.

        Parameters
        ----------
        i : int
            Index of the internal variable in Eps_vars.

        Returns
        -------
        arr : np.ndarray
            Array of shape (n_steps, ...) matching the shape of the i-th variable.
        """
        var_shape = self.Eps_vars[i].shape
        start = sum(np.prod(v.shape) for v in self.Eps_vars[:i])
        end = start + np.prod(var_shape)
        flat = self.Eps_t[:, start:end]
        return flat.reshape((flat.shape[0],) + var_shape)

    def get_conj_var(self, i):
        """
        Extract the i-th conjugate variable (thermodynamic force) as a time series.

        Parameters
        ----------
        i : int
            Index of the conjugate variable in Sig_vars.

        Returns
        -------
        arr : np.ndarray
            Array of shape (n_steps, ...) matching the shape of the i-th variable.
        """
        var_shape = self.Sig_vars[i].shape
        start = sum(np.prod(v.shape) for v in self.Sig_vars[:i])
        end = start + np.prod(var_shape)
        flat = self.Sig_t[:, start:end]
        return flat.reshape((flat.shape[0],) + var_shape)

    def to_dict(self):
        """
        Return all data and metadata as a dictionary for storage or further processing.
        """
        return dict(
            t_t=self.t_t,
            ctrl_var=self.ctrl_var,
            conj_var=self.conj_var,
            Eps_t=self.Eps_t,
            Sig_t=self.Sig_t,
            Eps_vars=self.Eps_vars,
            Sig_vars=self.Sig_vars,
            iter_t=self.iter_t,
            lam_t=self.lam_t
        )

    def __getitem__(self, key):
        """
        Allow dict-like access to main arrays.
        """
        return getattr(self, key)

    def __repr__(self):
        return (
            f"ResponseData(t_t.shape={self.t_t.shape}, "
            f"ctrl_var.shape={self.ctrl_var.shape}, "
            f"conj_var.shape={self.conj_var.shape}, "
            f"Eps_t.shape={self.Eps_t.shape}, "
            f"Sig_t.shape={self.Sig_t.shape})"
        )
