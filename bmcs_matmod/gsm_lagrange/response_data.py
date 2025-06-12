import numpy as np
import traits.api as tr
import bmcs_utils.api as bu

class ResponseDataContainer:
    def __init__(self, data_dict):
        self._data = data_dict
    def __getitem__(self, key):
        return self._data[key]
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"No attribute named {name}")
    def keys(self):
        return self._data.keys()
    def items(self):
        return self._data.items()
    def values(self):
        return self._data.values()

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
    eps_t : np.ndarray
        Control variable history (e.g., strain or stress, shape: (n_steps, ...)).
    sig_t : np.ndarray
        Conjugate variable history (e.g., stress or strain, shape: (n_steps, ...)).
    Eps_t_flat : np.ndarray
        Flattened internal variables history (shape: (n_steps, n_flat)).
    Sig_t_flat : np.ndarray
        Flattened conjugate variables to internal variables (shape: (n_steps, n_flat)).
    Eps_vars : tr.Tuple
        Tuple of internal variable symbols (from GSMEngine).
    Sig_vars : tr.Tuple
        Tuple of conjugate variable symbols (from GSMEngine).
    Eps_t : dict
        Dictionary mapping internal variable codenames to their time series.
    Sig_t : dict
        Dictionary mapping conjugate variable codenames to their time series.
    iter_t : np.ndarray
        Iteration counts per time step (optional).
    lam_t : np.ndarray
        Inelastic multipliers per time step (optional).
    """

    t_t = tr.Array
    eps_t = tr.Array
    sig_t = tr.Array
    Eps_t_flat = tr.Array
    Sig_t_flat = tr.Array
    Eps_vars = tr.Tuple
    Sig_vars = tr.Tuple
    Eps_t = tr.Any
    Sig_t = tr.Any
    iter_t = tr.Array
    lam_t = tr.Array

    @classmethod
    def from_engine_response(cls, engine, response):
        """
        Construct a ResponseData object from the output tuple of GSMEngine.get_response.

        Parameters
        ----------
        engine : GSMEngine
            The GSMEngine instance (F_engine or G_engine) used for the simulation.
        response : tuple
            The tuple returned by engine.get_response.

        Returns
        -------
        ResponseData
            An instance populated with data and metadata.
        """
        t_t, eps_t, sig_t, Eps_t, Sig_t, iter_t, lam_t, _ = response
        # Flatten Eps_t and Sig_t to (n_steps, n_flat)
        Eps_t_flat_local = Eps_t.reshape(Eps_t.shape[0], -1)
        Sig_t_flat_local = Sig_t.reshape(Sig_t.shape[0], -1)
        # Get variable metadata from the GSMEngine
        Eps_vars = engine.Eps_vars
        Sig_vars = engine.Sig_vars
        # Try to get codenames from the parent GSMDef class if available
        gsm_def = getattr(engine, 'gsm_def', None)
        if gsm_def is None:
            # Try to infer from the engine's module (works if engine is a class attribute)
            import sys
            for obj in sys.modules[engine.__class__.__module__].__dict__.values():
                if hasattr(obj, 'F_engine') and (obj.F_engine is engine or getattr(obj, 'G_engine', None) is engine):
                    gsm_def = obj
                    break

        # Build dicts for variable access by codename (supporting scalars, vectors, tensors)
        Eps_t_dict = {}
        col = 0
        for var in Eps_vars:
            var_shape = var.shape if hasattr(var, 'shape') else ()
            n_flat = np.prod(var_shape) if var_shape else 1
            # Use codename attribute from var
            codename = getattr(var, 'codename', str(var))
            print(f'Processing variable: {var}, codename = {codename}, shape = {var_shape}, n_flat = {n_flat}')
            Eps_t_dict[codename] = Eps_t_flat_local[:, col:col + n_flat].reshape((Eps_t_flat_local.shape[0],) + tuple(var_shape))
            col += n_flat

        Sig_t_dict = {}
        col = 0
        for var in Sig_vars:
            var_shape = var.shape if hasattr(var, 'shape') else ()
            n_flat = np.prod(var_shape) if var_shape else 1
            codename = getattr(var, 'codename', str(var))
            Sig_t_dict[codename] = Sig_t_flat_local[:, col:col + n_flat].reshape((Sig_t_flat_local.shape[0],) + tuple(var_shape))
            col += n_flat

        return cls(
            t_t=t_t,
            eps_t=eps_t,
            sig_t=sig_t,
            Eps_t_flat=Eps_t_flat_local,
            Sig_t_flat=Sig_t_flat_local,
            Eps_vars=Eps_vars,
            Sig_vars=Sig_vars,
            Eps_t=ResponseDataContainer(Eps_t_dict),
            Sig_t=ResponseDataContainer(Sig_t_dict),
            iter_t=iter_t,
            lam_t=lam_t
        )

    def get_internal_var(self, key):
        """
        Extract the internal variable as a time series, accessed by codename or index.

        Parameters
        ----------
        key : str or int
            Codename of the internal variable (as defined in GSMDef), or index.

        Returns
        -------
        arr : np.ndarray
            Array of shape (n_steps,) for scalar variables.
        """
        if isinstance(key, int):
            return self.get_internal_var_by_index(key)
        return self.Eps_t[key]

    def get_internal_var_by_index(self, i):
        var_shape = self.Eps_vars[i].shape
        start = sum(np.prod(v.shape) for v in self.Eps_vars[:i])
        end = start + np.prod(var_shape)
        flat = self.Eps_t_flat[:, start:end]
        return flat.reshape((flat.shape[0],) + var_shape)

    def get_conj_var(self, key):
        """
        Extract the conjugate variable (thermodynamic force) as a time series, accessed by codename or index.

        Parameters
        ----------
        key : str or int
            Codename of the conjugate variable (as defined in GSMDef), or index.

        Returns
        -------
        arr : np.ndarray
            Array of shape (n_steps,) for scalar variables.
        """
        if isinstance(key, int):
            return self.get_conj_var_by_index(key)
        return self.Sig_t[key]

    def get_conj_var_by_index(self, i):
        var_shape = self.Sig_vars[i].shape
        start = sum(np.prod(v.shape) for v in self.Sig_vars[:i])
        end = start + np.prod(var_shape)
        flat = self.Sig_t_flat[:, start:end]
        return flat.reshape((flat.shape[0],) + var_shape)

    def to_dict(self):
        """
        Return all data and metadata as a dictionary for storage or further processing.
        """
        return dict(
            t_t=self.t_t,
            eps_t=self.eps_t,
            sig_t=self.sig_t,
            Eps_t_flat=self.Eps_t_flat,
            Sig_t_flat=self.Sig_t_flat,
            Eps_vars=self.Eps_vars,
            Sig_vars=self.Sig_vars,
            Eps_t=self.Eps_t,
            Sig_t=self.Sig_t,
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
            f"eps_t.shape={self.eps_t.shape}, "
            f"sig_t.shape={self.sig_t.shape}, "
            f"Eps_t_flat.shape={self.Eps_t_flat.shape}, "
            f"Sig_t_flat.shape={self.Sig_t_flat.shape}, "
            f"Eps_vars={self.Eps_vars}, "
            f"Sig_vars={self.Sig_vars}, "
            f"Eps_t keys={list(self.Eps_t.keys())}, "
            f"Sig_t keys={list(self.Sig_t.keys())})"
        )
