import traits.api as tr
from .gsm_def import GSMDef
from .gsm_model import derive_trait_model_params

class MaterialParams(tr.HasTraits):
    """Record of parameter values for a GSM model."""
    gsm_def = tr.Type(GSMDef)
    param_values = tr.Dict

    def __init__(self, gsm_def, **param_values):
        super().__init__()
        self.gsm_def = gsm_def
        self.param_values = param_values
        # Optionally, validate param_values keys using derive_trait_model_params

    @property
    def trait_model_params(self):
        return derive_trait_model_params(self.gsm_def)
