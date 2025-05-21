from .material_params import MaterialParams
from .gsm_def import GSMDef
from .gsm_engine import GSMEngine

class Material:
    """Represents a real-world material, aggregates parameter records for models."""
    def __init__(self, name):
        self.name = name
        self.mat_params = []

    def add_param(self, param: MaterialParams):
        self.mat_params.append(param)

    def get_param_for_model(self, gsm_def):
        for param in self.mat_params:
            if param.gsm_def == gsm_def:
                return param
        return None
