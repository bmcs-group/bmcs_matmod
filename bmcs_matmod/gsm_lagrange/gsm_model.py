import traits.api as tr
import bmcs_utils.api as bu
import sympy as sp
import numpy as np
from typing import Type, Dict, List, Union, Any, Tuple
import inspect

from .gsm_def import GSMDef
from .gsm_engine import GSMEngine

def derive_trait_model_params(gsm_def):
    """Utility to derive trait_model_params mapping from a GSMDef."""
    # ...extract logic from old GSMMaterialModel.__new__...
    temp_instance = gsm_def()
    param_symbols = temp_instance.F_engine.m_params
    trait_model_params = {}
    for param_sym in param_symbols:
        param_name = param_sym.name
        trait_name = param_name
        if '\\' in param_name or '{' in param_name or '}' in param_name:
            if hasattr(temp_instance, 'param_codenames') and param_sym in temp_instance.param_codenames:
                trait_name = temp_instance.param_codenames[param_sym]
            else:
                trait_name = param_name.replace('\\', '').replace('{', '').replace('}', '')
        trait_model_params[param_sym] = trait_name
    return trait_model_params

class GSMModel(bu.Model):
    """
    Executable GSM model with assigned parameters.
    Bridges symbolic definition and simulation.
    """
    gsm_def = tr.Type(GSMDef)
    gsm_exec = tr.Property(tr.Instance(GSMDef), depends_on='gsm_def,+params')
    trait_model_params = tr.Dict

    def __new__(cls, gsm_def=None, **traits):
        if gsm_def is None:
            return super().__new__(cls)
        model_class_name = f"{gsm_def.__name__}MatMod"
        if model_class_name in globals():
            return globals()[model_class_name].__new__(globals()[model_class_name])
        trait_model_params = derive_trait_model_params(gsm_def)
        traits_dict = {
            '__doc__': f"Executable material model based on {gsm_def.__name__}",
            'gsm_def': gsm_def,
            'trait_model_params': trait_model_params,
            'param_names': list(trait_model_params.values())
        }
        for param_sym, trait_name in trait_model_params.items():
            traits_dict[trait_name] = tr.Float(1.0, desc=f"Material parameter {param_sym.name}")
        model_class = type(model_class_name, (cls,), traits_dict)
        globals()[model_class_name] = model_class
        instance = model_class.__new__(model_class)
        return instance

    def __init__(self, gsm_def=None, **traits):
        if gsm_def is not None:
            traits['gsm_def'] = gsm_def
        super().__init__(**traits)

    @tr.cached_property
    def _get_gsm_exec(self):
        return self.gsm_def()

    def get_param_dict(self):
        return {trait_name: getattr(self, trait_name) for trait_name in self.trait_model_params.values()}

    def get_args(self):
        return self.gsm_exec.get_args(**self.get_param_dict())

    # ...delegate methods as before, but use gsm_exec instead of model_instance...
    def get_F_sig(self, eps):
        return self.gsm_exec.get_F_sig(eps, *self.get_args())
    def get_F_response(self, eps_ta, t_t):
        return self.gsm_exec.get_F_response(eps_ta, t_t, *self.get_args())
    def get_F_Sig(self, eps):
        return self.gsm_exec.get_F_Sig(eps, *self.get_args())
    def get_G_eps(self, sig):
        return self.gsm_exec.get_G_eps(sig, *self.get_args())
    def get_G_response(self, sig_ta, t_t):
        return self.gsm_exec.get_G_response(sig_ta, t_t, *self.get_args())
    def get_G_Sig(self, sig):
        return self.gsm_exec.get_G_Sig(sig, *self.get_args())
    def print_potentials(self):
        self.gsm_exec.print_potentials()
    def markdown(self):
        return self.gsm_exec.markdown()
    def get_param_values(self):
        return {sym: getattr(self, trait_name) for sym, trait_name in self.trait_model_params.items()}
    def __str__(self):
        params_str = ", ".join(f"{name}={getattr(self, name)}" for name in self.trait_model_params.values())
        return f"{self.__class__.__name__}({params_str})"

    def set_params(self, **kw):
        """Set parameters from keyword arguments, ignoring unknown keys."""
        for name in getattr(self, 'param_names', []):
            if name in kw:
                setattr(self, name, kw[name])
