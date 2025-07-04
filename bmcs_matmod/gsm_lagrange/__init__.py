"""
GSM Lagrange Framework

This package provides a symbolic-to-executable framework for thermodynamically
consistent material modeling based on the Generalized Standard Material concept.
"""

from .material_params import MaterialParams
from .material import Material
from .gsm_def import GSMDef
from .gsm_engine import GSMEngine
from .gsm_model import GSMModel
from .gsm1d_ed import GSM1D_ED
from .gsm1d_ep import GSM1D_EP
from .gsm1d_epd import GSM1D_EPD
from .gsm1d_evp import GSM1D_EVP
from .gsm1d_evpd import GSM1D_EVPD
from .gsm1d_ve import GSM1D_VE
from .gsm1d_ved import GSM1D_VED
from .gsm1d_vevp import GSM1D_VEVP
from .gsm1d_vevpd import GSM1D_VEVPD

# CLI interface components
try:
    from .cli_interface import GSMModelCLI
    from .data_structures import MaterialParameterData, LoadingData, SimulationConfig, SimulationResults
    from .parameter_loader import ParameterLoader
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

__all__ = [
    'MaterialParams', 'Material', 'GSMDef', 'GSMEngine', 'GSMModel',
    'GSM1D_ED', 'GSM1D_EP', 'GSM1D_EPD', 'GSM1D_EVP', 'GSM1D_EVPD',
    'GSM1D_VE', 'GSM1D_VED', 'GSM1D_VEVP', 'GSM1D_VEVPD'
]

if CLI_AVAILABLE:
    __all__.extend([
        'GSMModelCLI',
        'MaterialParameterData',
        'LoadingData', 
        'SimulationConfig',
        'SimulationResults',
        'ParameterLoader'
    ])

__version__ = '0.1.0'
