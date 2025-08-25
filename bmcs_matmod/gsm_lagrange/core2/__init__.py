"""
GSM Core2 Package - Pure Architecture Implementation

This package provides the refactored GSM architecture with clear separation 
between symbolic definitions and numerical execution.

Core Components:
- GSMSymbDef: Pure symbolic thermodynamic definitions
- GSMEngine: Numerical execution engines
- MaterialParams: Parameter database repository
- GSMModel: High-level model interface
"""

from bmcs_matmod.gsm_lagrange.core2.gsm_symb_def import GSMSymbDef
from bmcs_matmod.gsm_lagrange.core2.gsm_engine import GSMEngine
from bmcs_matmod.gsm_lagrange.core2.material_params import MaterialParams, CommonMaterialParams
from bmcs_matmod.gsm_lagrange.core2.gsm_model import GSMModel, GSMModelFactory
from bmcs_matmod.gsm_lagrange.core2.gsm1d import GSM1D_ED, GSM1D_VE

__all__ = [
    'GSMSymbDef',
    'GSMEngine', 
    'MaterialParams',
    'CommonMaterialParams',
    'GSMModel',
    'GSMModelFactory',
    'GSM1D_ED',
    'GSM1D_VE'
]
