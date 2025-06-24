"""
Core GSM Lagrange framework components.

This module contains the fundamental classes and utilities for
defining and working with Generalized Standard Material models.
"""

# Import modules with error handling for environment issues
try:
    from .gsm_vars import Scalar, Vector
    from .response_data import ResponseData
    from .gsm_def import GSMDef
    from .gsm_engine import GSMEngine
    from .gsm_model import GSMModel
    from .gsm_def_registry import (
        get_available_gsm_defs,
        get_gsm_def_class,
        check_gsm_def_exists
    )
    
    __all__ = [
        'GSMDef',
        'GSMEngine', 
        'GSMModel',
        'Scalar',
        'Vector',
        'ResponseData',
        'get_available_gsm_defs',
        'get_gsm_def_class',
        'check_gsm_def_exists'
    ]
    
except ImportError as e:
    # Fallback for import issues - individual modules can still be imported directly
    print(f"Warning: Some core modules could not be imported: {e}")
    print("You can import individual modules directly, e.g.:")
    print("  from bmcs_matmod.gsm_lagrange.core.gsm_def import GSMDef")
    
    __all__ = []
