#!/usr/bin/env python3
"""
Test script to check CLI imports
"""
import sys
from pathlib import Path

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except Exception as e:
        print(f"✗ numpy import failed: {e}")
        return False

    try:
        import json
        print("✓ json imported successfully")
    except Exception as e:
        print(f"✗ json import failed: {e}")
        return False

    return True

def test_gsm_model_import():
    """Test GSM model imports"""
    print("\nTesting GSM model imports...")
    try:
        from bmcs_matmod.gsm_lagrange.core.gsm_model import GSMModel
        print("✓ GSMModel imported successfully")
    except Exception as e:
        print(f"✗ GSMModel import failed: {e}")
        return False

    try:
        from bmcs_matmod.gsm_lagrange.models.gsm1d_ed import GSM1D_ED
        print("✓ GSM1D_ED imported successfully")
    except Exception as e:
        print(f"✗ GSM1D_ED import failed: {e}")
        return False

    return True

def test_model_instantiation():
    """Test model instantiation"""
    print("\nTesting model instantiation...")
    try:
        from bmcs_matmod.gsm_lagrange.core.gsm_model import GSMModel
        from bmcs_matmod.gsm_lagrange.models.gsm1d_ed import GSM1D_ED
        
        material = GSMModel(GSM1D_ED)
        print("✓ GSMModel instantiated successfully")
        
        # Try setting basic parameters
        material.set_params(E=30000.0, S=1.0, c=1.0, eps_0=0.0)
        print("✓ Parameters set successfully")
        
        return True
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return False

if __name__ == "__main__":
    print("GSM CLI Import Test")
    print("=" * 30)
    
    success = True
    success &= test_basic_imports()
    success &= test_gsm_model_import()
    success &= test_model_instantiation()
    
    print("\n" + "=" * 30)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
