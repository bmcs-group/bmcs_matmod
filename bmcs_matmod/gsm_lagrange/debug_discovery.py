#!/usr/bin/env python3
"""Debug script for GSM model discovery"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Python path: {sys.path[0]}")
print(f"Current directory: {os.getcwd()}")

# Test basic imports
try:
    from bmcs_matmod.gsm_lagrange.gsm_def import GSMDef
    print("✓ GSMDef imported successfully")
except Exception as e:
    print(f"✗ Could not import GSMDef: {e}")
    sys.exit(1)

# Test model import
try:
    from bmcs_matmod.gsm_lagrange.gsm1d_ed import GSM1D_ED
    print("✓ GSM1D_ED imported successfully")
    print(f"  Class: {GSM1D_ED}")
    print(f"  Has F_engine: {hasattr(GSM1D_ED, 'F_engine')}")
    
    if hasattr(GSM1D_ED, 'F_engine'):
        engine = GSM1D_ED.F_engine
        print(f"  F_engine: {engine}")
        print(f"  F_engine type: {type(engine)}")
    
except Exception as e:
    print(f"✗ Could not import GSM1D_ED: {e}")
    import traceback
    traceback.print_exc()

# Test manual discovery
print("\nManual model discovery:")
print("-" * 30)

import importlib
import inspect

gsm_dir = Path(__file__).parent / "gsm_lagrange"
gsm_files = [f for f in os.listdir(gsm_dir) if f.startswith('gsm1d_') and f.endswith('.py')]

print(f"Found GSM files: {gsm_files}")

for file_name in gsm_files[:3]:  # Test first 3
    module_name = file_name[:-3]
    try:
        module_path = f"bmcs_matmod.gsm_lagrange.{module_name}"
        print(f"\nImporting {module_path}...")
        module = importlib.import_module(module_path)
        
        # Find classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, GSMDef) and obj is not GSMDef:
                print(f"  Found GSMDef subclass: {name}")
                print(f"    Has F_engine: {hasattr(obj, 'F_engine')}")
                if hasattr(obj, 'F_engine'):
                    print(f"    F_engine value: {obj.F_engine}")
                    print(f"    F_engine is not None: {obj.F_engine is not None}")
    
    except Exception as e:
        print(f"  Error with {module_name}: {e}")

print("\nDebug complete!")
