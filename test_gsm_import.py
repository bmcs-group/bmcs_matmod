#!/usr/bin/env python3
"""
Test script to isolate gsm_lagrange import issues
"""

print("Starting import test...")

try:
    print("1. Testing basic imports...")
    import traits.api as tr
    print("   - traits.api: OK")
    
    import sympy as sp
    print("   - sympy: OK")
    
    print("2. Testing gsm_lagrange package import...")
    import bmcs_matmod.gsm_lagrange
    print("   - bmcs_matmod.gsm_lagrange: OK")
    
    print("3. Testing individual core imports...")
    
    # Try importing each core file individually
    print("   - gsm_vars...")
    from bmcs_matmod.gsm_lagrange.core import gsm_vars
    print("     OK")
    
    print("   - response_data...")
    from bmcs_matmod.gsm_lagrange.core import response_data
    print("     OK")
    
    print("   - gsm_def...")
    from bmcs_matmod.gsm_lagrange.core import gsm_def
    print("     OK")
    
    print("   - gsm_engine...")
    from bmcs_matmod.gsm_lagrange.core import gsm_engine
    print("     OK")
    
    print("   - gsm_model...")
    from bmcs_matmod.gsm_lagrange.core import gsm_model
    print("     OK")
    
    print("4. Testing core __init__.py...")
    from bmcs_matmod.gsm_lagrange.core import GSMDef
    print("   - GSMDef import: OK")
    
    print("\nAll imports successful! ✅")

except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
