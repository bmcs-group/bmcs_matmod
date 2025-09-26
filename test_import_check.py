#!/usr/bin/env python3
"""Simple test to verify imports work correctly."""

try:
    print("Testing imports...")
    from bmcs_matmod.gsm_lagrange.core2.gsm_thermodyn_box import GSMThermodynBox
    print("✅ GSMThermodynBox2 imported successfully")
    
    from bmcs_matmod.gsm_lagrange.core2.gsm_state_fn import GSMStateFn, StateFunctionType
    print("✅ GSMStateFn and StateFunction imported successfully")
    
    from bmcs_matmod.gsm_lagrange.core2.gsm_vars import Scalar
    print("✅ Scalar imported successfully")
    
    # Try creating a simple instance
    T = Scalar('T', codename='T', real=True, positive=True)
    S = Scalar('S', codename='S', real=True)
    print("✅ Scalar instances created successfully")
    
    print("\n✅ ALL IMPORTS SUCCESSFUL!")
    print("The test file import paths have been corrected and should work now.")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
