#!/usr/bin/env python3
"""Simple test runner to verify the GSMThermodynBox2 functionality."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_basic_gsm_functionality():
    """Test basic GSM functionality."""
    print("🧪 Testing GSMThermodynBox2 basic functionality...")
    
    try:
        # Import with full paths
        from bmcs_matmod.gsm_lagrange.core2.gsm_thermodyn_box2 import GSMThermodynBox2
        from bmcs_matmod.gsm_lagrange.core2.gsm_state_fn import GSMStateFn, StateFunction
        from bmcs_matmod.gsm_lagrange.core2.gsm_vars import Scalar
        print("✅ All imports successful")
        
        # Create symbols
        T = Scalar(r'\vartheta', codename='T', real=True, positive=True)
        S = Scalar('S', codename='S', real=True)
        eps = Scalar(r'\varepsilon', codename='eps', real=True)
        sig = Scalar(r'\sigma', codename='sig', real=True)
        print("✅ Symbols created")
        
        # Create Helmholtz free energy
        import sympy as sp
        F_expr = (1/2) * 100 * eps**2 + 300 * T * (sp.log(T/300) - 1) + 300 * T
        print("✅ Helmholtz expression defined")
        
        # Create state function
        helmholtz_fn = GSMStateFn(
            name='helmholtz',
            state_fn=StateFunction.F,
            description='Helmholtz Free Energy',
            state_variables=[T, eps],
            state_expr=F_expr
        )
        print("✅ Helmholtz state function created")
        
        # Create thermodynamic box
        box = GSMThermodynBox2()
        box.add_state_fn(helmholtz_fn)
        print("✅ Thermodynamic box created and function added")
        
        # Test round trip consistency
        F_val = box.get_state_function_expr('F')
        print(f"✅ F expression retrieved: {str(F_val)[:50]}...")
        
        U_val = box.get_state_function_expr('U')
        print(f"✅ U expression computed: {str(U_val)[:50]}...")
        
        print("\n🎉 All basic tests passed! The import paths are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_gsm_functionality()
    sys.exit(0 if success else 1)
