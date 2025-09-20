#!/usr/bin/env python3
"""
Quick test verification script for GSMThermodynBox2 tests
"""

import sys
from pathlib import Path

# Add the core2 directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_import():
    """Test that all required modules can be imported."""
    try:
        print("Testing imports...")
        
        # Test core module imports
        from gsm_thermodyn_box2 import GSMThermodynBox2
        print("✓ GSMThermodynBox2 imported")
        
        from gsm_state_fn import GSMStateFn, StateFunction
        print("✓ GSMStateFn and StateFunction imported")
        
        from gsm_vars import Scalar
        print("✓ Scalar imported")
        
        # Test that test module can be imported
        from bmcs_matmod.gsm_lagrange.core2.tests.test_gsm_thermodyn_box import TestGSMThermodynBox2RoundTrips
        print("✓ Test classes imported")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic GSMThermodynBox2 functionality."""
    try:
        print("\nTesting basic functionality...")
        
        from gsm_thermodyn_box2 import GSMThermodynBox2
        from gsm_state_fn import GSMStateFn, StateFunction
        from gsm_vars import Scalar
        import sympy as sp
        
        # Create symbols
        T = Scalar(r'\vartheta', codename='T', real=True, positive=True)
        S = Scalar('S', codename='S', real=True)
        eps = Scalar(r'\varepsilon', codename='eps', real=True)
        sig = Scalar(r'\sigma', codename='sig', real=True)
        E = Scalar('E', codename='E', positive=True)
        
        # Simple elastic Helmholtz free energy: F(T,ε) = ½Eε²
        F_elastic = sp.Rational(1, 2) * E * eps**2
        
        # Create the initial state function instance
        F_state_fn = GSMStateFn(
            fn_expr=F_elastic,
            th_x_var=T,      # Temperature is natural variable
            th_y_var=S,      # Entropy is conjugate variable
            mc_x_var=eps,    # Strain is natural variable
            mc_y_var=sig,    # Stress is conjugate variable
            Eps_var=sp.Symbol('eps_dummy'),  # Dummy internal variable
            Sig_var=sp.Symbol('sig_dummy'),  # Dummy internal conjugate
            state_function_type=StateFunction.HELMHOLTZ
        )
        
        # Create the thermodynamic box
        box = GSMThermodynBox2(
            initial_state_fn=StateFunction.HELMHOLTZ,
            initial_state_instance=F_state_fn,
            T_var=T,
            S_var=S
        )
        
        print("✓ GSMThermodynBox2 instance created")
        
        # Test basic operations
        available_functions = box.get_available_state_functions()
        print(f"✓ Available functions: {[sf.value for sf in available_functions]}")
        
        # Test property access
        F_instance = box.F
        print(f"✓ F property access: {F_instance.fn_expr}")
        
        # Test transformation
        G_instance = box.G
        print(f"✓ G transformation: {G_instance.fn_expr}")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=== GSMThermodynBox2 Test Verification ===")
    
    success = True
    
    # Test imports
    success &= test_import()
    
    # Test basic functionality
    success &= test_basic_functionality()
    
    print(f"\n=== Summary ===")
    if success:
        print("✅ All verification tests passed!")
        print("The GSMThermodynBox2 tests should work correctly.")
        return 0
    else:
        print("❌ Some verification tests failed!")
        print("Please fix the issues before running the full test suite.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
