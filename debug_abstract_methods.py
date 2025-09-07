#!/usr/bin/env python3
"""Debug script to find the abstract method mismatch."""

import sys
sys.path.insert(0, '/home/rch/Coding/bmcs_matmod')

try:
    from bmcs_matmod.gsm_lagrange.core2.gsm_state_fn_ifc import GSMStateFnIfc
    from bmcs_matmod.gsm_lagrange.core2.gsm_state_fn import GSMStateFn
    print("✅ Import successful")
    
    print("\nAbstract methods required by GSMStateFnIfc:")
    for method in sorted(GSMStateFnIfc.__abstractmethods__):
        print(f"  - {method}")
    
    print(f"\nTotal abstract methods: {len(GSMStateFnIfc.__abstractmethods__)}")
    
    print("\nMethods implemented in GSMStateFn:")
    for method in sorted(dir(GSMStateFn)):
        if not method.startswith('_') and method.startswith('get_'):
            print(f"  - {method}")
    
    # Try to create an instance to see the exact error
    print("\nTrying to create GSMStateFn instance...")
    
    import sympy as sp
    from bmcs_matmod.gsm_lagrange.core2.gsm_vars import Scalar
    from bmcs_matmod.gsm_lagrange.core2.gsm_state_fn import StateFunction
    
    T = Scalar('T', codename='T')
    S = Scalar('S', codename='S')
    eps = Scalar('eps', codename='eps')
    sig = Scalar('sig', codename='sig')
    
    F_expr = sp.Rational(1, 2) * eps**2
    
    instance = GSMStateFn(
        fn_expr=F_expr,
        th_x_var=T, th_y_var=S, mc_x_var=eps, mc_y_var=sig,
        Eps_var=sp.Symbol('eps_dummy'), Sig_var=sp.Symbol('sig_dummy'),
        state_function_type=StateFunction.HELMHOLTZ
    )
    print("✅ GSMStateFn instance created successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
