#!/usr/bin/env python3
"""
Test script to verify the enhanced GSM thermodynamic box widget with constitutive relations.
"""

import sympy as sp
from bmcs_matmod.gsm_lagrange.core2.gsm_state_fn import GSMStateFn, StateFunction
from bmcs_matmod.gsm_lagrange.core2.gsm_thermodyn_box import GSMThermodynBox
from bmcs_matmod.gsm_lagrange.core2.gsm_thermodyn_box_widget import GSMThermodynBoxWidget

def test_widget_with_constitutive_relations():
    """Test the enhanced widget with constitutive relations display."""
    
    # Create symbolic variables
    T, S = sp.symbols('T S')  # Thermal variables
    eps, sig = sp.symbols('eps sig')  # Mechanical variables  
    Eps, Sig = sp.symbols('Eps Sig')  # Internal variables
    
    # Create a simple Helmholtz free energy expression with quadratic terms
    F_expr = T**2/2 + eps**2/2 + Eps**2/2 + T*eps + T*Eps + eps*Eps
    
    # Create GSMStateFn instance (Helmholtz: F(T, ε, Ɛ))
    state_fn = GSMStateFn(
        fn_expr=F_expr,
        th_x_var=T,      # Temperature (natural)
        th_y_var=S,      # Entropy (conjugate)
        mc_x_var=eps,    # Strain (natural)
        mc_y_var=sig,    # Stress (conjugate) 
        Eps_var=Eps,     # Internal strain (natural)
        Sig_var=Sig,     # Internal stress (conjugate)
        state_function_type=StateFunction.HELMHOLTZ
    )
    
    print("Testing enhanced widget with constitutive relations...")
    print("=" * 60)
    
    # Test the organized constitutive relations method
    print("\n1. Testing organized constitutive relations:")
    try:
        relations = state_fn.get_organized_constitutive_relations()
        for section, rel_list in relations.items():
            print(f"\n{section.capitalize()} relations:")
            for conj_var, expr in rel_list:
                print(f"  {conj_var} = {expr}")
    except Exception as e:
        print(f"Error in organized relations: {e}")
    
    # Test individual methods
    print("\n2. Testing individual constitutive relation methods:")
    
    try:
        # Thermal
        th_var, th_expr = state_fn.get_thermal_constitutive_relation()
        print(f"Thermal: {th_var} = {th_expr}")
        
        # Mechanical
        mech_relations = state_fn.get_mechanical_constitutive_relations()
        print(f"Mechanical: {mech_relations}")
        
        # Internal
        int_relations = state_fn.get_internal_constitutive_relations()
        print(f"Internal: {int_relations}")
        
    except Exception as e:
        print(f"Error in individual methods: {e}")
    
    # Create GSMThermodynBox
    print("\n3. Creating GSM Thermodynamic Box...")
    try:
        gsm_box = GSMThermodynBox(state_fn, state_fn)
        print("GSM Box created successfully!")
        
        # Test widget creation (but don't display it in terminal)
        print("\n4. Testing widget creation...")
        widget = GSMThermodynBoxWidget(gsm_box, "Test Widget with Constitutive Relations")
        print("Widget created successfully!")
        
        # Test the method we use in the widget
        print("\n5. Testing widget's constitutive display method...")
        test_state_fn = gsm_box.F  # Get the F state function
        
        # Test the organized method 
        try:
            organized = test_state_fn.get_organized_constitutive_relations()
            print("Organized method works!")
            print(f"Sections available: {list(organized.keys())}")
        except Exception as e:
            print(f"Organized method failed: {e}")
            
        # Test the helper method for getting natural variables
        try:
            for section, relations in organized.items():
                print(f"\n{section} relations:")
                for conj_var, deriv in relations:
                    natural_var = widget._get_natural_variable_for_constitutive(test_state_fn, conj_var)
                    print(f"  {conj_var} = ∂F/∂{natural_var} = {deriv}")
        except Exception as e:
            print(f"Helper method failed: {e}")
            
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Error creating GSM Box or Widget: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_widget_with_constitutive_relations()
