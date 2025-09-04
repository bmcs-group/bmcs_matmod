#!/usr/bin/env python3
"""
Simple test for GSMSymbBoxWidget2 cached properties.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, '.')

print("Testing imports...")

try:
    from bmcs_matmod.gsm_lagrange.core2.gsm_symb_box import GSMSymbBox
    print("✓ GSMSymbBox imported successfully")
    
    from bmcs_matmod.gsm_lagrange.core2.gsm_symb_box_widget import GSMSymbBoxWidget
    print("✓ GSMSymbBoxWidget2 imported successfully")
    
    print("Creating GSMSymbBox...")
    gsm_box = GSMSymbBox(state_vars=['rho', 'tau'], external_vars=['Beta'])
    print("✓ GSMSymbBox created successfully")
    
    print("Creating widget...")
    widget = GSMSymbBoxWidget(gsm_box, figsize=(6, 6))
    print("✓ Widget created successfully")
    
    print("Testing cached properties (this will create cache files)...")
    
    # Test one cached property at a time
    print("Testing F_image...")
    f_img = widget.F_image
    print(f"✓ F_image: {len(f_img)} bytes")
    
    print("All tests completed successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
