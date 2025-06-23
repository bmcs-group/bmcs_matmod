#!/usr/bin/env python3
"""Simple test for the registry"""

import sys
import os

# Add the project to path
sys.path.insert(0, '/home/rch/Coding/bmcs_matmod')

print("Current working directory:", os.getcwd())
print("Python path includes project:", '/home/rch/Coding/bmcs_matmod' in sys.path)

try:
    print("\nStep 1: Testing basic directory scanning...")
    gsm_dir = '/home/rch/Coding/bmcs_matmod/bmcs_matmod/gsm_lagrange'
    gsm_files = [f for f in os.listdir(gsm_dir) if f.endswith('.py') and f.startswith('gsm1d_')]
    print(f"Found {len(gsm_files)} GSM files: {gsm_files}")
    
    print("\nStep 2: Testing registry import...")
    from bmcs_matmod.gsm_lagrange.gsm_def_registry import get_gsm_defs, discover_gsm_defs
    print("✓ Successfully imported registry functions")
    
    print("\nStep 3: Testing registry discovery with debug...")
    registry = get_gsm_defs(debug=True)
    print(f"✓ Registry contains {len(set(registry.values()))} unique models")
    print(f"✓ Registry has {len(registry)} access keys")
    
    # Show some keys
    if registry:
        keys = sorted(registry.keys())
        print(f"\nFirst 10 keys: {keys[:10]}")
        
        # Test accessing a model
        test_key = keys[0]
        model_class = registry[test_key]
        print(f"✓ Successfully accessed model '{test_key}': {model_class}")
    else:
        print("⚠ Registry is empty!")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
