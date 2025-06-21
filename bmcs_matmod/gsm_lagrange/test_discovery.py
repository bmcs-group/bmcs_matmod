#!/usr/bin/env python3
"""Simple test of model discovery"""

import os
import sys
from pathlib import Path

# Add proper paths
sys.path.insert(0, '/home/rch/Coding/bmcs_matmod')

def test_single_model_import():
    """Test importing a single GSM model"""
    try:
        from bmcs_matmod.gsm_lagrange.gsm1d_ed import GSM1D_ED
        print(f"Successfully imported GSM1D_ED")
        print(f"  Class: {GSM1D_ED}")
        print(f"  Has F_engine: {hasattr(GSM1D_ED, 'F_engine')}")
        
        # Try to create instance
        instance = GSM1D_ED()
        print(f"  Instance created: {instance}")
        return True
    except Exception as e:
        print(f"Failed to import GSM1D_ED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_discovery():
    """Test the model discovery"""
    try:
        from bmcs_matmod.gsm_lagrange.model_registry import GSMModelRegistry
        
        registry = GSMModelRegistry()
        print(f"Registry created with {len(registry.models)} models")
        
        if registry.models:
            print("Found models:")
            for key, model_info in registry.models.items():
                print(f"  {key}: {model_info.name}")
        else:
            print("No models found - debugging discovery...")
            
            # Manual test of discovery logic
            current_dir = Path('/home/rch/Coding/bmcs_matmod/bmcs_matmod/gsm_lagrange')
            model_files = list(current_dir.glob('gsm1d_*.py'))
            print(f"Found {len(model_files)} GSM1D files:")
            for f in model_files:
                print(f"  {f.name}")
        
    except Exception as e:
        print(f"Error in model discovery: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Testing Single Model Import ===")
    test_single_model_import()
    
    print("\n=== Testing Model Discovery ===")
    test_model_discovery()
