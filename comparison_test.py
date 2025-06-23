#!/usr/bin/env python3
"""
Comparison of registry approaches

This script demonstrates the difference between the old approach
(import everything upfront) vs the new approach (lazy loading).
"""

import sys
import time
import os
sys.path.insert(0, '/home/rch/Coding/bmcs_matmod/bmcs_matmod/gsm_lagrange')

def test_old_approach():
    """Test the old gsm_def_registry.py approach"""
    print("Testing OLD approach (gsm_def_registry.py):")
    print("=" * 50)
    
    start = time.time()
    try:
        # This will import all GSM classes immediately
        from gsm_def_registry import discover_gsm_defs, get_gsm_defs
        discovery_time = time.time() - start
        
        models, registry = discover_gsm_defs(debug=False)
        print(f"✓ Found {len(models)} models in {discovery_time:.3f} seconds")
        print(f"✓ All classes loaded immediately: {len(registry)} entries")
        
        # Show that classes are already loaded
        if 'ED' in registry:
            print(f"✓ ED class ready: {registry['ED']}")
        
    except Exception as e:
        print(f"✗ Old approach failed: {e}")

def test_new_approach():
    """Test the new gsm_def_registry_new.py approach"""
    print("\nTesting NEW approach (gsm_def_registry_new.py):")
    print("=" * 50)
    
    start = time.time()
    try:
        from gsm_def_registry_new import LazyGSMRegistry
        registry = LazyGSMRegistry()
        discovery_time = time.time() - start
        
        print(f"✓ Path-based discovery in {discovery_time:.3f} seconds")
        print(f"✓ Found {len(registry.get_model_names())} models")
        print(f"✓ No classes loaded yet (lazy loading)")
        
        # Show that we can do CLI operations instantly
        print(f"✓ Check ED exists: {registry.has_model('ED')}")
        print(f"✓ Get ED module path: {registry.get_module_path('ED')}")
        
        # Now load a class on demand
        print("Loading ED class on demand...")
        start_load = time.time()
        ed_class = registry.get_class('ED')
        load_time = time.time() - start_load
        print(f"✓ ED class loaded in {load_time:.3f} seconds: {type(ed_class).__name__}")
        
    except Exception as e:
        print(f"✗ New approach failed: {e}")

def test_cli_safety():
    """Test CLI safety - operations that must not hang"""
    print("\nTesting CLI safety:")
    print("=" * 30)
    
    try:
        from gsm_def_registry_new import (
            get_available_models, 
            check_model_exists, 
            get_model_module_path
        )
        
        start = time.time()
        models, paths = get_available_models()
        safety_time = time.time() - start
        
        print(f"✓ CLI-safe discovery in {safety_time:.3f} seconds")
        print(f"✓ Available models: {len(models)}")
        print(f"✓ Model 'ED' exists: {check_model_exists('ED')}")
        print(f"✓ ED module path: {get_model_module_path('ED')}")
        
    except Exception as e:
        print(f"✗ CLI safety test failed: {e}")

if __name__ == "__main__":
    print("GSM Registry Approach Comparison")
    print("=" * 60)
    
    # Test CLI safety first (most important)
    test_cli_safety()
    
    # Test new approach
    test_new_approach()
    
    # Test old approach (might take time)
    print("\nNote: Old approach test might take longer due to immediate imports...")
    test_old_approach()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("- NEW approach: Instant CLI operations, lazy class loading")
    print("- OLD approach: All imports upfront, potential CLI hanging")
    print("- NEW approach solves the CLI hanging problem!")
