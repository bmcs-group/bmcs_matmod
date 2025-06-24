#!/usr/bin/env python3
"""Minimal test script to isolate CLI import issues"""

import sys
import os

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(__file__))

print("Testing imports step by step...")

try:
    print("1. Importing data_structures...")
    import data_structures
    print("   ✅ data_structures OK")
except Exception as e:
    print(f"   ❌ data_structures failed: {e}")
    sys.exit(1)

try:
    print("2. Importing gsm_def_registry...")
    import gsm_def_registry
    print("   ✅ gsm_def_registry OK")
except Exception as e:
    print(f"   ❌ gsm_def_registry failed: {e}")
    sys.exit(1)

try:
    print("3. Importing parameter_loader...")
    import parameter_loader
    print("   ✅ parameter_loader OK")
except Exception as e:
    print(f"   ❌ parameter_loader failed: {e}")
    sys.exit(1)

try:
    print("4. Importing response_data...")
    import response_data
    print("   ✅ response_data OK")
except Exception as e:
    print(f"   ❌ response_data failed: {e}")
    sys.exit(1)

print("5. Testing GSMParameterSpec class...")
try:
    # Import the specific parts step by step
    from cli_gsm import GSMParameterSpec
    print("   ✅ GSMParameterSpec imported OK")
except Exception as e:
    print(f"   ❌ GSMParameterSpec failed: {e}")
    sys.exit(1)

print("6. Testing GSMDefCLI class...")
try:
    from cli_gsm import GSMDefCLI
    print("   ✅ GSMDefCLI imported OK")
except Exception as e:
    print(f"   ❌ GSMDefCLI failed: {e}")
    sys.exit(1)

print("7. All imports successful! Testing basic CLI functionality...")
try:
    cli = GSMDefCLI()
    print("   ✅ GSMDefCLI instantiated OK")
except Exception as e:
    print(f"   ❌ GSMDefCLI instantiation failed: {e}")
    sys.exit(1)

print("8. Testing --help...")
try:
    # Mock sys.argv to test help
    import sys
    old_argv = sys.argv
    sys.argv = ['cli_gsm.py', '--help']
    
    # This should show help and exit
    cli.main()
    
except SystemExit as e:
    print(f"   ✅ Help command completed (exit code: {e.code})")
    sys.argv = old_argv
except Exception as e:
    print(f"   ❌ Help command failed: {e}")
    sys.argv = old_argv

print("✅ All tests completed successfully!")
