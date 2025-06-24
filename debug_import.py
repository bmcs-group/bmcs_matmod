#!/usr/bin/env python3
"""
Debug test for gsm_lagrange imports
"""

import sys
import time

def test_import_with_timeout(import_statement, timeout=3):
    """Test an import with a timeout"""
    print(f"Testing: {import_statement}")
    start_time = time.time()
    
    try:
        exec(import_statement)
        elapsed = time.time() - start_time
        print(f"  ✅ Success in {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ❌ Failed in {elapsed:.2f}s: {e}")
        return False

if __name__ == "__main__":
    print("=== GSM Lagrange Import Debug Test ===\n")
    
    # Test individual files
    tests = [
        "import bmcs_matmod.gsm_lagrange",
        "from bmcs_matmod.gsm_lagrange.core import gsm_vars",
        "from bmcs_matmod.gsm_lagrange.core.gsm_vars import Scalar",
        "from bmcs_matmod.gsm_lagrange.core import response_data",
        "from bmcs_matmod.gsm_lagrange.core import Scalar",
    ]
    
    for test in tests:
        test_import_with_timeout(test)
        print()
