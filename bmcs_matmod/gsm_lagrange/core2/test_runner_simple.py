#!/usr/bin/env python3
"""
Simple test execution script for GSMThermodynBox2
"""

import subprocess
import sys
from pathlib import Path

def run_simple_test():
    """Run a simple test to verify the setup works."""
    test_dir = Path(__file__).parent / "tests"
    test_file = test_dir / "test_gsm_thermodyn_box2.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return 1
    
    print(f"📁 Test file found: {test_file}")
    
    # Try to run just the collection to see if tests can be discovered
    cmd = [sys.executable, "-m", "pytest", str(test_file), "--collect-only", "-q"]
    
    try:
        print("🔍 Discovering tests...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Tests discovered successfully!")
            print("📋 Discovered tests:")
            print(result.stdout)
            
            # Now try to run one simple test
            print("🧪 Running one simple test...")
            cmd_run = [sys.executable, "-m", "pytest", str(test_file) + "::TestGSMThermodynBox2FrameworkValidation::test_simple_elastic_material_known_results", "-v"]
            result_run = subprocess.run(cmd_run, capture_output=True, text=True, timeout=60, cwd=Path(__file__).parent)
            
            if result_run.returncode == 0:
                print("✅ Simple test passed!")
                print(result_run.stdout)
                return 0
            else:
                print("❌ Simple test failed:")
                print("STDOUT:", result_run.stdout)
                print("STDERR:", result_run.stderr)
                return 1
        else:
            print("❌ Test discovery failed:")
            print("STDOUT:", result.stdout)  
            print("STDERR:", result.stderr)
            return 1
            
    except subprocess.TimeoutExpired:
        print("❌ Test execution timed out")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    print("=== Simple GSMThermodynBox2 Test Runner ===")
    sys.exit(run_simple_test())
