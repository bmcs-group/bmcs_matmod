#!/usr/bin/env python3
"""
Test runner for GSM Thermodynamic State Function Box

This script provides a convenient way to run GSM tests with different
configurations and reporting options. It focuses on the GSMThermodynBox2
implementation.

Usage:
    python run_tests.py                     # Run all tests
    python run_tests.py --fast              # Skip integration tests
    python run_tests.py --roundtrip-only    # Only round-trip tests
    python run_tests.py --constitutive-only # Only constitutive relation tests
    python run_tests.py --verbose           # Verbose output
    python run_tests.py --coverage          # Run with coverage report
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_tests(test_args=None, coverage=False, verbose=False, module=None):
    """Run the test suite with specified arguments."""
    # Base command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add test directory or specific test file
    test_dir = Path(__file__).parent / "tests"
    
    if module:
        # Run specific module tests - only gsm_thermodyn_box2 is supported
        if module == "gsm_thermodyn_box2":
            test_file = test_dir / "test_gsm_thermodyn_box2.py"
        else:
            raise ValueError(f"Unknown module: {module}. Available: gsm_thermodyn_box2")
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        cmd.append(str(test_file))
    else:
        # Run all tests in test directory
        cmd.append(str(test_dir))
    
    # Add coverage if requested
    if coverage:
        coverage_modules = ["gsm_thermodyn_box2"]
        if module:
            coverage_modules = [module]
        
        for cov_module in coverage_modules:
            cmd.extend(["--cov", cov_module])
        cmd.extend(["--cov-report=html", "--cov-report=term"])
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add custom test arguments
    if test_args:
        cmd.extend(test_args)
    
    module_info = f" ({module})" if module else ""
    print(f"Running GSM tests{module_info}: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, timeout=300)
        return result.returncode
    except subprocess.TimeoutExpired:
        print("\n\nTest run timed out after 5 minutes")
        return 1
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run GSM thermodynamic framework tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip slow/integration tests"
    )
    parser.add_argument(
        "--roundtrip-only", action="store_true",
        help="Only run round-trip consistency tests"
    )
    parser.add_argument(
        "--constitutive-only", action="store_true",
        help="Only run constitutive relation tests"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="Run with coverage report"
    )
    parser.add_argument(
        "--module", "-m",
        choices=["gsm_thermodyn_box2"],
        help="Run tests for GSM ThermodynBox2 module only"
    )
    parser.add_argument(
        "--markers", action="store_true",
        help="List available test markers"
    )
    
    args = parser.parse_args()
    
    if args.markers:
        print("Available test markers:")
        print("  integration: Integration tests (slower, comprehensive)")
        print("  roundtrip: Round-trip consistency tests") 
        print("  constitutive: Constitutive relation tests")
        print("  slow: Slow tests")
        print("\nAvailable modules:")
        print("  gsm_thermodyn_box2: Interface-based thermodynamic box implementation")
        print("\nExample usage:")
        print("  python run_tests.py --fast  # Skip integration tests")
        print("  python run_tests.py --module gsm_thermodyn_box2  # Test implementation")
        print("  pytest -m roundtrip         # Only round-trip tests")
        print("  pytest -m 'not slow'        # Skip slow tests")
        return 0
    
    # Build test arguments
    test_args = []
    
    if args.fast:
        test_args.extend(["-m", "not integration and not slow"])
    
    if args.roundtrip_only:
        test_args.extend(["-k", "round_trip"])
    
    if args.constitutive_only:
        test_args.extend(["-k", "constitutive"])
    
    # Run tests
    try:
        return_code = run_tests(
            test_args=test_args,
            coverage=args.coverage,
            verbose=args.verbose,
            module=args.module
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    
    if return_code == 0:
        module_info = f" for {args.module}" if args.module else ""
        print(f"\n✓ All tests passed{module_info}!")
    else:
        module_info = f" for {args.module}" if args.module else ""
        print(f"\n✗ Tests failed{module_info} with return code {return_code}")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())
