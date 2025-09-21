#!/usr/bin/env python3
"""
GSM Thermodynamic Framework Test Runner

This script runs comprehensive tests for the GSM thermodynamic framework and
reports on their success or failure. It focuses on the GSMThermoDynBox
implementation.

Usage:
    python run_gsm_tests.py [--verbose]
    
Options:
    --verbose: Enable detailed test output
"""

import sys
import argparse
import subprocess
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add the core2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def run_pytest_command(test_file: Path, verbose: bool = False) -> Tuple[int, str, str]:
    """
    Run pytest on a specific test file and return results.
    
    Args:
        test_file: Path to the test file
        verbose: Whether to use verbose output
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = ["python", "-m", "pytest", str(test_file)]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.append("-q")
    
    # Run ALL tests without marker filtering
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=test_file.parent
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Test execution timed out after 5 minutes"
    except Exception as e:
        return -1, "", f"Error running tests: {str(e)}"


def run_integration_tests(test_file: Path, verbose: bool = False) -> Tuple[int, str, str]:
    """
    Run integration tests separately.
    
    Args:
        test_file: Path to the test file
        verbose: Whether to use verbose output
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = ["python", "-m", "pytest", str(test_file), "-m", "integration"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.append("-q")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for integration tests
            cwd=test_file.parent
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Integration test execution timed out after 10 minutes"
    except Exception as e:
        return -1, "", f"Error running integration tests: {str(e)}"


def parse_test_results(stdout: str, stderr: str) -> Dict[str, any]:
    """
    Parse pytest output to extract test results summary.
    
    Args:
        stdout: Standard output from pytest
        stderr: Standard error from pytest
        
    Returns:
        Dictionary with parsed results
    """
    combined_output = stdout + "\n" + stderr
    
    results = {
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'skipped': 0,
        'warnings': 0,
        'duration': 0.0,
        'failures': [],
        'errors_list': []
    }
    
    # Use regex to find the summary line like "22 passed, 4 warnings in 13.47s"
    # Pattern matches lines with equals signs or test count summaries
    summary_pattern = r'(?:=+\s*)?(\d+)\s+passed(?:,\s*(\d+)\s+failed)?(?:,\s*(\d+)\s+errors?)?(?:,\s*(\d+)\s+skipped)?(?:,\s*(\d+)\s+warnings?)?\s+in\s+([\d.]+)s'
    
    match = re.search(summary_pattern, combined_output, re.IGNORECASE)
    if match:
        results['passed'] = int(match.group(1)) if match.group(1) else 0
        results['failed'] = int(match.group(2)) if match.group(2) else 0
        results['errors'] = int(match.group(3)) if match.group(3) else 0
        results['skipped'] = int(match.group(4)) if match.group(4) else 0
        results['warnings'] = int(match.group(5)) if match.group(5) else 0
        results['duration'] = float(match.group(6)) if match.group(6) else 0.0
    
    # Look for failure and error details
    for line in combined_output.split('\n'):
        if 'FAILED' in line and '::' in line:
            results['failures'].append(line.strip())
        elif 'ERROR' in line and '::' in line:
            results['errors_list'].append(line.strip())
    
    return results
    
    return results


def print_test_summary(module_name: str, results: Dict[str, any], integration_results: Optional[Dict[str, any]] = None):
    """
    Print a formatted test summary.
    
    Args:
        module_name: Name of the tested module
        results: Regular test results
        integration_results: Integration test results (optional)
    """
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {module_name}")
    print(f"{'='*60}")
    
    total_tests = results['passed'] + results['failed'] + results['errors']
    success_rate = (results['passed'] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Regular Tests:")
    print(f"  ✓ Passed:  {results['passed']}")
    print(f"  ✗ Failed:  {results['failed']}")
    print(f"  ⚠ Errors:  {results['errors']}")
    print(f"  ⊝ Skipped: {results['skipped']}")
    if results['warnings'] > 0:
        print(f"  ⚡ Warnings: {results['warnings']}")
    print(f"  Duration: {results['duration']:.2f}s")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    if integration_results:
        int_total = integration_results['passed'] + integration_results['failed'] + integration_results['errors']
        int_success_rate = (integration_results['passed'] / int_total * 100) if int_total > 0 else 0
        
        print(f"\nIntegration Tests:")
        print(f"  ✓ Passed:  {integration_results['passed']}")
        print(f"  ✗ Failed:  {integration_results['failed']}")
        print(f"  ⚠ Errors:  {integration_results['errors']}")
        print(f"  Duration: {integration_results['duration']:.2f}s")
        print(f"  Success Rate: {int_success_rate:.1f}%")
        
        # Overall statistics
        overall_passed = results['passed'] + integration_results['passed']
        overall_total = total_tests + int_total
        overall_success_rate = (overall_passed / overall_total * 100) if overall_total > 0 else 0
        
        print(f"\nOverall:")
        print(f"  Total Tests: {overall_total}")
        print(f"  Overall Success Rate: {overall_success_rate:.1f}%")
    
    # Print failures if any
    if results['failures'] or results['errors_list']:
        print(f"\nFAILURES/ERRORS:")
        for failure in results['failures'][:5]:  # Show first 5 failures
            print(f"  {failure}")
        for error in results['errors_list'][:5]:  # Show first 5 errors
            print(f"  {error}")
        if len(results['failures']) + len(results['errors_list']) > 10:
            remaining = len(results['failures']) + len(results['errors_list']) - 10
            print(f"  ... and {remaining} more failures/errors")
    
    # Status indicator
    overall_failed = results['failed'] + results['errors']
    if integration_results:
        overall_failed += integration_results['failed'] + integration_results['errors']
    
    if overall_failed == 0:
        status = "✅ ALL TESTS PASSED"
        print(f"\n{status}")
    else:
        status = f"❌ {overall_failed} TEST(S) FAILED"
        print(f"\n{status}")
    
    return overall_failed == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run GSM thermodynamic framework tests",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Enable verbose test output'
    )
    parser.add_argument(
        '--integration', 
        action='store_true',
        help='Run integration tests only'
    )
    parser.add_argument(
        '--no-integration',
        action='store_true',
        help='Skip integration tests'
    )
    
    args = parser.parse_args()
    
    # Find test files - use the actual test file with tests
    tests_dir = Path(__file__).parent / "tests"
    test_files = {
        'gsm_thermodyn_box': tests_dir / "test_gsm_thermodyn_box.py"  # This is the file with actual tests
    }
    
    # Check that test files exist
    missing_files = []
    for name, path in test_files.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("❌ Missing test files:")
        for missing in missing_files:
            print(f"  {missing}")
        return 1
    
    print("🧪 GSM Thermodynamic Framework Test Runner")
    print(f"Running tests for: {', '.join(test_files.keys())}")
    print(f"Verbose mode: {'ON' if args.verbose else 'OFF'}")
    
    all_passed = True
    start_time = time.time()
    
    # Run tests for each module
    for module_name, test_file in test_files.items():
        print(f"\n🔬 Testing {module_name}...")
        
        if args.integration:
            # Run only integration tests
            print("Running integration tests...")
            return_code, stdout, stderr = run_integration_tests(test_file, args.verbose)
            results = parse_test_results(stdout, stderr)
            module_passed = print_test_summary(module_name, results)
        else:
            # Run regular tests
            print("Running unit tests...")
            return_code, stdout, stderr = run_pytest_command(test_file, args.verbose)
            results = parse_test_results(stdout, stderr)
            
            integration_results = None
            if not args.no_integration:
                # Run integration tests
                print("Running integration tests...")
                int_return_code, int_stdout, int_stderr = run_integration_tests(test_file, args.verbose)
                integration_results = parse_test_results(int_stdout, int_stderr)
            
            module_passed = print_test_summary(module_name, results, integration_results)
        
        if not module_passed:
            all_passed = False
        
        # Print detailed output if verbose
        if args.verbose:
            print(f"\nDETAILED OUTPUT for {module_name}:")
            print("-" * 40)
            print("STDOUT:")
            print(stdout)
            if stderr:
                print("\nSTDERR:")
                print(stderr)
    
    total_duration = time.time() - start_time
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Modules tested: {len(test_files)}")
    print(f"Total duration: {total_duration:.2f}s")
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! The GSM thermodynamic framework is working correctly.")
        return 0
    else:
        print("💥 SOME TESTS FAILED! Please review the failures above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
