#!/usr/bin/env python3
"""
Comprehensive Parameter Validation Examples for GSM CLI

This script demonstrates various parameter validation scenarios,
including boundary testing, type validation, and error handling.
"""

import json
import subprocess
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class ValidationTestCase:
    """Represents a parameter validation test case"""
    name: str
    model: str
    formulation: str
    parameters: Dict[str, Any]
    loading: Dict[str, Any]
    expected_valid: bool
    description: str

class GSMParameterValidator:
    """Comprehensive parameter validation for GSM models"""
    
    def __init__(self, cli_path: str = "../cli_gsm.py"):
        self.cli_path = cli_path
        self.test_results = []
    
    def validate_parameters(self, test_case: ValidationTestCase) -> Dict[str, Any]:
        """Validate parameters for a test case"""
        params_json = json.dumps(test_case.parameters)
        loading_json = json.dumps(test_case.loading)
        
        cmd = [
            "python", self.cli_path,
            "--model", test_case.model,
            "--formulation", test_case.formulation,
            "--params-inline", params_json,
            "--loading-inline", loading_json,
            "--validate-only",
            "--json-output"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                validation_result = json.loads(result.stdout)
                return {
                    "status": "success",
                    "validation": validation_result,
                    "test_case": test_case.name
                }
            else:
                return {
                    "status": "error",
                    "error": result.stderr,
                    "test_case": test_case.name
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "Validation timeout",
                "test_case": test_case.name
            }
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"JSON decode error: {e}",
                "stdout": result.stdout,
                "test_case": test_case.name
            }
    
    def run_validation_suite(self, test_cases: List[ValidationTestCase]) -> Dict[str, Any]:
        """Run a complete validation test suite"""
        print("GSM Parameter Validation Suite")
        print("=" * 50)
        
        results = []
        passed = 0
        failed = 0
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{i+1:2d}. {test_case.name}")
            print(f"    {test_case.description}")
            print(f"    Expected: {'VALID' if test_case.expected_valid else 'INVALID'}")
            
            result = self.validate_parameters(test_case)
            results.append(result)
            
            if result["status"] == "success":
                validation = result["validation"]
                if validation["status"] == "validation_complete":
                    actual_valid = validation["valid"]
                    
                    if actual_valid == test_case.expected_valid:
                        print(f"    Result: ✅ PASS - {'Valid' if actual_valid else 'Invalid'} as expected")
                        passed += 1
                    else:
                        print(f"    Result: ❌ FAIL - Expected {'valid' if test_case.expected_valid else 'invalid'}, got {'valid' if actual_valid else 'invalid'}")
                        if not actual_valid and "errors" in validation:
                            print(f"    Errors: {validation['errors']}")
                        failed += 1
                else:
                    print(f"    Result: ❌ FAIL - Validation error: {validation.get('error', 'Unknown')}")
                    failed += 1
            else:
                print(f"    Result: ❌ FAIL - Execution error: {result['error']}")
                failed += 1
        
        # Summary
        total = len(test_cases)
        success_rate = passed / total if total > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"Validation Suite Summary")
        print(f"{'='*50}")
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success rate: {success_rate:.1%}")
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate": success_rate,
            "results": results
        }


def create_standard_loading() -> Dict[str, Any]:
    """Create standard loading scenario for validation tests"""
    return {
        "time_array": [0.0, 0.5, 1.0],
        "strain_history": [0.0, 0.005, 0.01],
        "loading_type": "strain_controlled"
    }


def create_gsm1d_ed_validation_tests() -> List[ValidationTestCase]:
    """Create validation test cases for GSM1D_ED model"""
    
    standard_loading = create_standard_loading()
    
    test_cases = [
        # Valid parameter sets
        ValidationTestCase(
            name="GSM1D_ED_Valid_Basic",
            model="GSM1D_ED",
            formulation="F",
            parameters={"E": 30000.0, "S": 1.0, "c": 2.0, "r": 0.5, "eps_0": 0.001},
            loading=standard_loading,
            expected_valid=True,
            description="Basic valid parameters within normal ranges"
        ),
        
        ValidationTestCase(
            name="GSM1D_ED_Valid_HighStrength",
            model="GSM1D_ED",
            formulation="F",
            parameters={"E": 45000.0, "S": 2.5, "c": 3.5, "r": 0.3, "eps_0": 0.0005},
            loading=standard_loading,
            expected_valid=True,
            description="High strength concrete parameters"
        ),
        
        ValidationTestCase(
            name="GSM1D_ED_Valid_LowStrength",
            model="GSM1D_ED",
            formulation="F",
            parameters={"E": 20000.0, "S": 0.5, "c": 1.0, "r": 0.7, "eps_0": 0.002},
            loading=standard_loading,
            expected_valid=True,
            description="Low strength concrete parameters"
        ),
        
        # Invalid parameter sets - negative values
        ValidationTestCase(
            name="GSM1D_ED_Invalid_NegativeE",
            model="GSM1D_ED",
            formulation="F",
            parameters={"E": -30000.0, "S": 1.0, "c": 2.0, "r": 0.5, "eps_0": 0.001},
            loading=standard_loading,
            expected_valid=False,
            description="Negative elastic modulus (physically impossible)"
        ),
        
        ValidationTestCase(
            name="GSM1D_ED_Invalid_NegativeS",
            model="GSM1D_ED",
            formulation="F",
            parameters={"E": 30000.0, "S": -1.0, "c": 2.0, "r": 0.5, "eps_0": 0.001},
            loading=standard_loading,
            expected_valid=False,
            description="Negative damage parameter S"
        ),
        
        # Invalid parameter sets - extreme values
        ValidationTestCase(
            name="GSM1D_ED_Invalid_ExtremeE",
            model="GSM1D_ED",
            formulation="F",
            parameters={"E": 1000000.0, "S": 1.0, "c": 2.0, "r": 0.5, "eps_0": 0.001},
            loading=standard_loading,
            expected_valid=False,
            description="Extremely high elastic modulus (unrealistic)"
        ),
        
        ValidationTestCase(
            name="GSM1D_ED_Invalid_PoissonRatio",
            model="GSM1D_ED",
            formulation="F",
            parameters={"E": 30000.0, "S": 1.0, "c": 2.0, "r": 0.5, "eps_0": 0.001, "nu": 0.6},
            loading=standard_loading,
            expected_valid=False,
            description="Poisson's ratio > 0.5 (violates thermodynamic constraints)"
        ),
        
        # Missing parameters
        ValidationTestCase(
            name="GSM1D_ED_Invalid_MissingE",
            model="GSM1D_ED",
            formulation="F",
            parameters={"S": 1.0, "c": 2.0, "r": 0.5, "eps_0": 0.001},
            loading=standard_loading,
            expected_valid=False,
            description="Missing elastic modulus E"
        ),
        
        ValidationTestCase(
            name="GSM1D_ED_Invalid_MissingMultiple",
            model="GSM1D_ED",
            formulation="F",
            parameters={"E": 30000.0},
            loading=standard_loading,
            expected_valid=False,
            description="Missing multiple required parameters"
        ),
        
        # Boundary value tests
        ValidationTestCase(
            name="GSM1D_ED_Boundary_MinE",
            model="GSM1D_ED",
            formulation="F",
            parameters={"E": 1000.0, "S": 1.0, "c": 2.0, "r": 0.5, "eps_0": 0.001},
            loading=standard_loading,
            expected_valid=True,
            description="Minimum realistic elastic modulus"
        ),
        
        ValidationTestCase(
            name="GSM1D_ED_Boundary_MaxE",
            model="GSM1D_ED",
            formulation="F",
            parameters={"E": 100000.0, "S": 1.0, "c": 2.0, "r": 0.5, "eps_0": 0.001},
            loading=standard_loading,
            expected_valid=True,
            description="Maximum realistic elastic modulus"
        ),
    ]
    
    return test_cases


def create_loading_validation_tests() -> List[ValidationTestCase]:
    """Create validation test cases for loading scenarios"""
    
    base_params = {"E": 30000.0, "S": 1.0, "c": 2.0, "r": 0.5, "eps_0": 0.001}
    
    test_cases = [
        # Valid loading scenarios
        ValidationTestCase(
            name="Loading_Valid_Monotonic",
            model="GSM1D_ED",
            formulation="F",
            parameters=base_params,
            loading={
                "time_array": [0.0, 0.5, 1.0],
                "strain_history": [0.0, 0.005, 0.01],
                "loading_type": "strain_controlled"
            },
            expected_valid=True,
            description="Valid monotonic loading"
        ),
        
        ValidationTestCase(
            name="Loading_Valid_Cyclic",
            model="GSM1D_ED",
            formulation="F",
            parameters=base_params,
            loading={
                "time_array": [0.0, 0.25, 0.5, 0.75, 1.0],
                "strain_history": [0.0, 0.005, 0.0, -0.003, 0.0],
                "loading_type": "strain_controlled"
            },
            expected_valid=True,
            description="Valid cyclic loading"
        ),
        
        # Invalid loading scenarios
        ValidationTestCase(
            name="Loading_Invalid_MismatchedArrays",
            model="GSM1D_ED",
            formulation="F",
            parameters=base_params,
            loading={
                "time_array": [0.0, 0.5, 1.0],
                "strain_history": [0.0, 0.005],  # Mismatched length
                "loading_type": "strain_controlled"
            },
            expected_valid=False,
            description="Mismatched time and strain array lengths"
        ),
        
        ValidationTestCase(
            name="Loading_Invalid_NonMonotonicTime",
            model="GSM1D_ED",
            formulation="F",
            parameters=base_params,
            loading={
                "time_array": [0.0, 1.0, 0.5],  # Non-monotonic
                "strain_history": [0.0, 0.005, 0.01],
                "loading_type": "strain_controlled"
            },
            expected_valid=False,
            description="Non-monotonic time array"
        ),
        
        ValidationTestCase(
            name="Loading_Invalid_NegativeTime",
            model="GSM1D_ED",
            formulation="F",
            parameters=base_params,
            loading={
                "time_array": [-0.5, 0.0, 0.5],  # Negative time
                "strain_history": [0.0, 0.005, 0.01],
                "loading_type": "strain_controlled"
            },
            expected_valid=False,
            description="Negative time values"
        ),
        
        ValidationTestCase(
            name="Loading_Invalid_EmptyArrays",
            model="GSM1D_ED",
            formulation="F",
            parameters=base_params,
            loading={
                "time_array": [],
                "strain_history": [],
                "loading_type": "strain_controlled"
            },
            expected_valid=False,
            description="Empty time and strain arrays"
        ),
    ]
    
    return test_cases


def create_formulation_validation_tests() -> List[ValidationTestCase]:
    """Create validation test cases for different formulations"""
    
    base_params = {"E": 30000.0, "S": 1.0, "c": 2.0, "r": 0.5, "eps_0": 0.001}
    strain_loading = create_standard_loading()
    stress_loading = {
        "time_array": [0.0, 0.5, 1.0],
        "stress_history": [0.0, 15.0, 30.0],
        "loading_type": "stress_controlled"
    }
    
    test_cases = [
        # Valid formulations
        ValidationTestCase(
            name="Formulation_Valid_F_StrainControl",
            model="GSM1D_ED",
            formulation="F",
            parameters=base_params,
            loading=strain_loading,
            expected_valid=True,
            description="F formulation with strain-controlled loading"
        ),
        
        ValidationTestCase(
            name="Formulation_Valid_G_StressControl",
            model="GSM1D_ED",
            formulation="G",
            parameters=base_params,
            loading=stress_loading,
            expected_valid=True,
            description="G formulation with stress-controlled loading"
        ),
        
        ValidationTestCase(
            name="Formulation_Valid_Helmholtz_StrainControl",
            model="GSM1D_ED",
            formulation="Helmholtz",
            parameters=base_params,
            loading=strain_loading,
            expected_valid=True,
            description="Helmholtz formulation with strain-controlled loading"
        ),
        
        ValidationTestCase(
            name="Formulation_Valid_Gibbs_StressControl",
            model="GSM1D_ED",
            formulation="Gibbs",
            parameters=base_params,
            loading=stress_loading,
            expected_valid=True,
            description="Gibbs formulation with stress-controlled loading"
        ),
        
        # Potentially problematic combinations
        ValidationTestCase(
            name="Formulation_Questionable_F_StressControl",
            model="GSM1D_ED",
            formulation="F",
            parameters=base_params,
            loading=stress_loading,
            expected_valid=True,  # Should work but may not be optimal
            description="F formulation with stress-controlled loading (may be suboptimal)"
        ),
    ]
    
    return test_cases


def run_comprehensive_validation_demo():
    """Run comprehensive validation demonstration"""
    print("GSM CLI Comprehensive Parameter Validation Demo")
    print("=" * 70)
    
    validator = GSMParameterValidator()
    
    # Run different validation test suites
    test_suites = [
        ("GSM1D_ED Parameter Validation", create_gsm1d_ed_validation_tests()),
        ("Loading Scenario Validation", create_loading_validation_tests()),
        ("Formulation Validation", create_formulation_validation_tests()),
    ]
    
    all_results = {}
    
    for suite_name, test_cases in test_suites:
        print(f"\n{suite_name}")
        print("=" * len(suite_name))
        
        suite_results = validator.run_validation_suite(test_cases)
        all_results[suite_name] = suite_results
        
        # Save individual suite results
        filename = f"{suite_name.lower().replace(' ', '_')}_results.json"
        with open(filename, 'w') as f:
            json.dump(suite_results, f, indent=2, default=str)
        print(f"Results saved to {filename}")
    
    # Overall summary
    total_tests = sum(suite["total_tests"] for suite in all_results.values())
    total_passed = sum(suite["passed"] for suite in all_results.values())
    overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"OVERALL VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total test suites: {len(test_suites)}")
    print(f"Total tests: {total_tests}")
    print(f"Total passed: {total_passed}")
    print(f"Overall success rate: {overall_success_rate:.1%}")
    
    # Suite breakdown
    print(f"\nSuite Breakdown:")
    for suite_name, results in all_results.items():
        rate = results["success_rate"]
        print(f"  {suite_name}: {results['passed']}/{results['total_tests']} ({rate:.1%})")
    
    # Save comprehensive results
    with open("comprehensive_validation_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nComprehensive results saved to: comprehensive_validation_results.json")
    
    return all_results


def main():
    """Run validation demonstration"""
    try:
        results = run_comprehensive_validation_demo()
        
        print("\n" + "=" * 70)
        print("Validation Demo Completed Successfully!")
        print("\nThis demo demonstrated:")
        print("  - Parameter boundary testing")
        print("  - Type validation")
        print("  - Missing parameter detection")
        print("  - Loading scenario validation")
        print("  - Formulation compatibility checking")
        print("  - Comprehensive error reporting")
        
    except KeyboardInterrupt:
        print("\nValidation demo interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
