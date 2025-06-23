#!/usr/bin/env python3
"""
Batch Processing Examples for GSM CLI

This script demonstrates how to process multiple simulations in batch mode,
useful for parameter studies, sensitivity analysis, and automated workflows.
"""

import json
import os
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import itertools
import numpy as np

class GSMBatchProcessor:
    """Batch processor for GSM simulations"""
    
    def __init__(self, cli_path: str = "../cli_gsm.py"):
        self.cli_path = cli_path
        self.results = []
    
    def create_parameter_study(self, base_params: Dict[str, Any], 
                             param_variations: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Create parameter combinations for parameter study"""
        param_sets = []
        
        # Get all parameter names and their variation values
        param_names = list(param_variations.keys())
        param_values = list(param_variations.values())
        
        # Generate all combinations
        for combination in itertools.product(*param_values):
            param_set = base_params.copy()
            for i, param_name in enumerate(param_names):
                param_set[param_name] = combination[i]
            param_sets.append(param_set)
        
        return param_sets
    
    def execute_simulation(self, model: str, formulation: str, 
                          parameters: Dict[str, Any], loading: Dict[str, Any],
                          config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a single simulation using CLI"""
        
        # Create temporary files for parameters and loading
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as param_file:
            json.dump({"parameters": parameters}, param_file)
            param_file_path = param_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as loading_file:
            json.dump(loading, loading_file)
            loading_file_path = loading_file.name
        
        try:
            # Build CLI command
            cmd = [
                "python", self.cli_path,
                "--model", model,
                "--formulation", formulation,
                "--params", param_file_path,
                "--loading", loading_file_path,
                "--json-output"
            ]
            
            if config:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
                    json.dump(config, config_file)
                    config_file_path = config_file.name
                cmd.extend(["--config", config_file_path])
            
            # Execute simulation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {
                    "status": "error",
                    "error": result.stderr,
                    "parameters": parameters
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "Simulation timeout",
                "parameters": parameters
            }
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"JSON decode error: {e}",
                "stdout": result.stdout,
                "parameters": parameters
            }
        finally:
            # Clean up temporary files
            os.unlink(param_file_path)
            os.unlink(loading_file_path)
            if config and 'config_file_path' in locals():
                os.unlink(config_file_path)
    
    def run_parameter_study(self, model: str, formulation: str,
                           base_params: Dict[str, Any], param_variations: Dict[str, List[float]],
                           loading: Dict[str, Any], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Run a parameter study"""
        
        param_sets = self.create_parameter_study(base_params, param_variations)
        results = []
        
        print(f"Running parameter study with {len(param_sets)} parameter combinations...")
        
        for i, params in enumerate(param_sets):
            print(f"  Simulation {i+1}/{len(param_sets)}: {params}")
            
            start_time = time.time()
            result = self.execute_simulation(model, formulation, params, loading, config)
            execution_time = time.time() - start_time
            
            result["simulation_id"] = i + 1
            result["execution_time_client"] = execution_time
            result["input_parameters"] = params
            
            results.append(result)
            
            if result["status"] == "success":
                print(f"    ✅ Completed in {execution_time:.3f}s")
            else:
                print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save batch results to file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze batch results"""
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]
        
        analysis = {
            "total_simulations": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "avg_execution_time": np.mean([r.get("execution_time_client", 0) for r in successful]) if successful else 0
        }
        
        if failed:
            analysis["failure_reasons"] = {}
            for result in failed:
                error = result.get("error", "Unknown error")
                analysis["failure_reasons"][error] = analysis["failure_reasons"].get(error, 0) + 1
        
        return analysis


def example_elastic_modulus_study():
    """Example: Study effect of elastic modulus variations"""
    print("=== Elastic Modulus Parameter Study ===")
    
    processor = GSMBatchProcessor()
    
    # Base parameters for GSM1D_ED
    base_params = {
        "S": 1.0,
        "c": 2.0,
        "r": 0.5,
        "eps_0": 0.001
    }
    
    # Vary elastic modulus
    param_variations = {
        "E": [20000.0, 25000.0, 30000.0, 35000.0, 40000.0]
    }
    
    # Loading scenario
    loading = {
        "time_array": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "strain_history": [0.0, 0.002, 0.004, 0.006, 0.008, 0.01],
        "loading_type": "strain_controlled"
    }
    
    # Run parameter study
    results = processor.run_parameter_study(
        model="GSM1D_ED",
        formulation="F",
        base_params=base_params,
        param_variations=param_variations,
        loading=loading
    )
    
    # Save and analyze results
    processor.save_results(results, "elastic_modulus_study.json")
    analysis = processor.analyze_results(results)
    
    print(f"\nStudy Analysis:")
    print(f"  Total simulations: {analysis['total_simulations']}")
    print(f"  Successful: {analysis['successful']}")
    print(f"  Success rate: {analysis['success_rate']:.1%}")
    print(f"  Average execution time: {analysis['avg_execution_time']:.3f}s")
    
    # Extract final stresses for comparison
    if analysis['successful'] > 0:
        print(f"\nElastic Modulus vs Final Stress:")
        for result in results:
            if result["status"] == "success":
                E = result["input_parameters"]["E"]
                final_stress = result["response"]["sig_t"][-1]
                print(f"  E = {E:5.0f} MPa → σ_final = {final_stress:.2f} MPa")


def example_damage_parameter_study():
    """Example: Study effect of damage parameters"""
    print("\n=== Damage Parameter Study ===")
    
    processor = GSMBatchProcessor()
    
    # Base parameters
    base_params = {
        "E": 30000.0,
        "eps_0": 0.001
    }
    
    # Vary damage parameters
    param_variations = {
        "S": [0.5, 1.0, 1.5, 2.0],
        "c": [1.0, 2.0, 3.0],
        "r": [0.3, 0.5, 0.7]
    }
    
    # Loading scenario
    loading = {
        "time_array": [0.0, 0.25, 0.5, 0.75, 1.0],
        "strain_history": [0.0, 0.0025, 0.005, 0.0075, 0.01],
        "loading_type": "strain_controlled"
    }
    
    # Run parameter study (this will generate 4×3×3 = 36 combinations)
    results = processor.run_parameter_study(
        model="GSM1D_ED",
        formulation="F",
        base_params=base_params,
        param_variations=param_variations,
        loading=loading
    )
    
    # Save and analyze results
    processor.save_results(results, "damage_parameter_study.json")
    analysis = processor.analyze_results(results)
    
    print(f"\nStudy Analysis:")
    print(f"  Total simulations: {analysis['total_simulations']}")
    print(f"  Successful: {analysis['successful']}")
    print(f"  Success rate: {analysis['success_rate']:.1%}")
    
    if analysis['failed'] > 0:
        print(f"  Failure reasons:")
        for reason, count in analysis.get('failure_reasons', {}).items():
            print(f"    {reason}: {count}")


def example_loading_scenario_comparison():
    """Example: Compare different loading scenarios"""
    print("\n=== Loading Scenario Comparison ===")
    
    processor = GSMBatchProcessor()
    
    # Fixed parameters
    parameters = {
        "E": 30000.0,
        "S": 1.0,
        "c": 2.0,
        "r": 0.5,
        "eps_0": 0.001
    }
    
    # Different loading scenarios
    loading_scenarios = {
        "monotonic_slow": {
            "time_array": [0.0, 0.5, 1.0, 1.5, 2.0],
            "strain_history": [0.0, 0.0025, 0.005, 0.0075, 0.01],
            "loading_type": "strain_controlled"
        },
        "monotonic_fast": {
            "time_array": [0.0, 0.25, 0.5],
            "strain_history": [0.0, 0.005, 0.01],
            "loading_type": "strain_controlled"
        },
        "cyclic": {
            "time_array": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
            "strain_history": [0.0, 0.005, 0.0, -0.005, 0.0, 0.005, 0.01],
            "loading_type": "strain_controlled"
        }
    }
    
    results = []
    
    for scenario_name, loading in loading_scenarios.items():
        print(f"\n  Running scenario: {scenario_name}")
        result = processor.execute_simulation(
            model="GSM1D_ED",
            formulation="F",
            parameters=parameters,
            loading=loading
        )
        result["scenario"] = scenario_name
        results.append(result)
        
        if result["status"] == "success":
            final_stress = result["response"]["sig_t"][-1]
            print(f"    ✅ Final stress: {final_stress:.2f} MPa")
        else:
            print(f"    ❌ Failed: {result.get('error', 'Unknown')}")
    
    # Save results
    processor.save_results(results, "loading_scenario_comparison.json")
    
    # Compare final stresses
    print(f"\nScenario Comparison:")
    for result in results:
        if result["status"] == "success":
            scenario = result["scenario"]
            final_stress = result["response"]["sig_t"][-1]
            duration = result["response"]["t_t"][-1]
            print(f"  {scenario:15} → σ_final = {final_stress:6.2f} MPa, duration = {duration:.2f}")


def main():
    """Run batch processing examples"""
    print("GSM CLI Batch Processing Examples")
    print("=" * 50)
    
    try:
        example_elastic_modulus_study()
        example_damage_parameter_study()
        example_loading_scenario_comparison()
        
        print(f"\n" + "=" * 50)
        print("Batch processing examples completed!")
        print("\nGenerated files:")
        print("  - elastic_modulus_study.json")
        print("  - damage_parameter_study.json")
        print("  - loading_scenario_comparison.json")
        
    except KeyboardInterrupt:
        print("\n\nBatch processing interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
