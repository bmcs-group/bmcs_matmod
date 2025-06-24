#!/usr/bin/env python3
"""
Remote Request Examples for GSM CLI Network Interface

This script demonstrates how to interact with the GSM CLI network server
using Python requests for remote simulation execution.
"""

import requests
import json
import time
from typing import Dict, Any, Optional

class GSMNetworkClient:
    """Client for interacting with GSM CLI network server"""
    
    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def check_server_health(self) -> bool:
        """Check if the server is running and responding"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def list_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        try:
            response = self.session.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)}
    
    def get_parameter_spec(self, model_name: str) -> Dict[str, Any]:
        """Get parameter specification for a model"""
        try:
            response = self.session.get(f"{self.base_url}/param-spec/{model_name}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)}
    
    def validate_parameters(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters without executing simulation"""
        try:
            request_data["validate_only"] = True
            response = self.session.post(
                f"{self.base_url}/validate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)}
    
    def execute_simulation(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a simulation"""
        try:
            response = self.session.post(
                f"{self.base_url}/simulate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)}


def example_basic_interaction():
    """Example: Basic server interaction"""
    print("=== Basic Server Interaction ===")
    
    client = GSMNetworkClient()
    
    # Check server health
    if not client.check_server_health():
        print("❌ Server is not running or not responding")
        print("Please start the server with: python cli_gsm.py --serve --port 8888")
        return
    
    print("✅ Server is running and responsive")
    
    # List available models
    print("\n--- Available Models ---")
    models_result = client.list_models()
    if models_result["status"] == "success":
        for model in models_result["models"]:
            print(f"- {model}")
    else:
        print(f"Error: {models_result['error']}")


def example_parameter_specification():
    """Example: Get parameter specifications"""
    print("\n=== Parameter Specification Retrieval ===")
    
    client = GSMNetworkClient()
    
    if not client.check_server_health():
        print("❌ Server not available")
        return
    
    # Get parameter specification for GSM1D_ED
    print("\n--- Parameter Spec for GSM1D_ED ---")
    spec_result = client.get_parameter_spec("GSM1D_ED")
    
    if spec_result["status"] == "success":
        spec = spec_result["specification"]
        print(f"Model: {spec['model_name']}")
        print("Parameters:")
        for param_name, param_info in spec["parameters"].items():
            desc = spec["parameter_descriptions"].get(param_name, "No description")
            bounds = spec["parameter_bounds"].get(param_name, "No bounds")
            units = spec["parameter_units"].get(param_name, "-")
            print(f"  {param_name}: {desc} [{units}]")
            print(f"    Type: {param_info['type']}, Required: {param_info['required']}")
            print(f"    Bounds: {bounds}")
    else:
        print(f"Error: {spec_result['error']}")


def example_parameter_validation():
    """Example: Parameter validation"""
    print("\n=== Parameter Validation ===")
    
    client = GSMNetworkClient()
    
    if not client.check_server_health():
        print("❌ Server not available")
        return
    
    # Test valid parameters
    valid_request = {
        "model": "GSM1D_ED",
        "formulation": "F",
        "parameters": {
            "E": 30000.0,
            "S": 1.0,
            "c": 2.0,
            "r": 0.5,
            "eps_0": 0.001
        },
        "loading": {
            "time_array": [0.0, 0.5, 1.0],
            "strain_history": [0.0, 0.005, 0.01]
        }
    }
    
    print("\n--- Testing Valid Parameters ---")
    validation_result = client.validate_parameters(valid_request)
    
    if validation_result["status"] == "validation_complete":
        if validation_result["valid"]:
            print("✅ Parameters are valid")
        else:
            print("❌ Parameters are invalid:")
            for error in validation_result["errors"]:
                print(f"  - {error}")
    else:
        print(f"Error: {validation_result.get('error', 'Unknown error')}")
    
    # Test invalid parameters
    invalid_request = valid_request.copy()
    invalid_request["parameters"]["E"] = -1000.0  # Invalid (negative)
    
    print("\n--- Testing Invalid Parameters ---")
    validation_result = client.validate_parameters(invalid_request)
    
    if validation_result["status"] == "validation_complete":
        if validation_result["valid"]:
            print("❌ Expected validation to fail, but it passed")
        else:
            print("✅ Correctly identified invalid parameters:")
            for error in validation_result["errors"]:
                print(f"  - {error}")
    else:
        print(f"Error: {validation_result.get('error', 'Unknown error')}")


def example_simulation_execution():
    """Example: Execute a simulation"""
    print("\n=== Simulation Execution ===")
    
    client = GSMNetworkClient()
    
    if not client.check_server_health():
        print("❌ Server not available")
        return
    
    # Simulation request
    simulation_request = {
        "model": "GSM1D_ED",
        "formulation": "F",
        "parameters": {
            "E": 30000.0,
            "S": 1.0,
            "c": 2.0,
            "r": 0.5,
            "eps_0": 0.001
        },
        "loading": {
            "time_array": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "strain_history": [0.0, 0.002, 0.004, 0.006, 0.008, 0.01]
        },
        "config": {
            "tolerance": 1e-6,
            "max_iterations": 100
        }
    }
    
    print("\n--- Executing Simulation ---")
    print(f"Model: {simulation_request['model']}")
    print(f"Formulation: {simulation_request['formulation']}")
    print(f"Max strain: {max(simulation_request['loading']['strain_history'])}")
    
    start_time = time.time()
    result = client.execute_simulation(simulation_request)
    execution_time = time.time() - start_time
    
    if result["status"] == "success":
        print(f"✅ Simulation completed in {execution_time:.3f} seconds")
        print(f"Model execution time: {result.get('execution_time', 'N/A')} seconds")
        
        if "response" in result:
            response = result["response"]
            final_stress = response["sig_t"][-1] if "sig_t" in response else "N/A"
            print(f"Final stress: {final_stress}")
        
        if "warnings" in result and result["warnings"]:
            print("Warnings:")
            for warning in result["warnings"]:
                print(f"  - {warning}")
    else:
        print(f"❌ Simulation failed: {result.get('error', 'Unknown error')}")


def main():
    """Run all examples"""
    print("GSM CLI Network Client Examples")
    print("=" * 50)
    
    try:
        example_basic_interaction()
        example_parameter_specification()
        example_parameter_validation()
        example_simulation_execution()
        
    except KeyboardInterrupt:
        print("\n\nExample execution interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run the server: python cli_gsm.py --serve --port 8888")
    print("Then run this script again to test the network interface.")


if __name__ == "__main__":
    main()
