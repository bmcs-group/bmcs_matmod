#!/usr/bin/env python3
"""
GSM CLI - Generalized Standard Material Simulation Interface

A command-line interface for executing real GSM simulations using the 
bmcs_matmod.gsm_lagrange framework.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

def add_project_to_path():
    """Add project root to Python path for imports"""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def discover_models():
    """Discover available GSM models"""
    try:
        models_dir = Path(__file__).parent.parent / "models"
        model_files = list(models_dir.glob("gsm1d_*.py"))
        models = []
        for file in model_files:
            if file.name != "__init__.py":
                model_name = file.stem.upper()
                models.append(model_name)
        return sorted(models)
    except Exception as e:
        print(f"Error discovering models: {e}")
        return []

def get_model_class(model_name):
    """Dynamically import and return the model class"""
    model_name = model_name.upper()
    
    if model_name == "GSM1D_ED":
        from bmcs_matmod.gsm_lagrange.models.gsm1d_ed import GSM1D_ED
        return GSM1D_ED
    elif model_name == "GSM1D_EP":
        from bmcs_matmod.gsm_lagrange.models.gsm1d_ep import GSM1D_EP
        return GSM1D_EP
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def execute_simulation(model_name, params_data=None, loading_data=None):
    """Execute GSM simulation"""
    try:
        add_project_to_path()
        from bmcs_matmod.gsm_lagrange.core.gsm_model import GSMModel
        
        # Get model class and create material
        model_class = get_model_class(model_name)
        material = GSMModel(model_class)
        
        # Set parameters
        if params_data:
            material.set_params(**params_data)
        else:
            # Use sensible defaults based on model type
            if model_name.upper() == "GSM1D_ED":
                material.set_params(E=20000.0, S=1, c=1, eps_0=0.0)
            elif model_name.upper() == "GSM1D_EP":
                material.set_params(E=20000.0, c=1, eps_0=0.0)
        
        # Setup loading history
        if loading_data:
            time_array = np.array(loading_data.get("time_array", [0, 1.0]))
            strain_history = np.array(loading_data.get("strain_history", [0, 0.01]))
        else:
            # Default monotonic tension test
            n_steps = 1000
            strain_max = 0.13
            strain_history = np.linspace(0, strain_max, n_steps)
            time_array = np.linspace(0, 1.0, n_steps)
        
        # Execute simulation
        rd = material.get_F_response(strain_history, time_array)
        
        # Extract results
        eps = rd.eps_t[:, 0]
        sig = rd.sig_t[:, 0, 0]
        
        # Get internal variables
        internal_vars = {}
        for var_name, var_data in rd.Eps_t.__dict__.items():
            if isinstance(var_data, np.ndarray) and var_data.ndim >= 2:
                internal_vars[var_name] = var_data[:, 0, 0].tolist()
        
        return {
            "model": model_name,
            "parameters": dict(params_data) if params_data else {},
            "loading": dict(loading_data) if loading_data else {"default": True},
            "results": {
                "strain": eps.tolist(),
                "stress": sig.tolist(),
                "internal_variables": internal_vars,
                "n_steps": len(eps)
            },
            "status": "success"
        }
        
    except Exception as e:
        return {
            "model": model_name,
            "parameters": params_data,
            "loading": loading_data,
            "status": "error",
            "error": str(e)
        }

def get_model_info(model_name):
    """Get information about a specific model"""
    try:
        add_project_to_path()
        model_class = get_model_class(model_name)
        
        # Try to get parameter information
        from bmcs_matmod.gsm_lagrange.core.gsm_model import GSMModel
        material = GSMModel(model_class)
        
        # Get default parameters
        params = {}
        for param_sym, name in material.trait_model_params.items():
            value = getattr(material, name)
            params[name] = value
        
        return {
            "model_name": model_name,
            "class_name": model_class.__name__,
            "parameters": params,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "model_name": model_name,
            "status": "error",
            "error": str(e)
        }

def parse_json_string(json_str):
    """Parse JSON string safely"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def show_help():
    """Show help message"""
    help_text = """
GSM CLI - Generalized Standard Material Simulation Interface

Usage: python cli_gsm_clean.py [options]

Options:
  --list-models               List all available GSM models
  --get-param-spec MODEL      Get parameter specification for a model
  --exec MODEL                Execute GSM simulation for specified model
  --params-inline JSON        Inline parameters as JSON string
  --loading-inline JSON       Inline loading as JSON string
  --json-output               Output results in JSON format
  --help, -h                  Show this help message
  --version                   Show version information

Examples:
  # List available models
  python cli_gsm_clean.py --list-models
  
  # Get model parameter info
  python cli_gsm_clean.py --get-param-spec GSM1D_ED
  
  # Execute simulation with default parameters
  python cli_gsm_clean.py --exec GSM1D_ED
  
  # Execute with custom parameters
  python cli_gsm_clean.py --exec GSM1D_ED --params-inline '{"E": 30000, "S": 1.0}'
  
  # Custom loading history
  python cli_gsm_clean.py --exec GSM1D_ED --params-inline '{"E": 25000}' \\
    --loading-inline '{"time_array": [0,0.5,1], "strain_history": [0,0.005,0.015]}'
  
  # JSON output for data processing
  python cli_gsm_clean.py --exec GSM1D_EP --params-inline '{"E": 20000}' --json-output

Available Models:
  GSM1D_ED    - 1D Elastic-Damage model
  GSM1D_EP    - 1D Elastic-Plastic model
"""
    print(help_text)

def main():
    """Main entry point"""
    if len(sys.argv) == 1 or "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        return
    
    if "--version" in sys.argv:
        print("GSM CLI v1.0 - Production Interface")
        return
    
    json_output = "--json-output" in sys.argv
    
    # List models
    if "--list-models" in sys.argv:
        models = discover_models()
        
        if json_output:
            result = {
                "available_models": models,
                "count": len(models),
                "status": "success"
            }
            print(json.dumps(result, indent=2))
        else:
            print("Available GSM Models:")
            print("=" * 30)
            for i, model in enumerate(models, 1):
                print(f"{i:2d}. {model}")
            print(f"\nTotal: {len(models)} models found")
        return
    
    # Get model parameter specification
    if "--get-param-spec" in sys.argv:
        try:
            idx = sys.argv.index("--get-param-spec")
            if idx + 1 >= len(sys.argv):
                print("Error: --get-param-spec requires a model name")
                sys.exit(1)
            model_name = sys.argv[idx + 1]
        except ValueError:
            print("Error: --get-param-spec not found")
            sys.exit(1)
        
        info = get_model_info(model_name)
        
        if json_output:
            print(json.dumps(info, indent=2))
        else:
            if info["status"] == "success":
                print(f"Model: {info['model_name']} ({info['class_name']})")
                print("=" * 50)
                print("Default Parameters:")
                for param, value in info["parameters"].items():
                    print(f"  {param}: {value}")
            else:
                print(f"Error getting info for {model_name}: {info['error']}")
        return
    
    # Execute simulation
    if "--exec" in sys.argv:
        try:
            idx = sys.argv.index("--exec")
            if idx + 1 >= len(sys.argv):
                print("Error: --exec requires a model name")
                sys.exit(1)
            model_name = sys.argv[idx + 1]
        except ValueError:
            print("Error: --exec requires a model name")
            sys.exit(1)
        
        # Parse parameters
        params_data = None
        if "--params-inline" in sys.argv:
            try:
                idx = sys.argv.index("--params-inline")
                if idx + 1 < len(sys.argv):
                    params_json = sys.argv[idx + 1]
                    params_data = parse_json_string(params_json)
                    if params_data is None:
                        sys.exit(1)
            except ValueError:
                pass
        
        # Parse loading
        loading_data = None
        if "--loading-inline" in sys.argv:
            try:
                idx = sys.argv.index("--loading-inline")
                if idx + 1 < len(sys.argv):
                    loading_json = sys.argv[idx + 1]
                    loading_data = parse_json_string(loading_json)
                    if loading_data is None:
                        sys.exit(1)
            except ValueError:
                pass
        
        # Execute simulation
        result = execute_simulation(model_name, params_data, loading_data)
        
        if json_output:
            print(json.dumps(result, indent=2))
        else:
            if result["status"] == "success":
                print(f"GSM Simulation Results for {model_name}")
                print("=" * 50)
                
                # Show parameters
                if result["parameters"]:
                    print("Parameters:")
                    for param, value in result["parameters"].items():
                        print(f"  {param}: {value}")
                    print()
                
                # Show simulation summary
                n_steps = result["results"]["n_steps"]
                max_strain = max(result["results"]["strain"])
                max_stress = max(result["results"]["stress"])
                print(f"Simulation completed successfully!")
                print(f"  Steps: {n_steps}")
                print(f"  Max strain: {max_strain:.6f}")
                print(f"  Max stress: {max_stress:.2f}")
                
                # Show internal variables
                if result["results"]["internal_variables"]:
                    print("\nInternal variables:")
                    for var_name in result["results"]["internal_variables"].keys():
                        print(f"  - {var_name}")
                
                print(f"\n✅ Simulation executed successfully")
                print("💡 Use --json-output for detailed numerical results")
            else:
                print(f"❌ Simulation failed for {model_name}")
                print(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    print("Error: Unknown command. Use --help for available options.")
    sys.exit(1)

if __name__ == "__main__":
    main()
