#!/usr/bin/env python3
"""
Standalone GSM CLI Script

This is a simplified version of the CLI that works around import issues
by providing basic functionality without relying on problematic sympy imports.
"""

import sys
import os
import json
from pathlib import Path

def add_project_to_path():
    """Add project root to Python path for imports"""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def discover_models_simple():
    """Simple model discovery without importing the classes"""
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

def execute_real_simulation(model_name, params_data=None, loading_data=None, json_output=False):
    """Execute real GSM simulation using the working framework from gsm_model.ipynb"""
    try:
        # Add project to path for imports
        add_project_to_path()
        
        # Import the required modules
        import numpy as np
        from bmcs_matmod.gsm_lagrange.core.gsm_model import GSMModel
        
        # Dynamically import the model class
        if model_name.upper() == "GSM1D_ED":
            from bmcs_matmod.gsm_lagrange.models.gsm1d_ed import GSM1D_ED
            model_class = GSM1D_ED
        elif model_name.upper() == "GSM1D_EP":
            from bmcs_matmod.gsm_lagrange.models.gsm1d_ep import GSM1D_EP
            model_class = GSM1D_EP
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Create material model - following the working notebook approach
        material = GSMModel(model_class)
        
        # Set parameters if provided
        if params_data:
            material.set_params(**params_data)
        else:
            # Use default parameters like in the notebook
            if model_name.upper() == "GSM1D_ED":
                material.set_params(E=20000.0, S=1, c=1, eps_0=0.0)
            elif model_name.upper() == "GSM1D_EP":
                material.set_params(E=20000.0, c=1, eps_0=0.0)
        
        # Setup loading history
        if loading_data:
            time_array = np.array(loading_data.get("time_array", [0, 1.0]))
            strain_history = np.array(loading_data.get("strain_history", [0, 0.01]))
        else:
            # Default loading like in the notebook
            n_steps = 1000
            strain_max = 0.13
            strain_history = np.linspace(0, strain_max, n_steps)
            time_array = np.linspace(0, 1.0, n_steps)
        
        # Run the simulation - exactly like in the notebook
        rd = material.get_F_response(strain_history, time_array)
        
        # Extract results
        eps = rd.eps_t[:, 0]
        sig = rd.sig_t[:, 0, 0]
        
        # Get internal variables
        internal_vars = {}
        for var_name, var_data in rd.Eps_t.__dict__.items():
            if isinstance(var_data, np.ndarray) and var_data.ndim >= 2:
                internal_vars[var_name] = var_data[:, 0, 0].tolist()
        
        # Prepare results
        result = {
            "model": model_name,
            "parameters": dict(params_data) if params_data else {},
            "loading": dict(loading_data) if loading_data else {"default": True},
            "results": {
                "strain": eps.tolist(),
                "stress": sig.tolist(),
                "internal_variables": internal_vars,
                "n_steps": len(eps)
            },
            "status": "success",
            "simulation_type": "real_gsm"
        }
        
        return result
        
    except Exception as e:
        return {
            "model": model_name,
            "parameters": params_data,
            "loading": loading_data,
            "status": "error",
            "error": str(e),
            "simulation_type": "real_gsm"
        }

def show_help():
    """Show help message"""
    help_text = """
GSM CLI - Real Simulation Interface

Usage: python cli_gsm_standalone.py [options]

Options:
  --list-models               List all available GSM models
  --get-param-spec MODEL      Get parameter specification for a model
  --exec MODEL                Execute real GSM simulation for specified model
  --model MODEL               Specify model for simulation (legacy)
  --formulation FORM          Specify formulation (F, Helmholtz, Gibbs)
  --params-inline JSON        Inline parameters as JSON string
  --loading-inline JSON       Inline loading as JSON string
  --simulate                  Run basic simulation (placeholder)
  --validate-only             Validate inputs without running simulation
  --json-output               Output results in JSON format
  --verbose, -v               Enable verbose output
  --help, -h                  Show this help message
  --version                   Show version information

Examples:
  # List models
  python cli_gsm_standalone.py --list-models
  
  # Get model info
  python cli_gsm_standalone.py --get-param-spec GSM1D_ED
  
  # Execute real GSM simulation
  python cli_gsm_standalone.py --exec GSM1D_ED --params-inline '{"E": 30000, "S": 1.0}'
  
  # Real simulation with custom loading
  python cli_gsm_standalone.py --exec GSM1D_ED --params-inline '{"E": 25000}' \\
    --loading-inline '{"time_array": [0,0.5,1], "strain_history": [0,0.005,0.015]}'
  
  # Get JSON output
  python cli_gsm_standalone.py --exec GSM1D_EP --params-inline '{"E": 20000, "c": 5}' --json-output

Note: --exec runs real GSM simulations using the working framework.
      Other options provide basic functionality that works around import issues.
"""
    print(help_text)

def get_model_info_simple(model_name):
    """Get basic model information without importing the class"""
    try:
        models_dir = Path(__file__).parent.parent / "models"
        model_file = models_dir / f"{model_name.lower()}.py"
        
        if not model_file.exists():
            return None
            
        # Read the file and extract basic information
        with open(model_file, 'r') as f:
            content = f.read()
            
        # Basic parameter extraction (simplified)
        # This is a simplified approach that looks for common parameter patterns
        info = {
            "model_name": model_name,
            "file": str(model_file),
            "description": f"{model_name} Material Model",
            "parameters": {
                "note": "Parameter specifications require full CLI import capabilities",
                "common_parameters": ["E", "S", "c", "r", "eps_0"],
                "suggestion": f"Use: python -m bmcs_matmod.gsm_lagrange.cli.cli_gsm --get-param-spec {model_name}"
            }
        }
        
        return info
        
    except Exception as e:
        return {"error": f"Could not read model info: {e}"}

def simulate_basic(model_name, params_data=None, loading_data=None):
    """Basic simulation placeholder - simplified version"""
    try:
        result = {
            "model": model_name,
            "status": "simulation_placeholder",
            "message": "Full simulation requires complete CLI import capabilities",
            "suggestion": f"Use: python -m bmcs_matmod.gsm_lagrange.cli.cli_gsm --model {model_name}",
            "inputs": {
                "parameters": params_data if params_data else "Not provided",
                "loading": loading_data if loading_data else "Not provided"
            },
            "note": "This is a placeholder. Real simulation requires sympy/traits imports."
        }
        return result
    except Exception as e:
        return {"error": f"Simulation placeholder failed: {e}", "status": "error"}

def parse_json_string(json_str):
    """Parse JSON string safely"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}

def execute_real_gsm_simulation(model_name, params_data, loading_data=None, verbose=False):
    """Execute real GSM simulation using subprocess to notebook kernel"""
    import subprocess
    import tempfile
    
    try:
        # Create Python script that uses the working GSM framework
        script_content = f'''
import sys
import json
import numpy as np

# Add project to path
sys.path.insert(0, '/home/rch/Coding/bmcs_matmod')

try:
    # Import as in the working notebook
    from bmcs_matmod.gsm_lagrange.core.gsm_model import GSMModel
    from bmcs_matmod.gsm_lagrange.core.gsm_def_registry import get_gsm_def_class
    
    # Get model class
    gsm_def_class = get_gsm_def_class('{model_name}')
    
    # Create material model (from notebook workflow)
    material = GSMModel(gsm_def_class)
    
    # Set parameters
    params = {params_data}
    material.set_params(**params)
    
    # Prepare loading
    if {loading_data is not None}:
        loading = {loading_data}
        time = np.array(loading.get("time_array", [0, 1]))
        strain = np.array(loading.get("strain_history", [0, 0.01]))
    else:
        # Default monotonic test
        n_steps = 100
        strain_max = 0.01
        strain = np.linspace(0, strain_max, n_steps)
        time = np.linspace(0, 1.0, n_steps)
    
    print(f"Executing real GSM simulation for {{material.gsm_def.__class__.__name__}}")
    print(f"Parameters: {{params}}")
    print(f"Time steps: {{len(time)}}")
    
    # Real simulation using get_F_response (from notebook)
    rd = material.get_F_response(strain, time)
    
    print("✅ Real GSM simulation completed!")
    
    # Extract results
    eps = rd.eps_t[:, 0]
    sig = rd.sig_t[:, 0, 0]
    
    # Internal variables
    internal_vars = {{}}
    for var_name, var_data in rd.Eps_t.items():
        if hasattr(var_data, 'shape') and len(var_data.shape) >= 2:
            if var_data.shape[1] > 0:
                internal_vars[var_name] = var_data[:, 0, 0].tolist()
            else:
                internal_vars[var_name] = var_data[:, 0].tolist()
    
    # Output results
    results = {{
        "status": "success",
        "model_name": "{model_name}",
        "execution_info": {{
            "time_steps": len(time),
            "max_strain": float(np.max(np.abs(eps))),
            "max_stress": float(np.max(np.abs(sig))),
            "simulation_type": "real_gsm"
        }},
        "parameters": params,
        "response": {{
            "time": time.tolist(),
            "strain": eps.tolist(), 
            "stress": sig.tolist(),
            "internal_variables": internal_vars
        }}
    }}
    
    print("JSON_RESULTS_START")
    print(json.dumps(results, indent=2))
    print("JSON_RESULTS_END")
    
except Exception as e:
    error_result = {{
        "status": "error",
        "error": str(e),
        "model_name": "{model_name}"
    }}
    print("JSON_RESULTS_START")
    print(json.dumps(error_result, indent=2))
    print("JSON_RESULTS_END")
    import traceback
    traceback.print_exc()
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        if verbose:
            print(f"Created simulation script: {script_path}")
        
        # Run with python (will use current environment)
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=30)
        
        # Clean up
        import os
        os.unlink(script_path)
        
        if result.returncode == 0:
            # Extract JSON results
            output_lines = result.stdout.split('\n')
            json_start = None
            json_end = None
            
            for i, line in enumerate(output_lines):
                if line.strip() == "JSON_RESULTS_START":
                    json_start = i + 1
                elif line.strip() == "JSON_RESULTS_END":
                    json_end = i
                    break
            
            if json_start is not None and json_end is not None:
                json_lines = output_lines[json_start:json_end]
                json_str = '\n'.join(json_lines)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: return text output
            return {
                "status": "completed",
                "output": result.stdout,
                "message": "Real simulation executed (output not parsed as JSON)"
            }
        else:
            return {
                "status": "error",
                "error": result.stderr,
                "stdout": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "Simulation timed out"
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e)
        }

def main():
    """Main entry point"""
    if len(sys.argv) == 1 or "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        return
    
    if "--version" in sys.argv:
        print("GSM CLI Standalone v1.0")
        return
    
    json_output = "--json-output" in sys.argv
    
    if "--list-models" in sys.argv:
        add_project_to_path()
        models = discover_models_simple()
        
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
    
    if "--get-param-spec" in sys.argv:
        try:
            # Find the model name after --get-param-spec
            idx = sys.argv.index("--get-param-spec")
            if idx + 1 < len(sys.argv):
                model_name = sys.argv[idx + 1]
            else:
                print("Error: --get-param-spec requires a model name")
                sys.exit(1)
        except ValueError:
            print("Error: --get-param-spec not found")
            sys.exit(1)
            
        add_project_to_path()
        info = get_model_info_simple(model_name)
        
        if info is None:
            if json_output:
                result = {"error": f"Model {model_name} not found", "status": "error"}
                print(json.dumps(result, indent=2))
            else:
                print(f"Error: Model {model_name} not found")
            sys.exit(1)
            
        if json_output:
            print(json.dumps(info, indent=2))
        else:
            print(f"Model Information: {info['model_name']}")
            print("=" * 40)
            print(f"Description: {info['description']}")
            print(f"File: {info['file']}")
            print("\nNote: This is simplified model info.")
            print("For detailed parameter specifications, use:")
            print(f"  python -m bmcs_matmod.gsm_lagrange.cli.cli_gsm --get-param-spec {model_name}")
        return
    
    # Handle --exec option for real simulation
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
        
        # Extract parameters
        params_data = None
        if "--params-inline" in sys.argv:
            try:
                idx = sys.argv.index("--params-inline")
                if idx + 1 < len(sys.argv):
                    params_json = sys.argv[idx + 1]
                    params_data = parse_json_string(params_json)
            except ValueError:
                pass
        
        # Extract loading
        loading_data = None
        if "--loading-inline" in sys.argv:
            try:
                idx = sys.argv.index("--loading-inline")
                if idx + 1 < len(sys.argv):
                    loading_json = sys.argv[idx + 1]
                    loading_data = parse_json_string(loading_json)
            except ValueError:
                pass
        
        # Execute real simulation
        result = execute_real_simulation(model_name, params_data, loading_data, json_output)
        
        if json_output:
            print(json.dumps(result, indent=2))
        else:
            if result["status"] == "success":
                print(f"GSM Real Simulation Results for {model_name}")
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
                print('max_stress', max_stress)
                print(f"Simulation completed successfully!")
                print(f"  Steps: {n_steps}")
                print(f"  Max strain: {max_strain:.6f}")
                print(f"  Max stress: {max_stress:.2f}")
                
                # Show internal variables
                if result["results"]["internal_variables"]:
                    print("\nInternal variables available:")
                    for var_name in result["results"]["internal_variables"].keys():
                        print(f"  - {var_name}")
                
                print(f"\n✅ Real GSM simulation executed successfully")
                print("💡 Use --json-output for detailed numerical results")
            else:
                print(f"❌ Simulation failed for {model_name}")
                print(f"Error: {result.get('error', 'Unknown error')}")
        return

    # Handle simulation-related commands (legacy placeholder)
    if "--model" in sys.argv or "--simulate" in sys.argv:
        add_project_to_path()
        
        # Extract model name
        model_name = None
        if "--model" in sys.argv:
            try:
                idx = sys.argv.index("--model")
                if idx + 1 < len(sys.argv):
                    model_name = sys.argv[idx + 1]
            except ValueError:
                pass
        
        if not model_name:
            if json_output:
                result = {"error": "Model name required for simulation", "status": "error"}
                print(json.dumps(result, indent=2))
            else:
                print("Error: --model MODEL_NAME is required for simulation")
            sys.exit(1)
        
        # Extract parameters
        params_data = None
        if "--params-inline" in sys.argv:
            try:
                idx = sys.argv.index("--params-inline")
                if idx + 1 < len(sys.argv):
                    params_json = sys.argv[idx + 1]
                    params_data = parse_json_string(params_json)
            except ValueError:
                pass
        
        # Extract loading
        loading_data = None
        if "--loading-inline" in sys.argv:
            try:
                idx = sys.argv.index("--loading-inline")
                if idx + 1 < len(sys.argv):
                    loading_json = sys.argv[idx + 1]
                    loading_data = parse_json_string(loading_json)
            except ValueError:
                pass
        
        # Run simulation placeholder
        if "--validate-only" in sys.argv:
            result = {
                "validation": "placeholder",
                "model": model_name,
                "parameters": params_data,
                "loading": loading_data,
                "status": "validation_placeholder",
                "message": "Full validation requires complete CLI capabilities"
            }
        else:
            result = simulate_basic(model_name, params_data, loading_data)
        
        if json_output:
            print(json.dumps(result, indent=2))
        else:
            print(f"Legacy Simulation Placeholder for {model_name}")
            print("=" * 40)
            print("⚠️  This is a simplified simulation placeholder.")
            print("✅ For actual simulation execution, use:")
            print(f"  python cli_gsm_standalone.py --exec {model_name}")
            if params_data:
                print(f"Parameters: {params_data}")
            if loading_data:
                print(f"Loading: {loading_data}")
            print("\n💡 The --exec option runs real GSM simulations!")
        return
    
    print("Error: Unknown command. Use --help for available options.")
    sys.exit(1)

if __name__ == "__main__":
    main()
