#!/usr/bin/env python3
"""
GSM Simulation Executor

This script executes real GSM simulations by running code that works
in the notebook environment. It bridges the CLI and the working GSM framework.
"""

import sys
import os
import json
import subprocess
import tempfile
from pathlib import Path

def create_simulation_script(model_name, params, loading_spec=None, formulation="F"):
    """Create a Python script that runs the GSM simulation"""
    
    # Default loading if not specified
    if loading_spec is None:
        time_array = "[0, 1.0]"
        strain_history = "[0, 0.01]"
        n_steps = 100
    else:
        time_array = str(loading_spec.get("time_array", [0, 1]))
        strain_history = str(loading_spec.get("strain_history", [0, 0.01]))
        n_steps = len(loading_spec.get("time_array", [0, 1]))
    
    script_content = f'''
import sys
import json
import numpy as np

# Add project path for imports
sys.path.insert(0, '/home/rch/Coding/bmcs_matmod')

try:
    # Import GSM framework (as in notebook)
    from bmcs_matmod.gsm_lagrange.core.gsm_model import GSMModel
    from bmcs_matmod.gsm_lagrange.core.gsm_def_registry import get_gsm_def_class
    
    # Get model class
    gsm_def_class = get_gsm_def_class('{model_name}')
    
    # Create GSMModel instance (from notebook)
    material = GSMModel(gsm_def_class)
    
    # Set parameters
    params = {params}
    material.set_params(**params)
    
    # Prepare loading
    if {n_steps} <= 2:
        # Simple loading
        time = np.array({time_array})
        strain = np.array({strain_history})
    else:
        # Interpolated loading  
        time_points = {time_array}
        strain_points = {strain_history}
        time = np.linspace(time_points[0], time_points[-1], {n_steps})
        strain = np.interp(time, time_points, strain_points)
    
    print(f"Running GSM simulation for {{material.gsm_def.__class__.__name__}}")
    print(f"Parameters: {{params}}")
    print(f"Time steps: {{len(time)}}")
    
    # Execute simulation (from notebook: get_F_response)
    rd = material.get_F_response(strain, time)
    
    print("Simulation completed successfully!")
    
    # Extract results (as in notebook)
    eps = rd.eps_t[:, 0]
    sig = rd.sig_t[:, 0]
    
    # Extract internal variables
    internal_vars = {{}}
    for var_name, var_data in rd.Eps_t.items():
        if hasattr(var_data, 'shape') and len(var_data.shape) >= 2:
            if var_data.shape[1] > 0:
                internal_vars[var_name] = var_data[:, 0, 0].tolist()
            else:
                internal_vars[var_name] = var_data[:, 0].tolist()
        else:
            internal_vars[var_name] = var_data.tolist() if hasattr(var_data, 'tolist') else str(var_data)
    
    # Build results
    results = {{
        "status": "success",
        "model_name": "{model_name}",
        "formulation": "{formulation}",
        "execution_info": {{
            "time_steps": int(len(time)),
            "max_strain": float(np.max(np.abs(eps))),
            "max_stress": float(np.max(np.abs(sig))),
            "simulation_type": "real_gsm_via_notebook"
        }},
        "parameters": params,
        "loading": {{
            "time_array": time.tolist(),
            "strain_history": strain.tolist(),
            "type": "strain_controlled"
        }},
        "response": {{
            "time": time.tolist(),
            "strain": eps.tolist(),
            "stress": sig.tolist(),
            "internal_variables": internal_vars
        }},
        "gsm_info": {{
            "engine": "GSMModel",
            "method": "get_F_response",
            "framework": "bmcs_matmod.gsm_lagrange"
        }}
    }}
    
    print(json.dumps(results, indent=2))
    
except Exception as e:
    error_result = {{
        "status": "error",
        "error": str(e),
        "model_name": "{model_name}",
        "message": "GSM simulation failed"
    }}
    print(json.dumps(error_result, indent=2))
    import traceback
    traceback.print_exc()
'''
    return script_content

def execute_gsm_simulation(model_name, params, loading_spec=None, formulation="F", verbose=False):
    """Execute GSM simulation using temporary script"""
    
    try:
        # Create temporary script
        script_content = create_simulation_script(model_name, params, loading_spec, formulation)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        if verbose:
            print(f"Created simulation script: {script_path}")
            print("Executing GSM simulation...")
        
        # Execute script
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=30)
        
        # Clean up
        os.unlink(script_path)
        
        if result.returncode == 0:
            # Parse JSON output
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                # Return text output if not JSON
                return {
                    "status": "success",
                    "output": result.stdout,
                    "message": "Simulation completed but output not JSON"
                }
        else:
            return {
                "status": "error", 
                "error": result.stderr,
                "stdout": result.stdout,
                "returncode": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "Simulation timed out after 30 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def discover_models():
    """Discover available models"""
    try:
        models_dir = Path(__file__).parent.parent / "models"
        model_files = list(models_dir.glob("gsm1d_*.py"))
        models = []
        for file in model_files:
            if file.name != "__init__.py":
                model_name = file.stem.upper()
                models.append(model_name)
        return sorted(models)
    except Exception:
        return ["GSM1D_ED", "GSM1D_EP", "GSM1D_VE", "GSM1D_VED", "GSM1D_EPD"]

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GSM Real Simulation Executor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Execute GSM simulation
  python gsm_executor.py --exec GSM1D_ED --params '{"E": 30000, "S": 1.0}'
  
  # With custom loading
  python gsm_executor.py --exec GSM1D_ED --params '{"E": 25000}' \\
    --loading '{"time_array": [0,0.5,1], "strain_history": [0,0.005,0.015]}'
  
  # List models
  python gsm_executor.py --list-models

Note: This executor runs real GSM simulations using the working notebook framework.
        """
    )
    
    parser.add_argument('--list-models', action='store_true',
                       help='List available GSM models')
    
    parser.add_argument('--exec', dest='model',
                       help='Execute GSM simulation for specified model')
    
    parser.add_argument('--params', required=False,
                       help='Material parameters as JSON string')
    
    parser.add_argument('--loading', 
                       help='Loading specification as JSON string')
    
    parser.add_argument('--formulation', choices=['F', 'Helmholtz', 'G', 'Gibbs'],
                       default='F', help='Energy formulation')
    
    parser.add_argument('--json-output', action='store_true',
                       help='Output in JSON format only')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.list_models:
        models = discover_models()
        if args.json_output:
            result = {"available_models": models, "count": len(models)}
            print(json.dumps(result, indent=2))
        else:
            print("Available GSM Models:")
            print("=" * 30)
            for i, model in enumerate(models, 1):
                print(f"{i:2d}. {model}")
            print(f"\\nTotal: {len(models)} models")
        return
    
    if args.model:
        if not args.params:
            print("Error: --params required for simulation")
            return
        
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"Error parsing parameters: {e}")
            return
        
        loading_spec = None
        if args.loading:
            try:
                loading_spec = json.loads(args.loading)
            except json.JSONDecodeError as e:
                print(f"Error parsing loading: {e}")
                return
        
        # Execute simulation
        result = execute_gsm_simulation(
            args.model, params, loading_spec, args.formulation, args.verbose
        )
        
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            if result["status"] == "success":
                print("🎉 Real GSM Simulation Results")
                print("=" * 40)
                print(f"Model: {result.get('model_name', 'Unknown')}")
                if 'execution_info' in result:
                    info = result['execution_info']
                    print(f"Time steps: {info.get('time_steps', 'Unknown')}")
                    print(f"Max strain: {info.get('max_strain', 0):.6f}")
                    print(f"Max stress: {info.get('max_stress', 0):.2f} MPa")
                    print(f"Type: {info.get('simulation_type', 'Unknown')}")
                print("✅ Simulation completed successfully!")
            else:
                print("❌ Simulation failed:")
                print(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    parser.print_help()

if __name__ == "__main__":
    main()
