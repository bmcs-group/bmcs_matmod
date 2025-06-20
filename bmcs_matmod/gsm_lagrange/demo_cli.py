#!/usr/bin/env python3
"""
Demo script for GSM CLI Interface

This script demonstrates the CLI interface functionality by creating example
data and running simulations programmatically.
"""

import json
import tempfile
from pathlib import Path
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

from data_structures import (
    MaterialParameterData, 
    LoadingData, 
    SimulationConfig,
    create_monotonic_loading,
    create_cyclic_loading
)
from parameter_loader import ParameterLoader
from cli_interface import GSMModelCLI

def create_demo_data():
    """Create demo data for testing"""
    
    # Create material parameters
    params = MaterialParameterData(
        parameters={
            'E': 30000.0,
            'nu': 0.2,
            'omega_0': 0.1,
            'S': 1000.0,
            'r': 0.01
        },
        material_name="Demo Concrete",
        model_type="ElasticDamage",
        description="Demo parameters for CLI testing"
    )
    
    # Create loading specifications
    strain_loading = create_monotonic_loading(max_strain=0.01, n_steps=20)
    stress_loading = create_cyclic_loading(amplitude=30.0, n_cycles=2, 
                                         steps_per_cycle=15, loading_type='stress')
    
    # Create configuration
    config = SimulationConfig(
        tolerance=1e-6,
        max_iterations=100,
        save_internal_variables=True,
        debug_output=True
    )
    
    return params, strain_loading, stress_loading, config

def demo_parameter_loading():
    """Demonstrate parameter loading from different sources"""
    print("Demo 1: Parameter Loading")
    print("-" * 30)
    
    loader = ParameterLoader()
    
    # Test inline JSON parameters
    inline_json = '{"E": 25000, "nu": 0.25, "omega_0": 0.05}'
    try:
        params = loader.load_parameters_from_string(inline_json)
        print(f"✓ Loaded inline parameters: {list(params.parameters.keys())}")
    except Exception as e:
        print(f"✗ Inline parameter loading failed: {e}")
    
    # Test file-based loading
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        demo_params, _, _, _ = create_demo_data()
        f.write(demo_params.to_json())
        f.flush()
        
        try:
            params = loader.load_parameters(f.name)
            print(f"✓ Loaded file parameters: {params.material_name}")
        except Exception as e:
            print(f"✗ File parameter loading failed: {e}")
        finally:
            os.unlink(f.name)

def demo_cli_validation():
    """Demonstrate CLI validation functionality"""
    print("\nDemo 2: CLI Validation")
    print("-" * 30)
    
    cli = GSMModelCLI()
    params, strain_loading, _, config = create_demo_data()
    
    # Get a model class for testing
    model_classes = cli.available_models
    if not model_classes:
        print("✗ No model classes available")
        return
    
    model_name = list(model_classes.keys())[0]
    model_class = model_classes[model_name]
    
    print(f"Testing with model: {model_name}")
    
    # Test validation
    try:
        is_valid = cli.validate_inputs(model_class, params, strain_loading, config)
        if is_valid:
            print("✓ Input validation passed")
        else:
            print("✗ Input validation failed")
    except Exception as e:
        print(f"✗ Validation error: {e}")

def demo_simulation_run():
    """Demonstrate running a simulation"""
    print("\nDemo 3: Simulation Execution")
    print("-" * 30)
    
    cli = GSMModelCLI()
    params, strain_loading, stress_loading, config = create_demo_data()
    
    # Get a model class for testing
    model_classes = cli.available_models
    if not model_classes:
        print("✗ No model classes available")
        return
    
    model_name = list(model_classes.keys())[0]
    model_class = model_classes[model_name]
    
    print(f"Running simulation with model: {model_name}")
    
    # Test strain-controlled simulation
    try:
        results = cli.run_simulation(
            model_class=model_class,
            params=params,
            loading=strain_loading,
            config=config,
            formulation='F'
        )
        print("✓ Strain-controlled simulation completed")
        print(f"  - Model: {results.model_name}")
        print(f"  - Formulation: {results.formulation}")
        print(f"  - Time steps: {len(results.response.t_t)}")
        print(f"  - Warnings: {len(results.warnings or [])}")
        
    except Exception as e:
        print(f"✗ Strain-controlled simulation failed: {e}")
    
    # Test stress-controlled simulation
    try:
        results = cli.run_simulation(
            model_class=model_class,
            params=params,
            loading=stress_loading,
            config=config,
            formulation='G'
        )
        print("✓ Stress-controlled simulation completed")
        print(f"  - Model: {results.model_name}")
        print(f"  - Formulation: {results.formulation}")
        print(f"  - Time steps: {len(results.response.t_t)}")
        
    except Exception as e:
        print(f"✗ Stress-controlled simulation failed: {e}")

def demo_file_operations():
    """Demonstrate file I/O operations"""
    print("\nDemo 4: File Operations")
    print("-" * 30)
    
    params, strain_loading, _, config = create_demo_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save example files
        params_file = temp_path / "demo_params.json"
        loading_file = temp_path / "demo_loading.json"
        config_file = temp_path / "demo_config.json"
        
        with open(params_file, 'w') as f:
            f.write(params.to_json())
        
        with open(loading_file, 'w') as f:
            f.write(strain_loading.to_json())
        
        with open(config_file, 'w') as f:
            f.write(json.dumps(config.to_dict(), indent=2))
        
        print(f"✓ Created demo files in {temp_dir}")
        print(f"  - Parameters: {params_file.name}")
        print(f"  - Loading: {loading_file.name}")
        print(f"  - Config: {config_file.name}")
        
        # Test loading files
        loader = ParameterLoader()
        
        try:
            loaded_params = loader.load_parameters(str(params_file))
            loaded_loading = loader.load_loading(str(loading_file))
            loaded_config = loader.load_config(str(config_file))
            
            print("✓ Successfully loaded all files")
            print(f"  - Parameters: {len(loaded_params.parameters)} items")
            print(f"  - Loading: {len(loaded_loading.time_array)} time steps")
            print(f"  - Config: tolerance = {loaded_config.tolerance}")
            
        except Exception as e:
            print(f"✗ File loading failed: {e}")

def demo_command_line_interface():
    """Demonstrate command line interface construction"""
    print("\nDemo 5: Command Line Interface")
    print("-" * 30)
    
    cli = GSMModelCLI()
    parser = cli.create_parser()
    
    # Show help
    print("Available CLI options:")
    print("---------------------")
    
    # Simulate help output
    test_args = ['--help']
    try:
        parser.parse_args(test_args)
    except SystemExit:
        pass  # argparse calls sys.exit() for --help
    
    # Show example command construction
    print("\nExample CLI commands:")
    print("--------------------")
    
    example_commands = [
        [
            '--model', 'ElasticDamage',
            '--formulation', 'F',
            '--params', 'material_params.json',
            '--loading', 'loading_spec.json',
            '--output', 'results.json',
            '--validate-only'
        ],
        [
            '--model', 'ElasticDamage',
            '--formulation', 'G',
            '--params-inline', '{"E": 30000, "nu": 0.2}',
            '--loading', 'stress_loading.json',
            '--output', 'stress_results.json',
            '--verbose'
        ]
    ]
    
    for i, cmd_args in enumerate(example_commands, 1):
        print(f"\nExample {i}:")
        print(f"python -m bmcs_matmod.gsm_lagrange.cli_interface {' '.join(cmd_args)}")
        
        # Test argument parsing
        try:
            args = parser.parse_args(cmd_args)
            print(f"  ✓ Arguments parsed successfully")
            print(f"    Model: {args.model}")
            print(f"    Formulation: {args.formulation}")
        except Exception as e:
            print(f"  ✗ Argument parsing failed: {e}")

def main():
    """Run all demonstrations"""
    print("GSM CLI Interface Demonstration")
    print("=" * 50)
    print()
    
    try:
        demo_parameter_loading()
        demo_cli_validation()
        demo_simulation_run()
        demo_file_operations()
        demo_command_line_interface()
        
        print("\n" + "=" * 50)
        print("✓ All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Create example files: python -m bmcs_matmod.gsm_lagrange.cli_utils generate-examples")
        print("2. Run CLI interface: python -m bmcs_matmod.gsm_lagrange.cli_interface --help")
        
    except Exception as e:
        print(f"\n✗ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
