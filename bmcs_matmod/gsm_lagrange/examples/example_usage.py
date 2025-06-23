#!/usr/bin/env python3
"""
Example script demonstrating GSM CLI interface usage

This script shows how to use the GSM CLI interface for various scenarios
including parameter loading from different sources and both strain and
stress-controlled simulations.
"""

import os
import json
import tempfile
from pathlib import Path

# Import CLI components
from bmcs_matmod.gsm_lagrange.cli_interface import GSMModelCLI
from bmcs_matmod.gsm_lagrange.parameter_loader import (
    generate_example_parameters, 
    generate_example_loading, 
    generate_example_config
)
from bmcs_matmod.gsm_lagrange.data_structures import create_monotonic_loading, create_cyclic_loading

def example_1_basic_usage():
    """Example 1: Basic strain-controlled simulation"""
    print("Example 1: Basic strain-controlled simulation")
    print("=" * 50)
    
    # Create temporary directory for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Generate example parameter file
        params = generate_example_parameters()
        params_file = temp_path / "params.json"
        with open(params_file, 'w') as f:
            f.write(params.to_json())
        
        # Generate loading specification
        loading = create_monotonic_loading(max_strain=0.01, n_steps=50)
        loading_file = temp_path / "loading.json"
        with open(loading_file, 'w') as f:
            f.write(loading.to_json())
        
        # Generate configuration
        config = generate_example_config()
        config_file = temp_path / "config.json"
        with open(config_file, 'w') as f:
            f.write(json.dumps(config.to_dict(), indent=2))
        
        # Results file
        results_file = temp_path / "results.json"
        
        print(f"Parameter file: {params_file}")
        print(f"Loading file: {loading_file}")
        print(f"Config file: {config_file}")
        print(f"Results file: {results_file}")
        
        # Prepare CLI arguments
        cli_args = [
            '--model', 'ElasticDamage',  # This would need to be implemented
            '--formulation', 'F',
            '--params', str(params_file),
            '--loading', str(loading_file),
            '--config', str(config_file),
            '--output', str(results_file),
            '--validate-only'  # Only validate for this example
        ]
        
        print(f"CLI command: python -m bmcs_matmod.gsm_lagrange.cli_interface {' '.join(cli_args)}")
        
        # Note: Actual execution would require GSM model implementations
        print("Note: This would execute the CLI with the generated files")

def example_2_inline_parameters():
    """Example 2: Using inline parameters"""
    print("\nExample 2: Using inline parameters")
    print("=" * 50)
    
    # Create inline parameter string
    inline_params = {
        "E": 30000.0,
        "nu": 0.2,
        "omega_0": 0.1,
        "S": 1000.0,
        "r": 0.01
    }
    
    # Create temporary loading file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        loading = create_cyclic_loading(amplitude=0.005, n_cycles=3, steps_per_cycle=20)
        loading_file = temp_path / "cyclic_loading.json"
        with open(loading_file, 'w') as f:
            f.write(loading.to_json())
        
        results_file = temp_path / "cyclic_results.json"
        
        cli_args = [
            '--model', 'ElasticDamage',
            '--formulation', 'F',
            '--params-inline', f"'{json.dumps(inline_params)}'",
            '--loading', str(loading_file),
            '--output', str(results_file),
            '--verbose'
        ]
        
        print(f"CLI command: python -m bmcs_matmod.gsm_lagrange.cli_interface {' '.join(cli_args)}")

def example_3_stress_controlled():
    """Example 3: Stress-controlled (Gibbs) simulation"""
    print("\nExample 3: Stress-controlled (Gibbs) simulation")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Generate parameters
        params = generate_example_parameters()
        params_file = temp_path / "params.json"
        with open(params_file, 'w') as f:
            f.write(params.to_json())
        
        # Create stress-controlled loading
        loading = create_cyclic_loading(amplitude=40.0, n_cycles=2, 
                                      steps_per_cycle=25, loading_type='stress')
        loading_file = temp_path / "stress_loading.json"
        with open(loading_file, 'w') as f:
            f.write(loading.to_json())
        
        results_file = temp_path / "stress_results.json"
        
        cli_args = [
            '--model', 'ElasticDamage',
            '--formulation', 'G',  # Gibbs formulation for stress control
            '--params', str(params_file),
            '--loading', str(loading_file),
            '--output', str(results_file)
        ]
        
        print(f"CLI command: python -m bmcs_matmod.gsm_lagrange.cli_interface {' '.join(cli_args)}")

def example_4_database_scenario():
    """Example 4: Database integration scenario"""
    print("\nExample 4: Database integration scenario")
    print("=" * 50)
    
    # This demonstrates how database URIs would be used
    database_scenarios = [
        {
            'name': 'SQLite local database',
            'uri': 'sqlite:///materials.sqlite?table=materials&id=1',
            'description': 'Load from local SQLite database'
        },
        {
            'name': 'PostgreSQL remote database',
            'uri': 'db://materials-server:5432/production?table=calibrated_params&id=C30_37_batch_A',
            'description': 'Load from remote PostgreSQL database'
        },
        {
            'name': 'HTTP API endpoint',
            'uri': 'https://api.materials-database.org/parameters/concrete/C30-37',
            'description': 'Load from REST API'
        }
    ]
    
    for scenario in database_scenarios:
        print(f"\n{scenario['name']}:")
        print(f"Description: {scenario['description']}")
        
        cli_args = [
            '--model', 'ElasticDamage',
            '--formulation', 'F',
            '--params', f'"{scenario["uri"]}"',
            '--loading', 'loading_spec.json',
            '--output', f'{scenario["name"].lower().replace(" ", "_")}_results.json'
        ]
        
        print(f"CLI command: python -m bmcs_matmod.gsm_lagrange.cli_interface {' '.join(cli_args)}")

def example_5_aiida_integration():
    """Example 5: AiiDA integration"""
    print("\nExample 5: AiiDA integration")
    print("=" * 50)
    
    print("AiiDA integration allows loading parameters from workflow nodes:")
    print()
    
    # Python code example
    python_code = '''
from bmcs_matmod.gsm_lagrange.parameter_loader import AiidaParameterLoader
from aiida import orm

# Load parameters from specific AiiDA node
loader = AiidaParameterLoader()
params = loader.load_parameters_from_node(node_id=12345)

# Query parameters from AiiDA database
results = loader.load_parameters_from_query(
    attributes={'material_type': 'concrete'},
    extras={'calibration_source': 'experimental'}
)

# Use in CLI via programmatic interface
cli = GSMModelCLI()
simulation_results = cli.run_simulation(
    model_class=ElasticDamage,
    params=params,
    loading=loading_spec,
    config=simulation_config,
    formulation='F'
)
'''
    
    print("Python code example:")
    print(python_code)

def example_6_remote_execution():
    """Example 6: Remote execution scenarios"""
    print("\nExample 6: Remote execution scenarios")
    print("=" * 50)
    
    remote_scenarios = [
        {
            'name': 'SSH execution with local files',
            'command': '''ssh compute-node "cd /simulation/workspace && \\
python -m bmcs_matmod.gsm_lagrange.cli_interface \\
    --model ElasticDamage \\
    --formulation F \\
    --params parameters.json \\
    --loading loading.json \\
    --output results.json"'''
        },
        {
            'name': 'SSH execution with network parameters',
            'command': '''ssh compute-node "cd /simulation/workspace && \\
python -m bmcs_matmod.gsm_lagrange.cli_interface \\
    --model ElasticDamage \\
    --formulation F \\
    --params 'https://materials-db.org/api/params/123' \\
    --loading loading.json \\
    --output results.json"'''
        },
        {
            'name': 'Docker container execution',
            'command': '''docker run -v $(pwd):/workspace gsm-materials \\
python -m bmcs_matmod.gsm_lagrange.cli_interface \\
    --model ElasticDamage \\
    --formulation F \\
    --params /workspace/params.json \\
    --loading /workspace/loading.json \\
    --output /workspace/results.json'''
        }
    ]
    
    for scenario in remote_scenarios:
        print(f"\n{scenario['name']}:")
        print(scenario['command'])

def main():
    """Run all examples"""
    print("GSM CLI Interface Examples")
    print("=" * 60)
    
    example_1_basic_usage()
    example_2_inline_parameters()
    example_3_stress_controlled()
    example_4_database_scenario()
    example_5_aiida_integration()
    example_6_remote_execution()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nTo run the actual CLI interface:")
    print("1. First generate example files:")
    print("   python -m bmcs_matmod.gsm_lagrange.cli_utils generate-examples")
    print("2. Then run simulations:")
    print("   python -m bmcs_matmod.gsm_lagrange.cli_interface --help")

if __name__ == "__main__":
    main()
