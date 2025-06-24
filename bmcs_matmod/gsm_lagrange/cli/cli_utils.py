#!/usr/bin/env python3
"""
Utility script for GSM CLI interface

This script provides utilities for generating example data, validating inputs,
and testing the CLI interface.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any

from .data_structures import MaterialParameterData, LoadingData, SimulationConfig
from .parameter_loader import generate_example_parameters, generate_example_loading, generate_example_config

def generate_examples(output_dir: str = "examples"):
    """Generate example input files for CLI testing"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate material parameters
    params = generate_example_parameters()
    with open(output_path / "example_params.json", 'w') as f:
        f.write(params.to_json())
    
    # Generate loading specifications
    monotonic_loading = generate_example_loading('monotonic')
    with open(output_path / "example_monotonic_loading.json", 'w') as f:
        f.write(monotonic_loading.to_json())
    
    cyclic_loading = generate_example_loading('cyclic')
    with open(output_path / "example_cyclic_loading.json", 'w') as f:
        f.write(cyclic_loading.to_json())
    
    # Generate simulation config
    config = generate_example_config()
    with open(output_path / "example_config.json", 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=2))
    
    print(f"Example files generated in {output_path}")

def create_database_schema(db_path: str = "materials.sqlite"):
    """Create SQLite database schema for material parameters"""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create materials table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            description TEXT,
            created_date TEXT,
            source TEXT,
            E REAL,
            nu REAL,
            omega_0 REAL,
            S REAL,
            r REAL,
            eta_v REAL,
            K REAL,
            G REAL
        )
    ''')
    
    # Insert example data
    example_materials = [
        (1, "Concrete C30/37", "ElasticDamage", "Standard concrete", "2024-12-20", "Lab Test A", 
         30000.0, 0.2, 0.1, 1000.0, 0.01, None, None, None),
        (2, "Steel S355", "ElasticPlastic", "Structural steel", "2024-12-20", "Material Database",
         210000.0, 0.3, None, None, None, None, 1000.0, None),
        (3, "Polymer PMMA", "Viscoelastic", "Polymethyl methacrylate", "2024-12-20", "Supplier Data",
         3000.0, 0.35, None, None, None, 100.0, None, 1200.0)
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO materials 
        (id, name, model_type, description, created_date, source, E, nu, omega_0, S, r, eta_v, K, G)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', example_materials)
    
    conn.commit()
    conn.close()
    
    print(f"Database schema created: {db_path}")

def validate_cli_examples():
    """Validate example files can be loaded correctly"""
    from .parameter_loader import ParameterLoader
    
    loader = ParameterLoader()
    examples_dir = Path("examples")
    
    if not examples_dir.exists():
        print("Examples directory not found. Run generate_examples() first.")
        return False
    
    # Test parameter loading
    try:
        params_file = examples_dir / "example_params.json"
        if params_file.exists():
            params = loader.load_parameters(str(params_file))
            print(f"✓ Parameters loaded: {len(params.parameters)} parameters")
            if not params.validate():
                print("✗ Parameter validation failed")
                return False
        else:
            print(f"✗ Parameters file not found: {params_file}")
    except Exception as e:
        print(f"✗ Parameter loading failed: {e}")
        return False
    
    # Test loading specification
    try:
        loading_file = examples_dir / "example_monotonic_loading.json"
        if loading_file.exists():
            loading = loader.load_loading(str(loading_file))
            print(f"✓ Loading loaded: {len(loading.time_array)} time steps")
            if not loading.validate():
                print("✗ Loading validation failed")
                return False
        else:
            print(f"✗ Loading file not found: {loading_file}")
    except Exception as e:
        print(f"✗ Loading specification loading failed: {e}")
        return False
    
    # Test configuration
    try:
        config_file = examples_dir / "example_config.json"
        if config_file.exists():
            config = loader.load_config(str(config_file))
            print(f"✓ Configuration loaded: tolerance={config.tolerance}")
            if not config.validate():
                print("✗ Configuration validation failed")
                return False
        else:
            print(f"✗ Configuration file not found: {config_file}")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False
    
    print("✓ All example files validated successfully")
    return True

def test_inline_parameters():
    """Test inline parameter specification"""
    from .parameter_loader import ParameterLoader
    
    loader = ParameterLoader()
    
    # Test inline parameters
    inline_params = '{"parameters": {"E": 30000, "nu": 0.2}, "material_name": "Test Material"}'
    
    try:
        params = loader.load_parameters_from_string(inline_params)
        print(f"✓ Inline parameters loaded: {params.material_name}")
        return True
    except Exception as e:
        print(f"✗ Inline parameter loading failed: {e}")
        return False

def create_aiida_examples():
    """Create example AiiDA data nodes (requires AiiDA installation)"""
    try:
        from aiida import orm
        from .parameter_loader import AiidaParameterLoader
        
        # Create Dict node with material parameters
        params_dict = {
            'parameters': {
                'E': 30000.0,
                'nu': 0.2,
                'omega_0': 0.1
            },
            'material_name': 'AiiDA Test Material',
            'model_type': 'ElasticDamage'
        }
        
        node = orm.Dict(dict=params_dict)
        node.store()
        
        print(f"✓ AiiDA parameter node created: {node.id}")
        
        # Test loading
        aiida_loader = AiidaParameterLoader()
        loaded_params = aiida_loader.load_parameters_from_node(node.id)
        print(f"✓ AiiDA parameters loaded: {loaded_params.material_name}")
        
        return node.id
        
    except ImportError:
        print("AiiDA not available. Skipping AiiDA examples.")
        return None
    except Exception as e:
        print(f"✗ AiiDA example creation failed: {e}")
        return None

def create_cli_test_script():
    """Create a test script that demonstrates CLI usage"""
    
    script_content = '''#!/bin/bash
# GSM CLI Test Script

echo "Testing GSM CLI Interface"
echo "========================="

# Test 1: Validate example inputs
echo "Test 1: Validating example inputs"
python -m bmcs_matmod.gsm_lagrange.cli_interface \\
    --model ElasticDamage \\
    --formulation F \\
    --params examples/example_params.json \\
    --loading examples/example_monotonic_loading.json \\
    --config examples/example_config.json \\
    --validate-only

# Test 2: Run simulation with file inputs
echo ""
echo "Test 2: Running simulation with file inputs"
python -m bmcs_matmod.gsm_lagrange.cli_interface \\
    --model ElasticDamage \\
    --formulation F \\
    --params examples/example_params.json \\
    --loading examples/example_monotonic_loading.json \\
    --config examples/example_config.json \\
    --output results_test2.json \\
    --verbose

# Test 3: Run with inline parameters
echo ""
echo "Test 3: Running with inline parameters"
python -m bmcs_matmod.gsm_lagrange.cli_interface \\
    --model ElasticDamage \\
    --formulation F \\
    --params-inline '{"E": 30000, "nu": 0.2, "omega_0": 0.1, "S": 1000, "r": 0.01}' \\
    --loading examples/example_monotonic_loading.json \\
    --output results_test3.json

# Test 4: Stress-controlled simulation
echo ""
echo "Test 4: Stress-controlled (Gibbs) simulation"
python -m bmcs_matmod.gsm_lagrange.cli_interface \\
    --model ElasticDamage \\
    --formulation G \\
    --params examples/example_params.json \\
    --loading examples/example_cyclic_stress_loading.json \\
    --output results_test4.json

# Test 5: Database parameters (if available)
echo ""
echo "Test 5: Database parameters"
python -m bmcs_matmod.gsm_lagrange.cli_interface \\
    --model ElasticDamage \\
    --formulation F \\
    --params "sqlite:///materials.sqlite?table=materials&id=1" \\
    --loading examples/example_monotonic_loading.json \\
    --output results_test5.json

echo ""
echo "CLI tests completed. Check output files for results."
'''
    
    with open("test_cli.sh", 'w') as f:
        f.write(script_content)
    
    # Make executable
    import os
    os.chmod("test_cli.sh", 0o755)
    
    print("CLI test script created: test_cli.sh")

def main():
    parser = argparse.ArgumentParser(description="GSM CLI Utilities")
    parser.add_argument('command', choices=[
        'generate-examples', 'create-database', 'validate-examples', 
        'test-inline', 'create-aiida', 'create-test-script'
    ])
    parser.add_argument('--output-dir', default='examples', help='Output directory for examples')
    parser.add_argument('--db-path', default='materials.sqlite', help='Database path')
    
    args = parser.parse_args()
    
    if args.command == 'generate-examples':
        generate_examples(args.output_dir)
    elif args.command == 'create-database':
        create_database_schema(args.db_path)
    elif args.command == 'validate-examples':
        validate_cli_examples()
    elif args.command == 'test-inline':
        test_inline_parameters()
    elif args.command == 'create-aiida':
        create_aiida_examples()
    elif args.command == 'create-test-script':
        create_cli_test_script()

if __name__ == "__main__":
    main()
