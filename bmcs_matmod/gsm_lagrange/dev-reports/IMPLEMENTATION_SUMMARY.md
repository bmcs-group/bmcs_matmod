# GSM CLI Interface Implementation Summary

## Overview

I've implemented a comprehensive Command Line Interface (CLI) for the GSM (Generalized Standard Material) framework that enables execution of material models with parameters from various sources including databases, JSON files, and network transfers.

## Key Components Created

### 1. Core CLI Interface (`cli_interface.py`)
- **GSMModelCLI**: Main CLI class with argument parsing and execution logic
- Supports both Helmholtz (strain-controlled) and Gibbs (stress-controlled) formulations
- Automatic model discovery and validation
- Comprehensive error handling and logging

### 2. Data Structures (`data_structures.py`)
- **MaterialParameterData**: Typed container for material parameters with metadata
- **LoadingData**: Container for loading specifications (strain/stress histories)
- **SimulationConfig**: Configuration for simulation control
- **SimulationResults**: Complete simulation results with metadata
- JSON serialization/deserialization for all data types

### 3. Parameter Loading (`parameter_loader.py`)
- **ParameterLoader**: Unified interface for loading from multiple sources
- Support for:
  - JSON files (local and remote via HTTP/HTTPS)
  - SQLite databases with URI syntax
  - Database queries (PostgreSQL/MySQL - placeholder implementation)
  - Inline JSON strings
  - Network URLs
- **AiidaParameterLoader**: Integration with AiiDA database (optional)

### 4. Utilities and Examples
- **cli_utils.py**: Utility functions for generating examples, database setup, validation
- **demo_cli.py**: Comprehensive demonstration script
- Example JSON files for parameters, loading, and configuration
- Test scripts and documentation

## CLI Usage Examples

### Basic Usage
```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params material_params.json \
    --loading loading_spec.json \
    --output results.json
```

### Inline Parameters
```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params-inline '{"E": 30000, "nu": 0.2, "omega_0": 0.1}' \
    --loading loading_spec.json
```

### Database Integration
```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params "sqlite:///materials.sqlite?table=materials&id=123" \
    --loading loading_spec.json
```

### Network Parameters
```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params "https://api.materials-db.org/parameters/123" \
    --loading loading_spec.json
```

## Data Type Vocabulary for AiiDA Integration

### Parameter Types
The implementation defines a clear vocabulary of input types that can be represented as AiiDA data nodes:

1. **Scalar Parameters**: Simple float/int values (E, nu, etc.)
2. **Time-Dependent Parameters**: Arrays representing parameter evolution
3. **Stress-Strain Relations**: Tabular data for complex constitutive laws
4. **Energy-Strain Relations**: Lookup tables for energy-based formulations

### Data Node Schema
```python
# Material Parameter Node
{
    "parameters": {
        "E": 30000.0,           # Young's modulus
        "nu": 0.2,              # Poisson's ratio  
        "omega_0": 0.1,         # Initial damage
        "S": 1000.0,            # Damage parameter
        "r": 0.01               # Damage evolution rate
    },
    "metadata": {
        "material_name": "Concrete C30/37",
        "model_type": "ElasticDamage",
        "calibration_date": "2024-12-20",
        "validation_data": {...}
    }
}

# Loading Specification Node  
{
    "time_array": [0.0, 0.5, 1.0],
    "strain_history": [0.0, 0.005, 0.01],
    "loading_metadata": {
        "loading_type": "monotonic",
        "max_amplitude": 0.01,
        "temperature": 20.0
    }
}
```

## Remote Execution Capabilities

### SSH Execution
```bash
ssh compute-node "cd /simulation && \
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params parameters.json \
    --loading loading.json \
    --output results.json"
```

### Network Parameter Transfer
The system supports fetching parameters from remote databases or APIs:
- REST API endpoints
- Shared database servers
- Cloud storage systems
- AiiDA remote repositories

### Database → Data Files → Python Objects Pipeline

The implementation provides a clear transformation pipeline:

1. **Database Storage**: Parameters stored in relational/document databases
2. **Network Transfer**: JSON/binary serialization for network transport
3. **File Format**: Standardized JSON schema for local storage
4. **Python Objects**: Typed dataclasses with validation
5. **GSM Execution**: Direct integration with GSM model interface

## Key Design Features

### Type Safety and Validation
- Comprehensive input validation before simulation
- Parameter bounds checking
- Loading data consistency verification
- Clear error messages and logging

### Extensibility
- Plugin architecture for new parameter sources
- Modular design for different database backends
- Support for custom data transformations
- Easy integration with workflow systems

### Network and Database Integration
- URI-based parameter source specification
- Connection pooling and error recovery
- Caching for performance optimization
- Authentication and security considerations

### AiiDA Compatibility
- Direct integration with AiiDA data nodes
- Provenance tracking through source metadata
- Workflow integration capabilities
- Query-based parameter discovery

## Installation and Setup

1. **Generate Examples**:
```bash
python -m bmcs_matmod.gsm_lagrange.cli_utils generate-examples
```

2. **Create Database**:
```bash
python -m bmcs_matmod.gsm_lagrange.cli_utils create-database
```

3. **Run Demo**:
```bash
python -m bmcs_matmod.gsm_lagrange.demo_cli
```

4. **Test CLI**:
```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface --help
```

## File Structure Created

```
bmcs_matmod/gsm_lagrange/
├── cli_interface.py          # Main CLI implementation
├── data_structures.py        # Typed data containers
├── parameter_loader.py       # Multi-source parameter loading
├── cli_utils.py             # Utilities and setup functions
├── demo_cli.py              # Demonstration script
├── CLI_README.md            # Comprehensive documentation
├── examples/
│   ├── material_params.json
│   ├── monotonic_loading.json
│   ├── cyclic_stress_loading.json
│   ├── simulation_config.json
│   └── example_usage.py
└── __init__.py              # Updated with CLI exports
```

## Benefits for Your Workflow

1. **Standardized Interface**: Consistent parameter specification across all models
2. **Database Integration**: Direct connection to experimental databases and calibration systems
3. **Remote Execution**: SSH and network-based parameter transfer for HPC environments
4. **AiiDA Compatibility**: Seamless integration with existing AiiDA workflows
5. **Type Safety**: Comprehensive validation prevents runtime errors
6. **Extensibility**: Easy to add new parameter sources and model types
7. **Documentation**: Complete examples and usage patterns

This implementation provides the foundation for a robust material modeling pipeline that can scale from local development to production HPC environments with database integration and workflow automation.
