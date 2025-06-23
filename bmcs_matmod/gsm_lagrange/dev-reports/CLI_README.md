# GSM CLI Interface

This document describes the Command Line Interface (CLI) for executing GSM (Generalized Standard Material) models with material parameters from various sources.

## Overview

The GSM CLI interface provides a standardized way to execute material model simulations with parameters sourced from:
- JSON files
- Database queries (SQLite, PostgreSQL, etc.)
- Network URLs (HTTP/HTTPS)
- Inline JSON strings
- AiiDA database nodes

## Installation

The CLI interface is part of the `bmcs_matmod.gsm_lagrange` package. No additional installation is required beyond the main package dependencies.

## Basic Usage

```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface [OPTIONS]
```

### Required Arguments

- `--model`: GSM model to execute (e.g., `ElasticDamage`, `ViscoElastic`)
- `--formulation`: Energy formulation (`F`/`Helmholtz` for strain-controlled, `G`/`Gibbs` for stress-controlled)
- `--params` or `--params-inline`: Material parameter source
- `--loading`: Loading specification source

### Optional Arguments

- `--output`: Output file path (JSON format)
- `--config`: Simulation configuration file
- `--validate-only`: Only validate inputs without running simulation
- `--verbose`: Verbose output

## Parameter Sources

### 1. JSON Files

```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params material_params.json \
    --loading loading_spec.json \
    --output results.json
```

Example parameter file (`material_params.json`):
```json
{
  "parameters": {
    "E": 30000.0,
    "nu": 0.2,
    "omega_0": 0.1,
    "S": 1000.0,
    "r": 0.01
  },
  "material_name": "Concrete C30/37",
  "model_type": "ElasticDamage",
  "description": "Calibrated from experimental data"
}
```

### 2. Inline JSON

```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params-inline '{"E": 30000, "nu": 0.2, "omega_0": 0.1, "S": 1000, "r": 0.01}' \
    --loading loading_spec.json
```

### 3. Database Sources

#### SQLite Database
```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params "sqlite:///materials.sqlite?table=materials&id=123" \
    --loading loading_spec.json
```

#### PostgreSQL/MySQL (placeholder for future implementation)
```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params "db://localhost:5432/materials?table=materials&id=123" \
    --loading loading_spec.json
```

### 4. Network URLs

```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params "https://api.materials-db.org/parameters/123" \
    --loading loading_spec.json
```

## Loading Specifications

Loading specifications define the mechanical loading history applied to the material.

### Strain-Controlled Loading (Helmholtz Formulation)

Example loading file (`strain_loading.json`):
```json
{
  "time_array": [0.0, 0.5, 1.0],
  "strain_history": [0.0, 0.005, 0.01],
  "loading_type": "monotonic",
  "max_amplitude": 0.01,
  "temperature": 20.0,
  "description": "Monotonic tensile loading to 1% strain"
}
```

### Stress-Controlled Loading (Gibbs Formulation)

Example loading file (`stress_loading.json`):
```json
{
  "time_array": [0.0, 1.0, 2.0, 3.0],
  "stress_history": [0.0, 40.0, -40.0, 0.0],
  "loading_type": "cyclic",
  "max_amplitude": 40.0,
  "frequency": 0.5,
  "description": "Cyclic stress loading ±40 MPa"
}
```

## Simulation Configuration

Optional configuration file to control simulation parameters:

```json
{
  "tolerance": 1e-6,
  "max_iterations": 200,
  "step_size_control": true,
  "save_internal_variables": true,
  "debug_output": false
}
```

## Output Format

Results are saved in JSON format containing:

```json
{
  "model_name": "ElasticDamage",
  "formulation": "F",
  "parameters": {...},
  "loading": {...},
  "response": {
    "time": [...],
    "strain": [...],
    "stress": [...],
    "internal_variables": [...],
    "thermodynamic_forces": [...]
  },
  "execution_time": 1.234,
  "convergence_info": {...}
}
```

## Examples

### Example 1: Basic Strain-Controlled Simulation

```bash
# Generate example files
python -m bmcs_matmod.gsm_lagrange.cli_utils generate-examples

# Run simulation
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params examples/example_params.json \
    --loading examples/example_monotonic_loading.json \
    --config examples/example_config.json \
    --output results.json \
    --verbose
```

### Example 2: Stress-Controlled Simulation

```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation G \
    --params examples/example_params.json \
    --loading examples/example_cyclic_stress_loading.json \
    --output stress_controlled_results.json
```

### Example 3: Database Parameters

```bash
# Create example database
python -m bmcs_matmod.gsm_lagrange.cli_utils create-database

# Run with database parameters
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params "sqlite:///materials.sqlite?table=materials&id=1" \
    --loading examples/example_monotonic_loading.json \
    --output db_results.json
```

### Example 4: Validation Only

```bash
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params examples/example_params.json \
    --loading examples/example_monotonic_loading.json \
    --validate-only
```

## AiiDA Integration

If AiiDA is installed, parameters can be loaded from AiiDA database nodes:

```python
from bmcs_matmod.gsm_lagrange.parameter_loader import AiidaParameterLoader

# Load parameters from AiiDA node
loader = AiidaParameterLoader()
params = loader.load_parameters_from_node(node_id=123)

# Query parameters
results = loader.load_parameters_from_query(
    attributes={'model_type': 'ElasticDamage'}
)
```

## Data Structure Design

### MaterialParameterData

Typed container for material parameters with metadata:
- Core parameters (dictionary of name -> value)
- Material metadata (name, type, description)
- Parameter bounds and validation info
- Source information (file, database, URL)

### LoadingData

Container for loading specifications:
- Time array
- Strain/stress history arrays
- Loading metadata (type, rate, amplitude)
- Environmental conditions (temperature, humidity)

### SimulationConfig

Configuration for simulation control:
- Numerical parameters (tolerance, iterations)
- Output control (frequency, variables to save)
- Parallel computation settings

### SimulationResults

Complete simulation results with metadata:
- Model identification and parameters
- Input data (loading, configuration)
- Response data from GSM engine
- Execution statistics and convergence info

## Remote Execution

The CLI interface is designed for remote execution scenarios:

### SSH Execution
```bash
ssh user@remote-host "cd /path/to/simulation && \
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params 'parameters.json' \
    --loading 'loading.json' \
    --output 'results.json'"
```

### Network Parameter Transfer
```bash
# Parameters fetched from network
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params "https://materials-db.example.com/api/params/123" \
    --loading local_loading.json \
    --output results.json
```

### Database Integration
```bash
# Parameters from shared database
python -m bmcs_matmod.gsm_lagrange.cli_interface \
    --model ElasticDamage \
    --formulation F \
    --params "db://shared-db:5432/materials?id=123" \
    --loading loading.json \
    --output results.json
```

## Utilities

The `cli_utils.py` module provides utilities for:
- Generating example files
- Creating database schemas
- Validating input files
- Testing CLI functionality

```bash
# Generate all examples
python -m bmcs_matmod.gsm_lagrange.cli_utils generate-examples

# Create SQLite database with example data
python -m bmcs_matmod.gsm_lagrange.cli_utils create-database

# Validate example files
python -m bmcs_matmod.gsm_lagrange.cli_utils validate-examples

# Create test script
python -m bmcs_matmod.gsm_lagrange.cli_utils create-test-script
```

## Error Handling

The CLI interface provides comprehensive error handling:
- Input validation before simulation
- Parameter bounds checking
- Loading data validation
- Detailed error messages and logging
- Graceful handling of network/database failures

## Future Extensions

The design allows for future extensions:
- Additional database backends (PostgreSQL, MongoDB)
- Cloud storage integration (S3, Azure Blob)
- Real-time parameter streaming
- Distributed computation support
- Advanced optimization workflows
