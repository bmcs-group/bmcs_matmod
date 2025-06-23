# GSM CLI Network Interface Demo

This notebook demonstrates the enhanced GSM CLI interface with network communication capabilities, parameter specification retrieval, and cross-network computational node communication.

## Overview

The new `cli_gsm.py` provides:
- **Dynamic Parameter Specification**: Retrieve parameter requirements from GSM definitions
- **Network Communication**: Execute simulations via network requests  
- **Parameter Validation**: Validate parameters against GSM specifications
- **Serialization Support**: Full JSON serialization for network transfer
- **Computational Node Mode**: Server mode for workchain manager integration

## 1. Basic CLI Operations

### List Available Models

```bash
python cli_gsm.py --list-models
```

### Get Parameter Specification (Dynamic Validation)

```bash
# Get parameter specification for GSM1D_ED model
python cli_gsm.py --get-param-spec GSM1D_ED --json-output
```

This returns the parameter specification that can be used for:
- Dynamic widget creation in Jupyter interfaces
- Input validation before submission
- Parameter bounds checking

## 2. Network Communication Examples

### Execute Simulation with Inline Parameters

```bash
python cli_gsm.py --model GSM1D_ED --formulation F \
  --params-inline '{"E": 30000, "S": 1.0, "c": 2.0, "r": 0.5, "eps_0": 0.001}' \
  --loading-inline '{"time_array": [0, 0.5, 1.0], "strain_history": [0, 0.005, 0.01], "loading_type": "monotonic"}' \
  --json-output
```

### Validate Parameters Only

```bash
python cli_gsm.py --model GSM1D_ED --formulation F \
  --params-inline '{"E": 30000, "S": 1.0, "c": 2.0}' \
  --loading-inline '{"time_array": [0, 1], "strain_history": [0, 0.01]}' \
  --validate-only --json-output
```

## 3. Server Mode for Computational Nodes

### Start GSM Server

```bash
python cli_gsm.py --serve --port 8888
```

This starts a HTTP server with endpoints:
- `GET /models` - List available models
- `GET /param-spec/<model>` - Get parameter specification
- `POST /` - Execute simulation from JSON request

### Network Request Examples

#### Get Available Models
```bash
curl http://localhost:8888/models
```

#### Get Parameter Specification
```bash
curl http://localhost:8888/param-spec/GSM1D_ED
```

#### Execute Simulation
```bash
curl -X POST http://localhost:8888/ \
  -H "Content-Type: application/json" \
  -d @simulation_request.json
```

## 4. File-Based Execution

### Create Request File

Create `simulation_request.json`:
```json
{
  "model": "GSM1D_ED",
  "formulation": "F",
  "parameters": {
    "parameters": {
      "E": 30000.0,
      "S": 1.0,
      "c": 2.0,
      "r": 0.5,
      "eps_0": 0.001
    },
    "material_name": "Test Material",
    "model_type": "ElasticDamage"
  },
  "loading": {
    "time_array": [0.0, 0.5, 1.0],
    "strain_history": [0.0, 0.005, 0.01],
    "loading_type": "monotonic",
    "max_amplitude": 0.01
  },
  "config": {
    "tolerance": 1e-6,
    "max_iterations": 100,
    "save_internal_variables": true
  }
}
```

### Execute from File
```bash
python cli_gsm.py --execute-request simulation_request.json --json-output
```

## 5. Integration with Workchain Managers

The CLI can be integrated with workchain managers (like AiiDA) for:

### Remote Execution
```python
# In workchain manager
import subprocess
import json

# Prepare request
request_data = {
    "model": "GSM1D_ED",
    "formulation": "F", 
    "parameters": {...},
    "loading": {...}
}

# Execute on remote computational node
result = subprocess.run([
    "ssh", "compute-node",
    f"cd /work/gsm && python cli_gsm.py --execute-request -",
    "--json-output"
], input=json.dumps(request_data), capture_output=True, text=True)

# Parse results
simulation_results = json.loads(result.stdout)
```

### Parameter Specification Retrieval
```python
# Get parameter specification for dynamic UI
spec_result = subprocess.run([
    "ssh", "compute-node", 
    "python cli_gsm.py --get-param-spec GSM1D_ED --json-output"
], capture_output=True, text=True)

param_spec = json.loads(spec_result.stdout)

# Use specification to create dynamic widgets
for param_name, spec in param_spec['parameter_specification']['parameters'].items():
    create_widget(param_name, spec)
```

## 6. Data Structures and Serialization

The enhanced CLI uses comprehensive data structures from `data_structures.py`:

### MaterialParameterData
- Parameter values with metadata
- Units and descriptions
- Validation bounds
- Source tracking (file, network, database)

### LoadingData  
- Time-dependent loading specifications
- Support for strain-controlled and stress-controlled
- Environmental conditions (temperature, humidity)
- Loading metadata (type, rate, frequency)

### SimulationConfig
- Numerical parameters (tolerance, iterations)
- Output control (save frequency, variables)
- Parallel computation settings

### SimulationResults
- Complete results with metadata
- Response data from GSM engine
- Execution statistics
- Full serialization support

## 7. Error Handling and Validation

The CLI provides comprehensive error handling:

### Parameter Validation
```bash
# This will return validation errors
python cli_gsm.py --model GSM1D_ED --formulation F \
  --params-inline '{"E": -1000, "S": 0}' \
  --loading-inline '{"time_array": [0, 1], "strain_history": [0, 0.01]}' \
  --validate-only --json-output
```

### Network Error Handling
All network operations return structured error responses:
```json
{
  "status": "error",
  "error": "Parameter validation failed: Parameter 'E' = -1000 is outside bounds [1000.0, 100000.0]",
  "message": "Simulation failed"
}
```

## 8. VS Code Integration

### Terminal Execution
Use VS Code's integrated terminal to run CLI commands:

```bash
# Terminal in VS Code
cd /path/to/bmcs_matmod/bmcs_matmod/gsm_lagrange
python cli_gsm.py --list-models --json-output
```

### Shell Script for Batch Operations
Create `test_gsm_cli.sh`:

```bash
#!/bin/bash
echo "Testing GSM CLI Network Interface"
echo "=================================="

# Test 1: List models
echo "1. Available Models:"
python cli_gsm.py --list-models

# Test 2: Get parameter specification  
echo -e "\n2. Parameter Specification for GSM1D_ED:"
python cli_gsm.py --get-param-spec GSM1D_ED

# Test 3: Validate parameters
echo -e "\n3. Parameter Validation:"
python cli_gsm.py --model GSM1D_ED --formulation F \
  --params-inline '{"E": 30000, "S": 1.0, "c": 2.0, "r": 0.5}' \
  --loading-inline '{"time_array": [0, 1], "strain_history": [0, 0.01]}' \
  --validate-only

# Test 4: Execute simulation
echo -e "\n4. Execute Simulation:"
python cli_gsm.py --model GSM1D_ED --formulation F \
  --params-inline '{"E": 30000, "S": 1.0, "c": 2.0, "r": 0.5}' \
  --loading-inline '{"time_array": [0, 0.5, 1], "strain_history": [0, 0.005, 0.01]}' \
  --output test_results.json

echo -e "\nAll tests completed!"
```

Make executable and run:
```bash
chmod +x test_gsm_cli.sh
./test_gsm_cli.sh
```

This provides a comprehensive interface for GSM simulations with full network communication support and dynamic parameter specification retrieval.
