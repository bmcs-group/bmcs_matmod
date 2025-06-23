# GSM CLI Examples

This directory contains examples demonstrating the various capabilities of the GSM CLI interface.

## Files Overview

### Basic Usage Examples
- `01_basic_usage.sh` - Basic CLI commands and model discovery
- `02_parameter_specs.sh` - Parameter specification retrieval and validation
- `03_simulation_execution.sh` - Running simulations with different formulations

### Data Files
- `parameters/` - Example parameter files in JSON format
- `loading/` - Example loading scenarios
- `configs/` - Simulation configuration examples

### Network Communication Examples
- `04_network_server.sh` - Starting and testing the network server
- `05_remote_requests.py` - Python client for remote simulation requests
- `06_batch_processing.py` - Batch processing multiple simulations

### Advanced Examples
- `07_jupyter_integration.ipynb` - Jupyter notebook demonstrating dynamic parameter specification
- `08_workchain_integration.py` - Example of workchain manager integration
- `09_validation_demo.py` - Comprehensive parameter validation examples

## Quick Start

1. **Basic model discovery:**
   ```bash
   python ../cli_gsm.py --list-models
   ```

2. **Get parameter specification:**
   ```bash
   python ../cli_gsm.py --get-param-spec GSM1D_ED --json-output
   ```

3. **Run a simple simulation:**
   ```bash
   python ../cli_gsm.py --model GSM1D_ED --formulation F \
     --params parameters/gsm1d_ed_basic.json \
     --loading loading/monotonic_tension.json
   ```

4. **Start network server:**
   ```bash
   python ../cli_gsm.py --serve --port 8888
   ```

## Network Endpoints

When running in server mode (`--serve`), the following endpoints are available:

- `GET /models` - List available models
- `GET /param-spec/<model>` - Get parameter specification for a model
- `POST /simulate` - Execute simulation (JSON request body)
- `POST /validate` - Validate parameters only

## Data Format Examples

All JSON files in this directory follow the GSM CLI data structure format:
- Parameters: `MaterialParameterData` format
- Loading: `LoadingData` format with time arrays and strain/stress histories
- Config: `SimulationConfig` format for simulation settings
