#!/bin/bash
# Simulation Execution Examples
# This script demonstrates running GSM simulations with different configurations

echo "GSM CLI Simulation Execution Examples"
echo "====================================="
echo ""

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Function to print section headers
print_section() {
    echo ""
    echo "## $1"
    echo "=================================================="
    echo ""
}

# Function to run command and show output
run_command() {
    echo "Command: $1"
    echo "----------------------------------------"
    eval "$1"
    echo ""
}

print_section "1. Basic Simulation Execution"

echo "Run GSM1D_ED simulation with monotonic tension:"
run_command "python cli_gsm.py --model GSM1D_ED --formulation F \
  --params examples/parameters/gsm1d_ed_basic.json \
  --loading examples/loading/monotonic_tension.json \
  --config examples/configs/default_config.json \
  --json-output"

print_section "2. Different Energy Formulations"

echo "Run simulation with Helmholtz (strain-controlled) formulation:"
run_command "python cli_gsm.py --model GSM1D_ED --formulation Helmholtz \
  --params examples/parameters/gsm1d_ed_basic.json \
  --loading examples/loading/monotonic_tension.json \
  --json-output"

echo "Run simulation with Gibbs (stress-controlled) formulation:"
run_command "python cli_gsm.py --model GSM1D_ED --formulation Gibbs \
  --params examples/parameters/gsm1d_ed_basic.json \
  --loading examples/loading/stress_controlled.json \
  --json-output"

print_section "3. Inline Parameter and Loading Specification"

echo "Run simulation with inline parameters and loading:"
run_command 'python cli_gsm.py --model GSM1D_ED --formulation F \
  --params-inline "{\"E\": 30000, \"S\": 1.0, \"c\": 2.0, \"r\": 0.5, \"eps_0\": 0.001}" \
  --loading-inline "{\"time_array\": [0, 0.5, 1.0], \"strain_history\": [0, 0.005, 0.01]}" \
  --json-output'

print_section "4. Cyclic Loading Simulation"

echo "Run simulation with cyclic tension-compression loading:"
run_command "python cli_gsm.py --model GSM1D_ED --formulation F \
  --params examples/parameters/gsm1d_ed_high_strength.json \
  --loading examples/loading/cyclic_tension_compression.json \
  --json-output"

print_section "5. Different Material Models"

echo "Run elastic-plastic simulation:"
run_command "python cli_gsm.py --model GSM1D_EP --formulation F \
  --params examples/parameters/gsm1d_ep_basic.json \
  --loading examples/loading/monotonic_tension.json \
  --json-output"

print_section "6. Output to File"

echo "Run simulation and save results to file:"
run_command "python cli_gsm.py --model GSM1D_ED --formulation F \
  --params examples/parameters/gsm1d_ed_basic.json \
  --loading examples/loading/monotonic_tension.json \
  --output examples/simulation_results.json"

echo "Display saved results:"
if [ -f "examples/simulation_results.json" ]; then
    echo "Results file created successfully:"
    echo "----------------------------------------"
    head -20 examples/simulation_results.json
    echo "..."
    echo "(Results truncated for display)"
else
    echo "Results file was not created."
fi

print_section "7. Validation Only (No Execution)"

echo "Validate parameters without running simulation:"
run_command "python cli_gsm.py --model GSM1D_ED --formulation F \
  --params examples/parameters/gsm1d_ed_basic.json \
  --loading examples/loading/monotonic_tension.json \
  --validate-only --json-output"

echo ""
echo "Simulation execution examples completed!"
echo "Next: Run 04_network_server.sh to explore network communication"
