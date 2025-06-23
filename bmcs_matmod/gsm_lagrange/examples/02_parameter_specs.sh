#!/bin/bash
# Parameter Specification Examples
# This script demonstrates parameter specification retrieval and validation

echo "GSM CLI Parameter Specification Examples"
echo "========================================"
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

print_section "1. Parameter Specification Retrieval"

echo "Get parameter specification for GSM1D_ED (Elastic-Damage):"
run_command "python cli_gsm.py --get-param-spec GSM1D_ED"

echo "Get parameter specification with JSON output:"
run_command "python cli_gsm.py --get-param-spec GSM1D_ED --json-output"

echo "Get parameter specification for GSM1D_EP (Elastic-Plastic):"
run_command "python cli_gsm.py --get-param-spec GSM1D_EP"

print_section "2. Parameter Validation Examples"

echo "Test valid parameters for GSM1D_ED:"
run_command 'python cli_gsm.py --model GSM1D_ED --formulation F \
  --params-inline "{\"E\": 30000, \"S\": 1.0, \"c\": 2.0, \"r\": 0.5, \"eps_0\": 0.001}" \
  --loading-inline "{\"time_array\": [0, 0.5, 1.0], \"strain_history\": [0, 0.005, 0.01]}" \
  --validate-only --json-output'

echo "Test invalid parameters (missing required parameter):"
run_command 'python cli_gsm.py --model GSM1D_ED --formulation F \
  --params-inline "{\"E\": 30000, \"S\": 1.0}" \
  --loading-inline "{\"time_array\": [0, 0.5, 1.0], \"strain_history\": [0, 0.005, 0.01]}" \
  --validate-only --json-output'

echo "Test invalid parameters (out of bounds):"
run_command 'python cli_gsm.py --model GSM1D_ED --formulation F \
  --params-inline "{\"E\": -1000, \"S\": 1.0, \"c\": 2.0, \"r\": 0.5, \"eps_0\": 0.001}" \
  --loading-inline "{\"time_array\": [0, 0.5, 1.0], \"strain_history\": [0, 0.005, 0.01]}" \
  --validate-only --json-output'

print_section "3. Different Model Parameter Specifications"

echo "Compare parameter specifications across different models:"

models=("GSM1D_ED" "GSM1D_EP" "GSM1D_VE")

for model in "${models[@]}"; do
    echo "Parameters for $model:"
    run_command "python cli_gsm.py --get-param-spec $model --json-output | jq '.specification.parameters'"
done

echo ""
echo "Parameter specification examples completed!"
echo "Next: Run 03_simulation_execution.sh to execute simulations"
