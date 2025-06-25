#!/bin/bash
# GSM CLI Simulation Execution Examples - Real Simulations Only
# This script demonstrates real GSM simulation execution

echo "GSM CLI Real Simulation Execution Examples"
echo "=========================================="
echo ""

# Set script directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running from project root: $PROJECT_ROOT"
echo ""

# Use the real CLI - no fallbacks or mocks
CLI_CMD="python bmcs_matmod/gsm_lagrange/cli/cli_gsm.py"

echo "Using real GSM CLI: $CLI_CMD"
echo ""

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

print_section "1. Basic Real GSM Simulation"

echo "Run GSM1D_ED simulation with realistic parameters:"
run_command "$CLI_CMD --exec GSM1D_ED \
  --params-inline '{\"E\": 30000, \"S\": 1.0, \"c\": 2.0, \"eps_0\": 0.001}' \
  --loading-inline '{\"time_array\": [0, 0.5, 1.0], \"strain_history\": [0, 0.005, 0.01]}'"

print_section "2. Different Material Models - Real Simulations"

echo "Run simulation with Elastic-Plastic model:"
run_command "$CLI_CMD --exec GSM1D_EP \
  --params-inline '{\"E\": 25000, \"c\": 1.5, \"eps_0\": 0.0}' \
  --loading-inline '{\"time_array\": [0, 1.0], \"strain_history\": [0, 0.015]}'"

echo "Run simulation with Viscoelastic model:"
run_command "$CLI_CMD --exec GSM1D_VE \
  --params-inline '{\"E\": 20000, \"gamma_v\": 0.1, \"eps_0\": 0.0}' \
  --loading-inline '{\"time_array\": [0, 2.0], \"strain_history\": [0, 0.01]}'"

print_section "3. JSON Output for Data Processing"

echo "Run simulation with JSON output for data processing:"
run_command "$CLI_CMD --exec GSM1D_ED \
  --params-inline '{\"E\": 30000, \"S\": 1.0, \"c\": 2.0, \"eps_0\": 0.001}' \
  --loading-inline '{\"time_array\": [0, 0.5, 1.0], \"strain_history\": [0, 0.005, 0.01]}' \
  --json-output"

print_section "4. Parameter Specification and Model Information"

echo "Get parameter specification for GSM1D_ED:"
run_command "$CLI_CMD --get-param-spec GSM1D_ED"

echo "List all available GSM models:"
run_command "$CLI_CMD --list-models"

print_section "5. Different Material Models with Various Parameters"

echo "Test different material models with realistic parameters:"

echo "Testing GSM1D_ED (Elastic-Damage):"
run_command "$CLI_CMD --exec GSM1D_ED \
  --params-inline '{\"E\": 30000, \"S\": 1.0}' \
  --loading-inline '{\"time_array\": [0, 1.0], \"strain_history\": [0, 0.01]}'"

echo "Testing GSM1D_EP (Elastic-Plastic):"
run_command "$CLI_CMD --exec GSM1D_EP \
  --params-inline '{\"E\": 30000, \"c\": 1.0}' \
  --loading-inline '{\"time_array\": [0, 1.0], \"strain_history\": [0, 0.01]}'"

echo "Testing GSM1D_VE (Viscoelastic):"
run_command "$CLI_CMD --exec GSM1D_VE \
  --params-inline '{\"E\": 30000, \"gamma_v\": 0.1}' \
  --loading-inline '{\"time_array\": [0, 1.0], \"strain_history\": [0, 0.01]}'"

print_section "6. Complex Loading Scenarios"

echo "Cyclic loading simulation:"
run_command "$CLI_CMD --exec GSM1D_ED \
  --params-inline '{\"E\": 25000, \"S\": 1.2, \"c\": 1.5, \"eps_0\": 0.0}' \
  --loading-inline '{\"time_array\": [0, 0.25, 0.5, 0.75, 1.0], \"strain_history\": [0, 0.01, 0, 0.015, 0]}' \
  --json-output"

print_section "7. Summary"

echo "Real GSM Simulation Features Demonstrated:"
echo "✅ Actual GSM model instantiation and execution"
echo "✅ Real material parameter specification"
echo "✅ Authentic loading history simulation"
echo "✅ JSON output for data processing"
echo "✅ Multiple material model types (ED, EP, VE, etc.)"
echo "✅ Parameter specification validation"
echo ""
echo "All simulations execute real GSM calculations - no mocks or placeholders!"
echo ""
echo "Real GSM simulation examples completed successfully!"
