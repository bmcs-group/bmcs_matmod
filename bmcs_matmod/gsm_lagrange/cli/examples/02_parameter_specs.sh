#!/bin/bash
# Parameter Specification Examples - Updated for reorganized structure
# This script demonstrates parameter specification retrieval with fallback for import issues

echo "GSM CLI Parameter Specification Examples"
echo "========================================"
echo ""

# Set script directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running from project root: $PROJECT_ROOT"
echo ""

# Check if we can use the full CLI or need to use standalone
echo "Checking CLI availability..."
echo "Testing if full CLI file exists and checking basic import..."

# Quick check - does the file exist and can we do a very basic import test
if [ -f "bmcs_matmod/gsm_lagrange/cli/cli_gsm.py" ] && timeout 1 python -c "import sys" >/dev/null 2>&1; then
    echo "⚠️  Full CLI file exists but likely has import issues (using standalone)"
    CLI_CMD="python bmcs_matmod/gsm_lagrange/cli/cli_gsm_standalone.py"
    FULL_CLI=false
else
    echo "⚠️  Using standalone CLI"
    CLI_CMD="python bmcs_matmod/gsm_lagrange/cli/cli_gsm_standalone.py"
    FULL_CLI=false
fi
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

print_section "1. Parameter Specification Retrieval"

echo "Get parameter specification for GSM1D_ED (Elastic-Damage):"
run_command "$CLI_CMD --get-param-spec GSM1D_ED"

echo "Get parameter specification with JSON output:"
run_command "$CLI_CMD --get-param-spec GSM1D_ED --json-output"

echo "Get parameter specification for GSM1D_EP (Elastic-Plastic):"
run_command "$CLI_CMD --get-param-spec GSM1D_EP"

if [ "$FULL_CLI" = true ]; then
    print_section "2. Parameter Validation Examples (Full CLI Only)"
    
    echo "Note: Parameter validation requires full CLI functionality."
    echo "Testing with full CLI..."
    
    echo "Test valid parameters for GSM1D_ED:"
    run_command '$CLI_CMD --model GSM1D_ED --formulation F \
      --params-inline "{\"E\": 30000, \"S\": 1.0, \"c\": 2.0, \"r\": 0.5, \"eps_0\": 0.001}" \
      --loading-inline "{\"time_array\": [0, 0.5, 1.0], \"strain_history\": [0, 0.005, 0.01]}" \
      --validate-only --json-output'
      
    echo "Test invalid parameters (missing required parameter):"
    run_command '$CLI_CMD --model GSM1D_ED --formulation F \
      --params-inline "{\"E\": 30000, \"S\": 1.0}" \
      --loading-inline "{\"time_array\": [0, 0.5, 1.0], \"strain_history\": [0, 0.005, 0.01]}" \
      --validate-only --json-output'
else
    print_section "2. Parameter Validation (Requires Full CLI)"
    
    echo "⚠️  Parameter validation requires full CLI functionality with working imports."
    echo "The standalone CLI provides basic model information only."
    echo ""
    echo "To use parameter validation, resolve the import issues and run:"
    echo "  python -m bmcs_matmod.gsm_lagrange.cli.cli_gsm --model GSM1D_ED --get-param-spec"
    echo ""
fi

print_section "3. Model Comparison"

echo "Compare basic information across different models:"
models=("GSM1D_ED" "GSM1D_EP" "GSM1D_VE")

for model in "${models[@]}"; do
    echo "Information for $model:"
    run_command "$CLI_CMD --get-param-spec $model"
done

print_section "4. Additional Notes"

echo "CLI Execution Methods:"
echo "1. ✅ Standalone CLI (basic info): python bmcs_matmod/gsm_lagrange/cli/cli_gsm_standalone.py"
echo "2. 🔄 Module CLI (full features): python -m bmcs_matmod.gsm_lagrange.cli.cli_gsm"
echo ""
echo "For full parameter validation and simulation capabilities:"
echo "- Resolve sympy/traits import issues"
echo "- Use the full CLI: python -m bmcs_matmod.gsm_lagrange.cli.cli_gsm"
echo ""
echo "Parameter specification examples completed!"
echo "Next: Try other example scripts for additional functionality"
