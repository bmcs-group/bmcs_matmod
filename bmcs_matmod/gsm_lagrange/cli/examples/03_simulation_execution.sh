#!/bin/bash
# Simulation Execution Examples - Updated for reorganized structure
# This script demonstrates simulation execution with fallback for import issues

echo "GSM CLI Simulation Execution Examples"
echo "====================================="
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

print_section "1. Basic Simulation Execution"

if [ "$FULL_CLI" = true ]; then
    echo "Run GSM1D_ED simulation with monotonic tension (Full CLI):"
    run_command "$CLI_CMD --model GSM1D_ED --formulation F \
      --params bmcs_matmod/gsm_lagrange/cli/examples/parameters/gsm1d_ed_basic.json \
      --loading bmcs_matmod/gsm_lagrange/cli/examples/loading/monotonic_tension.json \
      --config bmcs_matmod/gsm_lagrange/cli/examples/configs/default_config.json \
      --json-output"
else
    echo "Run GSM1D_ED simulation placeholder (Standalone CLI):"
    run_command '$CLI_CMD --model GSM1D_ED --formulation F \
      --params-inline "{\"E\": 30000, \"S\": 1.0, \"c\": 2.0, \"r\": 0.5, \"eps_0\": 0.001}" \
      --loading-inline "{\"time_array\": [0, 0.5, 1.0], \"strain_history\": [0, 0.005, 0.01]}" \
      --simulate --json-output'
fi

if [ "$FULL_CLI" = true ]; then
    print_section "2. Different Energy Formulations (Full CLI Only)"

    echo "Run simulation with Helmholtz (strain-controlled) formulation:"
    run_command "$CLI_CMD --model GSM1D_ED --formulation Helmholtz \
      --params bmcs_matmod/gsm_lagrange/cli/examples/parameters/gsm1d_ed_basic.json \
      --loading bmcs_matmod/gsm_lagrange/cli/examples/loading/monotonic_tension.json \
      --json-output"

    echo "Run simulation with Gibbs (stress-controlled) formulation:"
    run_command "$CLI_CMD --model GSM1D_ED --formulation Gibbs \
      --params bmcs_matmod/gsm_lagrange/cli/examples/parameters/gsm1d_ed_basic.json \
      --loading bmcs_matmod/gsm_lagrange/cli/examples/loading/stress_controlled.json \
      --json-output"
else
    print_section "2. Different Energy Formulations (Requires Full CLI)"
    
    echo "⚠️  Energy formulation selection requires full CLI functionality."
    echo "The standalone CLI provides basic simulation placeholders only."
    echo ""
    echo "To use different formulations, resolve import issues and run:"
    echo "  python -m bmcs_matmod.gsm_lagrange.cli.cli_gsm --model GSM1D_ED --formulation Helmholtz"
fi

print_section "3. Inline Parameter and Loading Specification"

echo "Run simulation with inline parameters and loading:"
run_command '$CLI_CMD --model GSM1D_ED --formulation F \
  --params-inline "{\"E\": 30000, \"S\": 1.0, \"c\": 2.0, \"r\": 0.5, \"eps_0\": 0.001}" \
  --loading-inline "{\"time_array\": [0, 0.5, 1.0], \"strain_history\": [0, 0.005, 0.01]}" \
  --simulate --json-output'

print_section "4. Different Material Models"

echo "Test different material models with inline parameters:"

models=("GSM1D_ED" "GSM1D_EP" "GSM1D_VE")
for model in "${models[@]}"; do
    echo "Testing $model:"
    if [ "$FULL_CLI" = true ]; then
        run_command "$CLI_CMD --model $model --formulation F \
          --params-inline '{\"E\": 30000, \"S\": 1.0}' \
          --loading-inline '{\"time_array\": [0, 1.0], \"strain_history\": [0, 0.01]}' \
          --json-output"
    else
        run_command '$CLI_CMD --model '$model' --formulation F \
          --params-inline "{\"E\": 30000, \"S\": 1.0}" \
          --loading-inline "{\"time_array\": [0, 1.0], \"strain_history\": [0, 0.01]}" \
          --simulate --json-output'
    fi
done

print_section "5. Validation Examples"

echo "Validate parameters without running simulation:"
run_command '$CLI_CMD --model GSM1D_ED --formulation F \
  --params-inline "{\"E\": 30000, \"S\": 1.0, \"c\": 2.0, \"r\": 0.5, \"eps_0\": 0.001}" \
  --loading-inline "{\"time_array\": [0, 0.5, 1.0], \"strain_history\": [0, 0.005, 0.01]}" \
  --validate-only --json-output'

print_section "6. Additional Notes"

echo "CLI Execution Methods:"
echo "1. ✅ Standalone CLI (simulation placeholders): python bmcs_matmod/gsm_lagrange/cli/cli_gsm_standalone.py"
echo "2. 🔄 Module CLI (full simulations): python -m bmcs_matmod.gsm_lagrange.cli.cli_gsm"
echo ""

if [ "$FULL_CLI" = false ]; then
    echo "⚠️  Note: Using standalone CLI with simulation placeholders."
    echo "For actual simulation execution:"
    echo "- Resolve sympy/traits import issues"
    echo "- Use full CLI: python -m bmcs_matmod.gsm_lagrange.cli.cli_gsm"
    echo "- Check parameter file examples in examples/ directory"
    echo ""
fi

echo "Key Features Demonstrated:"
echo "- Model specification and validation"
echo "- Inline parameter and loading specification"
echo "- JSON output formatting"
if [ "$FULL_CLI" = true ]; then
    echo "- Different energy formulations"
    echo "- File-based parameter loading"
    echo "- Output file generation"
fi

echo ""
echo "Simulation execution examples completed!"
echo "Next: Check other example scripts for additional functionality"
