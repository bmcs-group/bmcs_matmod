#!/bin/bash
# Fixed GSM CLI Basic Usage Examples
# This script demonstrates fundamental CLI operations using the fixed CLI

echo "GSM CLI Basic Usage Examples (Fixed Version)"
echo "============================================="
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

print_section "1. Model Discovery"

echo "List all available GSM models:"
run_command "python cli_gsm_fixed.py --list-models"

echo "List models with JSON output:"
run_command "python cli_gsm_fixed.py --list-models --json-output"

print_section "2. Model Information"

echo "Get detailed information about a specific model:"
run_command "python cli_gsm_fixed.py --model-info GSM1D_ED"

echo "Get information about another model:"
run_command "python cli_gsm_fixed.py --model-info GSM1D_EP"

echo "Test model access (warning: may hang due to import issues):"
echo "# python cli_gsm_fixed.py --test-access GSM1D_ED  # Commented out - may hang"
echo ""

print_section "3. Registry Exploration"

echo "Show all available keys in the registry:"
run_command "python cli_gsm_fixed.py --show-all-keys"

echo "Show keys with JSON output:"
run_command "python cli_gsm_fixed.py --show-all-keys --json-output"

print_section "4. Help and Version Information"

echo "Show version information:"
run_command "python cli_gsm_fixed.py --version"

echo "Show extended help:"
run_command "python cli_gsm_fixed.py --help-extended"

print_section "5. Error Handling"

echo "Test with non-existent model:"
run_command "python cli_gsm_fixed.py --model-info NONEXISTENT"

echo "Test with non-existent model (JSON):"
run_command "python cli_gsm_fixed.py --model-info NONEXISTENT --json-output"

echo ""
echo "Examples completed successfully!"
echo "Note: This fixed version avoids import issues while providing full model discovery."
