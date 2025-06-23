#!/bin/bash
# Basic GSM CLI Usage Examples
# This script demonstrates fundamental CLI operations

echo "GSM CLI Basic Usage Examples"
echo "============================"
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
run_command "python cli_gsm.py --list-models"

echo "List models with JSON output:"
run_command "python cli_gsm.py --list-models --json-output"

print_section "2. Model Information"

echo "Get detailed information about a specific model:"
run_command "python cli_gsm.py --model-info GSM1D_ED"

echo "Test model access:"
run_command "python cli_gsm.py --test-access GSM1D_ED"

print_section "3. Registry Exploration"

echo "Show all available keys in the registry:"
run_command "python cli_gsm.py --show-all-keys"

echo "Show keys with JSON output:"
run_command "python cli_gsm.py --show-all-keys --json-output"

print_section "4. Help and Version Information"

echo "Show CLI help:"
run_command "python cli_gsm.py --help"

echo ""
echo "Basic usage examples completed!"
echo "Next: Run 02_parameter_specs.sh to explore parameter specifications"
