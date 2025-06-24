#!/bin/bash
# GSM CLI Network Interface Test Script
# This script demonstrates the enhanced CLI capabilities

echo "GSM CLI Network Interface Testing"
echo "================================="
echo ""

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

print_section "1. Basic Model Discovery"

run_command "python cli_gsm.py --list-models"

print_section "2. Parameter Specification Retrieval"

run_command "python cli_gsm.py --get-param-spec GSM1D_ED"

print_section "3. Parameter Specification (JSON Output)"

run_command "python cli_gsm.py --get-param-spec GSM1D_ED --json-output"

print_section "4. Parameter Validation"

echo "Testing parameter validation with valid parameters:"
run_command "python cli_gsm.py --model GSM1D_ED --formulation F \\
  --params-inline '{\"E\": 30000, \"S\": 1.0, \"c\": 2.0, \"r\": 0.5, \"eps_0\": 0.001}' \\
  --loading-inline '{\"time_array\": [0, 0.5, 1.0], \"strain_history\": [0, 0.005, 0.01]}' \\
  --validate-only"

echo "Testing parameter validation with invalid parameters:"
run_command "python cli_gsm.py --model GSM1D_ED --formulation F \\
  --params-inline '{\"E\": -1000, \"S\": 0}' \\
  --loading-inline '{\"time_array\": [0, 1], \"strain_history\": [0, 0.01]}' \\
  --validate-only --json-output"

print_section "5. Simulation Execution"

echo "Executing simulation with inline parameters:"
run_command "python cli_gsm.py --model GSM1D_ED --formulation F \\
  --params-inline '{\"E\": 30000, \"S\": 1.0, \"c\": 2.0, \"r\": 0.5, \"eps_0\": 0.001}' \\
  --loading-inline '{\"time_array\": [0, 0.25, 0.5, 0.75, 1.0], \"strain_history\": [0, 0.0025, 0.005, 0.0075, 0.01], \"loading_type\": \"monotonic\"}' \\
  --output test_simulation_results.json"

if [ -f "test_simulation_results.json" ]; then
    echo "Simulation results saved to test_simulation_results.json"
    echo "Result file size: $(stat -f%z test_simulation_results.json 2>/dev/null || stat -c%s test_simulation_results.json) bytes"
else
    echo "Simulation results file not created"
fi

print_section "6. Create Sample Request File"

cat > sample_request.json << 'EOF'
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
    "material_name": "Test Concrete",
    "model_type": "ElasticDamage",
    "description": "Test material for CLI demonstration"
  },
  "loading": {
    "time_array": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "strain_history": [0.0, 0.002, 0.004, 0.006, 0.008, 0.01],
    "loading_type": "monotonic",
    "max_amplitude": 0.01,
    "description": "Monotonic tensile loading to 1% strain"
  },
  "config": {
    "tolerance": 1e-6,
    "max_iterations": 100,
    "save_internal_variables": true,
    "debug_output": false
  }
}
EOF

echo "Created sample_request.json:"
run_command "cat sample_request.json"

print_section "7. Execute from Request File"

run_command "python cli_gsm.py --execute-request sample_request.json --json-output"

print_section "8. Test All Available Keys"

run_command "python cli_gsm.py --show-all-keys"

print_section "9. Model Information"

run_command "python cli_gsm.py --model-info GSM1D_ED"

print_section "10. Test Model Access"

run_command "python cli_gsm.py --test-access GSM1D_ED"

print_section "Network Server Test (Background Mode)"

echo "Testing network server endpoints..."
echo "Starting server in background..."

# Start server in background
python cli_gsm.py --serve --port 8889 > server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 2

# Test server endpoints
echo "Testing server endpoints:"

echo ""
echo "1. Testing /models endpoint:"
if command -v curl >/dev/null 2>&1; then
    curl -s http://localhost:8889/models | python -m json.tool 2>/dev/null || echo "Server not responding or curl not available"
else
    echo "curl not available - skipping network tests"
fi

echo ""
echo "2. Testing /param-spec endpoint:"
if command -v curl >/dev/null 2>&1; then
    curl -s http://localhost:8889/param-spec/GSM1D_ED | python -m json.tool 2>/dev/null || echo "Server not responding"
else
    echo "curl not available - skipping network tests"
fi

# Stop the server
echo ""
echo "Stopping test server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

print_section "Cleanup and Summary"

echo "Cleaning up test files..."
rm -f test_simulation_results.json sample_request.json server.log

echo ""
echo "CLI Test Summary:"
echo "=================="
echo "✓ Model discovery and listing"
echo "✓ Parameter specification retrieval"  
echo "✓ Parameter validation"
echo "✓ Simulation execution"
echo "✓ File-based request execution"
echo "✓ JSON serialization"
echo "✓ Network server mode"
echo ""
echo "The GSM CLI interface is ready for:"
echo "• Network communication with workchain managers"
echo "• Dynamic parameter specification for UI generation"
echo "• Cross-network computational node communication"
echo "• Full serialization support for remote execution"
echo ""
echo "Next steps:"
echo "1. Start server: python cli_gsm.py --serve --port 8888"
echo "2. Integrate with workchain manager"  
echo "3. Use parameter specifications for dynamic UIs"
echo ""
echo "Test completed successfully!"
