#!/bin/bash
# Network Server Examples
# This script demonstrates starting and testing the GSM CLI network server

echo "GSM CLI Network Server Examples"
echo "==============================="
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

print_section "1. Starting the Network Server"

echo "The GSM CLI can run as a network server to provide remote access to simulations."
echo "To start the server, run:"
echo ""
echo "python cli_gsm.py --serve --port 8888"
echo ""
echo "This will start a server on http://localhost:8888 with the following endpoints:"
echo ""
echo "Available Endpoints:"
echo "- GET  /models              - List available GSM models"
echo "- GET  /param-spec/<model>  - Get parameter specification for a model"
echo "- POST /simulate            - Execute simulation (JSON request body)"
echo "- POST /validate            - Validate parameters only"
echo ""

print_section "2. Testing Server Endpoints (requires server to be running)"

echo "NOTE: The following commands require the server to be running in another terminal."
echo "Start the server with: python cli_gsm.py --serve --port 8888"
echo ""

echo "Test 1: List available models"
echo "curl http://localhost:8888/models"
echo ""

echo "Test 2: Get parameter specification for GSM1D_ED"
echo "curl http://localhost:8888/param-spec/GSM1D_ED"
echo ""

echo "Test 3: Submit simulation request"
echo 'curl -X POST http://localhost:8888/simulate \'
echo '  -H "Content-Type: application/json" \'
echo '  -d @examples/network_requests/simple_simulation.json'
echo ""

echo "Test 4: Validate parameters only"
echo 'curl -X POST http://localhost:8888/validate \'
echo '  -H "Content-Type: application/json" \'
echo '  -d @examples/network_requests/validation_request.json'
echo ""

print_section "3. Server Management"

echo "To stop the server, use Ctrl+C in the server terminal."
echo ""
echo "For production use, consider:"
echo "- Running behind a reverse proxy (nginx, Apache)"
echo "- Adding authentication and rate limiting"
echo "- Using a production WSGI server (gunicorn, uWSGI)"
echo "- Implementing proper logging and monitoring"
echo ""

print_section "4. Interactive Server Testing"

echo "Would you like to start the server now for testing? (y/N)"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo ""
    echo "Starting GSM Network Server on port 8888..."
    echo "Use Ctrl+C to stop the server."
    echo "In another terminal, you can test the endpoints with curl commands shown above."
    echo ""
    
    # Start the server (this will block until Ctrl+C)
    python cli_gsm.py --serve --port 8888
else
    echo ""
    echo "Server not started. To manually start the server later, run:"
    echo "python cli_gsm.py --serve --port 8888"
fi

echo ""
echo "Network server examples completed!"
echo "Next: Check out 05_remote_requests.py for Python client examples"
