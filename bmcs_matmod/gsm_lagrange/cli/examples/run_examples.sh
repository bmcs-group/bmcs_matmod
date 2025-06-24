#!/bin/bash
# Master script to run all GSM CLI examples
# This script provides an interactive menu to run different example categories

echo "GSM CLI Examples Master Script"
echo "=============================="
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

# Function to pause for user input
pause() {
    echo ""
    read -p "Press Enter to continue or Ctrl+C to exit..."
    echo ""
}

# Main menu
while true; do
    clear
    echo "GSM CLI Examples Master Script"
    echo "=============================="
    echo ""
    echo "Available Examples:"
    echo ""
    echo "1. Basic Usage Examples (01_basic_usage.sh)"
    echo "   - Model discovery and information"
    echo "   - Registry exploration"
    echo ""
    echo "2. Parameter Specification Examples (02_parameter_specs.sh)"
    echo "   - Parameter specification retrieval"
    echo "   - Parameter validation demos"
    echo ""
    echo "3. Simulation Execution Examples (03_simulation_execution.sh)"
    echo "   - Basic simulation execution"
    echo "   - Different energy formulations"
    echo "   - Cyclic loading scenarios"
    echo ""
    echo "4. Network Server Examples (04_network_server.sh)"
    echo "   - Start network server"
    echo "   - Test network endpoints"
    echo ""
    echo "5. Remote Request Examples (05_remote_requests.py)"
    echo "   - Python client for remote servers"
    echo "   - Network communication demos"
    echo ""
    echo "6. Batch Processing Examples (06_batch_processing.py)"
    echo "   - Parameter studies"
    echo "   - Automated batch simulations"
    echo ""
    echo "7. Jupyter Integration (07_jupyter_integration.ipynb)"
    echo "   - Interactive notebook examples"
    echo "   - Widget-based parameter input"
    echo ""
    echo "8. Workchain Integration (08_workchain_integration.py)"
    echo "   - Multi-step workflow examples"
    echo "   - Dependency management"
    echo ""
    echo "9. Validation Demo (09_validation_demo.py)"
    echo "   - Comprehensive parameter validation"
    echo "   - Boundary testing"
    echo ""
    echo "A. Run All Examples (Sequential)"
    echo "Q. Quit"
    echo ""
    read -p "Please select an option [1-9, A, Q]: " choice
    
    case $choice in
        1)
            print_section "Running Basic Usage Examples"
            ./01_basic_usage.sh
            pause
            ;;
        2)
            print_section "Running Parameter Specification Examples"
            ./02_parameter_specs.sh
            pause
            ;;
        3)
            print_section "Running Simulation Execution Examples"
            ./03_simulation_execution.sh
            pause
            ;;
        4)
            print_section "Running Network Server Examples"
            echo "Note: This will start an interactive server."
            echo "Use Ctrl+C to stop the server and return to menu."
            pause
            ./04_network_server.sh
            pause
            ;;
        5)
            print_section "Running Remote Request Examples"
            python3 05_remote_requests.py
            pause
            ;;
        6)
            print_section "Running Batch Processing Examples"
            python3 06_batch_processing.py
            pause
            ;;
        7)
            print_section "Opening Jupyter Integration Notebook"
            echo "Opening Jupyter notebook..."
            echo "If Jupyter is not installed, install with: pip install jupyter"
            if command -v jupyter &> /dev/null; then
                jupyter notebook 07_jupyter_integration.ipynb
            else
                echo "Jupyter not found. Please install Jupyter and run:"
                echo "jupyter notebook 07_jupyter_integration.ipynb"
            fi
            pause
            ;;
        8)
            print_section "Running Workchain Integration Examples"
            python3 08_workchain_integration.py
            pause
            ;;
        9)
            print_section "Running Validation Demo"
            python3 09_validation_demo.py
            pause
            ;;
        [Aa])
            print_section "Running All Examples Sequentially"
            echo "This will run all examples in sequence."
            echo "Note: Network server example will be skipped in batch mode."
            pause
            
            echo "1/8: Basic Usage Examples"
            ./01_basic_usage.sh
            echo "✅ Basic usage examples completed"
            echo ""
            
            echo "2/8: Parameter Specification Examples"
            ./02_parameter_specs.sh
            echo "✅ Parameter specification examples completed"
            echo ""
            
            echo "3/8: Simulation Execution Examples"
            ./03_simulation_execution.sh
            echo "✅ Simulation execution examples completed"
            echo ""
            
            echo "4/8: Remote Request Examples"
            python3 05_remote_requests.py
            echo "✅ Remote request examples completed"
            echo ""
            
            echo "5/8: Batch Processing Examples"
            python3 06_batch_processing.py
            echo "✅ Batch processing examples completed"
            echo ""
            
            echo "6/8: Workchain Integration Examples"
            python3 08_workchain_integration.py
            echo "✅ Workchain integration examples completed"
            echo ""
            
            echo "7/8: Validation Demo"
            python3 09_validation_demo.py
            echo "✅ Validation demo completed"
            echo ""
            
            echo "8/8: Summary"
            echo "All examples completed successfully!"
            echo "Check the generated files in this directory for results."
            echo ""
            echo "Generated files:"
            ls -la *.json 2>/dev/null | head -10
            echo ""
            pause
            ;;
        [Qq])
            echo "Exiting GSM CLI Examples."
            echo "Thank you for using the GSM CLI interface!"
            exit 0
            ;;
        *)
            echo "Invalid option. Please select 1-9, A, or Q."
            sleep 2
            ;;
    esac
done
