#!/bin/bash
# Quick test of real GSM simulations

echo "=== GSM CLI Real Simulation Test ==="
echo ""

cd /home/rch/Coding/bmcs_matmod

echo "1. List available models:"
python bmcs_matmod/gsm_lagrange/cli/cli_gsm.py --list-models
echo ""

echo "2. Get parameter specification for GSM1D_ED:"
python bmcs_matmod/gsm_lagrange/cli/cli_gsm.py --get-param-spec GSM1D_ED
echo ""

echo "3. Run a real GSM1D_ED simulation:"
echo "   This executes actual GSM calculations - no mocks!"
python bmcs_matmod/gsm_lagrange/cli/cli_gsm.py --exec GSM1D_ED \
  --params-inline '{"E": 30000, "S": 1.0}' \
  --loading-inline '{"time_array": [0, 1.0], "strain_history": [0, 0.01]}'
echo ""

echo "4. Run simulation with JSON output:"
python bmcs_matmod/gsm_lagrange/cli/cli_gsm.py --exec GSM1D_EP \
  --params-inline '{"E": 25000, "c": 1.0}' \
  --loading-inline '{"time_array": [0, 1.0], "strain_history": [0, 0.005]}' \
  --json-output | head -20
echo ""

echo "=== Real GSM simulation tests completed! ==="
echo "All simulations use actual material models and calculations."
