#!/usr/bin/env python3
"""
Test remote client functionality for GSM models
"""

import subprocess
import json
import sys
import os

def run_cli_command(args):
    """Run CLI command and return output"""
    cmd = [sys.executable, 'cli_clean.py'] + args
    result = subprocess.run(cmd, 
                          capture_output=True, 
                          text=True, 
                          cwd='/home/rch/Coding/bmcs_matmod/bmcs_matmod/gsm_lagrange')
    return result.returncode, result.stdout, result.stderr

def test_model_availability(model_key):
    """Test if a model is available"""
    returncode, stdout, stderr = run_cli_command(['--check-model', model_key, '--json-output'])
    
    if returncode == 0:
        try:
            result = json.loads(stdout)
            return result
        except json.JSONDecodeError:
            return {'error': 'Invalid JSON response'}
    else:
        return {'error': f'CLI error: {stderr}'}

def list_available_models():
    """Get list of available models"""
    returncode, stdout, stderr = run_cli_command(['--list-models'])
    
    if returncode == 0:
        # Parse the output to extract model names
        models = []
        for line in stdout.split('\n'):
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                if 'GSM1D_' in line:
                    model_name = line.split('GSM1D_')[1].split()[0]
                    models.append(f'GSM1D_{model_name}')
        return models
    else:
        return []

if __name__ == "__main__":
    print("Testing Remote Client Functionality")
    print("=" * 50)
    
    # Test 1: List available models
    print("\n1. Testing model listing:")
    models = list_available_models()
    print(f"Found {len(models)} models: {models}")
    
    # Test 2: Check specific models
    print("\n2. Testing model availability checks:")
    test_models = ['ED', 'GSM1D_VE', 'VEVPD', 'NONEXISTENT']
    
    for model in test_models:
        result = test_model_availability(model)
        status = "✓ Available" if result.get('available') else "✗ Not Available"
        print(f"  {model}: {status}")
        if result.get('error'):
            print(f"    Error: {result['error']}")
    
    # Test 3: Verify a complex model
    print("\n3. Testing complex model verification:")
    complex_model = 'VEVPD'
    result = test_model_availability(complex_model)
    
    if result.get('available'):
        print(f"✓ Model {complex_model} is ready for use")
        print(f"  Module: {result.get('module_path')}")
        print(f"  Description: {result.get('description')}")
        print("  -> Remote client can proceed with GSMModel construction")
    else:
        print(f"✗ Model {complex_model} is not available")
        print("  -> Remote client should handle gracefully")
    
    print("\n" + "=" * 50)
    print("Remote client testing completed successfully!")
