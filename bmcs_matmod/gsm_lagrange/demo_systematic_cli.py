#!/usr/bin/env python3
"""
Demo script for the enhanced GSM CLI interface

This script demonstrates the systematic model discovery and CLI capabilities
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from gsm_lagrange.cli_interface import GSMModelCLI
    from gsm_lagrange.model_registry import get_registry
except ImportError:
    # Try alternative import method
    sys.path.insert(0, str(Path(__file__).parent))
    from cli_interface import GSMModelCLI
    from model_registry import get_registry

def demo_model_discovery():
    """Demonstrate model discovery capabilities"""
    print("="*80)
    print("GSM Model Discovery Demo")
    print("="*80)
    
    registry = get_registry()
    
    print(f"\nTotal discovered models: {len(registry.models)}")
    print(f"Available keys: {registry.get_available_keys()}")
    print(f"Available mechanisms: {registry.get_available_mechanisms()}")
    
    print("\nFull model table:")
    print(registry.format_model_table())
    
    print("\nModels by mechanism:")
    for mechanism in registry.get_available_mechanisms():
        models = registry.list_by_mechanism(mechanism)
        print(f"  {mechanism}: {len(models)} models")
        for model in models:
            print(f"    - {model.name}")

def demo_cli_help():
    """Demonstrate CLI help and options"""
    print("\n" + "="*80)
    print("GSM CLI Interface Demo")
    print("="*80)
    
    cli = GSMModelCLI()
    parser = cli.create_parser()
    
    print("\nCLI Help:")
    parser.print_help()

def demo_cli_listing():
    """Demonstrate CLI model listing"""
    print("\n" + "="*80)  
    print("CLI Model Listing Demo")
    print("="*80)
    
    cli = GSMModelCLI()
    
    print("\nList all models (via CLI):")
    print("-" * 40)
    print(cli.list_available_models())
    
    # Test mechanism-specific listing
    registry = get_registry()
    available_mechanisms = registry.get_available_mechanisms()
    
    if available_mechanisms:
        test_mechanism = available_mechanisms[0]
        print(f"\nModels with mechanism '{test_mechanism}':")
        print("-" * 40)
        models = registry.list_by_mechanism(test_mechanism)
        for model in models:
            print(f"{model.name} - {model.description}")

def main():
    """Main demo function"""
    try:
        demo_model_discovery()
        demo_cli_help()
        demo_cli_listing()
        
        print("\n" + "="*80)
        print("Demo completed successfully!")
        print("="*80)
        
        print("\nTo test the CLI interface, try:")
        print("python -m bmcs_matmod.gsm_lagrange.cli_interface --list-models")
        print("python -m bmcs_matmod.gsm_lagrange.cli_interface --list-by-mechanism ED")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
