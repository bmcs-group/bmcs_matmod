#!/usr/bin/env python3
"""
Clean CLI Interface for GSM Definitions

This module provides a clean CLI interface using the unified GSM definition registry
without complex fallback mechanisms.
"""

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from .gsm_def_registry import get_available_gsm_defs, check_gsm_def_exists, get_gsm_def_module_path, get_mechanism_description
except ImportError:
    # Fallback for direct execution
    from gsm_def_registry import get_available_gsm_defs, check_gsm_def_exists, get_gsm_def_module_path, get_mechanism_description

class GSMDefRegistry:
    """Wrapper to provide registry interface for GSM definitions using lightweight discovery"""
    
    def __init__(self):
        self.def_list, self.registry_dict = get_available_gsm_defs(debug=False)
    
    def list_definitions(self) -> List[str]:
        """List all GSM definition names (full names only)"""
        return self.def_list
    
    def get_definition(self, key: str) -> str:
        """Get GSM definition module path by key, raise error if not found"""
        module_path = get_gsm_def_module_path(key)
        if module_path is None:
            available = list(self.registry_dict.keys())
            raise ValueError(f"GSM definition '{key}' not found. Available definitions: {available}")
        return module_path
    
    def get_all_keys(self) -> List[str]:
        """Get all available keys including aliases"""
        return sorted(self.registry_dict.keys())
    
    def definition_exists(self, key: str) -> bool:
        """Check if a GSM definition exists"""
        return check_gsm_def_exists(key)
    
    # Backward compatibility methods (keep old names for CLI compatibility)
    def list_models(self) -> List[str]:
        """List all GSM definition names (backward compatibility)"""
        return self.list_definitions()
    
    def get_model(self, key: str) -> str:
        """Get GSM definition module path (backward compatibility)"""
        return self.get_definition(key)
    
    def model_exists(self, key: str) -> bool:
        """Check if a GSM definition exists (backward compatibility)"""
        return self.definition_exists(key)


class GSMDefCLI:
    """Clean CLI for GSM Definitions"""
    
    def __init__(self):
        self.registry = GSMDefRegistry()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Execute GSM material models",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
GSM Model Nomenclature:
  GSM1D_[MECHANISM]
  
  Available Mechanisms:
    ED     - Elasto-Damage
    VE     - Visco-Elastic  
    VED    - Visco-Elasto-Damage
    EP     - Elasto-Plastic
    EPD    - Elasto-Plastic-Damage
    VEP    - Visco-Elasto-Plastic
    EVP    - Elasto-Visco-Plastic
    EVPD   - Elasto-Visco-Plastic-Damage
    VEVP   - Visco-Elasto-Visco-Plastic
    VEVPD  - Visco-Elasto-Visco-Plastic-Damage

Examples:
  # List all available models
  python cli_clean.py --list-models
  
  # Get model information
  python cli_clean.py --model-info ED
  
  # Test model access
  python cli_clean.py --test-access GSM1D_ED
            """
        )
        
        # Model operations
        parser.add_argument(
            '--list-models',
            action='store_true',
            help='List all available GSM models'
        )
        
        parser.add_argument(
            '--model-info',
            help='Show information about a specific model'
        )
        
        parser.add_argument(
            '--test-access',
            help='Test accessing a model by key'
        )
        
        parser.add_argument(
            '--show-all-keys',
            action='store_true',
            help='Show all available access keys'
        )
        
        # Model availability check
        parser.add_argument(
            '--check-model',
            help='Check if a specific model is available'
        )
        
        # Output options
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        
        parser.add_argument(
            '--json-output',
            action='store_true',
            help='Output results in JSON format for remote clients'
        )
        
        return parser
    
    def list_models(self):
        """List all available models"""
        models = self.registry.list_models()
        print("Available GSM Models:")
        print("=" * 50)
        
        for i, model_name in enumerate(models, 1):
            try:
                # Extract mechanism
                mechanism = model_name[6:] if model_name.startswith('GSM1D_') else "Unknown"
                
                # Get description
                desc = get_mechanism_description(mechanism)
                
                print(f"{i:2d}. {model_name}")
                print(f"    Mechanism: {mechanism}")
                print(f"    Description: {desc}")
                
                # Show module path
                module_path = self.registry.get_model(model_name)
                print(f"    Module: {module_path}")
                print()
                
            except Exception as e:
                print(f"{i:2d}. {model_name} - Error: {e}")
        
        print(f"Total: {len(models)} models")
    
    def show_model_info(self, key: str):
        """Show detailed information about a model"""
        try:
            if not self.registry.model_exists(key):
                print(f"Model '{key}' not found.")
                available = [k for k in self.registry.get_all_keys() if k.startswith('GSM1D_')]
                print(f"Available models: {available}")
                return
            
            module_path = self.registry.get_model(key)
            
            print(f"Model Information for '{key}':")
            print("=" * 60)
            print(f"Module Path: {module_path}")
            
            # Extract mechanism and show description
            if key.startswith('GSM1D_'):
                mechanism = key[6:]
            elif key.upper() in ['ED', 'VE', 'VED', 'EP', 'EPD', 'EVP', 'EVPD', 'VEVP', 'VEVPD']:
                mechanism = key.upper()
            else:
                mechanism = key
            
            description = get_mechanism_description(mechanism)
            print(f"Mechanism: {mechanism}")
            print(f"Description: {description}")
            
            print(f"\nModel Status: Available for import")
            print(f"Can be used in: GSMModel construction")
            
        except Exception as e:
            print(f"Error getting model info: {e}")
    
    def test_model_access(self, key: str):
        """Test accessing a model by key"""
        print(f"Testing access to model '{key}':")
        print("-" * 40)
        
        try:
            if not self.registry.model_exists(key):
                print(f"✗ Model '{key}' not found")
                available = [k for k in self.registry.get_all_keys() if k.startswith('GSM1D_')]
                print(f"Available models: {available}")
                return
            
            module_path = self.registry.get_model(key)
            print(f"✓ Model found: {key}")
            print(f"  Module path: {module_path}")
            
            # Try to import the module (this will test if it's actually importable)
            try:
                import importlib
                module = importlib.import_module(module_path)
                print(f"✓ Successfully imported module: {module}")
                
                # Look for GSMDef subclasses in the module
                import inspect
                classes_found = []
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if name.startswith('GSM1D_'):
                        classes_found.append(name)
                
                if classes_found:
                    print(f"✓ Found GSM classes: {classes_found}")
                    # Try to instantiate the first one
                    first_class = getattr(module, classes_found[0])
                    instance = first_class()
                    print(f"✓ Successfully created instance: {instance}")
                else:
                    print(f"⚠ No GSM1D classes found in module")
                    
            except Exception as e:
                print(f"✗ Could not import or instantiate: {e}")
                
        except Exception as e:
            print(f"✗ Error accessing model: {e}")
    
    def show_all_keys(self):
        """Show all available access keys"""
        keys = self.registry.get_all_keys()
        print("All Available Access Keys:")
        print("=" * 40)
        
        full_names = []
        mechanisms = []
        lowercase = []
        
        for key in keys:
            if key.startswith('GSM1D_'):
                full_names.append(key)
            elif key.isupper() and len(key) <= 6:
                mechanisms.append(key)
            else:
                lowercase.append(key)
        
        print("Full Names:")
        for name in sorted(full_names):
            print(f"  {name}")
        
        print("\nMechanism Codes:")
        for mech in sorted(mechanisms):
            print(f"  {mech}")
        
        print("\nLowercase Aliases:")
        for alias in sorted(lowercase):
            print(f"  {alias}")
        
        print(f"\nTotal keys: {len(keys)}")
    
    def check_model_availability(self, key: str, json_output: bool = False):
        """Check if a model is available for remote clients"""
        result = {
            'model_key': key,
            'available': False,
            'module_path': None,
            'mechanism': None,
            'description': None,
            'error': None
        }
        
        try:
            if self.registry.model_exists(key):
                result['available'] = True
                result['module_path'] = self.registry.get_model(key)
                
                # Extract mechanism
                if key.startswith('GSM1D_'):
                    mechanism = key[6:]
                elif key.upper() in ['ED', 'VE', 'VED', 'EP', 'EPD', 'EVP', 'EVPD', 'VEVP', 'VEVPD']:
                    mechanism = key.upper()
                else:
                    mechanism = key
                
                result['mechanism'] = mechanism
                result['description'] = get_mechanism_description(mechanism)
            else:
                result['error'] = f"Model '{key}' not found"
                
        except Exception as e:
            result['error'] = str(e)
        
        if json_output:
            import json
            print(json.dumps(result, indent=2))
        else:
            print(f"Model Availability Check for '{key}':")
            print("-" * 50)
            if result['available']:
                print(f"✓ Status: Available")
                print(f"✓ Module: {result['module_path']}")
                print(f"✓ Mechanism: {result['mechanism']}")
                print(f"✓ Description: {result['description']}")
            else:
                print(f"✗ Status: Not Available")
                if result['error']:
                    print(f"✗ Error: {result['error']}")
        
        return result
    
    def main(self):
        """Main CLI entry point"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        try:
            if args.list_models:
                self.list_models()
                return
            
            if args.model_info:
                self.show_model_info(args.model_info)
                return
            
            if args.test_access:
                self.test_model_access(args.test_access)
                return
            
            if args.show_all_keys:
                self.show_all_keys()
                return
            
            if args.check_model:
                self.check_model_availability(args.check_model, args.json_output)
                return
            
            # Show help if no action specified
            parser.print_help()
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

def main():
    """Entry point for command line execution"""
    cli = GSMDefCLI()
    cli.main()

if __name__ == "__main__":
    main()
