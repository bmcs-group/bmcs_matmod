#!/usr/bin/env python3
"""
Fixed GSM CLI for examples - avoids problematic imports
This version provides all the functionality needed for basic examples
without triggering imports that cause hangs.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional

# Add the current directory to Python path for direct execution
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import dependencies with fallback
try:
    from .gsm_def_registry import LazyGSMDefRegistry
except ImportError:
    from gsm_def_registry import LazyGSMDefRegistry


class FixedGSMCLI:
    """Fixed GSM CLI that avoids problematic imports while providing all needed functionality"""
    
    def __init__(self):
        self.parser = self._create_parser()
        # Use non-debug mode to reduce output
        self.registry = LazyGSMDefRegistry(debug=False)
    
    def _create_parser(self):
        """Create command line argument parser"""
        parser = argparse.ArgumentParser(
            description='GSM (Generalized Stress Model) Command Line Interface',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Basic commands (matching original CLI interface)
        parser.add_argument('--list-models', action='store_true',
                          help='List all available GSM models')
        parser.add_argument('--show-all-keys', action='store_true',
                          help='Show all available keys in the registry')
        parser.add_argument('--model-info', metavar='MODEL',
                          help='Show detailed information about a specific model')
        parser.add_argument('--test-access', metavar='MODEL',
                          help='Test access to a specific model')
        parser.add_argument('--json-output', action='store_true',
                          help='Output results in JSON format')
        
        # Additional CLI commands that might be expected
        parser.add_argument('--help-extended', action='store_true',
                          help='Show extended help information')
        parser.add_argument('--version', action='store_true',
                          help='Show version information')
        
        return parser
    
    def list_models(self, json_output: bool = False) -> None:
        """List all available GSM models"""
        try:
            model_names = self.registry.get_def_names()
            
            if json_output:
                result = {
                    "status": "success",
                    "models": model_names,
                    "count": len(model_names)
                }
                print(json.dumps(result, indent=2))
            else:
                print("Available GSM Models:")
                print("=" * 50)
                for name in sorted(model_names):
                    print(f"  - {name}")
                print(f"\nTotal: {len(model_names)} models")
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            if json_output:
                print(json.dumps({"status": "error", "message": str(e)}, indent=2))
            else:
                print(f"Error: {e}")
    
    def show_all_keys(self, json_output: bool = False) -> None:
        """Show all available keys in the registry"""
        try:
            all_keys = self.registry.get_access_keys()
            model_keys = [k for k in all_keys if k.startswith('GSM1D_')]
            alias_keys = [k for k in all_keys if not k.startswith('GSM1D_')]
            
            if json_output:
                result = {
                    "status": "success",
                    "gsm_models": model_keys,
                    "aliases": alias_keys,
                    "total_count": len(all_keys)
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"GSM Models ({len(model_keys)}):")
                print("=" * 50)
                for key in sorted(model_keys):
                    print(f"  - {key}")
                print(f"\nAliases ({len(alias_keys)}):")
                print("-" * 20)
                for key in sorted(alias_keys):
                    print(f"  - {key}")
                print(f"\nTotal: {len(all_keys)} access keys")
        except Exception as e:
            logger.error(f"Error showing keys: {e}")
            if json_output:
                print(json.dumps({"status": "error", "message": str(e)}, indent=2))
            else:
                print(f"Error: {e}")
    
    def model_info(self, model_name: str, json_output: bool = False) -> None:
        """Show information about a specific model (safe, no imports)"""
        try:
            if not self.registry.has_def(model_name):
                available = self.registry.get_access_keys()
                model_keys = [k for k in available if k.startswith('GSM1D_')]
                if json_output:
                    result = {
                        "status": "error",
                        "message": f"Model '{model_name}' not found",
                        "available": model_keys
                    }
                    print(json.dumps(result, indent=2))
                else:
                    print(f"Error: Model '{model_name}' not found")
                    print(f"Available models: {', '.join(model_keys[:5])}")
                    if len(model_keys) > 5:
                        print(f"... and {len(model_keys) - 5} more")
                return
            
            # Get module path (safe, no imports)
            module_path = self.registry.get_module_path(model_name)
            
            # Extract mechanism from model name
            if model_name.startswith('GSM1D_'):
                mechanism = model_name[6:]
            else:
                mechanism = model_name.upper()
            
            # Basic description lookup
            descriptions = {
                'ED': 'Elastic Damage model with isotropic damage evolution',
                'EP': 'Elastic Plastic model with isotropic hardening', 
                'EPD': 'Elastic Plastic Damage model combining plasticity and damage',
                'EVP': 'Elastic Viscoplastic model with rate-dependent response',
                'EVPD': 'Elastic Viscoplastic Damage model with rate and damage effects',
                'VE': 'Viscoelastic model with time-dependent response',
                'VED': 'Viscoelastic Damage model with time-dependent damage',
                'VEVP': 'Viscoelastic Viscoplastic model with multiple time scales',
                'VEVPD': 'Viscoelastic Viscoplastic Damage model with complex coupling'
            }
            
            description = descriptions.get(mechanism, f"GSM definition for {mechanism} mechanism")
            
            if json_output:
                result = {
                    "status": "success",
                    "model_name": model_name,
                    "mechanism": mechanism,
                    "description": description,
                    "module_path": module_path
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"Model Information:")
                print("=" * 50)
                print(f"Name: {model_name}")
                print(f"Mechanism: {mechanism}")
                print(f"Description: {description}")
                print(f"Module: {module_path}")
                print("\nNote: This is basic information without importing the actual model class.")
                print("      To access full model capabilities, use --test-access (may require dependencies).")
                
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            if json_output:
                print(json.dumps({"status": "error", "message": str(e)}, indent=2))
            else:
                print(f"Error: {e}")
    
    def test_access(self, model_name: str, json_output: bool = False) -> None:
        """Test access to a specific model (WILL attempt import - may hang)"""
        try:
            if not self.registry.has_def(model_name):
                available = self.registry.get_access_keys()
                model_keys = [k for k in available if k.startswith('GSM1D_')]
                if json_output:
                    result = {
                        "status": "error",
                        "message": f"Model '{model_name}' not found",
                        "available": model_keys
                    }
                    print(json.dumps(result, indent=2))
                else:
                    print(f"Error: Model '{model_name}' not found")
                    print(f"Available models: {', '.join(model_keys[:5])}")
                return
            
            print("WARNING: This command will attempt to import the GSM definition class.")
            print("         This may hang due to dependency issues (bmcs_utils.api).")
            print("         Use Ctrl+C to interrupt if needed.")
            print("")
            
            # Try to load the actual class (DANGEROUS - may hang)
            gsm_def_class = self.registry.get_def_class(model_name)
            
            if json_output:
                result = {
                    "status": "success",
                    "model_name": model_name,
                    "class_name": gsm_def_class.__name__,
                    "module": gsm_def_class.__module__
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"Access Test Successful:")
                print("=" * 50)
                print(f"Model: {model_name}")
                print(f"Class: {gsm_def_class.__name__}")
                print(f"Module: {gsm_def_class.__module__}")
                
        except Exception as e:
            logger.error(f"Error testing access: {e}")
            if json_output:
                print(json.dumps({"status": "error", "message": str(e)}, indent=2))
            else:
                print(f"Error: {e}")
    
    def show_version(self, json_output: bool = False) -> None:
        """Show version information"""
        version_info = {
            "cli_version": "1.0.0-fixed",
            "status": "Fixed version avoiding problematic imports"
        }
        
        if json_output:
            print(json.dumps(version_info, indent=2))
        else:
            print("GSM CLI Version Information:")
            print("=" * 40)
            print(f"CLI Version: {version_info['cli_version']}")
            print(f"Status: {version_info['status']}")
    
    def show_extended_help(self) -> None:
        """Show extended help information"""
        print("GSM CLI Extended Help")
        print("=" * 50)
        print("")
        print("This is a fixed version of the GSM CLI that avoids problematic imports.")
        print("It provides safe access to GSM model discovery and basic information.")
        print("")
        print("Available Operations:")
        print("  --list-models     : Lists all discovered GSM definitions (safe)")
        print("  --show-all-keys   : Shows all access keys including aliases (safe)")
        print("  --model-info MODEL: Shows basic model information (safe)")
        print("  --test-access MODEL: Attempts to import model class (may hang!)")
        print("")
        print("Known Issues:")
        print("  - Model imports may hang due to bmcs_utils.api dependency issues")
        print("  - Use Ctrl+C to interrupt hanging operations")
        print("")
        print("Safe Usage:")
        print("  Use --model-info instead of --test-access for basic model information")
        print("  All discovery operations (--list-models, --show-all-keys) are safe")
    
    def run(self):
        """Run the CLI"""
        args = self.parser.parse_args()
        
        # Handle commands
        if args.list_models:
            self.list_models(args.json_output)
        elif args.show_all_keys:
            self.show_all_keys(args.json_output)
        elif args.model_info:
            self.model_info(args.model_info, args.json_output)
        elif args.test_access:
            self.test_access(args.test_access, args.json_output)
        elif args.version:
            self.show_version(args.json_output)
        elif args.help_extended:
            self.show_extended_help()
        else:
            self.parser.print_help()


def main():
    """Main entry point"""
    try:
        cli = FixedGSMCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
