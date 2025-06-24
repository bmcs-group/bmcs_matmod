#!/usr/bin/env python3
"""
Working Minimal GSM CLI for testing
This version uses LazyGSMDefRegistry correctly
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


class WorkingMinimalGSMCLI:
    """Working Minimal GSM CLI for testing basic functionality"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.registry = LazyGSMDefRegistry(debug=True)
    
    def _create_parser(self):
        """Create command line argument parser"""
        parser = argparse.ArgumentParser(
            description='GSM (Generalized Stress Model) Command Line Interface',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Basic commands
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
            
            if json_output:
                result = {
                    "status": "success",
                    "keys": all_keys,
                    "count": len(all_keys)
                }
                print(json.dumps(result, indent=2))
            else:
                print("Available Access Keys:")
                print("=" * 50)
                for key in sorted(all_keys):
                    print(f"  - {key}")
                print(f"\nTotal: {len(all_keys)} keys")
        except Exception as e:
            logger.error(f"Error showing keys: {e}")
            if json_output:
                print(json.dumps({"status": "error", "message": str(e)}, indent=2))
            else:
                print(f"Error: {e}")
    
    def model_info(self, model_name: str, json_output: bool = False) -> None:
        """Show information about a specific model"""
        try:
            if not self.registry.has_def(model_name):
                available = self.registry.get_access_keys()
                if json_output:
                    result = {
                        "status": "error",
                        "message": f"Model '{model_name}' not found",
                        "available": available
                    }
                    print(json.dumps(result, indent=2))
                else:
                    print(f"Error: Model '{model_name}' not found")
                    print(f"Available models: {', '.join(available[:5])}")
                return
            
            # Get module path (safe, no imports)
            module_path = self.registry.get_module_path(model_name)
            
            if json_output:
                result = {
                    "status": "success",
                    "model_name": model_name,
                    "module_path": module_path
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"Model Information:")
                print("=" * 50)
                print(f"Name: {model_name}")
                print(f"Module: {module_path}")
                
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            if json_output:
                print(json.dumps({"status": "error", "message": str(e)}, indent=2))
            else:
                print(f"Error: {e}")
    
    def test_access(self, model_name: str, json_output: bool = False) -> None:
        """Test access to a specific model (will attempt to import)"""
        try:
            if not self.registry.has_def(model_name):
                available = self.registry.get_access_keys()
                if json_output:
                    result = {
                        "status": "error",
                        "message": f"Model '{model_name}' not found",
                        "available": available
                    }
                    print(json.dumps(result, indent=2))
                else:
                    print(f"Error: Model '{model_name}' not found")
                    print(f"Available models: {', '.join(available[:5])}")
                return
            
            # Try to load the actual class (will import)
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
        else:
            self.parser.print_help()


def main():
    """Main entry point"""
    try:
        cli = WorkingMinimalGSMCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
