#!/usr/bin/env python3
"""
Minimal GSM CLI for testing import fixes
This version removes dependencies that cause issues
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


class MinimalGSMCLI:
    """Minimal GSM CLI for testing basic functionality"""
    
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
            gsm_defs = get_gsm_defs(debug=False)
            gsm_keys = [key for key in gsm_defs.keys() if key.startswith('GSM1D_')]
            
            if json_output:
                result = {
                    "status": "success",
                    "models": gsm_keys,
                    "count": len(gsm_keys)
                }
                print(json.dumps(result, indent=2))
            else:
                print("Available GSM Models:")
                print("=" * 50)
                for key in sorted(gsm_keys):
                    print(f"- {key}")
                print(f"\\nTotal: {len(gsm_keys)} models available")
                
        except Exception as e:
            if json_output:
                result = {"status": "error", "error": str(e)}
                print(json.dumps(result, indent=2))
            else:
                print(f"Error listing models: {e}")
    
    def show_all_keys(self, json_output: bool = False) -> None:
        """Show all available keys in the registry"""
        try:
            gsm_defs = get_gsm_defs(debug=False)
            all_keys = list(gsm_defs.keys())
            gsm_keys = [key for key in all_keys if key.startswith('GSM1D_')]
            other_keys = [key for key in all_keys if not key.startswith('GSM1D_')]
            
            if json_output:
                result = {
                    "status": "success",
                    "gsm_models": gsm_keys,
                    "other_keys": other_keys,
                    "total_count": len(all_keys)
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"GSM Models ({len(gsm_keys)}):")
                for key in sorted(gsm_keys):
                    print(f"  - {key}")
                
                if other_keys:
                    print(f"\\nOther Keys ({len(other_keys)}):")
                    for key in sorted(other_keys):
                        print(f"  - {key}")
                
                print(f"\\nTotal: {len(all_keys)} keys available")
                
        except Exception as e:
            if json_output:
                result = {"status": "error", "error": str(e)}
                print(json.dumps(result, indent=2))
            else:
                print(f"Error showing keys: {e}")
    
    def model_info(self, model_name: str, json_output: bool = False) -> None:
        """Show information about a specific model"""
        try:
            gsm_defs = get_gsm_defs(debug=False)
            
            if model_name not in gsm_defs:
                if json_output:
                    result = {"status": "error", "error": f"Model '{model_name}' not found"}
                    print(json.dumps(result, indent=2))
                else:
                    print(f"Error: Model '{model_name}' not found")
                return
            
            gsm_def_class = gsm_defs[model_name]
            
            if json_output:
                result = {
                    "status": "success",
                    "model": model_name,
                    "class_name": gsm_def_class.__name__,
                    "module": gsm_def_class.__module__
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"Model Information: {model_name}")
                print("=" * 50)
                print(f"Class name: {gsm_def_class.__name__}")
                print(f"Module: {gsm_def_class.__module__}")
                
        except Exception as e:
            if json_output:
                result = {"status": "error", "error": str(e)}
                print(json.dumps(result, indent=2))
            else:
                print(f"Error getting model info: {e}")
    
    def test_access(self, model_name: str, json_output: bool = False) -> None:
        """Test access to a specific model"""
        try:
            gsm_defs = get_gsm_defs(debug=False)
            
            if model_name not in gsm_defs:
                if json_output:
                    result = {"status": "error", "error": f"Model '{model_name}' not found"}
                    print(json.dumps(result, indent=2))
                else:
                    print(f"Error: Model '{model_name}' not found")
                return
            
            gsm_def_class = gsm_defs[model_name]
            
            # Try to instantiate the class
            try:
                instance = gsm_def_class()
                access_ok = True
                error_msg = None
            except Exception as e:
                access_ok = False
                error_msg = str(e)
            
            if json_output:
                result = {
                    "status": "success",
                    "model": model_name,
                    "accessible": access_ok,
                    "error": error_msg
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"Model Access Test: {model_name}")
                print("=" * 50)
                if access_ok:
                    print("✅ Model is accessible and can be instantiated")
                else:
                    print(f"❌ Model access failed: {error_msg}")
                    
        except Exception as e:
            if json_output:
                result = {"status": "error", "error": str(e)}
                print(json.dumps(result, indent=2))
            else:
                print(f"Error testing model access: {e}")
    
    def main(self):
        """Main entry point"""
        args = self.parser.parse_args()
        
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


if __name__ == "__main__":
    cli = MinimalGSMCLI()
    cli.main()
