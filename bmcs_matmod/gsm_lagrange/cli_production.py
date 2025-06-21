#!/usr/bin/env python3
"""
Production CLI Interface for GSM Models

This module provides a robust CLI interface for executing GSM models with 
systematic nomenclature and comprehensive error handling.
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

# Import with fallback
try:
    from .model_registry_robust import get_registry, ModelInfo
    from .data_structures import MaterialParameterData, LoadingData, SimulationConfig, SimulationResults
    from .parameter_loader import ParameterLoader
except ImportError:
    # Fallback for standalone execution
    try:
        from model_registry_robust import get_registry, ModelInfo
    except ImportError:
        # Create minimal fallbacks
        logger.warning("Using minimal fallbacks for demonstration")
        
        from dataclasses import dataclass
        from typing import Any
        
        @dataclass
        class ModelInfo:
            name: str
            mechanism: str
            description: str
            is_mock: bool = True
            cls: Any = None
        
        class MockRegistry:
            def list_models(self):
                return [
                    ModelInfo("GSM1D_ED", "ED", "Elasto-Damage"),
                    ModelInfo("GSM1D_VED", "VED", "Visco-Elasto-Damage"),
                ]
            
            def get_available_keys(self):
                return ["gsm1d_ed", "ed", "gsm1d_ved", "ved"]
            
            def get_available_mechanisms(self):
                return ["ED", "VED"]
            
            def list_by_mechanism(self, mech):
                return [m for m in self.list_models() if m.mechanism == mech.upper()]
            
            def format_model_table(self):
                return "Mock model table - see enhanced_cli.py for working example"
        
        def get_registry():
            return MockRegistry()
        
        # Mock data structures
        class MaterialParameterData:
            def __init__(self, **kwargs):
                self.parameters = kwargs
        
        class LoadingData:
            def __init__(self, time_array=None, strain_history=None, stress_history=None):
                self.time_array = time_array
                self.strain_history = strain_history  
                self.stress_history = stress_history
        
        class SimulationConfig:
            def __init__(self, **kwargs):
                self.config = kwargs
        
        class SimulationResults:
            def __init__(self, **kwargs):
                self.data = kwargs
            
            def to_json(self):
                return json.dumps(self.data, indent=2)
            
            def save_to_file(self, filepath):
                with open(filepath, 'w') as f:
                    f.write(self.to_json())
        
        class ParameterLoader:
            def load_parameters_from_string(self, param_str):
                params = json.loads(param_str)
                return MaterialParameterData(**params)

class GSMModelCLI:
    """Production CLI for GSM Models"""
    
    def __init__(self):
        self.registry = get_registry()
        try:
            self.parameter_loader = ParameterLoader()
        except:
            self.parameter_loader = ParameterLoader() if 'ParameterLoader' in globals() else None
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser"""
        parser = argparse.ArgumentParser(
            description="Execute GSM material models with systematic nomenclature",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
GSM Model Nomenclature:
  GSM1D_[MECHANISM][_HARDENING]
  
  Mechanisms:
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
  
  Future Hardening Types:
    LI     - Linear Isotropic
    NI     - Nonlinear Isotropic
    LK     - Linear Kinematic
    NK     - Nonlinear Kinematic
    LIHK   - Linear Isotropic + Hardening Kinematic
    NILHK  - Nonlinear Isotropic + Linear Hardening Kinematic

Examples:
  # List all available models
  python cli_production.py --list-models
  
  # List models by mechanism
  python cli_production.py --list-by-mechanism ED
  python cli_production.py --list-by-mechanism VEVPD
  
  # Run simulation with demo data
  python cli_production.py --model ed --formulation F --demo
  python cli_production.py --model gsm1d_vevpd --formulation G --demo
  
  # Run with parameters (when parameter loading is implemented)
  python cli_production.py --model ed --formulation F \\
    --params-inline '{"E": 30000, "S": 1.0, "c": 2.0, "r": 0.5, "eps_0": 0.001}' \\
    --loading-inline '{"type": "monotonic", "max_strain": 0.01, "steps": 100}'
            """
        )
        
        # Model selection
        parser.add_argument(
            '--model',
            choices=self.registry.get_available_keys(),
            help='GSM model to execute (use --list-models to see options)'
        )
        
        # Discovery and listing options
        discovery_group = parser.add_argument_group('Model Discovery')
        discovery_group.add_argument(
            '--list-models',
            action='store_true',
            help='List all available GSM models and exit'
        )
        
        discovery_group.add_argument(
            '--list-by-mechanism',
            help='List models by mechanism type (e.g., ED, VED, VEVPD) and exit'
        )
        
        discovery_group.add_argument(
            '--list-mechanisms',
            action='store_true',
            help='List all available mechanism types and exit'
        )
        
        # Simulation options
        sim_group = parser.add_argument_group('Simulation Options')
        sim_group.add_argument(
            '--formulation',
            choices=['F', 'G', 'Helmholtz', 'Gibbs'],
            default='F',
            help='Energy formulation: F/Helmholtz (strain-controlled) or G/Gibbs (stress-controlled)'
        )
        
        # Parameter input options
        param_group = parser.add_argument_group('Parameter Input')
        param_exclusive = param_group.add_mutually_exclusive_group()
        param_exclusive.add_argument(
            '--params',
            help='Parameter source: JSON file path, database URI, or network URL'
        )
        param_exclusive.add_argument(
            '--params-inline',
            help='Inline JSON string with material parameters'
        )
        
        # Loading input options  
        loading_group = parser.add_argument_group('Loading Input')
        loading_exclusive = loading_group.add_mutually_exclusive_group()
        loading_exclusive.add_argument(
            '--loading',
            help='Loading specification: JSON file path or inline JSON string'
        )
        loading_exclusive.add_argument(
            '--loading-inline',
            help='Inline JSON string with loading specification'
        )
        
        # Demo and testing
        demo_group = parser.add_argument_group('Demo and Testing')
        demo_group.add_argument(
            '--demo',
            action='store_true',
            help='Run demonstration simulation with synthetic data'
        )
        
        demo_group.add_argument(
            '--validate-only',
            action='store_true',
            help='Only validate inputs without running simulation'
        )
        
        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument(
            '--output',
            help='Output file path (JSON format). If not specified, prints to stdout'
        )
        
        output_group.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        
        output_group.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress all output except results'
        )
        
        return parser
    
    def list_available_models(self) -> str:
        """List all available GSM models"""
        return self.registry.format_model_table()
    
    def get_model_info(self, model_key: str) -> Optional[ModelInfo]:
        """Get model information by key"""
        return getattr(self.registry, 'get_model_info', lambda x: None)(model_key)
    
    def create_demo_loading(self, formulation: str) -> Any:
        """Create demonstration loading data"""
        time_array = np.linspace(0, 1, 21)  # 21 time steps
        
        if formulation in ['F', 'Helmholtz']:
            # Strain-controlled loading
            strain_history = 0.01 * np.sin(2 * np.pi * time_array)
            return LoadingData(
                time_array=time_array,
                strain_history=strain_history,
                stress_history=None
            ) if 'LoadingData' in globals() else {
                'time_array': time_array,
                'strain_history': strain_history,
                'stress_history': None
            }
        else:
            # Stress-controlled loading
            stress_history = 1000.0 * np.sin(2 * np.pi * time_array)
            return LoadingData(
                time_array=time_array,
                strain_history=None,
                stress_history=stress_history
            ) if 'LoadingData' in globals() else {
                'time_array': time_array,
                'strain_history': None,
                'stress_history': stress_history
            }
    
    def run_demo_simulation(self, model_info: ModelInfo, formulation: str) -> Dict[str, Any]:
        """Run demonstration simulation"""
        logger.info(f"Running demonstration simulation:")
        logger.info(f"  Model: {model_info.name} ({model_info.description})")
        logger.info(f"  Formulation: {formulation}")
        logger.info(f"  Status: {'Mock' if model_info.is_mock else 'Real'} model")
        
        # Create demo loading
        loading = self.create_demo_loading(formulation)
        
        # Create demo parameters
        demo_params = {
            'E': 30000.0,    # Young's modulus
            'S': 1.0,        # Damage parameter
            'c': 2.0,        # Damage evolution parameter
            'r': 0.5,        # Rate parameter  
            'eps_0': 0.001,  # Damage threshold
        }
        
        # Mock simulation results
        if formulation in ['F', 'Helmholtz']:
            # Strain-controlled
            strain_history = loading.get('strain_history', loading.strain_history if hasattr(loading, 'strain_history') else np.array([]))
            stress_response = demo_params['E'] * strain_history * (1 - 0.1 * np.abs(strain_history))
            time_array = loading.get('time_array', loading.time_array if hasattr(loading, 'time_array') else np.array([]))
            
            results = {
                'formulation': formulation,
                'control_variable': 'strain',
                'time': time_array.tolist(),
                'strain': strain_history.tolist(),
                'stress': stress_response.tolist(),
                'max_strain': float(np.max(np.abs(strain_history))),
                'max_stress': float(np.max(np.abs(stress_response))),
                'num_steps': len(time_array),
                'parameters': demo_params,
                'model': {
                    'name': model_info.name,
                    'mechanism': model_info.mechanism,
                    'description': model_info.description,
                    'is_mock': getattr(model_info, 'is_mock', True)
                }
            }
        else:
            # Stress-controlled
            stress_history = loading.get('stress_history', loading.stress_history if hasattr(loading, 'stress_history') else np.array([]))
            strain_response = stress_history / demo_params['E']
            time_array = loading.get('time_array', loading.time_array if hasattr(loading, 'time_array') else np.array([]))
            
            results = {
                'formulation': formulation,
                'control_variable': 'stress',
                'time': time_array.tolist(),
                'stress': stress_history.tolist(), 
                'strain': strain_response.tolist(),
                'max_stress': float(np.max(np.abs(stress_history))),
                'max_strain': float(np.max(np.abs(strain_response))),
                'num_steps': len(time_array),
                'parameters': demo_params,
                'model': {
                    'name': model_info.name,
                    'mechanism': model_info.mechanism,
                    'description': model_info.description,
                    'is_mock': getattr(model_info, 'is_mock', True)
                }
            }
        
        return results
    
    def main(self):
        """Main CLI entry point"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        # Configure logging
        if args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        try:
            # Handle discovery operations
            if args.list_models:
                print("Available GSM Models:")
                print("=" * 100)
                print(self.list_available_models())
                print(f"\nTotal: {len(self.registry.list_models())} models")
                print("\nUse --help to see usage examples")
                return
            
            if args.list_mechanisms:
                mechanisms = self.registry.get_available_mechanisms()
                print("Available Mechanism Types:")
                print("=" * 40)
                for i, mech in enumerate(mechanisms, 1):
                    count = len(self.registry.list_by_mechanism(mech))
                    print(f"{i:2d}. {mech:<6} ({count} model{'s' if count != 1 else ''})")
                print(f"\nTotal: {len(mechanisms)} mechanism types")
                return
            
            if args.list_by_mechanism:
                models = self.registry.list_by_mechanism(args.list_by_mechanism)
                if models:
                    print(f"GSM Models with mechanism '{args.list_by_mechanism.upper()}':")
                    print("=" * 70)
                    for model in models:
                        status = "Mock" if model.is_mock else "Real" 
                        print(f"{model.name:<12} | {status:<4} | {model.description}")
                else:
                    print(f"No models found with mechanism '{args.list_by_mechanism}'")
                    print("Available mechanisms:", ", ".join(self.registry.get_available_mechanisms()))
                return
            
            # Handle simulation
            if args.model:
                model_info = self.get_model_info(args.model)
                if not model_info:
                    print(f"Error: Model '{args.model}' not found")
                    print("Available models:", ", ".join(self.registry.get_available_keys()[:10]))
                    print("Use --list-models to see all available models")
                    sys.exit(1)
                
                if args.demo:
                    # Run demonstration simulation
                    results = self.run_demo_simulation(model_info, args.formulation)
                    
                    if args.output:
                        with open(args.output, 'w') as f:
                            json.dump(results, f, indent=2)
                        logger.info(f"Results saved to {args.output}")
                    else:
                        print(json.dumps(results, indent=2))
                    return
                
                # TODO: Implement real simulation with parameter/loading files
                print(f"Model '{model_info.name}' selected")
                print("Real simulation not yet implemented - use --demo for demonstration")
                return
            
            # Show help if no specific action
            parser.print_help()
            
        except KeyboardInterrupt:
            if not args.quiet:
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
    cli = GSMModelCLI()
    cli.main()

if __name__ == "__main__":
    main()
