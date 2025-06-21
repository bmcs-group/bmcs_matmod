#!/usr/bin/env python3
"""
Enhanced CLI Interface for GSM Models - Stand-alone Version

This is a simplified version focusing on the systematic nomenclature and CLI functionality
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockGSMModel:
    """Mock GSM model for demonstration"""
    def __init__(self, name: str, mechanism: str, description: str):
        self.name = name
        self.mechanism = mechanism
        self.description = description
    
    def get_F_response(self, eps_history, time_array):
        """Mock Helmholtz response"""
        import numpy as np
        return {
            'stress': 30000.0 * eps_history * (1 - 0.1 * eps_history),
            'strain': eps_history,
            'time': time_array
        }
    
    def get_G_response(self, sig_history, time_array):
        """Mock Gibbs response"""
        import numpy as np
        return {
            'strain': sig_history / 30000.0,
            'stress': sig_history,
            'time': time_array
        }

class GSMModelRegistry:
    """Simplified model registry for demonstration"""
    
    def __init__(self):
        self.models = self._create_demo_models()
    
    def _create_demo_models(self) -> Dict[str, MockGSMModel]:
        """Create demonstration models following the nomenclature"""
        models = {}
        
        # Define model configurations
        model_configs = [
            ('GSM1D_ED', 'ED', 'Elasto-Damage'),
            ('GSM1D_VE', 'VE', 'Visco-Elastic'),
            ('GSM1D_VED', 'VED', 'Visco-Elasto-Damage'),
            ('GSM1D_EP', 'EP', 'Elasto-Plastic'),
            ('GSM1D_EPD', 'EPD', 'Elasto-Plastic-Damage'),
            ('GSM1D_VEP', 'VEP', 'Visco-Elasto-Plastic'),
            ('GSM1D_EVP', 'EVP', 'Elasto-Visco-Plastic'),
            ('GSM1D_EVPD', 'EVPD', 'Elasto-Visco-Plastic-Damage'),
            ('GSM1D_VEVP', 'VEVP', 'Visco-Elasto-Visco-Plastic'),
            ('GSM1D_VEVPD', 'VEVPD', 'Visco-Elasto-Visco-Plastic-Damage'),
        ]
        
        for name, mechanism, description in model_configs:
            model = MockGSMModel(name, mechanism, description)
            models[name.lower()] = model
            models[mechanism.lower()] = model  # Allow access by mechanism
        
        return models
    
    def get_model(self, key: str) -> Optional[MockGSMModel]:
        """Get model by key"""
        return self.models.get(key.lower())
    
    def list_models(self) -> List[MockGSMModel]:
        """List unique models"""
        seen = set()
        unique_models = []
        for model in self.models.values():
            if model.name not in seen:
                unique_models.append(model)
                seen.add(model.name)
        return sorted(unique_models, key=lambda m: m.name)
    
    def list_by_mechanism(self, mechanism: str) -> List[MockGSMModel]:
        """List models by mechanism"""
        mechanism = mechanism.upper()
        return [m for m in self.list_models() if m.mechanism == mechanism]
    
    def get_available_keys(self) -> List[str]:
        """Get all available keys"""
        return sorted(self.models.keys())
    
    def get_available_mechanisms(self) -> List[str]:
        """Get available mechanisms"""
        mechanisms = set()
        for model in self.list_models():
            mechanisms.add(model.mechanism)
        return sorted(mechanisms)
    
    def format_model_table(self) -> str:
        """Format models as table"""
        models = self.list_models()
        if not models:
            return "No GSM models found."
        
        # Calculate column widths
        name_width = max(len("Model Name"), max(len(m.name) for m in models))
        mech_width = max(len("Mechanism"), max(len(m.mechanism) for m in models))
        desc_width = max(len("Description"), max(len(m.description) for m in models))
        
        # Format table
        header = f"{'Model Name':<{name_width}} | {'Mechanism':<{mech_width}} | {'Description':<{desc_width}}"
        separator = "-" * len(header)
        
        lines = [header, separator]
        for model in models:
            line = f"{model.name:<{name_width}} | {model.mechanism:<{mech_width}} | {model.description:<{desc_width}}"
            lines.append(line)
        
        return "\n".join(lines)

class GSMModelCLI:
    """Enhanced CLI for GSM Models"""
    
    def __init__(self):
        self.registry = GSMModelRegistry()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Execute GSM material models with systematic nomenclature",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # List all available models
  python enhanced_cli.py --list-models
  
  # List models by mechanism
  python enhanced_cli.py --list-by-mechanism ED
  
  # Run a simulation (demonstration)
  python enhanced_cli.py --model ed --formulation F --demo
            """
        )
        
        # Model selection
        parser.add_argument(
            '--model',
            choices=self.registry.get_available_keys(),
            help='GSM model to execute'
        )
        
        # Listing options
        parser.add_argument(
            '--list-models',
            action='store_true',
            help='List all available GSM models'
        )
        
        parser.add_argument(
            '--list-by-mechanism',
            help='List models by mechanism type (e.g., ED, VED, VEVPD)'
        )
        
        # Formulation
        parser.add_argument(
            '--formulation',
            choices=['F', 'G', 'Helmholtz', 'Gibbs'],
            default='F',
            help='Energy formulation'
        )
        
        # Demo mode
        parser.add_argument(
            '--demo',
            action='store_true',
            help='Run in demonstration mode with mock data'
        )
        
        # Verbose output
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Verbose output'
        )
        
        return parser
    
    def run_demo_simulation(self, model: MockGSMModel, formulation: str):
        """Run a demonstration simulation"""
        import numpy as np
        
        print(f"Running demonstration simulation:")
        print(f"  Model: {model.name} ({model.description})")
        print(f"  Formulation: {formulation}")
        
        # Create mock loading
        time_array = np.linspace(0, 1, 11)
        
        if formulation in ['F', 'Helmholtz']:
            # Strain-controlled
            strain_history = 0.01 * np.sin(2 * np.pi * time_array)
            response = model.get_F_response(strain_history, time_array)
            
            print(f"\nStrain-controlled simulation results:")
            print(f"  Time steps: {len(time_array)}")
            print(f"  Max strain: {np.max(strain_history):.6f}")
            print(f"  Max stress: {np.max(response['stress']):.2f}")
            
        else:
            # Stress-controlled
            stress_history = 1000.0 * np.sin(2 * np.pi * time_array)
            response = model.get_G_response(stress_history, time_array)
            
            print(f"\nStress-controlled simulation results:")
            print(f"  Time steps: {len(time_array)}")
            print(f"  Max stress: {np.max(stress_history):.2f}")
            print(f"  Max strain: {np.max(response['strain']):.6f}")
    
    def main(self):
        """Main CLI entry point"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        try:
            # Handle listing operations
            if args.list_models:
                print("Available GSM Models:")
                print("=" * 80)
                print(self.registry.format_model_table())
                print(f"\nTotal: {len(self.registry.list_models())} models")
                return
            
            if args.list_by_mechanism:
                models = self.registry.list_by_mechanism(args.list_by_mechanism)
                if models:
                    print(f"GSM Models with mechanism '{args.list_by_mechanism.upper()}':")
                    print("=" * 60)
                    for model in models:
                        print(f"{model.name} - {model.description}")
                else:
                    print(f"No models found with mechanism '{args.list_by_mechanism}'")
                    print("Available mechanisms:", ", ".join(self.registry.get_available_mechanisms()))
                return
                
            # Handle simulation
            if args.model and args.demo:
                model = self.registry.get_model(args.model)
                if not model:
                    print(f"Model '{args.model}' not found")
                    print("Available models:", ", ".join(self.registry.get_available_keys()))
                    sys.exit(1)
                
                self.run_demo_simulation(model, args.formulation)
                return
            
            # Show help if no specific action
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
    """Entry point"""
    cli = GSMModelCLI()
    cli.main()

if __name__ == "__main__":
    main()
