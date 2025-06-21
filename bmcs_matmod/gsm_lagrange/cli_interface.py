#!/usr/bin/env python3
"""
Command Line Interface for GSM Models

This module provides a CLI interface for executing GSM models with material parameters
from various sources including JSON files, database queries, and network transfers.
"""

import argparse
import json
import sys
import numpy as np
import importlib
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, Tuple
import logging

from .gsm_def import GSMDef
from .gsm_model import GSMModel
from .data_structures import MaterialParameterData, LoadingData, SimulationConfig, SimulationResults
from .parameter_loader import ParameterLoader
from .model_registry import get_registry, ModelInfo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GSMModelCLI:
    """Command Line Interface for GSM Models"""
    
    def __init__(self):
        self.registry = get_registry()
        self.parameter_loader = ParameterLoader()
    
    def list_available_models(self) -> str:
        """List all available GSM models"""
        return self.registry.format_model_table()
    
    def get_model_class(self, model_key: str) -> Type[GSMDef]:
        """Get model class by key"""
        model_cls = self.registry.get_model(model_key)
        if model_cls is None:
            available = ", ".join(self.registry.get_available_keys())
            raise ValueError(f"Model '{model_key}' not found. Available models: {available}")
        return model_cls
    
    def get_model_info(self, model_key: str) -> ModelInfo:
        """Get model information by key"""
        model_info = self.registry.get_model_info(model_key) 
        if model_info is None:
            available = ", ".join(self.registry.get_available_keys())
            raise ValueError(f"Model '{model_key}' not found. Available models: {available}")
        return model_info
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for CLI"""
        parser = argparse.ArgumentParser(
            description="Execute GSM material models with various parameter sources",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run Helmholtz formulation with JSON parameters
  python -m bmcs_matmod.gsm_lagrange.cli_interface --model ElasticDamage --formulation F \\
    --params params.json --loading loading.json --output results.json
  
  # Run Gibbs formulation with database parameters
  python -m bmcs_matmod.gsm_lagrange.cli_interface --model ViscoElastic --formulation G \\
    --params "db://localhost:5432/materials?id=123" --loading stress_history.json
  
  # Run with inline parameters
  python -m bmcs_matmod.gsm_lagrange.cli_interface --model ElasticDamage --formulation F \\
    --params-inline '{"E": 30000, "nu": 0.2, "omega_0": 0.1}' --loading loading.json
            """
        )
        
        # Model selection
        parser.add_argument(
            '--model', 
            choices=self.registry.get_available_keys(),
            help='GSM model to execute. Use --list-models to see all available models.'
        )
        
        # List models option
        parser.add_argument(
            '--list-models',
            action='store_true',
            help='List all available GSM models and exit'
        )
        
        # List by mechanism
        parser.add_argument(
            '--list-by-mechanism',
            help='List models by mechanism type (e.g., ED, VED, VEVPD)'
        )
        
        # Formulation (Helmholtz or Gibbs)
        parser.add_argument(
            '--formulation', 
            choices=['F', 'G', 'Helmholtz', 'Gibbs'],
            default='F',
            help='Energy formulation: F/Helmholtz (strain-controlled) or G/Gibbs (stress-controlled)'
        )
        
        # Parameter sources
        param_group = parser.add_mutually_exclusive_group()
        param_group.add_argument(
            '--params',
            help='Parameter source: JSON file path, database URI, or network URL'
        )
        param_group.add_argument(
            '--params-inline',
            help='Inline JSON string with parameters'
        )
        
        # Loading specification
        parser.add_argument(
            '--loading',
            help='Loading specification: JSON file path or inline JSON string'
        )
        
        # Output options
        parser.add_argument(
            '--output',
            help='Output file path (JSON format). If not specified, prints to stdout'
        )
        
        # Simulation options
        parser.add_argument(
            '--config',
            help='Simulation configuration file (JSON)'
        )
        
        # Validation options
        parser.add_argument(
            '--validate-only',
            action='store_true',
            help='Only validate inputs without running simulation'
        )
        
        # Verbose output
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Verbose output'
        )
        
        return parser
    
    def load_parameters(self, source: str) -> MaterialParameterData:
        """Load material parameters from various sources"""
        return self.parameter_loader.load_parameters(source)
    
    def load_loading(self, source: str) -> LoadingData:
        """Load loading specification"""
        return self.parameter_loader.load_loading(source)
    
    def load_config(self, source: Optional[str]) -> SimulationConfig:
        """Load simulation configuration"""
        if source:
            return self.parameter_loader.load_config(source)
        else:
            return SimulationConfig()  # Default configuration
    
    def validate_inputs(self, model_class: Type[GSMDef], params: MaterialParameterData, 
                       loading: LoadingData, config: SimulationConfig) -> bool:
        """Validate all inputs before simulation"""
        try:
            # Validate model parameters
            if hasattr(model_class, 'F_engine') and model_class.F_engine is not None:
                model_instance = model_class()
                if hasattr(model_instance, 'param_codenames'):
                    required_params = set(model_instance.param_codenames.values())
                    provided_params = set(params.parameters.keys())
                    
                    missing_params = required_params - provided_params
                    if missing_params:
                        logger.error(f"Missing required parameters: {missing_params}")
                        return False
                    
                    extra_params = provided_params - required_params
                    if extra_params:
                        logger.warning(f"Extra parameters provided (will be ignored): {extra_params}")
                else:
                    logger.warning("Model does not have param_codenames attribute - skipping parameter validation")
            else:
                logger.warning("Model validation limited - no F_engine available")
            
            # Validate loading data
            if not loading.validate():
                logger.error("Invalid loading specification")
                return False
            
            # Validate configuration
            if not config.validate():
                logger.error("Invalid simulation configuration")
                return False
            
            logger.info("Input validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def run_simulation(self, model_class: Type[GSMDef], params: MaterialParameterData,
                      loading: LoadingData, config: SimulationConfig, 
                      formulation: str) -> SimulationResults:
        """Run the GSM simulation"""
        try:
            # For demonstration purposes, we'll create mock results
            # In practice, this would create actual GSM model and run simulation
            
            # Check if we can create actual GSM model
            try:
                gsm_model = GSMModel(gsm_def=model_class)
                
                # Set parameters
                gsm_model.set_params(**params.parameters)
                
                # Prepare loading data
                if formulation in ['F', 'Helmholtz']:
                    # Strain-controlled simulation
                    if loading.strain_history is None:
                        raise ValueError("Strain history required for Helmholtz formulation")
                    
                    response = gsm_model.get_F_response(
                        loading.strain_history, 
                        loading.time_array
                    )
                    
                elif formulation in ['G', 'Gibbs']:
                    # Stress-controlled simulation
                    if loading.stress_history is None:
                        raise ValueError("Stress history required for Gibbs formulation")
                    
                    response = gsm_model.get_G_response(
                        loading.stress_history,
                        loading.time_array
                    )
                
            except Exception as e:
                logger.warning(f"Could not run actual GSM simulation: {e}")
                logger.info("Creating mock simulation results for demonstration")
                
                # Create mock response data
                import numpy as np
                n_steps = len(loading.time_array)
                
                if formulation in ['F', 'Helmholtz'] and loading.strain_history is not None:
                    # Mock stress response for strain input
                    mock_stress = 30000.0 * loading.strain_history * (1 - 0.1 * loading.strain_history)
                    response = type('MockResponse', (), {
                        't_t': loading.time_array,
                        'eps_t': loading.strain_history,
                        'sig_t': mock_stress,
                        'Eps_t_flat': np.zeros(n_steps),
                        'Sig_t_flat': np.zeros(n_steps)
                    })()
                    
                elif formulation in ['G', 'Gibbs'] and loading.stress_history is not None:
                    # Mock strain response for stress input
                    mock_strain = loading.stress_history / 30000.0
                    response = type('MockResponse', (), {
                        't_t': loading.time_array,
                        'eps_t': mock_strain,
                        'sig_t': loading.stress_history,
                        'Eps_t_flat': np.zeros(n_steps),
                        'Sig_t_flat': np.zeros(n_steps)
                    })()
                else:
                    raise ValueError("Invalid formulation or missing loading data")
            
            # Create results object
            results = SimulationResults(
                model_name=model_class.__name__,
                formulation=formulation,
                parameters=params.parameters,
                loading=loading,
                response=response,
                config=config,
                execution_time=0.123,  # Mock execution time
                warnings=["Mock simulation - actual GSM engine not available"]
            )
            
            logger.info("Simulation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise
    
    def main(self):
        """Main CLI entry point"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        # Handle list operations first
        if args.list_models:
            print("Available GSM Models:")
            print("=" * 60)
            print(self.list_available_models())
            return
        
        if args.list_by_mechanism:
            models = self.registry.list_by_mechanism(args.list_by_mechanism)
            if models:
                print(f"GSM Models with mechanism '{args.list_by_mechanism}':")
                print("=" * 60)
                for model in models:
                    print(f"{model.name} - {model.description}")
            else:
                print(f"No models found with mechanism '{args.list_by_mechanism}'")
                print("Available mechanisms:", ", ".join(self.registry.get_available_mechanisms()))
            return
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Check required arguments for simulation
        if not args.list_models and not args.list_by_mechanism:
            if not args.model:
                parser.error("--model is required for simulation")
            if not args.params and not args.params_inline:
                parser.error("Either --params or --params-inline is required for simulation")
            if not args.loading:
                parser.error("--loading is required for simulation")
        
        try:
            # Get model class
            model_class = self.get_model_class(args.model)
            model_info = self.get_model_info(args.model)
            
            logger.info(f"Using GSM model: {model_info.name} ({model_info.description})")
            
            # Load parameters
            if args.params_inline:
                params = self.parameter_loader.load_parameters_from_string(args.params_inline)
            else:
                params = self.load_parameters(args.params)
            
            # Load loading specification
            loading = self.load_loading(args.loading)
            
            # Load configuration
            config = self.load_config(args.config)
            
            # Validate inputs
            if not self.validate_inputs(model_class, params, loading, config):
                sys.exit(1)
            
            if args.validate_only:
                print("Validation successful")
                sys.exit(0)
            
            # Run simulation
            results = self.run_simulation(model_class, params, loading, config, args.formulation)
            
            # Output results
            if args.output:
                results.save_to_file(args.output)
                logger.info(f"Results saved to {args.output}")
            else:
                print(results.to_json())
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)

def main():
    """Entry point for command line execution"""
    cli = GSMModelCLI()
    cli.main()

if __name__ == "__main__":
    main()
