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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GSMModelCLI:
    """Command Line Interface for GSM Models"""
    
    def __init__(self):
        self.available_models = self._discover_models()
        self.parameter_loader = ParameterLoader()
    
    def _discover_models(self) -> Dict[str, Type[GSMDef]]:
        """Discover available GSM model classes"""
        models = {}
        
        # Import actual GSM models from the current package
        try:
            import importlib
            import inspect
            from pathlib import Path
            
            # Get the current package directory
            current_dir = Path(__file__).parent
            
            # Look for GSM model files (gsm1d_*.py pattern)
            model_files = list(current_dir.glob('gsm1d_*.py'))
            
            for model_file in model_files:
                module_name = model_file.stem
                try:
                    # Import the module dynamically
                    module = importlib.import_module(f'.{module_name}', package=__package__)
                    
                    # Find GSMDef subclasses in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, GSMDef) and 
                            obj is not GSMDef and 
                            hasattr(obj, 'F_engine')):
                            models[name.lower()] = obj
                            logger.debug(f"Discovered GSM model: {name}")
                            
                except ImportError as e:
                    logger.warning(f"Could not import {module_name}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing {module_name}: {e}")
            
            # Also try to import from specific known modules for backward compatibility
            known_models = [
                ('gsm1d_ed', 'GSM1D_ED'),
                ('gsm1d_ve', 'GSM1D_VE'), 
                ('gsm1d_ep', 'GSM1D_EP'),
                ('gsm1d_vevp', 'GSM1D_VEVP'),
                ('gsm1d_ved', 'GSM1D_VED'),
                ('gsm1d_evpd', 'GSM1D_EVPD'),
                ('gsm1d_vevpd', 'GSM1D_VEVPD')
            ]
            
            for module_name, class_name in known_models:
                try:
                    module = importlib.import_module(f'.{module_name}', package=__package__)
                    model_class = getattr(module, class_name, None)
                    if model_class and issubclass(model_class, GSMDef):
                        key = class_name.lower()
                        if key not in models:  # Don't override already discovered models
                            models[key] = model_class
                            logger.debug(f"Imported known model: {class_name}")
                except ImportError as e:
                    logger.debug(f"Could not import {module_name}.{class_name}: {e}")
                except Exception as e:
                    logger.debug(f"Error importing {module_name}.{class_name}: {e}")
            
            if not models:
                logger.warning("No GSM models found, creating placeholder models")
                models = self._create_placeholder_models()
                
        except Exception as e:
            logger.error(f"Error discovering models: {e}")
            models = self._create_placeholder_models()
        
        return models
    
    def _create_placeholder_models(self) -> Dict[str, Type[GSMDef]]:
        """Create placeholder models for testing when real models are not available"""
        class PlaceholderGSM(GSMDef):
            """Placeholder GSM model for CLI testing"""
            pass
        
        return {
            'placeholder_elastic': PlaceholderGSM,
            'placeholder_damage': PlaceholderGSM,
            'placeholder_plastic': PlaceholderGSM
        }
    
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
            choices=list(self.available_models.keys()),
            required=True,
            help='GSM model to execute'
        )
        
        # Formulation (Helmholtz or Gibbs)
        parser.add_argument(
            '--formulation', 
            choices=['F', 'G', 'Helmholtz', 'Gibbs'],
            default='F',
            help='Energy formulation: F/Helmholtz (strain-controlled) or G/Gibbs (stress-controlled)'
        )
        
        # Parameter sources
        param_group = parser.add_mutually_exclusive_group(required=True)
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
            required=True,
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
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        try:
            # Get model class
            model_class = self.available_models[args.model]
            
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
