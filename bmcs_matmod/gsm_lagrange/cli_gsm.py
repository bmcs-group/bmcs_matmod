#!/usr/bin/env python3
"""
GSM CLI Interface for Network Communication

This module provides a comprehensive CLI interface for GSM definitions with support for:
- Network communication and serialization
- Dynamic parameter specification retrieval
- Parameter validation against GSM definitions
- Cross-network computational node communication
- Workchain manager integration
"""

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union
import logging
import importlib
import inspect
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import urllib.request

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import dependencies
try:
    # Try relative imports first (when run as module)
    from .gsm_def_registry import (
        get_available_gsm_defs, check_gsm_def_exists, 
        get_gsm_def_module_path, get_mechanism_description,
        get_gsm_def_class
    )
    from .data_structures import MaterialParameterData, LoadingData, SimulationConfig, SimulationResults
    from .parameter_loader import ParameterLoader
except ImportError:
    # Fall back to absolute imports (when run directly)
    from gsm_def_registry import (
        get_available_gsm_defs, check_gsm_def_exists, 
        get_gsm_def_module_path, get_mechanism_description,
        get_gsm_def_class
    )
    from data_structures import MaterialParameterData, LoadingData, SimulationConfig, SimulationResults
    from parameter_loader import ParameterLoader

try:
    from .response_data import ResponseData
except ImportError:
    from response_data import ResponseData


class GSMParameterSpec:
    """Specification of parameters required by a GSM definition"""
    
    def __init__(self, gsm_def_class):
        self.gsm_def_class = gsm_def_class
        self.parameters = {}
        self.parameter_bounds = {}
        self.parameter_descriptions = {}
        self.parameter_units = {}
        self._extract_parameter_spec()
    
    def _extract_parameter_spec(self):
        """Extract parameter specifications from GSM definition"""
        try:
            # Try to get parameter specifications from the GSM definition
            if hasattr(self.gsm_def_class, 'param_codenames'):
                model_params = self.gsm_def_class.param_codenames
                for latex_symbol, codename in model_params.items():
                    param_codename = str(codename)
    
                    # Set default bounds and metadata
                    self.parameters[param_codename] = {
                        'latex_symbol': latex_symbol,
                        'required': True,
                        'type': 'float',
                        'default': None
                    }
                    
                    # Try to extract bounds and descriptions from docstring or attributes
                    self._extract_param_metadata(param_codename)
            
        except Exception as e:
            logger.warning(f"Could not extract parameter spec from {self.gsm_def_class.__name__}: {e}")
    
    def _extract_param_metadata(self, param_name: str):
        """Extract parameter metadata from GSM definition"""
        # This would be extended based on the actual GSM definition structure
        # For now, set reasonable defaults based on common parameter patterns
        common_bounds = {
            'E': (1000.0, 100000.0),  # Young's modulus in MPa
            'nu': (0.0, 0.5),         # Poisson's ratio
            'K': (1.0, 10000.0),      # Bulk modulus
            'G': (1.0, 50000.0),      # Shear modulus
            'S': (0.1, 10000.0),      # Damage parameter
            'r': (0.001, 1.0),        # Evolution rate
            'c': (0.1, 100.0),        # Cohesion
            'eps_0': (0.0, 0.01),     # Initial strain
            'f_c': (1.0, 100.0),      # Compressive strength
            'eta_vp': (1.0, 1000.0),  # Viscoplastic viscosity
            'eta_ve': (1.0, 1000.0),  # Viscoelastic viscosity
        }
        
        common_units = {
            'E': 'MPa', 'K': 'MPa', 'G': 'MPa', 'S': 'MPa', 'f_c': 'MPa',
            'nu': '-', 'r': '-', 'eps_0': '-',
            'c': 'MPa', 'eta_vp': 'MPa·s', 'eta_ve': 'MPa·s'
        }
        
        common_descriptions = {
            'E': 'Young\'s modulus',
            'nu': 'Poisson\'s ratio',
            'K': 'Bulk modulus',
            'G': 'Shear modulus',
            'S': 'Damage parameter',
            'r': 'Damage evolution rate',
            'c': 'Cohesion parameter',
            'eps_0': 'Initial strain',
            'f_c': 'Compressive strength',
            'eta_vp': 'Viscoplastic viscosity',
            'eta_ve': 'Viscoelastic viscosity'
        }
        
        if param_name in common_bounds:
            self.parameter_bounds[param_name] = common_bounds[param_name]
        
        if param_name in common_units:
            self.parameter_units[param_name] = common_units[param_name]
            
        if param_name in common_descriptions:
            self.parameter_descriptions[param_name] = common_descriptions[param_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_name': self.gsm_def_class.__name__,
            'parameters': self.parameters,
            'parameter_bounds': self.parameter_bounds,
            'parameter_descriptions': self.parameter_descriptions,
            'parameter_units': self.parameter_units
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def validate_parameters(self, params: Dict[str, float]) -> tuple[bool, List[str]]:
        """Validate parameters against specification"""
        errors = []
        
        # Check required parameters
        for param_name, spec in self.parameters.items():
            if spec.get('required', False) and param_name not in params:
                errors.append(f"Required parameter '{param_name}' is missing")
        
        # Check parameter bounds
        for param_name, value in params.items():
            if param_name in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param_name]
                if not (min_val <= value <= max_val):
                    errors.append(f"Parameter '{param_name}' = {value} is outside bounds [{min_val}, {max_val}]")
        
        return len(errors) == 0, errors


class GSMNetworkInterface:
    """Network interface for GSM computational nodes"""
    
    def __init__(self):
        self.parameter_loader = ParameterLoader()
    
    def execute_simulation(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simulation from network request"""
        try:
            # Parse request
            model_key = request_data.get('model')
            formulation = request_data.get('formulation', 'F')
            parameters = request_data.get('parameters')
            loading = request_data.get('loading')
            config = request_data.get('config', {})
            
            # Validate inputs
            if not model_key:
                raise ValueError("Model key is required")
            if not parameters:
                raise ValueError("Parameters are required")
            if not loading:
                raise ValueError("Loading specification is required")
            
            # Get GSM definition
            if not check_gsm_def_exists(model_key):
                available = get_available_gsm_defs(debug=False)[0]
                raise ValueError(f"Model '{model_key}' not found. Available: {available}")
            
            gsm_def_class = get_gsm_def_class(model_key)
            
            # Create data structures
            param_data = MaterialParameterData.from_dict(parameters)
            loading_data = LoadingData.from_dict(loading)
            config_data = SimulationConfig.from_dict(config)
            
            # Validate parameters against GSM specification
            param_spec = GSMParameterSpec(gsm_def_class)
            is_valid, errors = param_spec.validate_parameters(param_data.parameters)
            if not is_valid:
                raise ValueError(f"Parameter validation failed: {'; '.join(errors)}")
            
            # Execute simulation (this would be implemented with actual GSM engine)
            results = self._execute_gsm_simulation(
                gsm_def_class, param_data, loading_data, config_data, formulation
            )
            
            return {
                'status': 'success',
                'results': results.to_dict(),
                'message': 'Simulation completed successfully'
            }
            
        except Exception as e:
            logger.error(f"Simulation execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message': 'Simulation failed'
            }
    
    def _execute_gsm_simulation(self, gsm_def_class, param_data, loading_data, config_data, formulation):
        """Execute the actual GSM simulation"""
        try:
            # Create GSM instance
            gsm_instance = gsm_def_class()
            
            # Set parameters
            for param_name, value in param_data.parameters.items():
                if hasattr(gsm_instance, param_name):
                    setattr(gsm_instance, param_name, value)
            
            # Create GSM model (this would use the actual GSMModel)
            # For now, create mock results
            results = SimulationResults(
                model_name=gsm_def_class.__name__,
                formulation=formulation,
                parameters=param_data.parameters,
                loading=loading_data,
                config=config_data,
                response=self._create_mock_response(loading_data, formulation),
                execution_time=0.123,
                warnings=["Mock simulation - actual GSM engine integration needed"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"GSM simulation execution failed: {e}")
            raise
    
    def _create_mock_response(self, loading_data, formulation):
        """Create mock response data for demonstration"""
        # This would be replaced with actual GSM engine response
        n_steps = len(loading_data.time_array)
        
        if formulation in ['F', 'Helmholtz']:
            # Strain-controlled
            strain = loading_data.strain_history
            stress = 30000.0 * strain * (1 - 0.1 * np.abs(strain))  # Simple elastic-damage
        else:
            # Stress-controlled
            stress = loading_data.stress_history
            strain = stress / 30000.0  # Simple elastic
        
        # Create mock response container
        mock_response = {
            't_t': loading_data.time_array,
            'eps_t': strain,
            'sig_t': stress,
            'Eps_t_flat': np.column_stack([strain, np.zeros(n_steps)]),  # Mock internal vars
            'Sig_t_flat': np.column_stack([stress, np.zeros(n_steps)]),  # Mock conjugate vars
            'iter_t': np.ones(n_steps),  # Mock iterations
            'lam_t': np.ones(n_steps)    # Mock lambda
        }
        
        return mock_response


class GSMHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for GSM network interface"""
    
    def __init__(self, *args, network_interface=None, **kwargs):
        self.network_interface = network_interface
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path
            query_params = urllib.parse.parse_qs(parsed_path.query)
            
            response_data = {}
            status_code = 200
            
            if path == '/health':
                response_data = {'status': 'healthy', 'service': 'GSM CLI Network Interface'}
            
            elif path == '/models':
                # List available models
                def_list, registry_dict = get_available_gsm_defs(debug=False)
                response_data = {
                    'models': def_list,
                    'registry': {k: str(v) for k, v in registry_dict.items()}
                }
            
            elif path == '/param-spec':
                # Get parameter specification for a model
                model = query_params.get('model', [None])[0]
                if not model:
                    response_data = {'error': 'Model parameter is required'}
                    status_code = 400
                else:
                    try:
                        gsm_def_class = get_gsm_def_class(model)
                        param_spec = GSMParameterSpec(gsm_def_class)
                        response_data = param_spec.to_dict()
                    except Exception as e:
                        response_data = {'error': f'Failed to get parameter spec: {str(e)}'}
                        status_code = 400
            
            else:
                response_data = {'error': 'Endpoint not found'}
                status_code = 404
            
            # Send response
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data, indent=2, default=str).encode())
            
        except Exception as e:
            logger.error(f"GET request error: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            request_body = self.rfile.read(content_length).decode('utf-8')
            request_data = json.loads(request_body) if request_body else {}
            
            response_data = {}
            status_code = 200
            
            if path == '/simulate':
                # Execute simulation
                if not self.network_interface:
                    response_data = {'error': 'Network interface not available'}
                    status_code = 500
                else:
                    response_data = self.network_interface.execute_simulation(request_data)
                    if response_data.get('status') == 'error':
                        status_code = 400
            
            else:
                response_data = {'error': 'Endpoint not found'}
                status_code = 404
            
            # Send response
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data, indent=2, default=str).encode())
            
        except Exception as e:
            logger.error(f"POST request error: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")


class GSMDefCLI:
    """Enhanced CLI for GSM Definitions with Network Support"""
    
    def __init__(self):
        self.def_list, self.registry_dict = get_available_gsm_defs(debug=False)
        self.parameter_loader = ParameterLoader()
        self.network_interface = GSMNetworkInterface()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser"""
        parser = argparse.ArgumentParser(
            description="GSM CLI Interface with Network Communication Support",
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

Network Communication Examples:
  # Get parameter specification for dynamic validation
  python cli_gsm.py --get-param-spec GSM1D_ED --json-output
  
  # Execute simulation from JSON request
  python cli_gsm.py --execute-request simulation_request.json
  
  # Network simulation with parameter validation
  python cli_gsm.py --model ED --formulation F \\
    --params-inline '{"E": 30000, "S": 1.0, "c": 2.0}' \\
    --loading-inline '{"time_array": [0,1], "strain_history": [0,0.01]}' \\
    --validate-params --json-output
            """
        )
        
        # Model operations
        discovery_group = parser.add_argument_group('Model Discovery')
        discovery_group.add_argument(
            '--list-models',
            action='store_true',
            help='List all available GSM models'
        )
        
        discovery_group.add_argument(
            '--model-info',
            help='Show information about a specific model'
        )
        
        discovery_group.add_argument(
            '--get-param-spec',
            help='Get parameter specification for a GSM model (for dynamic validation)'
        )
        
        discovery_group.add_argument(
            '--test-access',
            help='Test accessing a model by key'
        )
        
        discovery_group.add_argument(
            '--show-all-keys',
            action='store_true',
            help='Show all available access keys'
        )
        
        # Network communication
        network_group = parser.add_argument_group('Network Communication')
        network_group.add_argument(
            '--execute-request',
            help='Execute simulation from JSON request file'
        )
        
        network_group.add_argument(
            '--serve',
            action='store_true',
            help='Start network server mode (for computational nodes)'
        )
        
        network_group.add_argument(
            '--port',
            type=int,
            default=8888,
            help='Port for network server (default: 8888)'
        )
        
        # Simulation execution
        sim_group = parser.add_argument_group('Simulation Execution')
        sim_group.add_argument(
            '--model',
            choices=[key for key in self.registry_dict.keys() if key.startswith('GSM1D_')],
            help='GSM model to execute'
        )
        
        sim_group.add_argument(
            '--formulation',
            choices=['F', 'G', 'Helmholtz', 'Gibbs'],
            default='F',
            help='Energy formulation: F/Helmholtz (strain-controlled) or G/Gibbs (stress-controlled)'
        )
        
        # Parameter input
        param_group = parser.add_argument_group('Parameter Input')
        param_exclusive = param_group.add_mutually_exclusive_group()
        param_exclusive.add_argument(
            '--params',
            help='Parameter source: JSON file, database URI, or network URL'
        )
        param_exclusive.add_argument(
            '--params-inline',
            help='Inline JSON string with material parameters'
        )
        
        # Loading input
        loading_group = parser.add_argument_group('Loading Input')
        loading_exclusive = loading_group.add_mutually_exclusive_group()
        loading_exclusive.add_argument(
            '--loading',
            help='Loading specification: JSON file or inline JSON string'
        )
        loading_exclusive.add_argument(
            '--loading-inline',
            help='Inline JSON string with loading specification'
        )
        
        # Validation and output
        validation_group = parser.add_argument_group('Validation and Output')
        validation_group.add_argument(
            '--validate-params',
            action='store_true',
            help='Validate parameters against GSM specification'
        )
        
        validation_group.add_argument(
            '--validate-only',
            action='store_true',
            help='Only validate inputs without running simulation'
        )
        
        validation_group.add_argument(
            '--output',
            help='Output file path (JSON format)'
        )
        
        validation_group.add_argument(
            '--config',
            help='Simulation configuration file (JSON)'
        )
        
        # Output format
        output_group = parser.add_argument_group('Output Format')
        output_group.add_argument(
            '--json-output',
            action='store_true',
            help='Output results in JSON format for network communication'
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
    
    def get_parameter_specification(self, model_key: str, json_output: bool = False) -> Dict[str, Any]:
        """Get parameter specification for a GSM model"""
        try:
            if not check_gsm_def_exists(model_key):
                available = [k for k in self.registry_dict.keys() if k.startswith('GSM1D_')]
                raise ValueError(f"Model '{model_key}' not found. Available: {available}")
            
            gsm_def_class = get_gsm_def_class(model_key)
            param_spec = GSMParameterSpec(gsm_def_class)
            
            result = {
                'model_key': model_key,
                'parameter_specification': param_spec.to_dict(),
                'status': 'success'
            }
            
            if json_output:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"Parameter Specification for '{model_key}':")
                print("=" * 60)
                print(f"Model: {param_spec.gsm_def_class.__name__}")
                print(f"Parameters:")
                for param_name, spec in param_spec.parameters.items():
                    print(f"  {param_name}:")
                    print(f"    Required: {spec.get('required', False)}")
                    print(f"    Type: {spec.get('type', 'float')}")
                    
                    if param_name in param_spec.parameter_bounds:
                        bounds = param_spec.parameter_bounds[param_name]
                        print(f"    Bounds: [{bounds[0]}, {bounds[1]}]")
                    
                    if param_name in param_spec.parameter_units:
                        print(f"    Units: {param_spec.parameter_units[param_name]}")
                    
                    if param_name in param_spec.parameter_descriptions:
                        print(f"    Description: {param_spec.parameter_descriptions[param_name]}")
                    print()
            
            return result
            
        except Exception as e:
            error_result = {
                'model_key': model_key,
                'status': 'error',
                'error': str(e)
            }
            
            if json_output:
                print(json.dumps(error_result, indent=2))
            else:
                print(f"Error getting parameter specification: {e}")
            
            return error_result
    
    def execute_from_request(self, request_file: str) -> Dict[str, Any]:
        """Execute simulation from JSON request file"""
        try:
            with open(request_file, 'r') as f:
                request_data = json.load(f)
            
            result = self.network_interface.execute_simulation(request_data)
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute request from {request_file}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message': 'Request execution failed'
            }
    
    def execute_simulation(self, model_key: str, formulation: str, 
                          params_source: str, loading_source: str,
                          config_source: Optional[str] = None,
                          validate_only: bool = False) -> Dict[str, Any]:
        """Execute simulation with parameter and loading sources"""
        try:
            # Load parameters
            if params_source.startswith('{'):
                param_data = self.parameter_loader.load_parameters_from_string(params_source)
            else:
                param_data = self.parameter_loader.load_parameters(params_source)
            
            # Load loading
            if loading_source.startswith('{'):
                loading_data = LoadingData.from_json(loading_source)
            else:
                loading_data = self.parameter_loader.load_loading(loading_source)
            
            # Load config
            if config_source:
                config_data = self.parameter_loader.load_config(config_source)
            else:
                config_data = SimulationConfig()
            
            # Create request data
            request_data = {
                'model': model_key,
                'formulation': formulation,
                'parameters': param_data.to_dict(),
                'loading': loading_data.to_dict(),
                'config': config_data.to_dict()
            }
            
            if validate_only:
                # Only validate parameters
                gsm_def_class = get_gsm_def_class(model_key)
                param_spec = GSMParameterSpec(gsm_def_class)
                is_valid, errors = param_spec.validate_parameters(param_data.parameters)
                
                return {
                    'status': 'validation_complete',
                    'valid': is_valid,
                    'errors': errors,
                    'parameter_specification': param_spec.to_dict()
                }
            else:
                # Execute simulation
                return self.network_interface.execute_simulation(request_data)
                
        except Exception as e:
            logger.error(f"Simulation execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def list_models(self, json_output: bool = False):
        """List all available models"""
        models = [key for key in self.registry_dict.keys() if key.startswith('GSM1D_')]
        
        if json_output:
            result = {
                'status': 'success',
                'models': []
            }
            
            for model_name in models:
                mechanism = model_name[6:] if model_name.startswith('GSM1D_') else "Unknown"
                desc = get_mechanism_description(mechanism)
                module_path = get_gsm_def_module_path(model_name)
                
                result['models'].append({
                    'name': model_name,
                    'mechanism': mechanism,
                    'description': desc,
                    'module_path': module_path
                })
            
            print(json.dumps(result, indent=2))
        else:
            print("Available GSM Models:")
            print("=" * 50)
            for model_name in models:
                mechanism = model_name[6:] if model_name.startswith('GSM1D_') else "Unknown"
                desc = get_mechanism_description(mechanism)
                print(f"- {model_name}: {desc}")
            
            print(f"\nTotal: {len(models)} models available")
    
    def get_parameter_specification(self, model_name: str, json_output: bool = False):
        """Get parameter specification for a model"""
        try:
            if not check_gsm_def_exists(model_name):
                error_msg = f"Model '{model_name}' not found"
                if json_output:
                    print(json.dumps({'status': 'error', 'error': error_msg}))
                else:
                    print(f"Error: {error_msg}")
                return
            
            gsm_def_class = get_gsm_def_class(model_name)
            param_spec = GSMParameterSpec(gsm_def_class)
            
            if json_output:
                result = {
                    'status': 'success',
                    'specification': param_spec.to_dict()
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"Parameter Specification for {model_name}:")
                print("=" * 50)
                spec_dict = param_spec.to_dict()
                for param_name, param_info in spec_dict['parameters'].items():
                    desc = spec_dict['parameter_descriptions'].get(param_name, 'No description')
                    bounds = spec_dict['parameter_bounds'].get(param_name, 'No bounds specified')
                    units = spec_dict['parameter_units'].get(param_name, '-')
                    print(f"- {param_name}: {desc}")
                    print(f"  Type: {param_info['type']}")
                    print(f"  Required: {param_info['required']}")
                    print(f"  Units: {units}")
                    print(f"  Bounds: {bounds}")
                    print()
                    
        except Exception as e:
            error_msg = f"Failed to get parameter specification: {e}"
            if json_output:
                print(json.dumps({'status': 'error', 'error': error_msg}))
            else:
                print(f"Error: {error_msg}")
    
    def execute_from_request(self, request_file: str):
        """Execute simulation from a JSON request file"""
        try:
            with open(request_file, 'r') as f:
                request_data = json.load(f)
            
            return self.network_interface.execute_simulation(request_data)
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
            print(json.dumps(result, indent=2))
        else:
            print("Available GSM Models:")
            print("=" * 50)
            
            for i, model_name in enumerate(models, 1):
                try:
                    mechanism = model_name[6:] if model_name.startswith('GSM1D_') else "Unknown"
                    desc = get_mechanism_description(mechanism)
                    module_path = get_gsm_def_module_path(model_name)
                    
                    print(f"{i:2d}. {model_name}")
                    print(f"    Mechanism: {mechanism}")
                    print(f"    Description: {desc}")
                    print(f"    Module: {module_path}")
                    print()
                    
                except Exception as e:
                    print(f"{i:2d}. {model_name} - Error: {e}")
            
            print(f"Total: {len(models)} models")
    
    def start_network_server(self, port: int = 8888):
        """Start network server for computational node"""
        try:
            import http.server
            import socketserver
            from urllib.parse import urlparse, parse_qs
            
            class GSMRequestHandler(http.server.BaseHTTPRequestHandler):
                def do_POST(self):
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    
                    try:
                        request_data = json.loads(post_data.decode('utf-8'))
                        result = self.server.gsm_cli.network_interface.execute_simulation(request_data)
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(result, default=str).encode('utf-8'))
                        
                    except Exception as e:
                        self.send_response(500)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        error_response = {'status': 'error', 'error': str(e)}
                        self.wfile.write(json.dumps(error_response).encode('utf-8'))
                
                def do_GET(self):
                    parsed_path = urlparse(self.path)
                    
                    if parsed_path.path == '/models':
                        # Return available models
                        models = [key for key in self.server.gsm_cli.registry_dict.keys() 
                                 if key.startswith('GSM1D_')]
                        response = {'status': 'success', 'models': models}
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode('utf-8'))
                    
                    elif parsed_path.path.startswith('/param-spec/'):
                        # Return parameter specification
                        model_key = parsed_path.path.split('/')[-1]
                        result = self.server.gsm_cli.get_parameter_specification(model_key, json_output=True)
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(result, default=str).encode('utf-8'))
                    
                    else:
                        self.send_response(404)
                        self.end_headers()
            
            with socketserver.TCPServer(("", port), GSMRequestHandler) as httpd:
                httpd.gsm_cli = self
                print(f"GSM Network Server started on port {port}")
                print(f"Endpoints:")
                print(f"  GET  /models - List available models")
                print(f"  GET  /param-spec/<model> - Get parameter specification")
                print(f"  POST / - Execute simulation (JSON request)")
                print(f"Server running at http://localhost:{port}")
                httpd.serve_forever()
                
        except Exception as e:
            logger.error(f"Failed to start network server: {e}")
            sys.exit(1)
    
    def main(self):
        """Main CLI entry point"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        # Configure logging
        if args.quiet:
            logging.getLogger().setLevel(logging.WARNING)
        elif args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        try:
            # Handle network server mode
            if args.serve:
                self.start_network_server(args.port)
                return
            
            # Handle discovery operations
            if args.list_models:
                self.list_models(args.json_output)
                return
            
            if args.get_param_spec:
                self.get_parameter_specification(args.get_param_spec, args.json_output)
                return
            
            if args.model_info:
                # Existing model info functionality
                self.show_model_info(args.model_info, args.json_output)
                return
            
            if args.test_access:
                self.test_model_access(args.test_access, args.json_output)
                return
            
            if args.show_all_keys:
                self.show_all_keys(args.json_output)
                return
            
            # Handle network request execution
            if args.execute_request:
                result = self.execute_from_request(args.execute_request)
                if args.json_output:
                    print(json.dumps(result, indent=2, default=str))
                else:
                    print(f"Execution result: {result['status']}")
                    if result['status'] == 'error':
                        print(f"Error: {result['error']}")
                return
            
            # Handle simulation execution
            if args.model and (args.params or args.params_inline) and (args.loading or args.loading_inline):
                params_source = args.params_inline if args.params_inline else args.params
                loading_source = args.loading_inline if args.loading_inline else args.loading
                
                result = self.execute_simulation(
                    args.model, args.formulation,
                    params_source, loading_source,
                    args.config, args.validate_only
                )
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    print(f"Results saved to {args.output}")
                elif args.json_output:
                    print(json.dumps(result, indent=2, default=str))
                else:
                    print(f"Simulation result: {result['status']}")
                    if result['status'] == 'error':
                        print(f"Error: {result['error']}")
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
    
    def show_model_info(self, key: str, json_output: bool = False):
        """Show detailed information about a model"""
        try:
            if not check_gsm_def_exists(key):
                error_msg = f"Model '{key}' not found"
                if json_output:
                    print(json.dumps({'status': 'error', 'error': error_msg}))
                else:
                    print(f"Error: {error_msg}")
                return
            
            mechanism = key[6:] if key.startswith('GSM1D_') else "Unknown"
            desc = get_mechanism_description(mechanism)
            module_path = get_gsm_def_module_path(key)
            gsm_def_class = get_gsm_def_class(key)
            
            if json_output:
                info = {
                    'status': 'success',
                    'model_info': {
                        'name': key,
                        'mechanism': mechanism,
                        'description': desc,
                        'module_path': module_path,
                        'class_name': gsm_def_class.__name__ if gsm_def_class else 'Unknown'
                    }
                }
                print(json.dumps(info, indent=2))
            else:
                print(f"Model Information: {key}")
                print("=" * 50)
                print(f"Mechanism: {mechanism}")
                print(f"Description: {desc}")
                print(f"Module Path: {module_path}")
                print(f"Class: {gsm_def_class.__name__ if gsm_def_class else 'Unknown'}")
                
        except Exception as e:
            error_msg = f"Failed to get model info: {e}"
            if json_output:
                print(json.dumps({'status': 'error', 'error': error_msg}))
            else:
                print(f"Error: {error_msg}")
    
    def test_model_access(self, key: str, json_output: bool = False):
        """Test accessing a model by key"""
        try:
            exists = check_gsm_def_exists(key)
            if exists:
                gsm_def_class = get_gsm_def_class(key)
                success = gsm_def_class is not None
            else:
                success = False
            
            if json_output:
                result = {
                    'status': 'success',
                    'test_result': {
                        'key': key,
                        'exists': exists,
                        'accessible': success
                    }
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"Access Test for: {key}")
                print("=" * 30)
                print(f"Exists: {exists}")
                print(f"Accessible: {success}")
                if success:
                    print("✓ Model can be accessed successfully")
                else:
                    print("✗ Model access failed")
                    
        except Exception as e:
            error_msg = f"Model access test failed: {e}"
            if json_output:
                print(json.dumps({'status': 'error', 'error': error_msg}))
            else:
                print(f"Error: {error_msg}")
    
    def show_all_keys(self, json_output: bool = False):
        """Show all available access keys"""
        all_keys = list(self.registry_dict.keys())
        gsm_keys = [key for key in all_keys if key.startswith('GSM1D_')]
        
        if json_output:
            result = {
                'status': 'success',
                'keys': {
                    'gsm_models': gsm_keys,
                    'all_keys': all_keys
                }
            }
            print(json.dumps(result, indent=2))
        else:
            print("All Available Keys:")
            print("=" * 50)
            print(f"GSM Models ({len(gsm_keys)}):")
            for key in gsm_keys:
                print(f"  - {key}")
            
            other_keys = [key for key in all_keys if not key.startswith('GSM1D_')]
            if other_keys:
                print(f"\nOther Keys ({len(other_keys)}):")
                for key in other_keys:
                    print(f"  - {key}")
            
            print(f"\nTotal: {len(all_keys)} keys available")


def main():
    """Entry point for command line execution"""
    cli = GSMDefCLI()
    cli.main()


def start_network_server(port: int = 8888):
    """Entry point for network server execution"""
    cli = GSMDefCLI()
    cli.start_network_server(port)


if __name__ == "__main__":
    main()
