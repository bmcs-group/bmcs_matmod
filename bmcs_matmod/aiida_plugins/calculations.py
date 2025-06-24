"""
AiiDA calculation classes for GSM simulations.

This module provides the calculation plugin for running GSM simulations
through AiiDA, using the CLI backend for execution.
"""

from aiida import orm
from aiida.common import datastructures, exceptions
from aiida.engine import CalcJob
from aiida.plugins import DataFactory
import json
import os
from pathlib import Path


class GSMSimulationCalculation(CalcJob):
    """AiiDA calculation for GSM material model simulations"""
    
    # Default output file for 'verdi calcjob outputcat'
    _DEFAULT_OUTPUT_FILE = 'gsm_simulation.out'
    
    # Default input file for 'verdi calcjob inputcat'
    _DEFAULT_INPUT_FILE = 'simulation_request.json'
    
    # Additional output files that can be accessed
    _OUTPUT_FILES = {
        'stdout': 'gsm_simulation.out',
        'stderr': 'gsm_simulation.err', 
        'results': 'results.json',
        'input': 'simulation_request.json'
    }

    @classmethod
    def define(cls, spec):
        """Define the process specification"""
        super().define(spec)
        
        # Inputs
        spec.input('metadata.options.resources', valid_type=dict, default={'num_machines': 1, 'num_mpiprocs_per_machine': 1})
        spec.input('metadata.options.max_wallclock_seconds', valid_type=int, default=3600)
        spec.input('metadata.options.parser_name', valid_type=str, default='gsm.parser')
        
        spec.input('gsm_model', valid_type=orm.Str, help='GSM model identifier (e.g., GSM1D_ED)')
        spec.input('formulation', valid_type=orm.Str, help='Model formulation (e.g., F, G, Helmholtz, Gibbs)')
        spec.input('material_parameters', valid_type=orm.Dict, help='Material parameter dictionary')
        spec.input('loading_data', valid_type=orm.Dict, help='Loading history data')
        spec.input('simulation_config', valid_type=orm.Dict, required=False, 
                  help='Optional simulation configuration')
        
        # Outputs
        spec.output('results', valid_type=orm.Dict, help='Simulation results')
        spec.output('response_data', valid_type=orm.ArrayData, required=False, 
                   help='Time-dependent response arrays')
        
        # Exit codes
        spec.exit_code(300, 'ERROR_NO_RETRIEVED_FOLDER', 
                      message='The retrieved folder data node could not be accessed.')
        spec.exit_code(301, 'ERROR_OUTPUT_FILES', 
                      message='Required output files not found.')
        spec.exit_code(302, 'ERROR_SIMULATION_FAILED', 
                      message='GSM simulation failed during execution.')
        spec.exit_code(303, 'ERROR_PARAMETER_VALIDATION', 
                      message='Material parameters failed validation.')
        spec.exit_code(304, 'ERROR_INVALID_INPUT', 
                      message='Invalid input parameters provided.')
        spec.exit_code(305, 'ERROR_GSM_MODEL_NOT_FOUND', 
                      message='Specified GSM model not available.')

    def prepare_for_submission(self, folder):
        """Prepare the calculation for submission"""
        
        # Import the registry functions
        from bmcs_matmod.gsm_lagrange.gsm_def_registry import get_gsm_defs
        
        # Create simulation request JSON
        # Extract parameters for the specific model
        all_material_params = self.inputs.material_parameters.get_dict()
        model_key = self.inputs.gsm_model.value
        
        # Get model configuration from registry to determine valid parameters
        try:
            gsm_catalog = get_gsm_defs(debug=False)
            if model_key in gsm_catalog:
                # Get the GSMDef class to inspect its parameters
                gsm_def_class = gsm_catalog[model_key]
                # For now, use all provided parameters (could be refined to check GSMDef signature)
                model_parameters = all_material_params
            else:
                # If model not found in registry, use all parameters (backward compatibility)
                model_parameters = all_material_params
        except Exception as e:
            self.logger.warning(f"Could not load GSM registry: {e}. Using all parameters.")
            model_parameters = all_material_params
        
        # Filter loading data to only include valid LoadingData fields
        all_loading_data = self.inputs.loading_data.get_dict()
        valid_loading_fields = {
            'time', 'strain', 'stress', 'max_strain', 'min_strain', 'n_steps', 'type',
            'strain_rate', 'cycles', 'R_ratio', 'loading_type', 'loading_history'
        }
        filtered_loading_data = {k: v for k, v in all_loading_data.items() if k in valid_loading_fields}
        
        simulation_request = {
            "model": model_key,
            "formulation": self.inputs.formulation.value,
            "parameters": {"parameters": model_parameters},  # Wrap in parameters field
            "loading": filtered_loading_data  # Use "loading" key as expected by CLI
        }
        
        if 'simulation_config' in self.inputs:
            simulation_request["simulation_config"] = self.inputs.simulation_config.get_dict()
        
        # Write request to file with error handling
        try:
            self.logger.info(f"Creating simulation_request.json with content: {simulation_request}")
            with folder.open('simulation_request.json', 'w') as f:
                json.dump(simulation_request, f, indent=2)
            self.logger.info("Successfully created simulation_request.json")
        except Exception as e:
            self.logger.error(f"Failed to create simulation_request.json: {e}")
            raise
        
        # Create calculation info
        codeinfo = datastructures.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        
        # Check if we're using Python module execution
        if str(self.inputs.code.filepath_executable).endswith('python'):
            # For Python module execution: python -m bmcs_matmod.gsm_lagrange.cli_gsm [args]
            codeinfo.cmdline_params = [
                '-m', 'bmcs_matmod.gsm_lagrange.cli_gsm',
                '--execute-request', 'simulation_request.json',
                '--json-output',
                '--validate-only' if self.inputs.get('validate_only', False) else '--output', 'results.json'
            ]
        else:
            # For direct executable: gsm_cli [args]
            codeinfo.cmdline_params = [
                '--execute-request', 'simulation_request.json',
                '--json-output',
                '--validate-only' if self.inputs.get('validate_only', False) else '--output', 'results.json'
            ]
        codeinfo.stdout_name = 'gsm_simulation.out'
        codeinfo.stderr_name = 'gsm_simulation.err'

        # Create calculation info
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = []
        calcinfo.retrieve_list = [
            'gsm_simulation.out',
            'gsm_simulation.err',
            'results.json',  # Expected output from CLI
            'simulation_request.json'  # Include input file for debugging
        ]
        calcinfo.retrieve_temporary_list = []

        return calcinfo
