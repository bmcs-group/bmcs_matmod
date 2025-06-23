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

    @classmethod
    def define(cls, spec):
        """Define the process specification"""
        super().define(spec)
        
        # Inputs
        spec.input('metadata.options.resources', valid_type=dict, default={'num_machines': 1, 'num_mpiprocs_per_machine': 1})
        spec.input('metadata.options.max_wallclock_seconds', valid_type=int, default=3600)
        spec.input('metadata.options.parser_name', valid_type=str, default='gsm_parser')
        
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
        
        # Create simulation request JSON
        simulation_request = {
            "model": self.inputs.gsm_model.value,
            "formulation": self.inputs.formulation.value,
            "material_parameters": self.inputs.material_parameters.get_dict(),
            "loading_data": self.inputs.loading_data.get_dict()
        }
        
        if 'simulation_config' in self.inputs:
            simulation_request["simulation_config"] = self.inputs.simulation_config.get_dict()
        
        # Write request to file
        with folder.open('simulation_request.json', 'w') as f:
            json.dump(simulation_request, f, indent=2)
        
        # Create calculation info
        codeinfo = datastructures.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
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
            'results.json'  # Expected output from CLI
        ]
        calcinfo.retrieve_temporary_list = []

        return calcinfo
