"""
AiiDA parser for GSM simulation results.

This module provides parsing functionality for GSM simulation outputs,
converting CLI JSON results into AiiDA data structures.
"""

from aiida import orm
from aiida.parsers.parser import Parser
from aiida.common import exceptions
import json
import numpy as np


class GSMParser(Parser):
    """Parser for GSM simulation calculation outputs"""

    def parse(self, **kwargs):
        """Parse the outputs of a GSM simulation calculation"""
        
        try:
            output_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        # Check for required output files
        files_retrieved = output_folder.list_object_names()
        if 'results.json' not in files_retrieved:
            return self.exit_codes.ERROR_OUTPUT_FILES

        # Parse main results file
        try:
            with output_folder.open('results.json', 'r') as f:
                results_data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to parse results.json: {e}")
            return self.exit_codes.ERROR_OUTPUT_FILES

        # Check if simulation was successful
        if not results_data.get('success', False):
            error_msg = results_data.get('error', 'Unknown simulation error')
            self.logger.error(f"Simulation failed: {error_msg}")
            return self.exit_codes.ERROR_SIMULATION_FAILED

        # Extract results
        simulation_results = results_data.get('results', {})
        response_data = results_data.get('response_data', {})

        # Create output dictionary with metadata
        output_dict = {
            'model': results_data.get('model'),
            'formulation': results_data.get('formulation'),
            'execution_time': results_data.get('execution_time'),
            'timestamp': results_data.get('timestamp'),
            'parameters_validated': results_data.get('parameters_validated', False)
        }

        # Add simulation results
        if simulation_results:
            output_dict.update(simulation_results)

        self.out('results', orm.Dict(dict=output_dict))

        # Process response data arrays if available
        if response_data and isinstance(response_data, dict):
            try:
                # Convert response data to ArrayData
                array_data = orm.ArrayData()
                
                for key, values in response_data.items():
                    if isinstance(values, list):
                        array_data.set_array(key, np.array(values))
                    elif isinstance(values, dict) and 'data' in values:
                        array_data.set_array(key, np.array(values['data']))
                        # Store metadata if available
                        if 'units' in values:
                            array_data.set_attribute(f'{key}_units', values['units'])
                        if 'description' in values:
                            array_data.set_attribute(f'{key}_description', values['description'])

                self.out('response_data', array_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to process response data arrays: {e}")

        return None  # Success
