"""
AiiDA data exporters for GSM simulation results.

This module provides exporters for converting AiiDA data
to various formats for analysis and visualization.
"""

from aiida import orm
import json
import numpy as np
import csv
from pathlib import Path


class GSMJSONExporter:
    """Export GSM simulation data to JSON format"""
    
    @staticmethod
    def export_simulation_results(node, filepath=None):
        """Export simulation results to JSON"""
        if not isinstance(node, orm.Dict):
            raise ValueError("Node must be a Dict data type")
        
        data = node.get_dict()
        
        # Add metadata
        export_data = {
            'aiida_node_uuid': node.uuid,
            'creation_time': node.ctime.isoformat(),
            'data': data
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            return filepath
        else:
            return json.dumps(export_data, indent=2)
    
    @staticmethod
    def export_array_data(node, filepath=None):
        """Export ArrayData to JSON with arrays"""
        if not isinstance(node, orm.ArrayData):
            raise ValueError("Node must be an ArrayData type")
        
        export_data = {
            'aiida_node_uuid': node.uuid,
            'creation_time': node.ctime.isoformat(),
            'arrays': {},
            'attributes': dict(node.attributes)
        }
        
        # Export all arrays
        for array_name in node.get_arraynames():
            array = node.get_array(array_name)
            export_data['arrays'][array_name] = array.tolist()
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            return filepath
        else:
            return json.dumps(export_data, indent=2)
    
    @staticmethod
    def export_sn_curve(node, filepath=None, format='json'):
        """Export S-N curve data to JSON or CSV"""
        if not isinstance(node, orm.ArrayData):
            raise ValueError("Node must be an ArrayData type")
        
        stress = node.get_array('stress_amplitude')
        cycles = node.get_array('cycles_to_failure')
        
        if format.lower() == 'csv':
            if not filepath:
                filepath = 'sn_curve.csv'
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Stress_Amplitude_MPa', 'Cycles_to_Failure'])
                for s, n in zip(stress, cycles):
                    writer.writerow([s, n])
            return filepath
        
        else:  # JSON format
            export_data = {
                'aiida_node_uuid': node.uuid,
                'creation_time': node.ctime.isoformat(),
                'sn_curve': {
                    'stress_amplitude': stress.tolist(),
                    'cycles_to_failure': cycles.tolist()
                },
                'metadata': dict(node.attributes)
            }
            
            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                return filepath
            else:
                return json.dumps(export_data, indent=2)
