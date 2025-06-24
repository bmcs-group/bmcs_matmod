"""
Parameter loading utilities for GSM CLI interface

This module provides functionality to load material parameters, loading specifications,
and configurations from various sources including JSON files, databases, network URLs,
and inline strings.
"""

import json
import sqlite3
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import numpy as np

from .data_structures import MaterialParameterData, LoadingData, SimulationConfig

logger = logging.getLogger(__name__)

class ParameterLoader:
    """Loads parameters from various sources"""
    
    def __init__(self):
        self.supported_schemes = ['file', 'http', 'https', 'db', 'sqlite']
    
    def load_parameters(self, source: str) -> MaterialParameterData:
        """Load material parameters from various sources"""
        try:
            if source.startswith('http://') or source.startswith('https://'):
                return self._load_parameters_from_url(source)
            elif source.startswith('db://'):
                return self._load_parameters_from_database(source)
            elif source.startswith('sqlite://'):
                return self._load_parameters_from_sqlite(source)
            elif source.startswith('{') and source.endswith('}'):
                # Inline JSON
                return self.load_parameters_from_string(source)
            else:
                # Assume file path
                return self._load_parameters_from_file(source)
                
        except Exception as e:
            logger.error(f"Failed to load parameters from {source}: {e}")
            raise
    
    def load_parameters_from_string(self, json_str: str) -> MaterialParameterData:
        """Load parameters from JSON string"""
        try:
            data = json.loads(json_str)
            
            # If data is just a dict of parameter values, wrap it
            if 'parameters' not in data:
                data = {'parameters': data}
            
            return MaterialParameterData.from_dict(data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in parameter string: {e}")
            raise
    
    def _load_parameters_from_file(self, filepath: str) -> MaterialParameterData:
        """Load parameters from JSON file"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Parameter file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Add source information
        if isinstance(data, dict):
            data['source'] = str(path.absolute())
        
        return MaterialParameterData.from_dict(data)
    
    def _load_parameters_from_url(self, url: str) -> MaterialParameterData:
        """Load parameters from HTTP/HTTPS URL"""
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            
            # Add source information
            if isinstance(data, dict):
                data['source'] = url
            
            return MaterialParameterData.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load parameters from URL {url}: {e}")
            raise
    
    def _load_parameters_from_database(self, db_uri: str) -> MaterialParameterData:
        """Load parameters from database URI
        
        Format: db://host:port/database?table=materials&id=123
        """
        try:
            # Parse database URI
            parsed = urllib.parse.urlparse(db_uri)
            query_params = urllib.parse.parse_qs(parsed.query)
            
            # This is a placeholder implementation
            # In practice, you would use appropriate database drivers
            # (PostgreSQL, MySQL, etc.)
            
            # For now, simulate loading from database
            logger.warning("Database loading not fully implemented. Using mock data.")
            
            # Mock database response
            mock_data = {
                'parameters': {
                    'E': 30000.0,
                    'nu': 0.2,
                    'omega_0': 0.1
                },
                'source': db_uri,
                'material_name': f"Material_{query_params.get('id', ['unknown'])[0]}",
                'model_type': 'ElasticDamage'
            }
            
            return MaterialParameterData.from_dict(mock_data)
            
        except Exception as e:
            logger.error(f"Failed to load parameters from database {db_uri}: {e}")
            raise
    
    def _load_parameters_from_sqlite(self, sqlite_uri: str) -> MaterialParameterData:
        """Load parameters from SQLite database
        
        Format: sqlite:///path/to/db.sqlite?table=materials&id=123
        """
        try:
            # Parse SQLite URI
            parsed = urllib.parse.urlparse(sqlite_uri)
            db_path = parsed.path
            query_params = urllib.parse.parse_qs(parsed.query)
            
            table = query_params.get('table', ['materials'])[0]
            material_id = query_params.get('id', [None])[0]
            
            if not material_id:
                raise ValueError("Material ID must be specified in SQLite URI")
            
            # Connect to SQLite database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query for material parameters
            cursor.execute(f"SELECT * FROM {table} WHERE id = ?", (material_id,))
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Material with ID {material_id} not found in {table}")
            
            # Get column names
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Create parameter dictionary
            param_data = dict(zip(columns, row))
            
            # Extract parameters (exclude metadata columns)
            metadata_columns = {'id', 'name', 'description', 'created_date', 'source'}
            parameters = {k: v for k, v in param_data.items() 
                         if k not in metadata_columns and isinstance(v, (int, float))}
            
            conn.close()
            
            # Create MaterialParameterData
            data = {
                'parameters': parameters,
                'source': sqlite_uri,
                'material_name': param_data.get('name'),
                'description': param_data.get('description')
            }
            
            return MaterialParameterData.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load parameters from SQLite {sqlite_uri}: {e}")
            raise
    
    def load_loading(self, source: str) -> LoadingData:
        """Load loading specification"""
        try:
            if source.startswith('http://') or source.startswith('https://'):
                return self._load_loading_from_url(source)
            elif source.startswith('{') and source.endswith('}'):
                # Inline JSON
                return self._load_loading_from_string(source)
            else:
                # Assume file path
                return self._load_loading_from_file(source)
                
        except Exception as e:
            logger.error(f"Failed to load loading specification from {source}: {e}")
            raise
    
    def _load_loading_from_file(self, filepath: str) -> LoadingData:
        """Load loading from JSON file"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Loading file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return LoadingData.from_dict(data)
    
    def _load_loading_from_url(self, url: str) -> LoadingData:
        """Load loading from HTTP/HTTPS URL"""
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            
            return LoadingData.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load loading from URL {url}: {e}")
            raise
    
    def _load_loading_from_string(self, json_str: str) -> LoadingData:
        """Load loading from JSON string"""
        try:
            data = json.loads(json_str)
            return LoadingData.from_dict(data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in loading string: {e}")
            raise
    
    def load_config(self, source: str) -> SimulationConfig:
        """Load simulation configuration"""
        try:
            if source.startswith('http://') or source.startswith('https://'):
                return self._load_config_from_url(source)
            elif source.startswith('{') and source.endswith('}'):
                # Inline JSON
                return self._load_config_from_string(source)
            else:
                # Assume file path
                return self._load_config_from_file(source)
                
        except Exception as e:
            logger.error(f"Failed to load configuration from {source}: {e}")
            raise
    
    def _load_config_from_file(self, filepath: str) -> SimulationConfig:
        """Load configuration from JSON file"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return SimulationConfig.from_dict(data)
    
    def _load_config_from_url(self, url: str) -> SimulationConfig:
        """Load configuration from HTTP/HTTPS URL"""
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            
            return SimulationConfig.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load configuration from URL {url}: {e}")
            raise
    
    def _load_config_from_string(self, json_str: str) -> SimulationConfig:
        """Load configuration from JSON string"""
        try:
            data = json.loads(json_str)
            return SimulationConfig.from_dict(data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration string: {e}")
            raise

# AiiDA integration utilities (if AiiDA is available)
try:
    from aiida import orm
    AIIDA_AVAILABLE = True
except ImportError:
    AIIDA_AVAILABLE = False

class AiidaParameterLoader:
    """Load parameters from AiiDA database"""
    
    def __init__(self):
        if not AIIDA_AVAILABLE:
            raise ImportError("AiiDA is not available. Install aiida-core to use this functionality.")
    
    def load_parameters_from_node(self, node_id: int) -> MaterialParameterData:
        """Load parameters from AiiDA node"""
        try:
            node = orm.load_node(node_id)
            
            if isinstance(node, orm.Dict):
                data = node.get_dict()
            else:
                raise ValueError(f"Node {node_id} is not a Dict node")
            
            # Add AiiDA metadata
            if 'source' not in data:
                data['source'] = f"aiida://node/{node_id}"
            
            return MaterialParameterData.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load parameters from AiiDA node {node_id}: {e}")
            raise
    
    def load_parameters_from_query(self, **query_kwargs) -> List[MaterialParameterData]:
        """Load parameters from AiiDA query"""
        try:
            # Example query for Dict nodes containing material parameters
            qb = orm.QueryBuilder()
            qb.append(orm.Dict, filters=query_kwargs)
            
            results = []
            for (node,) in qb.all():
                data = node.get_dict()
                data['source'] = f"aiida://node/{node.id}"
                results.append(MaterialParameterData.from_dict(data))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to query parameters from AiiDA: {e}")
            raise

# Example data generators for testing

def generate_example_parameters() -> MaterialParameterData:
    """Generate example material parameters"""
    return MaterialParameterData(
        parameters={
            'E': 30000.0,      # Young's modulus
            'nu': 0.2,         # Poisson's ratio
            'omega_0': 0.1,    # Initial damage
            'S': 1000.0,       # Damage parameter
            'r': 0.01          # Damage evolution rate
        },
        material_name="Example Concrete",
        model_type="ElasticDamage",
        description="Example parameters for elastic-damage model",
        units={
            'E': 'MPa',
            'nu': '-',
            'omega_0': '-',
            'S': 'MPa',
            'r': '-'
        },
        parameter_bounds={
            'E': (1000.0, 100000.0),
            'nu': (0.0, 0.5),
            'omega_0': (0.0, 1.0),
            'S': (100.0, 10000.0),
            'r': (0.001, 0.1)
        }
    )

def generate_example_loading(loading_type: str = 'monotonic') -> LoadingData:
    """Generate example loading specification"""
    if loading_type == 'monotonic':
        time_array = np.linspace(0, 1, 100)
        strain_history = np.linspace(0, 0.01, 100)  # 1% strain
        
        return LoadingData(
            time_array=time_array,
            strain_history=strain_history,
            loading_type='monotonic',
            max_amplitude=0.01,
            description="Monotonic tensile loading to 1% strain"
        )
    
    elif loading_type == 'cyclic':
        time_array = np.linspace(0, 10, 1000)  # 10 cycles
        strain_history = 0.005 * np.sin(2 * np.pi * time_array)  # ±0.5% strain
        
        return LoadingData(
            time_array=time_array,
            strain_history=strain_history,
            loading_type='cyclic',
            max_amplitude=0.005,
            frequency=1.0,
            description="Cyclic loading ±0.5% strain at 1 Hz"
        )
    
    else:
        raise ValueError(f"Unknown loading type: {loading_type}")

def generate_example_config() -> SimulationConfig:
    """Generate example simulation configuration"""
    return SimulationConfig(
        tolerance=1e-6,
        max_iterations=100,
        step_size_control=True,
        save_internal_variables=True,
        debug_output=False
    )
