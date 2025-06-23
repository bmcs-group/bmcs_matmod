"""
AiiDA data types for GSM material modeling.

This module provides specialized data types for material parameters
and loading histories used in GSM simulations.
"""

from aiida import orm
import json
import numpy as np


class GSMMaterialData(orm.Data):
    """Data type for GSM material parameters with validation"""

    def __init__(self, parameters=None, model=None, **kwargs):
        super().__init__(**kwargs)
        if parameters is not None:
            self.set_parameters(parameters, model)

    def set_parameters(self, parameters, model=None):
        """Set material parameters with optional model validation"""
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary")
        
        self.base.attributes.set('parameters', parameters)
        if model:
            self.base.attributes.set('gsm_model', model)
    
    def get_parameters(self):
        """Get material parameters"""
        return self.base.attributes.get('parameters', {})
    
    def get_model(self):
        """Get associated GSM model"""
        return self.base.attributes.get('gsm_model', None)
    
    def validate_parameters(self, gsm_model=None):
        """Validate parameters against GSM model specification"""
        # This could call the CLI validation functionality
        model = gsm_model or self.get_model()
        if not model:
            raise ValueError("GSM model must be specified for validation")
        
        # Implementation would call CLI parameter validation
        # For now, basic type checking
        params = self.get_parameters()
        for key, value in params.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Parameter {key} must be numeric, got {type(value)}")
        
        return True


class GSMLoadingData(orm.Data):
    """Data type for GSM loading histories"""

    def __init__(self, loading_data=None, **kwargs):
        super().__init__(**kwargs)
        if loading_data is not None:
            self.set_loading_data(loading_data)

    def set_loading_data(self, loading_data):
        """Set loading history data"""
        if not isinstance(loading_data, dict):
            raise ValueError("Loading data must be a dictionary")
        
        # Validate required fields
        if 'time_array' not in loading_data:
            raise ValueError("Loading data must contain 'time_array'")
        
        # Convert numpy arrays to lists for storage
        stored_data = {}
        for key, value in loading_data.items():
            if isinstance(value, np.ndarray):
                stored_data[key] = value.tolist()
            else:
                stored_data[key] = value
        
        self.base.attributes.set('loading_data', stored_data)
    
    def get_loading_data(self):
        """Get loading history data"""
        return self.base.attributes.get('loading_data', {})
    
    def get_time_array(self):
        """Get time array as numpy array"""
        loading_data = self.get_loading_data()
        time_array = loading_data.get('time_array', [])
        return np.array(time_array)
    
    def get_history(self, variable_name):
        """Get specific loading history variable as numpy array"""
        loading_data = self.get_loading_data()
        history = loading_data.get(variable_name, [])
        return np.array(history)
    
    def get_loading_type(self):
        """Determine the type of loading (strain-controlled, stress-controlled, etc.)"""
        loading_data = self.get_loading_data()
        
        if 'strain_history' in loading_data:
            return 'strain_controlled'
        elif 'stress_history' in loading_data:
            return 'stress_controlled'
        elif 'displacement_history' in loading_data:
            return 'displacement_controlled'
        else:
            return 'unknown'
