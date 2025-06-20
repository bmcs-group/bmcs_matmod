"""
Data structures for GSM CLI interface

This module defines typed data structures for material parameters, loading specifications,
simulation configurations, and results that can be serialized/deserialized from various
sources including JSON files, databases, and network transfers.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class MaterialParameterData:
    """Container for material parameters with metadata"""
    
    # Core parameter data
    parameters: Dict[str, float]
    
    # Metadata
    source: Optional[str] = None
    material_name: Optional[str] = None
    model_type: Optional[str] = None
    calibration_date: Optional[str] = None
    units: Optional[Dict[str, str]] = None
    description: Optional[str] = None
    
    # Parameter constraints/bounds
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    
    # Validation info
    validation_data: Optional[Dict[str, Any]] = None
    
    def validate(self) -> bool:
        """Validate parameter data"""
        try:
            # Check all parameters are numeric
            for key, value in self.parameters.items():
                if not isinstance(value, (int, float)):
                    logger.error(f"Parameter {key} is not numeric: {value}")
                    return False
                
                # Check bounds if specified
                if self.parameter_bounds and key in self.parameter_bounds:
                    min_val, max_val = self.parameter_bounds[key]
                    if not (min_val <= value <= max_val):
                        logger.error(f"Parameter {key}={value} outside bounds [{min_val}, {max_val}]")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaterialParameterData':
        """Create from dictionary"""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MaterialParameterData':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

@dataclass
class LoadingData:
    """Container for loading specifications"""
    
    # Time array
    time_array: np.ndarray
    
    # Loading histories (one or both depending on formulation)
    strain_history: Optional[np.ndarray] = None  # For Helmholtz (F) formulation
    stress_history: Optional[np.ndarray] = None  # For Gibbs (G) formulation
    
    # Loading metadata
    loading_type: Optional[str] = None  # 'monotonic', 'cyclic', 'creep', etc.
    loading_rate: Optional[float] = None
    max_amplitude: Optional[float] = None
    frequency: Optional[float] = None  # For cyclic loading
    
    # Environmental conditions
    temperature: Optional[Union[float, np.ndarray]] = None
    humidity: Optional[Union[float, np.ndarray]] = None
    
    # Description
    description: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate loading data"""
        try:
            # Check time array
            if self.time_array is None or len(self.time_array) == 0:
                logger.error("Time array is empty or None")
                return False
            
            # Check monotonically increasing time
            if not np.all(np.diff(self.time_array) >= 0):
                logger.error("Time array is not monotonically increasing")
                return False
            
            # Check that at least one loading history is provided
            if self.strain_history is None and self.stress_history is None:
                logger.error("Either strain_history or stress_history must be provided")
                return False
            
            # Validate strain history if provided
            if self.strain_history is not None:
                if len(self.strain_history) != len(self.time_array):
                    logger.error("Strain history length doesn't match time array length")
                    return False
            
            # Validate stress history if provided
            if self.stress_history is not None:
                if len(self.stress_history) != len(self.time_array):
                    logger.error("Stress history length doesn't match time array length")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Loading validation error: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.time_array is not None:
            data['time_array'] = self.time_array.tolist()
        if self.strain_history is not None:
            data['strain_history'] = self.strain_history.tolist()
        if self.stress_history is not None:
            data['stress_history'] = self.stress_history.tolist()
        if isinstance(self.temperature, np.ndarray):
            data['temperature'] = self.temperature.tolist()
        if isinstance(self.humidity, np.ndarray):
            data['humidity'] = self.humidity.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoadingData':
        """Create from dictionary"""
        # Convert lists back to numpy arrays
        if 'time_array' in data and data['time_array'] is not None:
            data['time_array'] = np.array(data['time_array'])
        if 'strain_history' in data and data['strain_history'] is not None:
            data['strain_history'] = np.array(data['strain_history'])
        if 'stress_history' in data and data['stress_history'] is not None:
            data['stress_history'] = np.array(data['stress_history'])
        if 'temperature' in data and isinstance(data['temperature'], list):
            data['temperature'] = np.array(data['temperature'])
        if 'humidity' in data and isinstance(data['humidity'], list):
            data['humidity'] = np.array(data['humidity'])
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LoadingData':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

@dataclass
class SimulationConfig:
    """Configuration for GSM simulation"""
    
    # Numerical parameters
    tolerance: float = 1e-6
    max_iterations: int = 100
    step_size_control: bool = True
    min_step_size: float = 1e-8
    max_step_size: float = 1.0
    
    # Output control
    output_frequency: int = 1  # Every nth step
    save_internal_variables: bool = True
    save_stiffness_matrix: bool = False
    
    # Parallel computation
    use_parallel: bool = False
    num_threads: Optional[int] = None
    
    # Debugging
    debug_output: bool = False
    convergence_history: bool = False
    
    def validate(self) -> bool:
        """Validate configuration"""
        try:
            if self.tolerance <= 0:
                logger.error("Tolerance must be positive")
                return False
            
            if self.max_iterations <= 0:
                logger.error("Max iterations must be positive")
                return False
            
            if self.output_frequency <= 0:
                logger.error("Output frequency must be positive")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class SimulationResults:
    """Container for simulation results"""
    
    # Model identification
    model_name: str
    formulation: str  # 'F' or 'G'
    
    # Input data
    parameters: Dict[str, float]
    loading: LoadingData
    config: SimulationConfig
    
    # Response data (from GSM engine)
    response: Any  # ResponseData object from gsm engine
    
    # Metadata
    execution_time: Optional[float] = None
    convergence_info: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {
            'model_name': self.model_name,
            'formulation': self.formulation,
            'parameters': self.parameters,
            'loading': self.loading.to_dict(),
            'config': self.config.to_dict(),
            'execution_time': self.execution_time,
            'convergence_info': self.convergence_info,
            'warnings': self.warnings
        }
        
        # Convert response data if available
        if self.response is not None:
            if hasattr(self.response, 'to_dict'):
                data['response'] = self.response.to_dict()
            else:
                # Try to extract main arrays
                try:
                    data['response'] = {
                        'time': self.response.t_t.tolist() if hasattr(self.response, 't_t') else None,
                        'strain': self.response.eps_t.tolist() if hasattr(self.response, 'eps_t') else None,
                        'stress': self.response.sig_t.tolist() if hasattr(self.response, 'sig_t') else None,
                        'internal_variables': self.response.Eps_t_flat.tolist() if hasattr(self.response, 'Eps_t_flat') else None,
                        'thermodynamic_forces': self.response.Sig_t_flat.tolist() if hasattr(self.response, 'Sig_t_flat') else None,
                    }
                except Exception as e:
                    logger.warning(f"Could not serialize response data: {e}")
                    data['response'] = None
        
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save results to file"""
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                f.write(self.to_json())
        else:
            raise ValueError("Only JSON output format is currently supported")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationResults':
        """Create from dictionary"""
        # Reconstruct complex objects
        data['loading'] = LoadingData.from_dict(data['loading'])
        data['config'] = SimulationConfig.from_dict(data['config'])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SimulationResults':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

# Utility functions for data conversion

def create_monotonic_loading(max_strain: float, n_steps: int = 100, 
                           loading_type: str = 'strain') -> LoadingData:
    """Create monotonic loading specification"""
    time_array = np.linspace(0, 1, n_steps)
    
    if loading_type == 'strain':
        strain_history = np.linspace(0, max_strain, n_steps)
        return LoadingData(
            time_array=time_array,
            strain_history=strain_history,
            loading_type='monotonic',
            max_amplitude=max_strain
        )
    elif loading_type == 'stress':
        stress_history = np.linspace(0, max_strain, n_steps)  # max_strain interpreted as max_stress
        return LoadingData(
            time_array=time_array,
            stress_history=stress_history,
            loading_type='monotonic',
            max_amplitude=max_strain
        )
    else:
        raise ValueError("loading_type must be 'strain' or 'stress'")

def create_cyclic_loading(amplitude: float, n_cycles: int = 5, steps_per_cycle: int = 20,
                         loading_type: str = 'strain') -> LoadingData:
    """Create cyclic loading specification"""
    n_steps = n_cycles * steps_per_cycle
    time_array = np.linspace(0, n_cycles, n_steps)
    
    if loading_type == 'strain':
        strain_history = amplitude * np.sin(2 * np.pi * time_array)
        return LoadingData(
            time_array=time_array,
            strain_history=strain_history,
            loading_type='cyclic',
            max_amplitude=amplitude,
            frequency=1.0
        )
    elif loading_type == 'stress':
        stress_history = amplitude * np.sin(2 * np.pi * time_array)
        return LoadingData(
            time_array=time_array,
            stress_history=stress_history,
            loading_type='cyclic',
            max_amplitude=amplitude,
            frequency=1.0
        )
    else:
        raise ValueError("loading_type must be 'strain' or 'stress'")
